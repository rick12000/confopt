import pytest
import numpy as np
from typing import Dict, Tuple, Optional

from confopt.tuning import ConformalTuner
from confopt.wrapping import CategoricalRange
from confopt.selection.acquisition import QuantileConformalSearcher, LowerBoundSampler

DRAW_OR_WIN_RATE_THRESHOLD = 0.75
WINDOW_SIZE = 20
TARGET_ALPHAS = [0.25, 0.5, 0.75]
ADAPTER_TYPES = ["DtACI", "ACI"]


def complex_objective(configuration: Dict) -> float:
    x1 = configuration["x1"]
    x2 = configuration["x2"]
    categorical_val = {"A": 1.0, "B": 2.5, "C": 4.0}[configuration["categorical"]]

    term1 = np.sin(x1 * np.pi) * np.cos(x2 * np.pi)
    term2 = 0.5 * (x1 - 0.3) ** 2 + 0.8 * (x2 - 0.7) ** 2
    term3 = categorical_val * np.exp(-((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2))

    return term1 + term2 + term3 + np.random.normal(0, 0.05)


def calculate_coverage_rate_from_study(study) -> float:
    breach_count = 0
    total_intervals = 0

    for trial in study.trials:
        if trial.lower_bound is not None and trial.upper_bound is not None:
            total_intervals += 1
            if not (trial.lower_bound <= trial.performance <= trial.upper_bound):
                breach_count += 1

    return 1 - (breach_count / total_intervals)


def calculate_windowed_deviations_from_study(
    study, alpha: float, window_size: int
) -> float:
    target_coverage = 1 - alpha
    trials = [
        t
        for t in study.trials
        if t.lower_bound is not None and t.upper_bound is not None
    ]

    if len(trials) < window_size:
        return 0.0

    n_windows = len(trials) // window_size
    deviations = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_trials = trials[start_idx:end_idx]

        breaches = sum(
            1
            for t in window_trials
            if not (t.lower_bound <= t.performance <= t.upper_bound)
        )
        window_coverage = 1 - (breaches / window_size)
        deviation = abs(window_coverage - target_coverage)
        deviations.append(deviation)

    return np.mean(deviations)


def run_experiment(
    adapter_type: Optional[str], seed: int, alpha: float
) -> Tuple[float, float]:
    np.random.seed(seed)

    search_space = {
        "x1": CategoricalRange(choices=np.linspace(0, 1, 15).tolist()),
        "x2": CategoricalRange(choices=np.linspace(0, 1, 15).tolist()),
        "categorical": CategoricalRange(choices=["A", "B", "C"]),
    }

    interval_width = 1 - alpha

    sampler = LowerBoundSampler(
        interval_width=interval_width,
        adapter=adapter_type,
        c=0,
    )

    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="qgbm",
        sampler=sampler,
        n_pre_conformal_trials=32,
        calibration_split_strategy="train_test_split",
    )

    tuner = ConformalTuner(
        objective_function=complex_objective,
        search_space=search_space,
        metric_optimization="minimize",
        n_candidate_configurations=2000,
        dynamic_sampling=True,
    )

    tuner.tune(
        n_random_searches=15,
        conformal_retraining_frequency=1,
        searcher=searcher,
        random_state=seed,
        max_searches=60,
        verbose=False,
    )

    coverage_rate = calculate_coverage_rate_from_study(tuner.study)
    windowed_deviation = calculate_windowed_deviations_from_study(
        tuner.study, alpha, WINDOW_SIZE
    )

    return coverage_rate, windowed_deviation


@pytest.mark.slow
@pytest.mark.parametrize("target_alpha", TARGET_ALPHAS)
@pytest.mark.parametrize("adapter_type", ADAPTER_TYPES)
def test_adaptive_vs_nonadaptive_coverage(target_alpha, adapter_type):
    print(f"Testing {adapter_type} with target alpha {target_alpha}")
    n_seeds = 5
    adaptive_wins_global = 0
    adaptive_wins_local = 0

    for seed in range(n_seeds):
        adaptive_coverage, adaptive_local_dev = run_experiment(
            adapter_type, seed, target_alpha
        )
        nonadaptive_coverage, nonadaptive_local_dev = run_experiment(
            None, seed, target_alpha
        )

        target_coverage = 1 - target_alpha
        adaptive_global_dev = abs(adaptive_coverage - target_coverage)
        nonadaptive_global_dev = abs(nonadaptive_coverage - target_coverage)

        if adaptive_global_dev <= nonadaptive_global_dev:
            adaptive_wins_global += 1
        if adaptive_local_dev <= nonadaptive_local_dev:
            adaptive_wins_local += 1

    global_win_rate = adaptive_wins_global / n_seeds
    local_win_rate = adaptive_wins_local / n_seeds

    print(f"Global win rate: {global_win_rate}, Local win rate: {local_win_rate}")

    if adapter_type is not None:
        assert (
            global_win_rate >= DRAW_OR_WIN_RATE_THRESHOLD
        ), f"Global win rate: {global_win_rate}"
        assert (
            local_win_rate >= DRAW_OR_WIN_RATE_THRESHOLD
        ), f"Local win rate: {local_win_rate}"


def test_dtaci_parameter_evolution():
    search_space = {
        "x1": CategoricalRange(choices=np.linspace(0, 1, 8).tolist()),
        "x2": CategoricalRange(choices=np.linspace(0, 1, 8).tolist()),
        "categorical": CategoricalRange(choices=["A", "B", "C"]),
    }

    sampler = LowerBoundSampler(
        interval_width=0.8,
        adapter="DtACI",
        c=0,
    )

    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=sampler,
        n_pre_conformal_trials=32,
    )

    tuner = ConformalTuner(
        objective_function=complex_objective,
        search_space=search_space,
        metric_optimization="minimize",
        n_candidate_configurations=500,
    )

    tuner.tune(
        n_random_searches=15,
        conformal_retraining_frequency=1,
        searcher=searcher,
        random_state=42,
        max_searches=100,
        verbose=False,
    )

    adapter = sampler.adapter

    assert adapter is not None
    assert adapter.update_count > 0
    assert len(adapter.alpha_history) > 0

    for alpha_val in adapter.alpha_history:
        assert 0.001 <= alpha_val <= 0.999

    assert np.var(adapter.alpha_history) != 0
