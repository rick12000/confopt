import pytest
import numpy as np
from typing import Dict
from itertools import product

from confopt.tuning import ConformalTuner, stop_search
from confopt.wrapping import CategoricalRange
from confopt.utils.tracking import RuntimeTracker
from confopt.selection.acquisition import QuantileConformalSearcher, LowerBoundSampler


def test_stop_search_no_remaining_configurations():
    assert stop_search(
        n_remaining_configurations=0,
        current_iter=5,
        current_runtime=10.0,
        max_runtime=100.0,
        max_searches=50,
    )


@pytest.mark.parametrize("max_runtime", [10.0, 15.0, 20.0])
def test_stop_search_runtime_exceeded(max_runtime):
    current_runtime = 25.0
    should_stop = current_runtime >= max_runtime
    assert (
        stop_search(
            n_remaining_configurations=10,
            current_iter=5,
            current_runtime=current_runtime,
            max_runtime=max_runtime,
            max_searches=50,
        )
        == should_stop
    )


@pytest.mark.parametrize("max_searches", [10, 20, 30])
def test_stop_search_iterations_exceeded(max_searches):
    current_iter = 25
    should_stop = current_iter >= max_searches
    assert (
        stop_search(
            n_remaining_configurations=10,
            current_iter=current_iter,
            current_runtime=5.0,
            max_runtime=100.0,
            max_searches=max_searches,
        )
        == should_stop
    )


def test_stop_search_continue_search():
    assert not stop_search(
        n_remaining_configurations=10,
        current_iter=5,
        current_runtime=10.0,
        max_runtime=100.0,
        max_searches=50,
    )


def test_check_objective_function_wrong_argument_count(dummy_parameter_grid):
    def invalid_objective(config1, config2):
        return 1.0

    with pytest.raises(
        ValueError, match="Objective function must take exactly one argument"
    ):
        ConformalTuner(
            objective_function=invalid_objective,
            search_space=dummy_parameter_grid,
            metric_optimization="minimize",
        )


def test_check_objective_function_wrong_argument_name(dummy_parameter_grid):
    def invalid_objective(config):
        return 1.0

    with pytest.raises(
        ValueError,
        match="The objective function must take exactly one argument named 'configuration'",
    ):
        ConformalTuner(
            objective_function=invalid_objective,
            search_space=dummy_parameter_grid,
            metric_optimization="minimize",
        )


def test_evaluate_configuration(tuner):
    config = {"param_1": 0.5, "param_2": 10, "param_3": "option1"}

    performance, runtime = tuner._evaluate_configuration(config)

    assert performance == 2
    assert runtime >= 0


def test_random_search_with_warm_start(
    mock_constant_objective_function, dummy_parameter_grid
):
    warm_start_configs = [
        ({"param_1": 0.5, "param_2": 10, "param_3": "option1"}, 0.8),
    ]

    tuner = ConformalTuner(
        objective_function=mock_constant_objective_function,
        search_space=dummy_parameter_grid,
        metric_optimization="minimize",
        warm_start_configurations=warm_start_configs,
    )

    tuner.initialize_tuning_resources()
    tuner.search_timer = RuntimeTracker()

    assert len(tuner.study.trials) == 1
    assert tuner.study.trials[0].acquisition_source == "warm_start"

    tuner.random_search(
        max_random_iter=3,
        verbose=False,
    )

    assert len(tuner.study.trials) == 4
    assert tuner.study.trials[0].acquisition_source == "warm_start"
    assert all(trial.acquisition_source == "rs" for trial in tuner.study.trials[1:])


def test_random_search_with_nan_performance(dummy_parameter_grid):
    def nan_objective(configuration: Dict) -> float:
        return np.nan

    tuner = ConformalTuner(
        objective_function=nan_objective,
        search_space=dummy_parameter_grid,
        metric_optimization="minimize",
    )

    tuner.initialize_tuning_resources()
    tuner.search_timer = RuntimeTracker()

    tuner.random_search(
        max_random_iter=3,
        verbose=False,
    )

    # Should handle NaN gracefully and not crash
    assert len(tuner.study.trials) == 0


@pytest.mark.parametrize("random_state", [42, 123, 999])
def test_tune_method_reproducibility(dummy_parameter_grid, random_state):
    """Test that tune method produces identical results with same random seed"""

    def complex_objective(configuration: Dict) -> float:
        # Complex objective with multiple terms
        x1 = configuration["param_1"]
        x2 = configuration["param_2"]
        x3_val = {"option1": 1, "option2": 2, "option3": 3}[configuration["param_3"]]
        return x1**2 + np.sin(x2) + x3_val * 0.5

    def run_tune_session():
        # Create fresh searcher for each run to avoid state contamination
        searcher = QuantileConformalSearcher(
            quantile_estimator_architecture="ql",
            sampler=LowerBoundSampler(
                interval_width=0.1,
                adapter="DtACI",
                beta_decay="logarithmic_decay",
                c=1,
            ),
            n_pre_conformal_trials=5,
        )

        tuner = ConformalTuner(
            objective_function=complex_objective,
            search_space=dummy_parameter_grid,
            metric_optimization="minimize",
            n_candidate_configurations=200,
        )

        tuner.tune(
            n_random_searches=10,
            conformal_retraining_frequency=3,
            searcher=searcher,
            optimizer_framework=None,
            random_state=random_state,
            max_searches=25,
            max_runtime=None,
            verbose=False,
        )

        return tuner.study

    # Run twice with same seed
    study1 = run_tune_session()
    study2 = run_tune_session()

    # Verify identical results
    assert len(study1.trials) == len(study2.trials)

    for trial1, trial2 in zip(study1.trials, study2.trials):
        assert trial1.configuration == trial2.configuration
        assert trial1.performance == trial2.performance
        # Skip acquisition_source comparison as it contains object addresses


@pytest.mark.slow
@pytest.mark.parametrize("dynamic_sampling", [True, False])
def test_tune_method_comprehensive_integration(
    comprehensive_tuning_setup, dynamic_sampling
):
    """Comprehensive integration test for tune method (single run, logic only)"""
    tuner, searcher, warm_start_configs, _ = comprehensive_tuning_setup(
        dynamic_sampling
    )

    tuner.tune(
        n_random_searches=15,
        conformal_retraining_frequency=1,
        searcher=searcher,
        optimizer_framework=None,
        random_state=42,
        max_searches=50,
        max_runtime=5 * 60,
        verbose=False,
    )
    study = tuner.study

    # Test 1: Verify correct number of trials
    assert len(study.trials) == 50

    # Test 2: Verify warm starts are present
    warm_start_trials = [
        t for t in study.trials if t.acquisition_source == "warm_start"
    ]
    assert len(warm_start_trials) == 3
    warm_start_performances = [t.performance for t in warm_start_trials]
    expected_performances = [perf for _, perf in warm_start_configs]
    assert set(warm_start_performances) == set(expected_performances)

    # Test 3: Verify trial sources
    rs_trials = [t for t in study.trials if t.acquisition_source == "rs"]
    conformal_trials = [
        t for t in study.trials if t.acquisition_source not in ["warm_start", "rs"]
    ]
    assert len(rs_trials) == 12
    assert len(conformal_trials) == 35

    # Test 4: Verify configurations are diverse
    all_configs = [t.configuration for t in study.trials]
    unique_configs = set(str(config) for config in all_configs)
    assert len(unique_configs) == len(all_configs)

    # Test 5: Verify study methods work correctly
    best_config = study.get_best_configuration()
    best_value = study.get_best_performance()
    assert best_config in all_configs
    assert best_value == min(t.performance for t in study.trials)


@pytest.mark.slow
@pytest.mark.parametrize("dynamic_sampling", [True, False])
def test_conformal_vs_random_performance_averaged(
    comprehensive_tuning_setup, dynamic_sampling
):
    """Compare conformal vs random search performance over multiple runs (averaged)."""
    n_repeats = 20
    min_conformal, min_random = [], []
    avg_conformal, avg_random = [], []
    for seed in range(n_repeats):
        tuner, searcher, _, _ = comprehensive_tuning_setup(dynamic_sampling)
        tuner.tune(
            n_random_searches=15,
            conformal_retraining_frequency=1,
            searcher=searcher,
            optimizer_framework=None,
            random_state=seed,
            max_searches=50,
            max_runtime=5 * 60,
            verbose=False,
        )
        study = tuner.study
        rs_trials = [t for t in study.trials if t.acquisition_source == "rs"]
        conformal_trials = [
            t for t in study.trials if t.acquisition_source not in ["warm_start", "rs"]
        ]
        if len(rs_trials) == 0 or len(conformal_trials) == 0:
            continue
        min_random.append(min(t.performance for t in rs_trials))
        min_conformal.append(min(t.performance for t in conformal_trials))
        avg_random.append(np.mean([t.performance for t in rs_trials]))
        avg_conformal.append(np.mean([t.performance for t in conformal_trials]))

    assert np.mean(avg_conformal) < np.mean(avg_random)
    assert np.mean(min_conformal) <= np.mean(min_random)


@pytest.mark.parametrize("metric_optimization", ["minimize", "maximize"])
def test_best_fetcher_methods(metric_optimization):
    grid = {
        "x": CategoricalRange(choices=[0, 1]),
        "y": CategoricalRange(choices=[0, 1, 2]),
    }

    def objective(configuration):
        return configuration["x"] + configuration["y"] * 10

    tuner = ConformalTuner(
        objective_function=objective,
        search_space=grid,
        metric_optimization=metric_optimization,
        n_candidate_configurations=100,
    )
    tuner.initialize_tuning_resources()
    tuner.search_timer = RuntimeTracker()

    total_configs = len(list(product([0, 1], [0, 1, 2])))
    tuner.random_search(max_random_iter=total_configs, verbose=False)

    # Use built-in methods to get best config and value
    best_config = tuner.get_best_params()
    best_value = tuner.get_best_value()

    if metric_optimization == "minimize":
        expected_config = {"x": 0, "y": 0}
    else:
        expected_config = {"x": 1, "y": 2}
    expected_value = objective(expected_config)

    assert best_config == expected_config
    assert best_value == expected_value
