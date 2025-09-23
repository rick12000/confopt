import numpy as np
import pytest
from confopt.selection.conformalization import (
    QuantileConformalEstimator,
    alpha_to_quantiles,
)
from confopt.wrapping import ConformalBounds
from confopt.utils.preprocessing import train_val_split
from conftest import (
    AMENDED_SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    AMENDED_QUANTILE_ESTIMATOR_ARCHITECTURES,
)

POINT_ESTIMATOR_COVERAGE_TOLERANCE = 0.15
QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE = 0.15
MINIMUM_CONFORMAL_WIN_RATE = 0.51

# Optional per-architecture tolerance overrides for rare problematic estimators
ARCH_TOLERANCE_OVERRIDES: dict[str, float] = {
    # Example only (keep empty unless specific architectures are identified):
    # "problem_arch": 0.10,
}


def validate_intervals(
    intervals: list[ConformalBounds],
    y_true: np.ndarray,
    alphas: list[float],
    tolerance: float,
) -> tuple[float, bool]:
    coverages = []
    errors = []
    for i, alpha in enumerate(alphas):
        lower_bound = intervals[i].lower_bounds
        upper_bound = intervals[i].upper_bounds
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        error = abs(coverage - (1 - alpha)) > tolerance

        coverages.append(coverage)
        errors.append(error)

    return coverages, errors


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3])
def test_alpha_to_quantiles(alpha):
    lower, upper = alpha_to_quantiles(alpha)
    assert lower == alpha / 2
    assert upper == 1 - alpha / 2
    assert lower <= upper


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_fixture_name",
    ["diabetes_data"],
)
@pytest.mark.parametrize(
    "estimator_architecture", AMENDED_QUANTILE_ESTIMATOR_ARCHITECTURES
)
@pytest.mark.parametrize("tuning_iterations", [0])
@pytest.mark.parametrize("alphas", [[0.1], [0.1, 0.3, 0.9]])
@pytest.mark.parametrize(
    "calibration_split_strategy", ["train_test_split", "cv", "adaptive"]
)
def test_quantile_fit_and_predict_intervals_shape_and_coverage(
    request,
    data_fixture_name,
    estimator_architecture,
    tuning_iterations,
    alphas,
    calibration_split_strategy,
):
    X, y = request.getfixturevalue(data_fixture_name)
    (X_train, y_train, X_test, y_test,) = train_val_split(
        X, y, train_split=0.8, normalize=False, ordinal=False, random_state=42
    )

    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=estimator_architecture,
        alphas=alphas,
        n_pre_conformal_trials=15,
        n_calibration_folds=3,
        calibration_split_strategy=calibration_split_strategy,
    )
    estimator.fit(
        X=X_train,
        y=y_train,
        tuning_iterations=tuning_iterations,
        random_state=42,
    )
    assert len(estimator.fold_scores_per_alpha) == len(alphas)

    intervals = estimator.predict_intervals(X_test)
    assert len(intervals) == len(alphas)

    tol = ARCH_TOLERANCE_OVERRIDES.get(
        estimator_architecture, QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE
    )
    _, errors = validate_intervals(intervals, y_test, alphas, tol)
    assert not any(errors)


def test_quantile_calculate_betas_output_properties(
    dummy_expanding_quantile_gaussian_dataset,
):
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=AMENDED_QUANTILE_ESTIMATOR_ARCHITECTURES[0],
        alphas=[0.1, 0.2, 0.3],
        n_pre_conformal_trials=15,
    )
    X, y = dummy_expanding_quantile_gaussian_dataset
    X_train, y_train, X_val, y_val = train_val_split(
        X, y, train_split=0.8, normalize=False, ordinal=False, random_state=42
    )
    estimator.fit(X=X_train, y=y_train, random_state=42)
    test_point = X_val[0]
    test_value = y_val[0]
    betas = estimator.calculate_betas(test_point, test_value)
    assert len(betas) == len(estimator.alphas)
    assert all(0 <= beta <= 1 for beta in betas)


@pytest.mark.parametrize(
    "n_trials,expected_conformalize",
    [
        (5, False),
        (50, True),
    ],
)
def test_quantile_conformalization_decision_logic(n_trials, expected_conformalize):
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=AMENDED_SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES[
            0
        ],
        alphas=[0.2],
        n_pre_conformal_trials=20,
    )
    total_size = n_trials
    X = np.random.rand(total_size, 3)
    y = np.random.rand(total_size)
    X_train, y_train, _, _ = train_val_split(
        X, y, train_split=0.8, normalize=False, ordinal=False, random_state=42
    )
    estimator.fit(X=X_train, y=y_train)
    assert estimator.conformalize_predictions == expected_conformalize


@pytest.mark.parametrize(
    "initial_alphas,new_alphas",
    [
        ([0.2], [0.15, 0.25]),
        ([0.1, 0.2], [0.05, 0.15, 0.3]),
        ([0.3], [0.1]),
    ],
)
def test_quantile_alpha_update_mechanism(initial_alphas, new_alphas):
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=AMENDED_QUANTILE_ESTIMATOR_ARCHITECTURES[0],
        alphas=initial_alphas,
    )
    estimator.update_alphas(new_alphas)
    assert estimator.updated_alphas == new_alphas
    assert estimator.alphas == initial_alphas


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_fixture_name",
    [
        # "heteroscedastic_data",
        "diabetes_data",
    ],
)
@pytest.mark.parametrize("estimator_architecture", ["qrf", "qgbm"])
@pytest.mark.parametrize("alphas", [[0.2, 0.4, 0.6, 0.8]])
@pytest.mark.parametrize("calibration_split_strategy", ["cv"])
def test_conformalized_vs_non_conformalized_quantile_estimator_coverage(
    request,
    data_fixture_name,
    estimator_architecture,
    alphas,
    calibration_split_strategy,
):
    X, y = request.getfixturevalue(data_fixture_name)

    n_repeats = 10
    np.random.seed(42)
    random_states = [np.random.randint(0, 10000) for _ in range(n_repeats)]
    better_or_equal_count = 0
    for random_state in random_states:
        (X_train, y_train, X_test, y_test,) = train_val_split(
            X,
            y,
            # A low value, given we care about distributional coverage
            # on hold out set and we want to simulate a finite training dataset:
            train_split=0.7,
            normalize=False,
            ordinal=False,
            random_state=random_state,
        )

        conformalized_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=estimator_architecture,
            alphas=alphas,
            n_pre_conformal_trials=32,
            calibration_split_strategy=calibration_split_strategy,
            n_calibration_folds=5,
            normalize_features=True,
        )

        conformalized_estimator.fit(
            X=X_train,
            y=y_train,
            random_state=random_state,
        )

        non_conformalized_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=estimator_architecture,
            alphas=alphas,
            n_pre_conformal_trials=10000,
            calibration_split_strategy=calibration_split_strategy,
            n_calibration_folds=5,
            normalize_features=True,
        )

        non_conformalized_estimator.fit(
            X=X_train,
            y=y_train,
            random_state=random_state,
        )

        assert conformalized_estimator.conformalize_predictions
        assert not non_conformalized_estimator.conformalize_predictions

        conformalized_intervals = conformalized_estimator.predict_intervals(X_test)
        non_conformalized_intervals = non_conformalized_estimator.predict_intervals(
            X_test
        )
        conformalized_coverages, _ = validate_intervals(
            conformalized_intervals,
            y_test,
            alphas,
            QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE,
        )
        non_conformalized_coverages, _ = validate_intervals(
            non_conformalized_intervals,
            y_test,
            alphas,
            QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE,
        )

        for i, alpha in enumerate(alphas):
            target_coverage = 1 - alpha
            conformalized_coverage = conformalized_coverages[i]
            non_conformalized_coverage = non_conformalized_coverages[i]

            conformalized_error = abs(conformalized_coverage - target_coverage)
            non_conformalized_error = abs(non_conformalized_coverage - target_coverage)

            if conformalized_error <= non_conformalized_error:
                better_or_equal_count += 1

    total_comparisons = n_repeats * len(alphas)
    percentage_better_or_equal = better_or_equal_count / total_comparisons
    assert percentage_better_or_equal >= MINIMUM_CONFORMAL_WIN_RATE
