import numpy as np
import pytest
from confopt.selection.conformalization import (
    LocallyWeightedConformalEstimator,
    QuantileConformalEstimator,
    alpha_to_quantiles,
)
from confopt.wrapping import ConformalBounds

from conftest import (
    POINT_ESTIMATOR_ARCHITECTURES,
    SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    QUANTILE_ESTIMATOR_ARCHITECTURES,
)

POINT_ESTIMATOR_COVERAGE_TOLERANCE = 0.2
QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE = 0.05


def create_train_val_split(
    X: np.ndarray, y: np.ndarray, train_split: float, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split_idx = round(len(X) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    return X_train, y_train, X_val, y_val


def validate_intervals(
    intervals: list[ConformalBounds],
    y_true: np.ndarray,
    alphas: list[float],
    tolerance: float,
) -> bool:
    assert len(intervals) == len(alphas)
    for i, alpha in enumerate(alphas):
        lower_bound = intervals[i].lower_bounds
        upper_bound = intervals[i].upper_bounds
        assert np.all(lower_bound <= upper_bound)
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        assert abs(coverage - (1 - alpha)) < tolerance
    return True


def calculate_coverage(
    intervals: list[ConformalBounds], y_true: np.ndarray, alphas: list[float]
) -> list[float]:
    """Calculate empirical coverage for each alpha level.

    Args:
        intervals: List of ConformalBounds objects from prediction
        y_true: True target values
        alphas: List of miscoverage levels

    Returns:
        List of empirical coverage rates, one per alpha level
    """
    coverages = []
    for i, alpha in enumerate(alphas):
        lower_bound = intervals[i].lower_bounds
        upper_bound = intervals[i].upper_bounds
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        coverages.append(coverage)
    return coverages


def calculate_interval_properties(intervals: list[ConformalBounds]) -> dict:
    """Calculate comprehensive interval properties for analysis.

    Args:
        intervals: List of ConformalBounds objects

    Returns:
        Dictionary with interval statistics
    """
    properties = {
        "negative_widths": [],
        "mean_widths": [],
        "min_widths": [],
        "max_widths": [],
        "width_std": [],
    }

    for interval in intervals:
        widths = interval.upper_bounds - interval.lower_bounds
        properties["negative_widths"].append(np.sum(widths < 0))
        properties["mean_widths"].append(np.mean(widths))
        properties["min_widths"].append(np.min(widths))
        properties["max_widths"].append(np.max(widths))
        properties["width_std"].append(np.std(widths))

    return properties


def calculate_monotonicity_violations(
    intervals: list[ConformalBounds],
) -> tuple[int, int]:
    """Calculate hard and soft monotonicity violations in intervals.

    Returns:
        tuple: (hard_violations, soft_violations) where hard = lower > upper, soft = lower ≈ upper
    """
    hard_violations = 0
    soft_violations = 0
    tolerance = 1e-6

    for interval in intervals:
        widths = interval.upper_bounds - interval.lower_bounds
        hard_violations += np.sum(widths < -tolerance)
        soft_violations += np.sum(np.abs(widths) <= tolerance)

    return hard_violations, soft_violations


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3])
def test_alpha_to_quantiles(alpha):
    lower, upper = alpha_to_quantiles(alpha)
    assert lower == alpha / 2
    assert upper == 1 - alpha / 2
    assert lower < upper


@pytest.mark.parametrize("alpha,cap", [(0.2, 0.85), (0.1, 0.95), (0.3, 0.8)])
def test_alpha_to_quantiles_with_cap(alpha, cap):
    lower, upper = alpha_to_quantiles(alpha, upper_quantile_cap=cap)
    assert lower == alpha / 2
    assert upper == min(1 - alpha / 2, cap)
    assert lower <= upper


def test_alpha_to_quantiles_invalid_cap():
    with pytest.raises(
        ValueError, match="Upper quantile cap.*resulted in an upper quantile"
    ):
        alpha_to_quantiles(0.9, upper_quantile_cap=0.1)


# LocallyWeightedConformalEstimator tests as standalone functions
@pytest.mark.parametrize("point_arch", POINT_ESTIMATOR_ARCHITECTURES)
@pytest.mark.parametrize("variance_arch", POINT_ESTIMATOR_ARCHITECTURES)
@pytest.mark.parametrize("tuning_iterations", [0, 1])
@pytest.mark.parametrize("alphas", [[0.5], [0.1, 0.9]])
def test_locally_weighted_fit_and_predict_intervals_shape_and_coverage(
    point_arch,
    variance_arch,
    tuning_iterations,
    alphas,
    dummy_expanding_quantile_gaussian_dataset,
):
    estimator = LocallyWeightedConformalEstimator(
        point_estimator_architecture=point_arch,
        variance_estimator_architecture=variance_arch,
        alphas=alphas,
    )
    X, y = dummy_expanding_quantile_gaussian_dataset
    X_train, y_train, X_val, y_val = create_train_val_split(
        X, y, train_split=0.8, random_state=42
    )
    estimator.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=tuning_iterations,
        random_state=42,
    )
    intervals = estimator.predict_intervals(X=X_val)
    validate_intervals(intervals, y_val, alphas, POINT_ESTIMATOR_COVERAGE_TOLERANCE)


def test_locally_weighted_calculate_betas_output_properties(
    dummy_expanding_quantile_gaussian_dataset,
):
    estimator = LocallyWeightedConformalEstimator(
        point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        alphas=[0.1, 0.2, 0.3],
    )
    X, y = dummy_expanding_quantile_gaussian_dataset
    X_train, y_train, X_val, y_val = create_train_val_split(
        X, y, train_split=0.8, random_state=42
    )
    estimator.fit(X_train, y_train, X_val, y_val, random_state=42)
    test_point = X_val[0]
    test_value = y_val[0]
    betas = estimator.calculate_betas(test_point, test_value)
    assert len(betas) == len(estimator.alphas)
    assert all(0 <= beta <= 1 for beta in betas)


@pytest.mark.parametrize(
    "initial_alphas,new_alphas",
    [
        ([0.2], [0.1, 0.3]),
        ([0.1, 0.2], [0.05, 0.15, 0.25]),
        ([0.3], [0.2]),
    ],
)
def test_locally_weighted_alpha_update_mechanism(initial_alphas, new_alphas):
    estimator = LocallyWeightedConformalEstimator(
        point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        alphas=initial_alphas,
    )
    estimator.update_alphas(new_alphas)
    assert estimator.updated_alphas == new_alphas
    assert estimator.alphas == initial_alphas
    fetched_alphas = estimator._fetch_alphas()
    assert fetched_alphas == new_alphas
    assert estimator.alphas == new_alphas


def test_locally_weighted_prediction_errors_before_fitting():
    estimator = LocallyWeightedConformalEstimator(
        point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        alphas=[0.2],
    )
    X_test = np.random.rand(5, 3)
    with pytest.raises(ValueError, match="Estimators must be fitted before prediction"):
        estimator.predict_intervals(X_test)
    with pytest.raises(
        ValueError, match="Estimators must be fitted before calculating beta"
    ):
        estimator.calculate_betas(X_test[0], 1.0)


# QuantileConformalEstimator tests as standalone functions
@pytest.mark.parametrize("estimator_architecture", QUANTILE_ESTIMATOR_ARCHITECTURES)
@pytest.mark.parametrize("tuning_iterations", [0, 1])
@pytest.mark.parametrize("alphas", [[0.5], [0.1, 0.9]])
@pytest.mark.parametrize("upper_quantile_cap", [None, 0.95])
def test_quantile_fit_and_predict_intervals_shape_and_coverage(
    estimator_architecture,
    tuning_iterations,
    alphas,
    upper_quantile_cap,
    dummy_expanding_quantile_gaussian_dataset,
):
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=estimator_architecture,
        alphas=alphas,
        n_pre_conformal_trials=15,
    )
    X, y = dummy_expanding_quantile_gaussian_dataset
    X_train, y_train, X_val, y_val = create_train_val_split(
        X, y, train_split=0.8, random_state=42
    )
    estimator.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=tuning_iterations,
        upper_quantile_cap=upper_quantile_cap,
        random_state=42,
    )
    assert len(estimator.nonconformity_scores) == len(alphas)
    intervals = estimator.predict_intervals(X_val)
    validate_intervals(intervals, y_val, alphas, QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE)


def test_quantile_calculate_betas_output_properties(
    dummy_expanding_quantile_gaussian_dataset,
):
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=QUANTILE_ESTIMATOR_ARCHITECTURES[0],
        alphas=[0.1, 0.2, 0.3],
        n_pre_conformal_trials=15,
    )
    X, y = dummy_expanding_quantile_gaussian_dataset
    X_train, y_train, X_val, y_val = create_train_val_split(
        X, y, train_split=0.8, random_state=42
    )
    estimator.fit(X_train, y_train, X_val, y_val, random_state=42)
    test_point = X_val[0]
    test_value = y_val[0]
    betas = estimator.calculate_betas(test_point, test_value)
    assert len(betas) == len(estimator.alphas)
    assert all(0 <= beta <= 1 for beta in betas)


@pytest.mark.parametrize(
    "n_trials,expected_conformalize",
    [
        (5, False),
        (25, True),
    ],
)
def test_quantile_conformalization_decision_logic(n_trials, expected_conformalize):
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES[0],
        alphas=[0.2],
        n_pre_conformal_trials=20,
    )
    total_size = n_trials
    X = np.random.rand(total_size, 3)
    y = np.random.rand(total_size)
    X_train, y_train, X_val, y_val = create_train_val_split(
        X, y, train_split=0.8, random_state=42
    )
    estimator.fit(X_train, y_train, X_val, y_val)
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
        quantile_estimator_architecture=QUANTILE_ESTIMATOR_ARCHITECTURES[0],
        alphas=initial_alphas,
    )
    estimator.update_alphas(new_alphas)
    assert estimator.updated_alphas == new_alphas
    assert estimator.alphas == initial_alphas
    fetched_alphas = estimator._fetch_alphas()
    assert fetched_alphas == new_alphas
    assert estimator.alphas == new_alphas


def test_quantile_prediction_errors_before_fitting():
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=QUANTILE_ESTIMATOR_ARCHITECTURES[0],
        alphas=[0.2],
    )
    X_test = np.random.rand(5, 3)
    with pytest.raises(ValueError, match="Estimator must be fitted before prediction"):
        estimator.predict_intervals(X_test)
    with pytest.raises(
        ValueError, match="Estimator must be fitted before calculating beta"
    ):
        estimator.calculate_betas(X_test[0], 1.0)


@pytest.mark.parametrize(
    "alpha,cap",
    [
        (0.2, 0.85),
        (0.1, 0.95),
        (0.3, None),
    ],
)
def test_quantile_upper_quantile_cap_behavior(
    alpha, cap, dummy_expanding_quantile_gaussian_dataset
):
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=QUANTILE_ESTIMATOR_ARCHITECTURES[0],
        alphas=[alpha],
        n_pre_conformal_trials=15,
    )
    X, y = dummy_expanding_quantile_gaussian_dataset
    X_train, y_train, X_val, y_val = create_train_val_split(
        X, y, train_split=0.8, random_state=42
    )
    estimator.fit(
        X_train, y_train, X_val, y_val, upper_quantile_cap=cap, random_state=42
    )
    assert estimator.upper_quantile_cap == cap
    expected_lower, expected_upper = alpha_to_quantiles(alpha, cap)
    assert expected_lower in estimator.quantile_indices
    assert expected_upper in estimator.quantile_indices


@pytest.mark.parametrize("estimator_architecture", QUANTILE_ESTIMATOR_ARCHITECTURES)
@pytest.mark.parametrize("alphas", [[0.1], [0.1, 0.9]])
def test_conformalized_vs_non_conformalized_quantile_estimator_coverage(
    estimator_architecture,
    alphas,
    dummy_expanding_quantile_gaussian_dataset,
):
    X, y = dummy_expanding_quantile_gaussian_dataset
    X_train, y_train, X_val, y_val = create_train_val_split(
        X, y, train_split=0.8, random_state=42
    )

    # Conformalized estimator (n_pre_conformal_trials=15)
    conformalized_estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=estimator_architecture,
        alphas=alphas,
        n_pre_conformal_trials=15,
    )

    conformalized_estimator.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=42,
    )

    # Non-conformalized estimator (n_pre_conformal_trials=10000)
    non_conformalized_estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=estimator_architecture,
        alphas=alphas,
        n_pre_conformal_trials=10000,
    )

    non_conformalized_estimator.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=42,
    )

    # Verify conformalization status
    assert conformalized_estimator.conformalize_predictions
    assert not non_conformalized_estimator.conformalize_predictions

    # Generate predictions for both estimators
    conformalized_intervals = conformalized_estimator.predict_intervals(X_val)
    non_conformalized_intervals = non_conformalized_estimator.predict_intervals(X_val)

    # Calculate coverage for both estimators
    conformalized_coverages = calculate_coverage(conformalized_intervals, y_val, alphas)
    non_conformalized_coverages = calculate_coverage(
        non_conformalized_intervals, y_val, alphas
    )

    # Verify that conformalized estimator has better or equal coverage
    for i, alpha in enumerate(alphas):
        target_coverage = 1 - alpha
        conformalized_coverage = conformalized_coverages[i]
        non_conformalized_coverage = non_conformalized_coverages[i]

        # Conformalized estimator should have coverage closer to or better than target
        conformalized_error = abs(conformalized_coverage - target_coverage)
        non_conformalized_error = abs(non_conformalized_coverage - target_coverage)

        # Assert that conformalized estimator performs better or equal
        assert conformalized_error <= non_conformalized_error

    # Check monotonicity properties
    conf_hard_violations, conf_soft_violations = calculate_monotonicity_violations(
        conformalized_intervals
    )
    (
        non_conf_hard_violations,
        non_conf_soft_violations,
    ) = calculate_monotonicity_violations(non_conformalized_intervals)

    # Conformalized should have better monotonicity than non-conformalized
    assert conf_hard_violations <= non_conf_hard_violations

    # Single-fit estimators should have perfect hard monotonicity
    if estimator_architecture in ["qgp", "qrf"]:
        assert conf_hard_violations == 0
        assert non_conf_hard_violations == 0
