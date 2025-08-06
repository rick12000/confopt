import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
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

    # Standardize features to avoid penalizing scale-sensitive estimators
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val


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
    assert len(intervals) == len(alphas)
    _, errors = validate_intervals(
        intervals, y_val, alphas, POINT_ESTIMATOR_COVERAGE_TOLERANCE
    )
    assert not any(errors)


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
@pytest.mark.parametrize("alphas", [[0.1], [0.1, 0.3, 0.9]])
def test_quantile_fit_and_predict_intervals_shape_and_coverage(
    estimator_architecture,
    tuning_iterations,
    alphas,
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
        random_state=42,
    )
    assert len(estimator.nonconformity_scores) == len(alphas)

    intervals = estimator.predict_intervals(X_val)
    assert len(intervals) == len(alphas)

    _, errors = validate_intervals(
        intervals, y_val, alphas, QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE
    )
    assert not any(errors)


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


@pytest.mark.parametrize(
    "data_fixture_name",
    [
        "linear_regression_data",
        "heteroscedastic_data",
        "diabetes_data",
    ],
)
@pytest.mark.parametrize("estimator_architecture", QUANTILE_ESTIMATOR_ARCHITECTURES)
@pytest.mark.parametrize("alphas", [[0.1, 0.9]])
def test_conformalized_vs_non_conformalized_quantile_estimator_coverage(
    request,
    data_fixture_name,
    estimator_architecture,
    alphas,
):
    X, y = request.getfixturevalue(data_fixture_name)
    X_train, y_train, X_val, y_val = create_train_val_split(
        X, y, train_split=0.8, random_state=42
    )

    # Conformalized estimator (n_pre_conformal_trials=15)
    conformalized_estimator = QuantileConformalEstimator(
        quantile_estimator_architecture=estimator_architecture,
        alphas=alphas,
        n_pre_conformal_trials=32,
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

    assert conformalized_estimator.conformalize_predictions
    assert not non_conformalized_estimator.conformalize_predictions

    conformalized_intervals = conformalized_estimator.predict_intervals(X_val)
    non_conformalized_intervals = non_conformalized_estimator.predict_intervals(X_val)
    conformalized_coverages, _ = validate_intervals(
        conformalized_intervals, y_val, alphas, QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE
    )
    non_conformalized_coverages, _ = validate_intervals(
        non_conformalized_intervals,
        y_val,
        alphas,
        QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE,
    )

    # Verify that conformalized estimator has better or equal coverage
    for i, alpha in enumerate(alphas):
        target_coverage = 1 - alpha
        conformalized_coverage = conformalized_coverages[i]
        non_conformalized_coverage = non_conformalized_coverages[i]

        conformalized_error = abs(conformalized_coverage - target_coverage)
        non_conformalized_error = abs(non_conformalized_coverage - target_coverage)

        assert conformalized_error <= non_conformalized_error
