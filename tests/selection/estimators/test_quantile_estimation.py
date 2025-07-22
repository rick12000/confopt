import pytest
import numpy as np
from unittest.mock import Mock
from confopt.selection.estimators.quantile_estimation import (
    BaseMultiFitQuantileEstimator,
    BaseSingleFitQuantileEstimator,
    QuantileLasso,
    QuantileGBM,
    QuantileLightGBM,
    QuantileForest,
    QuantileKNN,
    GaussianProcessQuantileEstimator,
    QuantileLeaf,
    QuantRegWrapper,
    _param_for_white_kernel_in_sum,
)
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel


class MockMultiFitEstimator(BaseMultiFitQuantileEstimator):
    """Mock implementation for testing abstract base class behavior."""

    def __init__(self):
        self.trained_estimators = []

    def _fit_quantile_estimator(self, X, y, quantile):
        mock_estimator = Mock()
        mock_estimator.predict = lambda X_test: np.full(len(X_test), quantile)
        return mock_estimator


class MockSingleFitEstimator(BaseSingleFitQuantileEstimator):
    """Mock implementation for testing abstract base class behavior."""

    def _fit_implementation(self, X, y):
        self.X_train = X
        self.y_train = y

    def _get_candidate_local_distribution(self, X):
        n_samples, n_candidates = len(X), 100
        return np.random.uniform(0, 1, size=(n_samples, n_candidates))


def calculate_breach_status(predictions, quantiles, tolerance=1e-6):
    """Calculate hard and soft monotonicity violations.

    Args:
        predictions: Array of shape (n_samples, n_quantiles) with quantile predictions
        quantiles: List of quantile levels (should be sorted)
        tolerance: Tolerance for floating-point comparisons

    Returns:
        tuple: (hard_violations, soft_violations, hard_rate, soft_rate)
    """
    n_samples, n_quantiles = predictions.shape
    hard_violations = 0
    soft_violations = 0

    for i in range(n_samples):
        pred_row = predictions[i, :]

        for j in range(n_quantiles - 1):
            diff = pred_row[j + 1] - pred_row[j]
            if diff < -tolerance:  # Hard violation: lower > upper
                hard_violations += 1
            elif abs(diff) <= tolerance:  # Soft violation: approximately equal
                soft_violations += 1

    total_comparisons = n_samples * (n_quantiles - 1)
    hard_rate = hard_violations / total_comparisons
    soft_rate = soft_violations / total_comparisons

    return hard_violations, soft_violations, hard_rate, soft_rate


def calculate_winkler_components(predictions, quantiles):
    """Calculate Winkler score components for interval quality assessment.

    Args:
        predictions: Array of shape (n_samples, n_quantiles) with quantile predictions
        quantiles: List of quantile levels (should be sorted)

    Returns:
        dict: Dictionary with interval width statistics
    """
    n_samples, n_quantiles = predictions.shape
    components = {"mean_widths": [], "negative_widths": 0, "total_intervals": 0}

    for j in range(n_quantiles - 1):
        widths = predictions[:, j + 1] - predictions[:, j]
        components["mean_widths"].append(np.mean(widths))
        components["negative_widths"] += np.sum(widths < 0)
        components["total_intervals"] += len(widths)

    return components


# Tolerance lookup for estimators that perform poorly on specific data
VIOLATION_TOLERANCES = {
    # Single-fit estimators should have perfect monotonicity
    "GaussianProcessQuantileEstimator": {"hard": 0.0, "soft": 0.05},
    "QuantileForest": {
        "hard": 0.0,
        "soft": 0.30,
    },  # Can have many soft violations due to discrete nature
    "QuantileLeaf": {"hard": 0.0, "soft": 0.25},
    "QuantileKNN": {"hard": 0.0, "soft": 0.20},
    # Multi-fit estimators can have violations but should be limited
    "QuantileGBM": {"hard": 0.15, "soft": 0.25},
    "QuantileLightGBM": {"hard": 0.15, "soft": 0.25},
    "QuantileLasso": {
        "hard": 0.30,
        "soft": 0.40,
    },  # Higher tolerance for linear methods
}

# Data-specific tolerance adjustments
DATA_SPECIFIC_ADJUSTMENTS = {
    "challenging_monotonicity_data": {
        "QuantileLasso": {"hard": 0.45, "soft": 0.55},
        "QuantileGBM": {"hard": 0.20, "soft": 0.30},
    },
    "skewed_regression_data": {
        "QuantileLasso": {"hard": 0.25, "soft": 0.35},
    },
}


@pytest.mark.parametrize(
    "data_fixture_name",
    [
        "heteroscedastic_regression_data",
        "multimodal_regression_data",
        "skewed_regression_data",
        "high_dimensional_regression_data",
        "sparse_regression_data",
        "challenging_monotonicity_data",
        "quantile_test_data",
    ],
)
@pytest.mark.parametrize(
    "estimator_class,init_params",
    [
        # Single-fit estimators
        (GaussianProcessQuantileEstimator, {"random_state": 42}),
        (QuantileForest, {"n_estimators": 10, "random_state": 42}),
        (QuantileLeaf, {"n_estimators": 10, "random_state": 42}),
        (QuantileKNN, {"n_neighbors": 5}),
        # Multi-fit estimators
        (
            QuantileGBM,
            {
                "learning_rate": 0.1,
                "n_estimators": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_depth": 3,
                "random_state": 42,
            },
        ),
        (
            QuantileLightGBM,
            {"learning_rate": 0.1, "n_estimators": 15, "random_state": 42},
        ),
        (QuantileLasso, {"max_iter": 200, "random_state": 42}),
    ],
)
def test_monotonicity_across_data_distributions(
    request,
    data_fixture_name,
    estimator_class,
    init_params,
    monotonicity_test_quantiles,
):
    """Test monotonicity behavior across all estimators and data distributions."""
    # Get the data fixture - handle quantile_test_data specially
    data_fixture = request.getfixturevalue(data_fixture_name)
    if data_fixture_name == "quantile_test_data":
        X, y, _ = data_fixture  # Unpack the true_quantiles
    else:
        X, y = data_fixture

    # Use subset for testing to keep tests fast
    n_test = min(50, len(X))
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test = X[-n_test:]

    quantiles = monotonicity_test_quantiles

    estimator = estimator_class(**init_params)
    estimator.fit(X_train, y_train, quantiles)
    predictions = estimator.predict(X_test)

    # Calculate violation statistics
    hard_violations, soft_violations, hard_rate, soft_rate = calculate_breach_status(
        predictions, quantiles
    )

    # Calculate interval quality
    winkler_components = calculate_winkler_components(predictions, quantiles)

    # Get tolerances for this estimator and data combination
    estimator_name = estimator_class.__name__
    base_tolerances = VIOLATION_TOLERANCES[estimator_name]

    # Apply data-specific adjustments if they exist
    if data_fixture_name in DATA_SPECIFIC_ADJUSTMENTS:
        if estimator_name in DATA_SPECIFIC_ADJUSTMENTS[data_fixture_name]:
            adjusted_tolerances = DATA_SPECIFIC_ADJUSTMENTS[data_fixture_name][
                estimator_name
            ]
            hard_tolerance = adjusted_tolerances["hard"]
            soft_tolerance = adjusted_tolerances["soft"]
        else:
            hard_tolerance = base_tolerances["hard"]
            soft_tolerance = base_tolerances["soft"]
    else:
        hard_tolerance = base_tolerances["hard"]
        soft_tolerance = base_tolerances["soft"]

    # Basic shape and validity checks
    assert predictions.shape == (len(X_test), len(quantiles))
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))

    # Hard violation checks (lower bound > upper bound)
    assert hard_rate <= hard_tolerance

    # Soft violation checks (bounds approximately equal)
    assert soft_rate <= soft_tolerance

    # Interval quality checks
    negative_rate = (
        winkler_components["negative_widths"] / winkler_components["total_intervals"]
    )

    # Single-fit estimators should have no negative intervals
    if estimator_name in [
        "GaussianProcessQuantileEstimator",
        "QuantileForest",
        "QuantileLeaf",
        "QuantileKNN",
    ]:
        assert negative_rate <= 0.01
    else:
        # Multi-fit estimators can have some negative intervals
        assert negative_rate <= 0.40


@pytest.mark.parametrize("n_samples", [1, 10, 1000])
@pytest.mark.parametrize("n_features", [1, 5, 20])
@pytest.mark.parametrize("n_quantiles", [1, 3, 9])
def test_multi_fit_base_predict_output_shape(
    toy_regression_data, n_samples, n_features, n_quantiles
):
    """Test that multi-fit estimators produce correctly shaped outputs."""
    X_train, y_train = toy_regression_data(n_samples=100, n_features=n_features)
    X_test = np.random.randn(n_samples, n_features)
    quantiles = np.linspace(0.1, 0.9, n_quantiles).tolist()

    estimator = MockMultiFitEstimator()
    estimator.fit(X_train, y_train, quantiles)
    predictions = estimator.predict(X_test)

    assert predictions.shape == (n_samples, n_quantiles)
    assert isinstance(predictions, np.ndarray)


@pytest.mark.parametrize("n_samples", [1, 10, 1000])
@pytest.mark.parametrize("n_features", [1, 5, 20])
@pytest.mark.parametrize("n_quantiles", [1, 3, 9])
def test_single_fit_base_predict_output_shape(
    toy_regression_data, n_samples, n_features, n_quantiles
):
    """Test that single-fit estimators produce correctly shaped outputs."""
    X_train, y_train = toy_regression_data(n_samples=100, n_features=n_features)
    X_test = np.random.randn(n_samples, n_features)
    quantiles = np.linspace(0.1, 0.9, n_quantiles).tolist()

    estimator = MockSingleFitEstimator()
    estimator.fit(X_train, y_train, quantiles)
    predictions = estimator.predict(X_test)

    assert predictions.shape == (n_samples, n_quantiles)
    assert isinstance(predictions, np.ndarray)


def test_multi_fit_base_unfitted_prediction_raises_error():
    """Test that predicting before fitting raises appropriate error."""
    estimator = MockMultiFitEstimator()
    X_test = np.random.randn(10, 3)

    with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
        estimator.predict(X_test)


@pytest.mark.parametrize(
    "estimator_class,init_params",
    [
        (QuantileLasso, {"max_iter": 100, "p_tol": 1e-4}),
        (
            QuantileGBM,
            {
                "learning_rate": 0.1,
                "n_estimators": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_depth": 3,
            },
        ),
        (QuantileLightGBM, {"learning_rate": 0.1, "n_estimators": 10}),
    ],
)
def test_multi_fit_estimators_fit_predict_consistency(
    heteroscedastic_regression_data, estimator_class, init_params
):
    """Test that multi-fit estimators maintain fitting-prediction consistency."""
    X_train, y_train = heteroscedastic_regression_data
    X_test = X_train[:50]  # Use subset for testing
    quantiles = [0.25, 0.75]

    estimator = estimator_class(**init_params)
    estimator.fit(X_train, y_train, quantiles)
    predictions = estimator.predict(X_test)

    assert predictions.shape == (len(X_test), len(quantiles))
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))


@pytest.mark.parametrize(
    "estimator_class,init_params",
    [
        (QuantileForest, {"n_estimators": 10, "max_depth": 3, "random_state": 42}),
        (QuantileKNN, {"n_neighbors": 5}),
        (QuantileLeaf, {"n_estimators": 10, "max_depth": 3, "random_state": 42}),
        (
            GaussianProcessQuantileEstimator,
            {"kernel": "rbf", "n_inducing_points": 10, "random_state": 42},
        ),
    ],
)
def test_single_fit_estimators_fit_predict_consistency(
    heteroscedastic_regression_data, estimator_class, init_params
):
    """Test that single-fit estimators maintain fitting-prediction consistency."""
    X_train, y_train = heteroscedastic_regression_data
    X_test = X_train[:50]  # Use subset for testing
    quantiles = [0.25, 0.75]

    estimator = estimator_class(**init_params)
    estimator.fit(X_train, y_train, quantiles)
    predictions = estimator.predict(X_test)

    assert predictions.shape == (len(X_test), len(quantiles))
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))


@pytest.mark.parametrize(
    "quantiles",
    [
        [0.1, 0.9],
        [0.05, 0.25, 0.5, 0.75, 0.95],
        [0.01, 0.99],
    ],
)
def test_quantile_ordering_consistency(uniform_regression_data, quantiles):
    """Test that quantile predictions maintain monotonic ordering."""
    X_train, y_train = uniform_regression_data
    X_test = X_train[:100]

    # Test multiple estimators
    estimators = [
        QuantileForest(n_estimators=20, random_state=42),
        QuantileKNN(n_neighbors=10),
        QuantileLeaf(n_estimators=20, random_state=42),
        GaussianProcessQuantileEstimator(
            kernel="rbf", n_inducing_points=10, random_state=42
        ),
    ]

    for estimator in estimators:
        estimator.fit(X_train, y_train, quantiles)
        predictions = estimator.predict(X_test)

        # Check monotonic ordering for each prediction
        for i in range(len(predictions)):
            pred_row = predictions[i]
            assert np.all(
                pred_row[:-1] <= pred_row[1:]
            ), f"Quantile ordering violated for {type(estimator).__name__}"


def test_quantreg_wrapper_with_intercept():
    """Test QuantRegWrapper handles intercept correctly."""
    mock_results = Mock()
    mock_results.params = np.array([1.0, 2.0, 3.0])  # intercept + 2 features

    wrapper = QuantRegWrapper(mock_results, has_intercept=True)
    X_test = np.array([[1, 2], [3, 4]])

    predictions = wrapper.predict(X_test)
    expected = np.array(
        [1 + 1 * 2 + 2 * 3, 1 + 3 * 2 + 4 * 3]
    )  # intercept + feature products

    np.testing.assert_array_equal(predictions, expected)


def test_quantreg_wrapper_without_intercept():
    """Test QuantRegWrapper handles no intercept correctly."""
    mock_results = Mock()
    mock_results.params = np.array([2.0, 3.0])  # 2 features only

    wrapper = QuantRegWrapper(mock_results, has_intercept=False)
    X_test = np.array([[1, 2], [3, 4]])

    predictions = wrapper.predict(X_test)
    expected = np.array([1 * 2 + 2 * 3, 3 * 2 + 4 * 3])  # feature products only

    np.testing.assert_array_equal(predictions, expected)


@pytest.mark.parametrize("n_neighbors", [1, 5, 20])
def test_quantile_knn_neighbor_sensitivity(
    heteroscedastic_regression_data, n_neighbors
):
    """Test that KNN estimator behavior changes appropriately with neighbor count."""
    X_train, y_train = heteroscedastic_regression_data
    X_test = X_train[:10]
    quantiles = [0.25, 0.5, 0.75]

    estimator = QuantileKNN(n_neighbors=n_neighbors)
    estimator.fit(X_train, y_train, quantiles)
    predictions = estimator.predict(X_test)

    assert predictions.shape == (len(X_test), len(quantiles))

    # With fewer neighbors, predictions should be more variable
    if n_neighbors == 1:
        # Single neighbor predictions should be more extreme
        variance = np.var(predictions, axis=0)
        assert np.all(
            variance > 0
        ), "Single neighbor should produce variable predictions"


@pytest.mark.parametrize(
    "kernel_name", ["rbf", "matern", "rational_quadratic", "exp_sine_squared"]
)
def test_gaussian_process_kernel_string_initialization(
    toy_regression_data, kernel_name
):
    """Test GP estimator initializes correctly with string kernel specifications."""
    X_train, y_train = toy_regression_data(n_samples=50, n_features=2)

    estimator = GaussianProcessQuantileEstimator(kernel=kernel_name, random_state=42)
    estimator.fit(X_train, y_train, quantiles=[0.25, 0.75])

    assert hasattr(estimator, "gp")
    assert estimator.gp.kernel_ is not None


def test_gaussian_process_custom_kernel_initialization(toy_regression_data):
    """Test GP estimator works with custom kernel objects."""
    X_train, y_train = toy_regression_data(n_samples=50, n_features=2)
    custom_kernel = RBF(length_scale=2.0) + WhiteKernel(noise_level=0.1)

    estimator = GaussianProcessQuantileEstimator(kernel=custom_kernel, random_state=42)
    estimator.fit(X_train, y_train, quantiles=[0.5])

    assert hasattr(estimator, "gp")


@pytest.mark.parametrize("noise_spec", [None, "gaussian", 0.1])
def test_gaussian_process_noise_handling(toy_regression_data, noise_spec):
    """Test GP estimator handles different noise specifications."""
    X_train, y_train = toy_regression_data(n_samples=50, n_features=2)

    estimator = GaussianProcessQuantileEstimator(noise=noise_spec, random_state=42)
    estimator.fit(X_train, y_train, quantiles=[0.5])

    if noise_spec == "gaussian":
        assert hasattr(estimator, "noise_")
        assert estimator.noise_ > 0
    elif isinstance(noise_spec, (int, float)):
        assert hasattr(estimator, "noise_")
        assert estimator.noise_ == noise_spec


def test_param_for_white_kernel_in_sum_detects_white_kernel():
    """Test utility function correctly identifies WhiteKernel in Sum kernels."""
    kernel_with_white = RBF() + WhiteKernel()
    has_white, param_key = _param_for_white_kernel_in_sum(kernel_with_white)

    assert has_white
    assert "white_kernel" in param_key.lower() or "k2" in param_key


def test_param_for_white_kernel_in_sum_no_white_kernel():
    """Test utility function correctly identifies absence of WhiteKernel."""
    kernel_without_white = RBF() + Matern()
    has_white, param_key = _param_for_white_kernel_in_sum(kernel_without_white)

    assert not has_white
