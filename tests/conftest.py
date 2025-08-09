import random

import numpy as np
import pytest
from typing import Dict
from confopt.tuning import (
    ConformalTuner,
)
from confopt.utils.configurations.sampling import get_tuning_configurations
from confopt.selection.acquisition import QuantileConformalSearcher, LowerBoundSampler
from confopt.wrapping import FloatRange, IntRange, CategoricalRange, ConformalBounds
from sklearn.base import BaseEstimator
from confopt.selection.estimator_configuration import (
    ESTIMATOR_REGISTRY,
)
from confopt.selection.estimators.quantile_estimation import (
    BaseSingleFitQuantileEstimator,
    BaseMultiFitQuantileEstimator,
)
from confopt.selection.estimators.ensembling import (
    QuantileEnsembleEstimator,
    PointEnsembleEstimator,
)
from unittest.mock import Mock
from confopt.selection.adaptation import DtACI

DEFAULT_SEED = 1234


def build_estimator_architectures(amended: bool = False):
    """Build estimator architecture lists from ESTIMATOR_REGISTRY.

    Args:
        amended: If True, creates modified versions with n_estimators=25 for faster testing.
                If False, creates standard architecture lists.

    Returns:
        Tuple containing:
        - point_estimator_architectures: List of point estimator names
        - single_fit_quantile_estimator_architectures: List of single-fit quantile estimator names
        - multi_fit_quantile_estimator_architectures: List of multi-fit quantile estimator names
        - quantile_estimator_architectures: List of all quantile estimator names
        - estimator_registry: Registry of estimator configurations (amended if requested)
    """
    from copy import deepcopy

    point_estimator_architectures = []
    single_fit_quantile_estimator_architectures = []
    multi_fit_quantile_estimator_architectures = []
    quantile_estimator_architectures = []

    # Create registry (amended if requested)
    if amended:
        estimator_registry = {}
        for estimator_name, estimator_config in ESTIMATOR_REGISTRY.items():
            amended_config = deepcopy(estimator_config)

            # Check if the estimator has n_estimators parameter
            if (
                hasattr(amended_config, "default_params")
                and "n_estimators" in amended_config.default_params
            ):
                amended_config.default_params["n_estimators"] = 15

            # Also check ensemble components if it's an ensemble estimator
            if (
                hasattr(amended_config, "ensemble_components")
                and amended_config.ensemble_components
            ):
                for component in amended_config.ensemble_components:
                    if "params" in component and "n_estimators" in component["params"]:
                        component["params"]["n_estimators"] = 15

            if estimator_name in ["gp", "qgp"]:
                continue

            if "qens" in estimator_name:
                continue

            estimator_registry[estimator_name] = amended_config
    else:
        estimator_registry = ESTIMATOR_REGISTRY

    # Build architecture lists
    for estimator_name, estimator_config in estimator_registry.items():
        if issubclass(
            estimator_config.estimator_class,
            (
                BaseMultiFitQuantileEstimator,
                BaseSingleFitQuantileEstimator,
                QuantileEnsembleEstimator,
            ),
        ):
            quantile_estimator_architectures.append(estimator_name)
        if issubclass(
            estimator_config.estimator_class,
            (BaseMultiFitQuantileEstimator),
        ):
            multi_fit_quantile_estimator_architectures.append(estimator_name)
        elif issubclass(
            estimator_config.estimator_class,
            (BaseSingleFitQuantileEstimator),
        ):
            single_fit_quantile_estimator_architectures.append(estimator_name)
        elif issubclass(
            estimator_config.estimator_class, (BaseEstimator, PointEnsembleEstimator)
        ):
            point_estimator_architectures.append(estimator_name)

    return (
        point_estimator_architectures,
        single_fit_quantile_estimator_architectures,
        multi_fit_quantile_estimator_architectures,
        quantile_estimator_architectures,
        estimator_registry,
    )


# Create original architecture lists
(
    POINT_ESTIMATOR_ARCHITECTURES,
    SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    QUANTILE_ESTIMATOR_ARCHITECTURES,
    _,
) = build_estimator_architectures(amended=False)

# Create amended architecture lists for faster testing
(
    AMENDED_POINT_ESTIMATOR_ARCHITECTURES,
    AMENDED_SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    AMENDED_MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    AMENDED_QUANTILE_ESTIMATOR_ARCHITECTURES,
    AMENDED_ESTIMATOR_REGISTRY,
) = build_estimator_architectures(amended=True)


def rastrigin(x, A=20):
    n = len(x)
    rastrigin_value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return rastrigin_value


class ObjectiveSurfaceGenerator:
    def __init__(self, generator: str):
        self.generator = generator

    def predict(self, params):
        x = np.array(list(params.values()), dtype=float)

        if self.generator == "rastrigin":
            y = rastrigin(x=x)

        return y


@pytest.fixture
def mock_constant_objective_function():
    def objective(configuration: Dict):
        return 2

    return objective


@pytest.fixture
def toy_dataset():
    # Create a small toy dataset with deterministic values
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([2, 4, 6, 8])
    return X, y


@pytest.fixture
def big_toy_dataset():
    # Create a larger toy dataset with 80 observations and 2 features
    X = np.linspace(0, 10, 80).reshape(-1, 1)  # Capped at 80
    X = np.hstack([X, X + np.random.normal(0, 1, 80).reshape(-1, 1)])  # Capped at 80
    # Make y always negative by using negative coefficients and subtracting a constant
    y = -5 * X[:, 0] - 3 * X[:, 1] - 10 + np.random.normal(0, 1, 80)  # Capped at 80
    return X, y


@pytest.fixture
def quantiles():
    return [0.1, 0.5, 0.9]


@pytest.fixture
def dummy_expanding_quantile_gaussian_dataset():
    np.random.seed(DEFAULT_SEED)
    random.seed(DEFAULT_SEED)

    X, y = [], []
    # Reduce to 80 total observations (16 per x_observation)
    for x_observation in range(1, 6):
        for _ in range(0, 20):  # Adjusted to make total 80
            X.append(x_observation)
            y.append(x_observation * np.random.normal(0, 10))

    X_array = np.array(X).reshape(-1, 1)
    # Normalize X to have zero mean and unit variance
    X_normalized = (X_array - np.mean(X_array)) / np.std(X_array)
    return X_normalized, np.array(y)


@pytest.fixture
def dummy_parameter_grid():
    return {
        "param_1": FloatRange(min_value=0.01, max_value=100, log_scale=True),
        "param_2": IntRange(min_value=1, max_value=100),
        "param_3": CategoricalRange(choices=["option1", "option2", "option3"]),
    }


@pytest.fixture
def linear_data_drift():
    np.random.seed(42)
    n = 500
    X = np.linspace(0, 10, n).reshape(-1, 1)

    noise_level = np.linspace(0.5, 3, n)
    noise = np.random.normal(0, 1, n) * noise_level

    y = np.zeros(n)

    first_segment = int(0.3 * n)
    y[:first_segment] = 2 * X[:first_segment].flatten() + 5 + noise[:first_segment]

    second_segment = int(0.6 * n)
    y[first_segment:second_segment] = (
        3 * X[first_segment:second_segment].flatten()
        + 2
        + noise[first_segment:second_segment]
    )

    y[second_segment:] = 2.5 * X[second_segment:].flatten() + 8 + noise[second_segment:]

    return X, y


@pytest.fixture
def simple_conformal_bounds():
    lower_bounds1 = np.array([0.1, 0.3, 0.5])
    upper_bounds1 = np.array([0.4, 0.6, 0.8])

    lower_bounds2 = np.array([0.2, 0.4, 0.6])
    upper_bounds2 = np.array([0.5, 0.7, 0.9])

    return [
        ConformalBounds(lower_bounds=lower_bounds1, upper_bounds=upper_bounds1),
        ConformalBounds(lower_bounds=lower_bounds2, upper_bounds=upper_bounds2),
    ]


@pytest.fixture
def estimator1():
    """Mock point estimator that returns deterministic values scaled to input size."""
    mock = Mock()

    def scaled_predict(X):
        # Return values that scale based on input length
        n_samples = len(X)
        return np.arange(1, n_samples + 1) * 2  # [2, 4, 6, 8, ...] based on input size

    mock.predict = Mock(side_effect=scaled_predict)
    mock.fit = Mock(return_value=mock)
    return mock


@pytest.fixture
def estimator2():
    """Mock point estimator that returns different deterministic values scaled to input size."""
    mock = Mock()

    def scaled_predict(X):
        # Return values that scale based on input length
        n_samples = len(X)
        return np.arange(2, n_samples + 2) * 2  # [4, 6, 8, 10, ...] based on input size

    mock.predict = Mock(side_effect=scaled_predict)
    mock.fit = Mock(return_value=mock)
    return mock


@pytest.fixture
def quantile_estimator1(quantiles):
    """Mock quantile estimator that returns deterministic quantile predictions for any input size."""
    mock = Mock()

    def scaled_predict(X):
        # Return values for any size of X
        n_samples = len(X)
        result = np.zeros((n_samples, len(quantiles)))
        for i, q in enumerate(quantiles):
            result[:, i] = (i + 1) * 2  # Values 2, 4, 6 for quantiles
        return result

    mock.fit = Mock(return_value=mock)
    mock.predict = Mock(side_effect=scaled_predict)
    return mock


@pytest.fixture
def quantile_estimator2(quantiles):
    """Mock quantile estimator that returns constant values across quantiles."""
    mock = Mock()

    def scaled_predict(X):
        # Return values for any size of X
        n_samples = len(X)
        return np.ones((n_samples, len(quantiles))) * 4

    mock.fit = Mock(return_value=mock)
    mock.predict = Mock(side_effect=scaled_predict)
    return mock


@pytest.fixture
def competing_estimator():
    """Mock point estimator with different performance characteristics."""
    mock = Mock()

    def scaled_predict(X):
        # Return values that scale based on input length
        n_samples = len(X)
        return (
            np.arange(0.5, n_samples + 0.5) * 2
        )  # [1, 3, 5, 7, ...] based on input size

    mock.predict = Mock(side_effect=scaled_predict)
    mock.fit = Mock(return_value=mock)
    return mock


@pytest.fixture
def tuner(mock_constant_objective_function, dummy_parameter_grid):
    # Create a standard tuner instance that can be reused across tests
    return ConformalTuner(
        objective_function=mock_constant_objective_function,
        search_space=dummy_parameter_grid,
        metric_optimization="minimize",
        n_candidate_configurations=100,
    )


@pytest.fixture
def small_parameter_grid():
    """Small parameter grid for focused configuration testing"""
    return {
        "x": FloatRange(min_value=0.0, max_value=1.0),
        "y": IntRange(min_value=1, max_value=3),
        "z": CategoricalRange(choices=["A", "B"]),
    }


@pytest.fixture
def dynamic_tuner(mock_constant_objective_function, small_parameter_grid):
    """Tuner configured for dynamic sampling with small candidate count"""
    return ConformalTuner(
        objective_function=mock_constant_objective_function,
        search_space=small_parameter_grid,
        metric_optimization="minimize",
        n_candidate_configurations=5,
        dynamic_sampling=True,
    )


@pytest.fixture
def static_tuner(mock_constant_objective_function, small_parameter_grid):
    """Tuner configured for static sampling with small candidate count"""
    return ConformalTuner(
        objective_function=mock_constant_objective_function,
        search_space=small_parameter_grid,
        metric_optimization="minimize",
        n_candidate_configurations=10,
        dynamic_sampling=False,
    )


# Fixtures for quantile estimation testing


@pytest.fixture
def uniform_regression_data():
    """Generate uniform regression data for quantile testing."""
    np.random.seed(42)
    n_samples = 300
    n_features = 3

    X = np.random.uniform(-1, 1, size=(n_samples, n_features))
    y = np.random.uniform(0, 1, size=n_samples)

    return X, y


@pytest.fixture
def heteroscedastic_regression_data():
    """Generate heteroscedastic regression data where variance changes with X."""
    np.random.seed(42)
    n_samples = 200
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)

    # Heteroscedastic noise: variance increases with |X|
    noise_std = 0.5 + 1.5 * np.abs(X.flatten())
    noise = np.random.normal(0, 1, n_samples) * noise_std

    # True function: quadratic with heteroscedastic noise
    y = 2 * X.flatten() ** 2 + 1.5 * X.flatten() + noise

    return X, y


@pytest.fixture
def multimodal_regression_data():
    """Generate multimodal regression data with multiple peaks and valleys."""
    np.random.seed(42)
    n_samples = 300
    X = np.linspace(-4, 4, n_samples).reshape(-1, 1)

    # Multimodal function: mixture of Gaussians
    y = (
        2 * np.exp(-0.5 * (X.flatten() + 2) ** 2)
        + 1.5 * np.exp(-0.5 * (X.flatten() - 1) ** 2)
        + np.exp(-0.5 * (X.flatten() - 3) ** 2)
        + np.random.normal(0, 0.3, n_samples)
    )

    return X, y


@pytest.fixture
def skewed_regression_data():
    """Generate regression data with skewed noise distribution."""
    np.random.seed(42)
    n_samples = 250
    X = np.linspace(0, 5, n_samples).reshape(-1, 1)

    # Skewed noise using exponential distribution
    skewed_noise = np.random.exponential(0.5, n_samples) - 0.5

    # True function with skewed residuals
    y = np.sin(X.flatten()) + 0.5 * X.flatten() + skewed_noise

    return X, y


@pytest.fixture
def high_dimensional_regression_data():
    """Generate high-dimensional regression data for testing scalability."""
    np.random.seed(42)
    n_samples = 150
    n_features = 8
    X = np.random.randn(n_samples, n_features)

    # Linear combination with interaction terms
    true_coef = np.array([2, -1, 0.5, -0.5, 1, 0, -0.3, 0.8])
    y = X @ true_coef + 0.5 * X[:, 0] * X[:, 1] + np.random.normal(0, 0.5, n_samples)

    return X, y


@pytest.fixture
def sparse_regression_data():
    """Generate sparse regression data with few informative features."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)

    # Only first 3 features are informative
    true_coef = np.zeros(n_features)
    true_coef[:3] = [3, -2, 1.5]
    y = X @ true_coef + np.random.normal(0, 0.3, n_samples)

    return X, y


@pytest.fixture
def toy_regression_data():
    """Generate simple toy regression data for basic testing."""

    def _generate_data(n_samples=100, n_features=2, noise_std=0.1, random_state=42):
        np.random.seed(random_state)
        X = np.random.randn(n_samples, n_features)
        true_coef = np.ones(n_features)
        y = X @ true_coef + np.random.normal(0, noise_std, n_samples)
        return X, y

    return _generate_data


@pytest.fixture
def quantile_test_data():
    """Generate data with known quantile structure for validation."""
    np.random.seed(42)
    n_samples = 500
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)

    # Create data where we know the true quantiles
    # Use a location-scale model: Y = μ(X) + σ(X) * ε
    mu = 2 * X.flatten()  # Mean function
    sigma = 0.5 + 0.3 * np.abs(X.flatten())  # Scale function
    epsilon = np.random.normal(0, 1, n_samples)  # Standard normal noise

    y = mu + sigma * epsilon

    # Store true quantiles for validation
    true_quantiles = {}
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        from scipy.stats import norm

        true_quantiles[q] = mu + sigma * norm.ppf(q)

    return X, y, true_quantiles


@pytest.fixture
def monotonicity_test_quantiles():
    """Standard quantiles for monotonicity testing."""
    return [0.1, 0.25, 0.5, 0.75, 0.9]


@pytest.fixture
def alpha_levels_for_conformalization():
    """Standard alpha levels for conformalization testing."""
    return [0.1, 0.2, 0.3]  # Corresponding to 90%, 80%, 70% coverage


@pytest.fixture
def estimation_test_data():
    """Generate test data for estimation module tests."""
    np.random.seed(42)
    X = np.random.rand(50, 5)
    y = X.sum(axis=1) + np.random.normal(0, 0.1, 50)
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.25, random_state=42)


@pytest.fixture
def point_tuner():
    """Create a PointTuner instance for testing."""
    from confopt.selection.estimation import PointTuner

    return PointTuner(random_state=42)


@pytest.fixture
def quantile_tuner_with_quantiles():
    """Create a QuantileTuner instance with quantiles for testing."""
    from confopt.selection.estimation import QuantileTuner

    quantiles = [0.1, 0.9]
    return QuantileTuner(quantiles=quantiles, random_state=42), quantiles


@pytest.fixture
def multi_interval_bounds():
    """Create multiple ConformalBounds objects for multi-interval testing."""
    n_obs = 30
    bounds_list = []
    for i in range(3):
        width_factor = (i + 1) * 0.5
        lower = np.random.uniform(-1, 0, n_obs)
        upper = lower + np.random.uniform(0.2 * width_factor, 1.0 * width_factor, n_obs)
        bounds_list.append(ConformalBounds(lower_bounds=lower, upper_bounds=upper))
    return bounds_list


@pytest.fixture
def nested_intervals():
    """Create properly nested intervals for testing interval relationships."""
    n_obs = 20
    # Create nested intervals: each inner interval contained within outer
    center = np.random.uniform(-1, 1, n_obs)

    # Widest interval (lowest confidence)
    wide_lower = center - 2.0
    wide_upper = center + 2.0

    # Medium interval
    med_lower = center - 1.0
    med_upper = center + 1.0

    # Narrowest interval (highest confidence)
    narrow_lower = center - 0.5
    narrow_upper = center + 0.5

    return [
        ConformalBounds(lower_bounds=wide_lower, upper_bounds=wide_upper),
        ConformalBounds(lower_bounds=med_lower, upper_bounds=med_upper),
        ConformalBounds(lower_bounds=narrow_lower, upper_bounds=narrow_upper),
    ]


@pytest.fixture
def coverage_feedback():
    """Sample coverage feedback for adaptation testing."""
    return [0.85, 0.78, 0.92]


@pytest.fixture
def small_dataset():
    """Small dataset for computational testing."""
    n_obs = 10
    bounds = []
    for _ in range(2):
        lower = np.random.uniform(-0.5, 0, n_obs)
        upper = lower + np.random.uniform(0.1, 0.5, n_obs)
        bounds.append(ConformalBounds(lower_bounds=lower, upper_bounds=upper))
    return bounds


@pytest.fixture
def test_predictions_and_widths():
    """Combined point predictions and interval widths for LCB testing."""
    np.random.seed(42)
    n_points = 15
    point_estimates = np.random.uniform(-2, 2, n_points)
    interval_widths = np.random.uniform(0.2, 1.5, n_points)
    return point_estimates, interval_widths


@pytest.fixture
def entropy_samples_gaussian():
    """Gaussian samples for entropy calculation testing."""
    np.random.seed(42)
    return np.random.normal(0, 1, 100)


@pytest.fixture
def entropy_samples_uniform():
    """Uniform samples for entropy calculation testing."""
    np.random.seed(42)
    return np.random.uniform(-2, 2, 50)


@pytest.fixture
def entropy_samples_identical():
    """Identical samples for entropy edge case testing."""
    return np.array([3.14, 3.14, 3.14, 3.14, 3.14])


@pytest.fixture
def entropy_samples_linear():
    """Linear samples for deterministic entropy testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def conformal_bounds_deterministic():
    """Deterministic conformal bounds for reproducible testing."""
    lower_bounds1 = np.array([1.0, 2.0, 3.0, 4.0])
    upper_bounds1 = np.array([1.5, 2.5, 3.5, 4.5])

    lower_bounds2 = np.array([0.8, 1.8, 2.8, 3.8])
    upper_bounds2 = np.array([1.3, 2.3, 3.3, 4.3])

    return [
        ConformalBounds(lower_bounds=lower_bounds1, upper_bounds=upper_bounds1),
        ConformalBounds(lower_bounds=lower_bounds2, upper_bounds=upper_bounds2),
    ]


@pytest.fixture
def monte_carlo_bounds_simple():
    """Simple bounds for Monte Carlo entropy testing."""
    # Create bounds that will yield predictable minimum values
    lower_bounds1 = np.array([10.0, 20.0, 5.0])  # min will be 5.0
    upper_bounds1 = np.array([15.0, 25.0, 8.0])

    lower_bounds2 = np.array([12.0, 18.0, 6.0])  # min will be 6.0
    upper_bounds2 = np.array([17.0, 23.0, 9.0])

    return [
        ConformalBounds(lower_bounds=lower_bounds1, upper_bounds=upper_bounds1),
        ConformalBounds(lower_bounds=lower_bounds2, upper_bounds=upper_bounds2),
    ]


@pytest.fixture
def comprehensive_tuning_setup(dummy_parameter_grid):
    """Fixture for comprehensive integration test setup (objective, warm starts, tuner, searcher)."""

    def optimization_objective(configuration: Dict) -> float:
        x1 = configuration["param_1"]
        x2 = configuration["param_2"]
        x3_val = {"option1": 0, "option2": 1, "option3": 2}[configuration["param_3"]]
        return (x1 - 1) ** 2 + (x2 - 10) ** 2 * 0.01 + x3_val * 0.5

    warm_start_configs_raw = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=3,
        random_state=123,
        sampling_method="uniform",
    )
    warm_start_configs = []
    for config in warm_start_configs_raw:
        performance = optimization_objective(config)
        warm_start_configs.append((config, performance))

    def make_tuner_and_searcher(dynamic_sampling):
        tuner = ConformalTuner(
            objective_function=optimization_objective,
            search_space=dummy_parameter_grid,
            metric_optimization="minimize",
            n_candidate_configurations=500,
            warm_start_configurations=warm_start_configs,
            dynamic_sampling=dynamic_sampling,
        )
        searcher = QuantileConformalSearcher(
            quantile_estimator_architecture="ql",
            sampler=LowerBoundSampler(
                interval_width=0.9,
                adapter="DtACI",
                beta_decay="logarithmic_decay",
                c=1,
            ),
            n_pre_conformal_trials=20,
        )
        return tuner, searcher, warm_start_configs, optimization_objective

    return make_tuner_and_searcher


@pytest.fixture
def moderate_shift_data():
    """Create data with moderate distribution shift (0.1 -> 0.5 noise)."""
    np.random.seed(42)
    n_points = 200
    shift_point = 100

    X1 = np.random.randn(shift_point, 2)
    y1 = X1.sum(axis=1) + 0.1 * np.random.randn(shift_point)

    X2 = np.random.randn(n_points - shift_point, 2)
    y2 = X2.sum(axis=1) + 0.5 * np.random.randn(n_points - shift_point)

    return np.vstack([X1, X2]), np.hstack([y1, y2])


@pytest.fixture
def high_shift_data():
    """Create data with high distribution shift (0.1 -> 0.8 -> 0.1 noise)."""
    np.random.seed(42)
    n_points = 300
    shift_points = [100, 200]

    X1 = np.random.randn(shift_points[0], 2)
    y1 = X1.sum(axis=1) + 0.1 * np.random.randn(shift_points[0])

    X2 = np.random.randn(shift_points[1] - shift_points[0], 2)
    y2 = X2.sum(axis=1) + 0.8 * np.random.randn(shift_points[1] - shift_points[0])

    X3 = np.random.randn(n_points - shift_points[1], 2)
    y3 = X3.sum(axis=1) + 0.1 * np.random.randn(n_points - shift_points[1])

    return np.vstack([X1, X2, X3]), np.hstack([y1, y2, y3])


@pytest.fixture
def dtaci_instance():
    """Standard DtACI instance for testing."""
    return DtACI(alpha=0.1, gamma_values=[0.01, 0.05, 0.1])


# Quantile Estimation Test Data Fixtures
@pytest.fixture
def linear_regression_data():
    """Simple linear regression with homoscedastic noise."""
    np.random.seed(42)
    n_samples = 200
    X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
    y = 2.5 * X.flatten() + 1.0 + np.random.normal(0, 0.5, n_samples)
    return X, y


@pytest.fixture
def heteroscedastic_data():
    """Heteroscedastic data where variance increases with |X|."""
    np.random.seed(42)
    n_samples = 300
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    noise_std = 0.3 + 1.2 * np.abs(X.flatten())
    noise = np.random.normal(0, 1, n_samples) * noise_std
    y = 1.5 * X.flatten() ** 2 + 0.5 * X.flatten() + noise
    return X, y


@pytest.fixture
def diabetes_data():
    """Scikit-learn diabetes dataset for regression testing."""
    from sklearn.datasets import load_diabetes

    diabetes = load_diabetes()
    return diabetes.data, diabetes.target


@pytest.fixture
def comprehensive_test_quantiles():
    """Comprehensive set of quantiles for testing."""
    return [0.05, 0.25, 0.5, 0.75, 0.95]


@pytest.fixture
def ensemble_test_quantiles():
    return [0.25, 0.5, 0.75]
