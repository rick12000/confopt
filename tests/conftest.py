import random

import numpy as np
import pytest

from confopt.acquisition import (
    MultiFitQuantileConformalSearcher,
    LocallyWeightedConformalSearcher,
)
from confopt.tuning import (
    ObjectiveConformalSearcher,
)
from confopt.utils import get_tuning_configurations
from hashlib import sha256
from confopt.conformalization import (
    LocallyWeightedConformalEstimator,
    SingleFitQuantileConformalEstimator,
    MultiFitQuantileConformalEstimator,
    QuantileInterval,
)
from confopt.config import QGBM_NAME, GBM_NAME, QRF_NAME
from confopt.ranges import FloatRange

DEFAULT_SEED = 1234


def noisy_rastrigin(x, A=20, noise_seed=42, noise=0):
    n = len(x)
    x_bytes = x.tobytes()
    combined_bytes = x_bytes + noise_seed.to_bytes(4, "big")
    hash_value = int.from_bytes(sha256(combined_bytes).digest()[:4], "big")
    rng = np.random.default_rng(hash_value)
    rastrigin_value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    noise = rng.normal(loc=0.0, scale=noise)
    return rastrigin_value + noise


class ObjectiveSurfaceGenerator:
    def __init__(self, generator: str):
        self.generator = generator

    def predict(self, params):
        x = np.array(list(params.values()), dtype=float)

        if self.generator == "rastrigin":
            y = noisy_rastrigin(x=x)

        return y


@pytest.fixture
def dummy_stationary_gaussian_dataset():
    np.random.seed(DEFAULT_SEED)
    random.seed(DEFAULT_SEED)

    X, y = [], []
    for x_observation in range(1, 11):
        for _ in range(0, 1000):
            X.append(x_observation)
            y.append(np.random.normal(0, 101))
    dataset = np.column_stack([X, y])
    np.random.shuffle(dataset)
    return dataset


@pytest.fixture
def dummy_fixed_quantile_dataset():
    np.random.seed(DEFAULT_SEED)
    random.seed(DEFAULT_SEED)

    X, y = [], []
    for x_observation in range(1, 11):
        for _ in range(0, 1000):
            X.append(x_observation)
            y.append(random.choice(range(1, 101)))
    dataset = np.column_stack([X, y])
    np.random.shuffle(dataset)
    return dataset


@pytest.fixture
def dummy_init_quantile_regression():
    qcr = MultiFitQuantileConformalSearcher(quantile_estimator_architecture="qgbm")
    return qcr


@pytest.fixture
def dummy_init_locally_weighted_regression():
    lwr = LocallyWeightedConformalSearcher(
        point_estimator_architecture="gbm",
        demeaning_estimator_architecture="gbm",
        variance_estimator_architecture="gbm",
    )
    return lwr


@pytest.fixture
def dummy_configuration_performance_bounds():
    """
    Dummy performance bounds, where each set of
    bounds is meant to represent upper and lower
    expectations of a certain parameter configuration's
    performance.
    """
    performance_lower_bounds = np.arange(0, 100, 0.5)
    performance_upper_bounds = performance_lower_bounds + 10
    return performance_lower_bounds, performance_upper_bounds


@pytest.fixture
def dummy_parameter_grid():
    """Create a parameter grid for testing using the new ParameterRange classes"""
    return {
        "param_1": FloatRange(min_value=0.01, max_value=100, log_scale=True),
        "param_2": FloatRange(min_value=0.01, max_value=100, log_scale=True),
        "param_3": FloatRange(min_value=0.01, max_value=100, log_scale=True),
    }


@pytest.fixture
def dummy_configurations(dummy_parameter_grid):
    """Create dummy configurations for testing"""

    return get_tuning_configurations(
        parameter_grid=dummy_parameter_grid, n_configurations=50, random_state=42
    )


@pytest.fixture
def dummy_tuner(dummy_parameter_grid):
    """
    Creates a conformal searcher instance from dummy raw X, y data
    and a dummy parameter grid.

    This particular fixture is set to optimize a GBM base model on
    regression data, using an MSE objective. The model architecture
    and type of data are arbitrarily pinned; more fixtures could
    be created to test other model or data types.
    """

    def objective_function(configuration):
        generator = ObjectiveSurfaceGenerator(generator="rastrigin")
        return generator.predict(params=configuration)

    searcher = ObjectiveConformalSearcher(
        objective_function=objective_function,
        search_space=dummy_parameter_grid,
        metric_optimization="inverse",
    )

    return searcher


@pytest.fixture
def sample_quantile_interval():
    """Sample quantile interval with lower=0.1, upper=0.9"""
    return QuantileInterval(lower_quantile=0.1, upper_quantile=0.9)


@pytest.fixture
def sample_locally_weighted_estimator():
    """Initialize a locally weighted conformal estimator with GBM architectures"""
    return LocallyWeightedConformalEstimator(
        point_estimator_architecture=GBM_NAME, variance_estimator_architecture=GBM_NAME
    )


@pytest.fixture
def sample_single_fit_estimator():
    """Initialize a single fit quantile conformal estimator with QRF architecture"""
    return SingleFitQuantileConformalEstimator(
        quantile_estimator_architecture=QRF_NAME, n_pre_conformal_trials=20
    )


@pytest.fixture
def sample_multi_fit_estimator(sample_quantile_interval):
    """Initialize a multi-fit quantile conformal estimator with QGBM architecture"""
    return MultiFitQuantileConformalEstimator(
        quantile_estimator_architecture=QGBM_NAME,
        interval=sample_quantile_interval,
        n_pre_conformal_trials=20,
    )
