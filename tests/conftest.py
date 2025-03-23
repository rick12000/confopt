import random

import numpy as np
import pytest

from confopt.tuning import (
    ObjectiveConformalSearcher,
)
from confopt.utils import get_tuning_configurations
from hashlib import sha256

from confopt.data_classes import FloatRange
from sklearn.base import BaseEstimator
from confopt.config import ESTIMATOR_REGISTRY
from confopt.quantile_wrappers import (
    BaseSingleFitQuantileEstimator,
    BaseMultiFitQuantileEstimator,
)
from confopt.ensembling import (
    MultiFitQuantileEnsembleEstimator,
    SingleFitQuantileEnsembleEstimator,
    PointEnsembleEstimator,
)

DEFAULT_SEED = 1234

POINT_ESTIMATOR_ARCHITECTURES = []
SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES = []
MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES = []
for estimator_name, estimator_config in ESTIMATOR_REGISTRY.items():
    if isinstance(
        estimator_config.estimator_instance,
        (BaseMultiFitQuantileEstimator, MultiFitQuantileEnsembleEstimator),
    ):
        MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES.append(estimator_name)
    elif isinstance(
        estimator_config.estimator_instance,
        (BaseSingleFitQuantileEstimator, SingleFitQuantileEnsembleEstimator),
    ):
        SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES.append(estimator_name)
    elif isinstance(
        estimator_config.estimator_instance, (BaseEstimator, PointEnsembleEstimator)
    ):
        POINT_ESTIMATOR_ARCHITECTURES.append(estimator_name)
    else:
        raise ValueError(
            f"Unknown estimator type: {estimator_config.estimator_instance}"
        )


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
def dummy_expanding_quantile_gaussian_dataset():
    np.random.seed(DEFAULT_SEED)
    random.seed(DEFAULT_SEED)

    X, y = [], []
    for x_observation in range(1, 6):
        for _ in range(0, 100):
            X.append(x_observation)
            y.append(x_observation * np.random.normal(0, 101))
    return np.array(X).reshape(-1, 1), np.array(y)


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
