import random

import numpy as np
import pytest

from confopt.tuning import (
    ConformalTuner,
)
from confopt.utils.encoding import get_tuning_configurations

from confopt.data_classes import FloatRange
from sklearn.base import BaseEstimator
from confopt.selection.estimator_configuration import ESTIMATOR_REGISTRY
from confopt.selection.quantile_estimators import (
    BaseSingleFitQuantileEstimator,
    BaseMultiFitQuantileEstimator,
)
from confopt.selection.ensembling import (
    QuantileEnsembleEstimator,
    PointEnsembleEstimator,
)

DEFAULT_SEED = 1234

POINT_ESTIMATOR_ARCHITECTURES = []
SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES = []
MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES = []
QUANTILE_ESTIMATOR_ARCHITECTURES = []
for estimator_name, estimator_config in ESTIMATOR_REGISTRY.items():
    if isinstance(
        estimator_config.estimator_instance,
        (
            BaseMultiFitQuantileEstimator,
            BaseSingleFitQuantileEstimator,
            QuantileEnsembleEstimator,
        ),
    ):
        QUANTILE_ESTIMATOR_ARCHITECTURES.append(estimator_name)
    if isinstance(
        estimator_config.estimator_instance,
        (BaseMultiFitQuantileEstimator),
    ):
        MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES.append(estimator_name)
    elif isinstance(
        estimator_config.estimator_instance,
        (BaseSingleFitQuantileEstimator),
    ):
        SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES.append(estimator_name)
    elif isinstance(
        estimator_config.estimator_instance, (BaseEstimator, PointEnsembleEstimator)
    ):
        POINT_ESTIMATOR_ARCHITECTURES.append(estimator_name)


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
    performance_lower_bounds = np.arange(0, 100, 0.5)
    performance_upper_bounds = performance_lower_bounds + 10
    return performance_lower_bounds, performance_upper_bounds


@pytest.fixture
def dummy_parameter_grid():
    return {
        "param_1": FloatRange(min_value=0.01, max_value=100, log_scale=True),
        "param_2": FloatRange(min_value=0.01, max_value=100, log_scale=True),
        "param_3": FloatRange(min_value=0.01, max_value=100, log_scale=True),
    }


@pytest.fixture
def dummy_configurations(dummy_parameter_grid):
    return get_tuning_configurations(
        parameter_grid=dummy_parameter_grid, n_configurations=50, random_state=42
    )


@pytest.fixture
def dummy_tuner(dummy_parameter_grid):
    def objective_function(configuration):
        generator = ObjectiveSurfaceGenerator(generator="rastrigin")
        return generator.predict(params=configuration)

    searcher = ConformalTuner(
        objective_function=objective_function,
        search_space=dummy_parameter_grid,
        metric_optimization="inverse",
    )

    return searcher


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
