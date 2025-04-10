import random

import numpy as np
import pytest
from typing import Dict
from confopt.tuning import (
    ConformalTuner,
)
from confopt.utils.encoding import get_tuning_configurations

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

DEFAULT_SEED = 1234

POINT_ESTIMATOR_ARCHITECTURES = []
SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES = []
MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES = []
QUANTILE_ESTIMATOR_ARCHITECTURES = []
for estimator_name, estimator_config in ESTIMATOR_REGISTRY.items():
    if issubclass(
        estimator_config.estimator_class,
        (
            BaseMultiFitQuantileEstimator,
            BaseSingleFitQuantileEstimator,
            QuantileEnsembleEstimator,
        ),
    ):
        QUANTILE_ESTIMATOR_ARCHITECTURES.append(estimator_name)
    if issubclass(
        estimator_config.estimator_class,
        (BaseMultiFitQuantileEstimator),
    ):
        MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES.append(estimator_name)
    elif issubclass(
        estimator_config.estimator_class,
        (BaseSingleFitQuantileEstimator),
    ):
        SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES.append(estimator_name)
    elif issubclass(
        estimator_config.estimator_class, (BaseEstimator, PointEnsembleEstimator)
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
def mock_random_objective_function():
    def objective(configuration: Dict):
        return random.uniform(0, 1)

    return objective


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
def quantiles():
    return [0.1, 0.5, 0.9]


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
        "param_2": IntRange(min_value=1, max_value=100),
        "param_3": CategoricalRange(choices=["option1", "option2", "option3"]),
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


@pytest.fixture
def conformal_bounds():
    # Create three deterministic conformal bounds
    predictions = []
    for i in range(3):
        bounds = ConformalBounds(
            lower_bounds=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * (i + 1),
            upper_bounds=np.array([1.1, 1.2, 1.3, 1.4, 1.5]) * (i + 1),
        )
        predictions.append(bounds)
    return predictions


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
