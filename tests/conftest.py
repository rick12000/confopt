import random
from typing import Dict

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from confopt.estimation import (
    QuantileConformalRegression,
    LocallyWeightedConformalRegression,
)
from confopt.tuning import ConformalSearcher
from confopt.utils import get_tuning_configurations

DEFAULT_SEED = 1234

# Dummy made up search space:
DUMMY_PARAMETER_GRID: Dict = {
    "int_parameter": [1, 2, 3, 4, 5],
    "float_parameter": [1.1, 2.2, 3.3, 4.4],
}

# Dummy search space for a GBM model:
DUMMY_GBM_PARAMETER_GRID: Dict = {
    "n_estimators": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
}


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
    qcr = QuantileConformalRegression(quantile_estimator_architecture="qgbm")
    return qcr


@pytest.fixture
def dummy_init_locally_weighted_regression():
    lwr = LocallyWeightedConformalRegression(
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
    return DUMMY_PARAMETER_GRID


@pytest.fixture
def dummy_configurations(dummy_parameter_grid):
    """
    Samples unique configurations from broader
    possible values in dummy hyperparameter search space.
    """
    max_configurations = 100
    tuning_configurations = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=max_configurations,
        random_state=DEFAULT_SEED,
    )
    return tuning_configurations


@pytest.fixture
def dummy_gbm_parameter_grid():
    return DUMMY_GBM_PARAMETER_GRID


@pytest.fixture
def dummy_gbm_configurations(dummy_gbm_parameter_grid):
    max_configurations = 60
    gbm_tuning_configurations = get_tuning_configurations(
        parameter_grid=dummy_gbm_parameter_grid,
        n_configurations=max_configurations,
        random_state=DEFAULT_SEED,
    )
    return gbm_tuning_configurations


@pytest.fixture
def dummy_initialized_conformal_searcher__gbm_mse(
    dummy_stationary_gaussian_dataset, dummy_gbm_parameter_grid
):
    """
    Creates a conformal searcher instance from dummy raw X, y data
    and a dummy parameter grid.

    This particular fixture is set to optimize a GBM base model on
    regression data, using an MSE objective. The model architecture
    and type of data are arbitrarily pinned; more fixtures could
    be created to test other model or data types.
    """
    custom_loss_function = "mean_squared_error"
    prediction_type = "regression"
    model = GradientBoostingRegressor()

    X, y = (
        dummy_stationary_gaussian_dataset[:, 0].reshape(-1, 1),
        dummy_stationary_gaussian_dataset[:, 1],
    )
    train_split = 0.5
    X_train, y_train = (
        X[: round(len(X) * train_split), :],
        y[: round(len(y) * train_split)],
    )
    X_val, y_val = X[round(len(X) * train_split) :, :], y[round(len(y) * train_split) :]

    searcher = ConformalSearcher(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        search_space=dummy_gbm_parameter_grid,
        prediction_type=prediction_type,
        custom_loss_function=custom_loss_function,
    )

    return searcher
