import pytest
import numpy as np
from acho.estimation import (
    QuantileConformalRegression,
    LocallyWeightedConformalRegression,
)
import random
from typing import Dict
from acho.utils import get_tuning_configurations
from acho.tuning import ConformalSearcher
from sklearn.ensemble import GradientBoostingRegressor

DEFAULT_SEED = 1234

# Only use numerical values for the grid fixtures, so their outputs don't need further postprocessing:
DUMMY_PARAMETER_GRID: Dict = {
    "int_parameter": [1, 2, 3, 4, 5],
    "float_parameter": [1.1, 2.2, 3.3, 4.4],
}

DUMMY_GBM_PARAMETER_GRID: Dict = {
    "n_estimators": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
}


# DUMMY_PARAMETER_GRID_COMPLEX_TYPES: Dict = {"tuple_": [1,2,3,4,5],
#                               "string_parameter":["value1", "value2"],
#                               "float_parameter":[1.1, 2.2, 3.3, 4.4]}


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
def dummy_performance_bounds():
    performance_lower_bounds = np.arange(0, 100, 0.5)
    performance_higher_bounds = performance_lower_bounds + 10
    return performance_lower_bounds, performance_higher_bounds


@pytest.fixture
def dummy_init_quantile_regression():
    qcr = QuantileConformalRegression(
        quantile_estimator_architecture="qgbm", random_state=DEFAULT_SEED
    )
    return qcr


@pytest.fixture
def dummy_init_locally_weighted_regression():
    lwr = LocallyWeightedConformalRegression(
        point_estimator_architecture="gbm",
        demeaning_estimator_architecture="gbm",
        variance_estimator_architecture="gbm",
        random_state=DEFAULT_SEED,
    )
    return lwr


@pytest.fixture
def dummy_parameter_grid():
    return DUMMY_PARAMETER_GRID


@pytest.fixture
def dummy_configurations(dummy_parameter_grid):
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


@pytest.fixture
def dummy_double_initialized_conformal_searcher__gbm_mse(
    dummy_stationary_gaussian_dataset, dummy_gbm_parameter_grid
):
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

    searcher_tuple = ()
    for _ in range(2):
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
        searcher_tuple = searcher_tuple + (searcher,)

    return searcher_tuple