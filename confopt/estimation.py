import logging
from typing import Dict, Optional, List, Tuple

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from confopt.config import (
    GBM_NAME,
    QRF_NAME,
    QGBM_NAME,
    QKNN_NAME,
    DNN_NAME,
    GP_NAME,
    KNN_NAME,
    KR_NAME,
    RF_NAME,
    QL_NAME,
    QLGBM_NAME,
    LGBM_NAME,
    QUANTILE_ESTIMATOR_ARCHITECTURES,
)
from confopt.quantile_wrappers import (
    QuantileGBM,
    QuantileLightGBM,
    QuantileForest,
    QuantileKNN,
    BaseSingleFitQuantileEstimator,
)
from confopt.utils import get_perceptron_layers, get_tuning_configurations

logger = logging.getLogger(__name__)

SEARCH_MODEL_TUNING_SPACE: Dict[str, Dict] = {
    DNN_NAME: {
        "solver": ["adam", "sgd"],
        "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 3, 10],
        "hidden_layer_sizes": get_perceptron_layers(
            n_layers_grid=[2, 3, 4], layer_size_grid=[16, 32, 64, 128]
        ),
    },
    RF_NAME: {
        "n_estimators": [25, 50, 100, 150, 200],
        "max_features": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 3],
    },
    KNN_NAME: {"n_neighbors": [1, 2, 3]},
    LGBM_NAME: {
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8],
        "n_estimators": [25, 50, 100, 200],
        "max_depth": [2, 3, 5, 10],
    },
    GBM_NAME: {
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8],
        "n_estimators": [25, 50, 100, 200],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 3, 5],
        "max_depth": [2, 3, 5, 10],
    },
    GP_NAME: {"kernel": [RBF(), RationalQuadratic()]},
    KR_NAME: {"alpha": [0.001, 0.1, 1, 10], "kernel": ["linear", "rbf", "polynomial"]},
    QRF_NAME: {"n_estimators": [25, 50, 100, 150, 200]},
    QKNN_NAME: {"n_neighbors": [5]},
    QL_NAME: {
        "alpha": [0.01, 0.1, 1.0],
        "max_iter": [500, 1000],
    },
    QGBM_NAME: {
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8],
        "n_estimators": [25, 50, 100, 200],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 3, 5],
        "max_depth": [2, 3, 5, 10],
    },
    QLGBM_NAME: {
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8],
        "n_estimators": [25, 50, 100, 200],
        "max_depth": [2, 3, 5, 10],
    },
}

SEARCH_MODEL_DEFAULT_CONFIGURATIONS: Dict[str, Dict] = {
    DNN_NAME: {
        "solver": "adam",
        "learning_rate_init": 0.001,
        "alpha": 0.1,
        "hidden_layer_sizes": (32, 16),
    },
    RF_NAME: {
        "n_estimators": 50,
        "max_features": 0.8,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
    },
    KNN_NAME: {"n_neighbors": 2},
    GBM_NAME: {
        "learning_rate": 0.1,
        "n_estimators": 50,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_depth": 3,
    },
    LGBM_NAME: {
        "learning_rate": 0.1,
        "n_estimators": 50,
        "max_depth": 3,
    },
    GP_NAME: {"kernel": RBF()},
    KR_NAME: {"alpha": 0.1, "kernel": "rbf"},
    QRF_NAME: {"n_estimators": 50},
    QKNN_NAME: {"n_neighbors": 5},
    QL_NAME: {
        "alpha": 0.1,
        "max_iter": 1000,
    },
    QGBM_NAME: {
        "learning_rate": 0.1,
        "n_estimators": 50,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_depth": 3,
    },
    QLGBM_NAME: {
        "learning_rate": 0.1,
        "n_estimators": 50,
        "max_depth": 3,
    },
}


def tune(
    X: np.array,
    y: np.array,
    estimator_architecture: str,
    n_searches: int,
    quantiles: Optional[List[float]] = None,
    k_fold_splits: int = 3,
    random_state: Optional[int] = None,
) -> Dict:
    tuning_configurations = get_tuning_configurations(
        parameter_grid=SEARCH_MODEL_TUNING_SPACE[estimator_architecture],
        n_configurations=n_searches,
        random_state=random_state,
    )
    tuning_configurations.append(
        SEARCH_MODEL_DEFAULT_CONFIGURATIONS[estimator_architecture]
    )

    scored_configurations, scores = cross_validate_configurations(
        configurations=tuning_configurations,
        estimator_architecture=estimator_architecture,
        X=X,
        y=y,
        k_fold_splits=k_fold_splits,
        quantiles=quantiles,
        random_state=random_state,
    )
    best_configuration = scored_configurations[scores.index(min(scores))]

    return best_configuration


def initialize_quantile_estimator(
    estimator_architecture: str,
    initialization_params: Dict,
    pinball_loss_alpha: List[float],
    random_state: Optional[int] = None,
):
    """
    Initialize a quantile estimator from an input dictionary.

    Classes are usually external dependancies or custom wrappers or
    scikit-learn estimator classes. Passed dictionaries must
    contain all required inputs for the class, in addition to any
    optional inputs to be overridden.

    Parameters
    ----------
    estimator_architecture :
        String name for the type of estimator to initialize.
    initialization_params :
        Dictionary of initialization parameters, where each key and
        value pair corresponds to a variable name and variable value
        to pass to the estimator class to initialize.
    pinball_loss_alpha :
        List of pinball loss alpha levels that will result in the
        estimator predicting the alpha-corresponding quantiles.
        For eg. passing [0.25, 0.75] will initialize a quantile
        estimator that predicts the 25th and 75th percentiles of
        the data.
    random_state :
        Random generation seed.

    Returns
    -------
    initialized_model :
        An initialized estimator class instance.
    """
    if estimator_architecture == QGBM_NAME:
        initialized_model = QuantileGBM(
            **initialization_params,
            quantiles=pinball_loss_alpha,
            random_state=random_state,
        )
    elif estimator_architecture == QLGBM_NAME:
        initialized_model = QuantileLightGBM(
            **initialization_params,
            quantiles=pinball_loss_alpha,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"{estimator_architecture} is not a valid estimator architecture."
        )

    return initialized_model


def average_scores_across_folds(
    scored_configurations: List[List[Tuple[str, float]]], scores: List[float]
) -> Tuple[List[List[Tuple[str, float]]], List[float]]:
    # Use a list to store aggregated scores and fold counts
    aggregated_scores = []
    fold_counts = []
    aggregated_configurations = []

    for configuration, score in zip(scored_configurations, scores):
        # Check if the configuration already exists in the aggregated_configurations list
        if configuration in aggregated_configurations:
            index = aggregated_configurations.index(configuration)
            aggregated_scores[index] += score
            fold_counts[index] += 1
        else:
            aggregated_configurations.append(configuration)
            aggregated_scores.append(score)
            fold_counts.append(1)

    # Calculate the average scores
    for i in range(len(aggregated_scores)):
        aggregated_scores[i] /= fold_counts[i]

    return aggregated_configurations, aggregated_scores


def initialize_point_estimator(
    estimator_architecture: str,
    initialization_params: Dict,
    random_state: Optional[int] = None,
):
    """
    Initialize a point estimator from an input dictionary.

    Classes are usually scikit-learn estimators and dictionaries must
    contain all required inputs for the class, in addition to any
    optional inputs to be overridden.

    Parameters
    ----------
    estimator_architecture :
        String name for the type of estimator to initialize.
    initialization_params :
        Dictionary of initialization parameters, where each key and
        value pair corresponds to a variable name and variable value
        to pass to the estimator class to initialize.
    random_state :
        Random generation seed.

    Returns
    -------
    initialized_model :
        An initialized estimator class instance.
    """
    if estimator_architecture == DNN_NAME:
        initialized_model = MLPRegressor(
            **initialization_params, random_state=random_state
        )
    elif estimator_architecture == RF_NAME:
        initialized_model = RandomForestRegressor(
            **initialization_params, random_state=random_state
        )
    elif estimator_architecture == KNN_NAME:
        initialized_model = KNeighborsRegressor(**initialization_params)
    elif estimator_architecture == GBM_NAME:
        initialized_model = GradientBoostingRegressor(
            **initialization_params, random_state=random_state
        )
    elif estimator_architecture == LGBM_NAME:
        initialized_model = LGBMRegressor(
            **initialization_params, random_state=random_state, verbose=-1
        )
    elif estimator_architecture == GP_NAME:
        initialized_model = GaussianProcessRegressor(
            **initialization_params, random_state=random_state
        )
    elif estimator_architecture == KR_NAME:
        initialized_model = KernelRidge(**initialization_params)
    elif estimator_architecture == QRF_NAME:
        initialized_model = QuantileForest(
            **initialization_params, random_state=random_state
        )
    elif estimator_architecture == QKNN_NAME:
        initialized_model = QuantileKNN(**initialization_params)
    else:
        raise ValueError(
            f"{estimator_architecture} is not a valid point estimator architecture."
        )

    return initialized_model


def cross_validate_configurations(
    configurations: List[Dict],
    estimator_architecture: str,
    X: np.array,
    y: np.array,
    k_fold_splits: int = 3,
    quantiles: Optional[List[float]] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[Dict], List[float]]:
    """
    Cross validate a specified estimator on a passed X, y dataset.

    Cross validation loops through a list of passed hyperparameter
    configurations for the previously specified estimator and returns
    an average score across folds for each.

    Parameters
    ----------
    configurations :
        List of estimator parameter configurations, where each
        configuration contains all parameter values necessary
        to create an estimator instance.
    estimator_architecture :
        String name for the type of estimator to cross validate.
    X :
        Explanatory variables to train estimator on.
    y :
        Target variable to train estimator on.
    k_fold_splits :
        Number of cross validation data splits.
    quantiles :
        If the estimator to cross validate is a quantile estimator,
        specify the quantiles it should estimate as a list in this
        variable (eg. [0.25, 0.75] will cross validate an estimator
        predicting the 25th and 75th percentiles of the target variable).
    random_state :
        Random generation seed.

    Returns
    -------
    cross_fold_scored_configurations :
        List of cross validated configurations.
    cross_fold_scores :
        List of corresponding cross validation scores (averaged across
        folds).
    """
    scored_configurations, scores = [], []
    kf = KFold(n_splits=k_fold_splits, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index, :], X[test_index, :]
        Y_train, Y_val = y[train_index], y[test_index]

        for configuration in configurations:
            logger.debug(
                f"Evaluating search model parameter configuration: {configuration}"
            )
            if estimator_architecture in QUANTILE_ESTIMATOR_ARCHITECTURES:
                if quantiles is None:
                    raise ValueError(
                        "'quantiles' cannot be None if passing a quantile regression estimator."
                    )
                else:
                    model = initialize_quantile_estimator(
                        estimator_architecture=estimator_architecture,
                        initialization_params=configuration,
                        pinball_loss_alpha=quantiles,
                        random_state=random_state,
                    )
            else:
                model = initialize_point_estimator(
                    estimator_architecture=estimator_architecture,
                    initialization_params=configuration,
                    random_state=random_state,
                )
            model.fit(X_train, Y_train)

            try:
                if estimator_architecture in QUANTILE_ESTIMATOR_ARCHITECTURES:
                    if quantiles is None:
                        raise ValueError(
                            "'quantiles' cannot be None if passing a quantile regression estimator."
                        )
                    else:
                        # Then evaluate on pinball loss:
                        prediction = model.predict(X_val)
                        lo_y_pred = prediction[:, 0]
                        hi_y_pred = prediction[:, 1]
                        lo_score = mean_pinball_loss(
                            Y_val, lo_y_pred, alpha=quantiles[0]
                        )
                        hi_score = mean_pinball_loss(
                            Y_val, hi_y_pred, alpha=quantiles[1]
                        )
                        score = (lo_score + hi_score) / 2
                elif isinstance(model, BaseSingleFitQuantileEstimator):
                    prediction = model.predict(X_val, quantiles=quantiles)
                    scores = []
                    for i, quantile in enumerate(quantiles):
                        y_pred = prediction[:, i]
                        quantile_score = mean_pinball_loss(
                            Y_val, y_pred, alpha=quantile
                        )
                        scores.append(quantile_score)
                    score = sum(scores) / len(scores)
                else:
                    # Then evaluate on MSE:
                    y_pred = model.predict(X=X_val)
                    score = mean_squared_error(Y_val, y_pred)

                scored_configurations.append(configuration)
                scores.append(score)

            except Exception as e:
                logger.warning(
                    "Scoring failed and result was not appended."
                    f"Caught exception: {e}"
                )
                continue

    cross_fold_scored_configurations, cross_fold_scores = average_scores_across_folds(
        scored_configurations=scored_configurations, scores=scores
    )

    return cross_fold_scored_configurations, cross_fold_scores
