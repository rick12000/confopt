import logging
import math
from typing import Dict, Optional, List

import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from acho.config import (
    GBM_NAME,
    QRF_NAME,
    QGBM_NAME,
    DNN_NAME,
    GP_NAME,
    KNN_NAME,
    KR_NAME,
    RF_NAME,
    QUANTILE_MODEL_TYPES,
)
from acho.optimization import RuntimeTracker
from acho.quantile_wrappers import QuantileGBM
from acho.utils import get_tuning_configurations, get_perceptron_layers

logger = logging.getLogger(__name__)

CONFORMAL_MODEL_SEARCH_SPACE: Dict = {
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
    GBM_NAME: {
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8],
        "n_estimators": [25, 50, 100, 200],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 3, 5],
        "max_depth": [2, 3, 5, 10],
    },
    GP_NAME: {"kernel": [RBF(), RationalQuadratic()]},
    KR_NAME: {"alpha": [0.001, 0.1, 1, 10]},
    QRF_NAME: {"n_estimators": [25, 50, 100, 150, 200]},
    QGBM_NAME: {
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8],
        "n_estimators": [25, 50, 100, 200],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 3, 5],
        "max_depth": [2, 3, 5, 10],
    },
}


def initialize_point_estimator(
    estimator_name: str, initialization_params: Dict, random_state: Optional[int] = None
):
    if estimator_name == DNN_NAME:
        initialized_model = MLPRegressor(
            **initialization_params, random_state=random_state
        )
    elif estimator_name == RF_NAME:
        initialized_model = RandomForestRegressor(
            **initialization_params, random_state=random_state
        )
    elif estimator_name == KNN_NAME:
        initialized_model = KNeighborsRegressor(**initialization_params)
    elif estimator_name == GBM_NAME:
        initialized_model = GradientBoostingRegressor(
            **initialization_params, random_state=random_state
        )
    elif estimator_name == GP_NAME:
        initialized_model = GaussianProcessRegressor(
            **initialization_params, random_state=random_state
        )
    elif estimator_name == KR_NAME:
        initialized_model = KernelRidge(**initialization_params)
    else:
        raise ValueError(f"{estimator_name} is not a valid point estimator name.")

    return initialized_model


def initialize_quantile_estimator(
    estimator_name: str,
    initialization_params: Dict,
    pinball_loss_alpha: List,
    random_state: Optional[int] = None,
):
    if estimator_name == QRF_NAME:
        initialized_model = RandomForestQuantileRegressor(
            **initialization_params,
            default_quantiles=pinball_loss_alpha,
            random_state=random_state,
        )
    elif estimator_name == QGBM_NAME:
        initialized_model = QuantileGBM(
            **initialization_params,
            quantiles=pinball_loss_alpha,
            random_state=random_state,
        )
    else:
        raise ValueError(f"{estimator_name} is not a valid estimator name.")

    return initialized_model


def cross_validate_estimator_configurations(
    configurations: List[Dict],
    estimator_type: str,
    X: np.array,
    y: np.array,
    k_fold_splits: int,
    scoring_function: str,
    quantiles: List[float] = None,
    random_state: int = None,
):
    if k_fold_splits is None:
        k_fold_splits = min(3, len(X))
    logger.debug(f"Cross validating with {k_fold_splits} splits...")

    kf = KFold(n_splits=k_fold_splits, random_state=random_state, shuffle=True)

    logger.debug(f"Length of X: {X.shape}. " f"Length of y: {y.shape}.")
    scored_configurations, scores = [], []
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index, :], X[test_index, :]
        Y_train, Y_val = y[train_index], y[test_index]

        for configuration in configurations:
            logger.debug(
                f"Searching conformal estimator parameter configuration: {configuration}"
            )
            if estimator_type in QUANTILE_MODEL_TYPES:
                fitted_model = initialize_quantile_estimator(
                    estimator_name=estimator_type,
                    initialization_params=configuration,
                    pinball_loss_alpha=quantiles,
                    random_state=random_state,
                )
            else:
                fitted_model = initialize_point_estimator(
                    estimator_name=estimator_type,
                    initialization_params=configuration,
                    random_state=random_state,
                )
            fitted_model.fit(X_train, Y_train)
            y_pred = fitted_model.predict(X_val)
            try:
                if scoring_function == "root_mean_squared_error":
                    score = math.sqrt(metrics.mean_squared_error(Y_val, y_pred))
                elif scoring_function == "mean_pinball_loss":
                    y_pred_lo = fitted_model.predict(X_val)[:, 0]
                    y_pred_hi = fitted_model.predict(X_val)[:, 1]
                    score_lo = mean_pinball_loss(Y_val, y_pred_lo, alpha=quantiles[0])
                    score_hi = mean_pinball_loss(Y_val, y_pred_hi, alpha=quantiles[1])
                    score = (score_lo + score_hi) / 2
                elif scoring_function == "mean_squared_error":
                    score = metrics.mean_squared_error(Y_val, y_pred)
                else:
                    raise ValueError(
                        f"{scoring_function} is not a supported scoring function."
                    )

                scored_configurations.append(configuration)
                scores.append(score)

            except Exception as e:
                logger.warning(
                    "Scoring failed and result was not appended."
                    f"Caught exception: {e}"
                )

    cross_fold_scored_configurations, cross_fold_scores = [], []
    for configuration in configurations:
        configuration_scores = [
            score
            for scored_configuration, score in zip(scored_configurations, scores)
            if scored_configuration == configuration
        ]
        if len(configuration_scores) > 0:
            configuration_average_score = sum(configuration_scores) / len(
                configuration_scores
            )
            cross_fold_scored_configurations.append(configuration)
            cross_fold_scores.append(configuration_average_score)

    return cross_fold_scored_configurations, cross_fold_scores


def tune_combinations(
    estimator_type: str,
    X: np.array,
    y: np.array,
    confidence_level: float,
    scoring_function: str,
    n_of_param_combinations: int,
    custom_best_param_combination: Optional[Dict] = None,
    random_state: Optional[int] = None,
):
    tuning_runtime_log = RuntimeTracker()

    parameter_grid = CONFORMAL_MODEL_SEARCH_SPACE[estimator_type]

    hyperparameter_configuration_list = get_tuning_configurations(
        parameter_grid=parameter_grid,
        n_configurations=n_of_param_combinations,
        random_state=random_state,
    )

    if custom_best_param_combination is not None:
        hyperparameter_configuration_list.append(custom_best_param_combination)

    logger.info(f"Tuning hyperparameters for {estimator_type} estimator...")

    if estimator_type in QUANTILE_MODEL_TYPES:
        quantiles = [
            ((1 - confidence_level) / 2),
            confidence_level + ((1 - confidence_level) / 2),
        ]
    else:
        quantiles = None
    scored_configurations, scores = cross_validate_estimator_configurations(
        configurations=hyperparameter_configuration_list,
        estimator_type=estimator_type,
        X=X,
        y=y,
        k_fold_splits=3,
        scoring_function=scoring_function,
        quantiles=quantiles,
        random_state=random_state,
    )
    logger.info(f"Tuning completed with optimal error of: {min(scores)}")

    optimal_parameters = scored_configurations[scores.index(max(scores))]
    logger.debug(f"Optimal conformal hyperparameters: {optimal_parameters}")

    tuning_runtime_per_configuration = tuning_runtime_log.return_runtime() / (
        len(hyperparameter_configuration_list)
    )

    return optimal_parameters, tuning_runtime_per_configuration


class LocallyWeightedConformalRegression:
    def __init__(
        self,
        point_estimator_architecture: str,
        demeaning_estimator_architecture: str,
        variance_estimator_architecture: str,
        random_state: int,
    ):
        self.point_estimator_architecture = point_estimator_architecture
        self.demeaning_estimator_architecture = demeaning_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture
        self.random_state = random_state

        self.nonconformity_scores = None
        self.trained_pe_estimator = None
        self.trained_ve_estimator = None
        self.tuning_runtime = None

        self.scoring_function = "mean_squared_error"

    def tune_fit(
        self,
        X_pe: np.array,
        y_pe: np.array,
        X_ve: np.array,
        y_ve: np.array,
        X_val: np.array,
        y_val: np.array,
        confidence_level: float,
        tuning_param_combinations: Optional[int] = 0,
        custom_best_pe_param_combination: Optional[Dict] = None,
        custom_best_de_param_combination: Optional[Dict] = None,
        custom_best_ve_param_combination: Optional[Dict] = None,
    ):
        if tuning_param_combinations > 1:
            optimal_pe_config, pe_tuning_runtime = tune_combinations(
                estimator_type=self.point_estimator_architecture,
                X=X_pe,
                y=y_pe,
                confidence_level=confidence_level,
                scoring_function=self.scoring_function,
                n_of_param_combinations=tuning_param_combinations,
                custom_best_param_combination=custom_best_pe_param_combination,
                random_state=self.random_state,
            )
        else:
            optimal_pe_config = custom_best_pe_param_combination.copy()
            pe_tuning_runtime = None
        optimal_pe_estimator = initialize_point_estimator(
            estimator_name=self.point_estimator_architecture,
            initialization_params=optimal_pe_config,
            random_state=self.random_state,
        )
        optimal_pe_estimator.fit(X_pe, y_pe)
        pe_residuals = np.array(y_ve) - np.array(optimal_pe_estimator.predict(X_ve))

        if tuning_param_combinations > 1:
            optimal_de_config, de_tuning_runtime = tune_combinations(
                estimator_type=self.demeaning_estimator_architecture,
                X=X_ve,
                y=pe_residuals,
                confidence_level=confidence_level,
                scoring_function=self.scoring_function,
                n_of_param_combinations=tuning_param_combinations,
                custom_best_param_combination=custom_best_de_param_combination,
                random_state=self.random_state,
            )
        else:
            optimal_de_config = custom_best_de_param_combination.copy()
            de_tuning_runtime = None

        optimal_de_estimator = initialize_point_estimator(
            estimator_name=self.demeaning_estimator_architecture,
            initialization_params=optimal_de_config,
            random_state=self.random_state,
        )
        optimal_de_estimator.fit(X_ve, pe_residuals)
        demeaned_pe_residuals = abs(pe_residuals - optimal_de_estimator.predict(X_ve))

        if tuning_param_combinations > 1:
            optimal_ve_config, ve_tuning_runtime = tune_combinations(
                estimator_type=self.variance_estimator_architecture,
                X=X_ve,
                y=demeaned_pe_residuals,
                confidence_level=confidence_level,
                scoring_function=self.scoring_function,
                n_of_param_combinations=tuning_param_combinations,
                custom_best_param_combination=custom_best_ve_param_combination,
                random_state=self.random_state,
            )
        else:
            optimal_ve_config = custom_best_ve_param_combination.copy()
            ve_tuning_runtime = None
        optimal_ve_estimator = initialize_point_estimator(
            estimator_name=self.variance_estimator_architecture,
            initialization_params=optimal_ve_config,
            random_state=self.random_state,
        )
        optimal_ve_estimator.fit(X_ve, demeaned_pe_residuals)

        if tuning_param_combinations > 1:
            self.tuning_runtime = (
                pe_tuning_runtime + de_tuning_runtime + ve_tuning_runtime
            )

        var_array = optimal_ve_estimator.predict(X_val)
        var_array = np.array([1 if x <= 0 else x for x in var_array])
        self.nonconformity_scores = (
            abs(np.array(y_val) - optimal_pe_estimator.predict(X_val)) / var_array
        )
        self.trained_pe_estimator = optimal_pe_estimator
        self.trained_ve_estimator = optimal_ve_estimator

        return optimal_pe_config, optimal_de_config, optimal_ve_config

    def predict(self, X: np.array, confidence_level: float):
        conformal_quantile = np.percentile(
            self.nonconformity_scores, confidence_level * 100
        )

        y_full_pred = np.array(self.trained_pe_estimator.predict(X))

        var_array = self.trained_ve_estimator.predict(X)
        var_array = np.array([max(x, 0) for x in var_array])
        intervals = conformal_quantile * var_array

        max_bound_y = y_full_pred + intervals
        min_bound_y = y_full_pred - intervals

        return min_bound_y, max_bound_y


class QuantileConformalRegression:
    def __init__(self, quantile_estimator_architecture: str, random_state: int):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.random_state = random_state

        self.scoring_function = "mean_pinball_loss"
        self.tuning_runtime = None

    def tune_fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        confidence_level: float,
        tuning_param_combinations: Optional[int] = 0,
        custom_best_param_combination: Optional[Dict] = None,
    ):
        if tuning_param_combinations > 1:
            optimal_config, tuning_runtime = tune_combinations(
                estimator_type=self.quantile_estimator_architecture,
                X=X_train,
                y=y_train,
                confidence_level=confidence_level,
                scoring_function=self.scoring_function,
                n_of_param_combinations=tuning_param_combinations,
                custom_best_param_combination=custom_best_param_combination,
                random_state=self.random_state,
            )
            self.tuning_runtime = tuning_runtime
        else:
            optimal_config = custom_best_param_combination.copy()
        optimal_quantile_estimator = initialize_quantile_estimator(
            estimator_name=self.quantile_estimator_architecture,
            initialization_params=optimal_config,
            pinball_loss_alpha=[
                ((1 - confidence_level) / 2),
                confidence_level + ((1 - confidence_level) / 2),
            ],
            random_state=self.random_state,
        )
        optimal_quantile_estimator.fit(X_train, y_train)

        self.quant_reg = optimal_quantile_estimator

        lo_errors = list(self.quant_reg.predict(X_val)[:, 0] - y_val)
        hi_errors = list(y_val - self.quant_reg.predict(X_val)[:, 1])
        errors = []
        for lo, hi in zip(lo_errors, hi_errors):
            errors.append(max(lo, hi))
        errors = np.array(errors)

        self.errors = errors

        return optimal_config

    def predict(self, X: np.array, confidence_level: float):
        conformal_quantile = np.quantile(self.errors, confidence_level)
        min_bound_y = np.array(self.quant_reg.predict(X)[:, 0]) - conformal_quantile
        max_bound_y = np.array(self.quant_reg.predict(X)[:, 1]) + conformal_quantile

        return min_bound_y, max_bound_y
