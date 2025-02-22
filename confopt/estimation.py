import logging
from typing import Dict, Optional, List, Tuple, Union, Literal
from pydantic import BaseModel

import random
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
from confopt.preprocessing import train_val_split
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
from confopt.tracking import RuntimeTracker
from confopt.quantile_wrappers import (
    QuantileGBM,
    QuantileLightGBM,
    QuantileForest,
    QuantileKNN,
    BaseSingleFitQuantileEstimator,
)  # , QuantileKNN, QuantileLasso
from confopt.utils import get_tuning_configurations, get_perceptron_layers
from confopt.adaptation import ACI, DtACI

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


class QuantileInterval(BaseModel):
    lower_quantile: float
    upper_quantile: float


class UCBSampler:
    def __init__(
        self,
        beta_decay: str = "logarithmic_decay",
        beta: float = 1,
        c: float = 1,
        interval_width: float = 0.2,
        adapter_framework: Optional[str] = None,
    ):
        self.beta_decay = beta_decay
        self.beta = beta
        self.c = c
        self.interval_width = interval_width
        self.alpha = 1 - interval_width
        self.t = 1

        # Initialize adapter if specified
        self.adapter = self._initialize_adapter(adapter_framework)
        self.quantiles = self._calculate_quantiles()

    def _initialize_adapter(self, framework: Optional[str]):
        if framework == "ACI":
            return ACI(alpha=self.alpha)
        elif framework == "DtACI":
            return DtACI(alpha=self.alpha)
        return None

    def _calculate_quantiles(self) -> QuantileInterval:
        return QuantileInterval(
            lower_quantile=self.alpha / 2, upper_quantile=1 - (self.alpha / 2)
        )

    def fetch_alpha(self) -> float:
        return self.alpha

    def fetch_interval(self) -> QuantileInterval:
        return self.quantiles

    def update_exploration_step(self):
        if self.beta_decay == "logarithmic_decay":
            self.beta = self.c * np.log(self.t) / self.t
        elif self.beta_decay == "logarithmic_growth":
            self.beta = 2 * np.log(self.t + 1)
        self.t += 1

    def update_interval_width(self, breach: int):
        if self.adapter:
            self.alpha = self.adapter.update(breach_indicator=breach)
            self.quantiles = self._calculate_quantiles()


class ThompsonSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter_framework: Optional[str] = None,
        enable_optimistic_sampling: bool = False,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of Thompson quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.enable_optimistic_sampling = enable_optimistic_sampling

        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, n_quantiles + 1)
        ]
        self.quantiles, self.alphas = self._initialize_quantiles_and_alphas(
            starting_quantiles
        )
        self.adapters = self._initialize_adapters(adapter_framework)

    def _initialize_quantiles_and_alphas(self, starting_quantiles: List[float]):
        quantiles = []
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            quantiles.append(
                QuantileInterval(lower_quantile=lower, upper_quantile=upper)
            )
            alphas.append(1 - (upper - lower))
        return quantiles, alphas

    def _initialize_adapters(self, framework: Optional[str]):
        if not framework:
            return []

        adapter_class = (
            ACI if framework == "ACI" else DtACI if framework == "DtACI" else None
        )
        if not adapter_class:
            raise ValueError(f"Unknown adapter framework: {framework}")

        return [adapter_class(alpha=alpha) for alpha in self.alphas]

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def fetch_intervals(self) -> List[QuantileInterval]:
        return self.quantiles

    def update_interval_width(self, breaches: List[int]):
        for i, (adapter, breach) in enumerate(zip(self.adapters, breaches)):
            updated_alpha = adapter.update(breach_indicator=breach)
            self.alphas[i] = updated_alpha
            self.quantiles[i] = QuantileInterval(
                lower_quantile=updated_alpha / 2, upper_quantile=1 - (updated_alpha / 2)
            )


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


class LocallyWeightedConformalSearcher:
    """
    Locally weighted conformal regression.

    Fits sequential estimators on X and y data to form point and
    variability predictions for y.

    The class contains tuning, fitting and prediction methods.
    """

    def __init__(
        self,
        point_estimator_architecture: str,
        variance_estimator_architecture: str,
        sampler: Union[UCBSampler, ThompsonSampler],
    ):
        self.point_estimator_architecture = point_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture
        self.sampler = sampler

        self.training_time = None

    def _fit_component_estimator(
        self,
        X,
        y,
        estimator_architecture,
        tuning_iterations,
        random_state: Optional[int] = None,
    ):
        """
        Fit component estimator with option to tune.

        Component estimators are loosely defined, general use
        point estimators. Their final purpose is dependent on
        what X and y data is passed to the function (eg. if y is
        a target, a residual, etc.).

        Parameters
        ----------
        X :
            Explanatory variables.
        y :
            Target variable.
        estimator_architecture :
            String name for the type of estimator to tune.
        tuning_iterations :
            Number of tuning searches to perform (eg. 5 means
            the model will randomly select 5 hyperparameter
            configurations for the estimator to evaluate).
            To skip tuning during fitting, set this to 0.
        random_state :
            Random generation seed.

        Returns
        -------
        estimator :
            Fitted estimator object.
        """
        if tuning_iterations > 1 and len(X) > 10:
            initialization_params = tune(
                X=X,
                y=y,
                estimator_architecture=estimator_architecture,
                n_searches=tuning_iterations,
                quantiles=None,
                random_state=random_state,
            )
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                estimator_architecture
            ].copy()
        estimator = initialize_point_estimator(
            estimator_architecture=estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )
        estimator.fit(X, y)

        return estimator

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit conformal regression model on specified data.

        Fitting process involves the following sequential steps:
            1.  Fitting an estimator on a first portion of the
                data, training on X to predict y.
            2.  Obtaining residuals between the estimator and
                observed y's on a second portion of the data.
            3.  Fitting a conditional mean estimator on the
                residual data.
            4.  Using the mean estimator to de-mean the residual
                data.
            5.  Fitting an estimator to predict absolute, de-meaned
                residuals (residual spread around the local mean).
            6.  Using a third portion of the data as a conformal
                hold out set to calibrate intervals for the estimator.

        Parameters
        ----------
        X_pe :
            Explanatory variables used to train the point estimator.
        y_pe :
            Target variable used to train the point estimator.
        X_ve :
            Explanatory variables used to train the residual spread
            (variability) estimator.
        y_ve :
            Target variable used to train the residual spread
            (variability) estimator.
        X_val :
            Explanatory variables used to calibrate the point estimator.
        y_val :
            Target variable used to calibrate the point estimator.
        tuning_iterations :
            Number of tuning searches to perform (eg. 5 means
            the model will randomly select 5 hyperparameter
            configurations for the estimator to evaluate).
            To skip tuning during fitting, set this to 0.
        random_state :
            Random generation seed.
        """
        (X_pe, y_pe, X_ve, y_ve,) = train_val_split(
            X_train,
            y_train,
            train_split=0.75,
            normalize=False,
            random_state=random_state,
        )
        logger.debug(
            f"Obtained sub training set of size {X_pe.shape} "
            f"and sub validation set of size {X_ve.shape}"
        )

        training_time_tracker = RuntimeTracker()

        self.pe_estimator = self._fit_component_estimator(
            X=X_pe,
            y=y_pe,
            estimator_architecture=self.point_estimator_architecture,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )

        pe_residuals = y_ve - self.pe_estimator.predict(X_ve)
        abs_pe_residuals = abs(pe_residuals)

        self.ve_estimator = self._fit_component_estimator(
            X=X_ve,
            y=abs_pe_residuals,
            estimator_architecture=self.variance_estimator_architecture,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )
        var_pred = self.ve_estimator.predict(X_val)
        var_pred = np.array([1 if x <= 0 else x for x in var_pred])

        self.nonconformity_scores = (
            abs(np.array(y_val) - self.pe_estimator.predict(X_val)) / var_pred
        )
        self.training_time = training_time_tracker.return_runtime()

        # TODO: TEMP
        self.primary_estimator_error = mean_squared_error(
            self.pe_estimator.predict(X=X_val), y_val
        )
        # TODO: END OF TEMP

    def predict(self, X: np.array):
        y_pred = np.array(self.pe_estimator.predict(X)).reshape(-1, 1)
        var_pred = self.ve_estimator.predict(X)
        var_pred = np.array([max(x, 0) for x in var_pred]).reshape(-1, 1)
        if isinstance(self.sampler, UCBSampler):
            return self._predict_with_ucb(y_pred, var_pred)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(y_pred, var_pred)

    def _predict_with_ucb(self, y_pred: np.array, var_pred: np.array):
        score_quantile = np.quantile(
            self.nonconformity_scores, self.sampler.fetch_alpha()
        )
        scaled_score = score_quantile * var_pred
        self.predictions_per_interval = [
            np.hstack(
                [
                    y_pred - self.sampler.beta * scaled_score,
                    y_pred + self.sampler.beta * scaled_score,
                ]
            )
        ]
        lower_bound = y_pred - self.sampler.beta * scaled_score
        self.sampler.update_exploration_step()
        return lower_bound

    def _predict_with_thompson(self, y_pred: np.array, var_pred: np.array):
        self.predictions_per_interval = []
        for alpha in self.sampler.fetch_alphas():
            score_quantile = np.quantile(self.nonconformity_scores, alpha)
            scaled_score = score_quantile * var_pred
            self.predictions_per_interval.append(
                np.hstack([y_pred - scaled_score, y_pred + scaled_score])
            )

        predictions_per_quantile = np.hstack(self.predictions_per_interval)
        lower_bound = []
        for i in range(predictions_per_quantile.shape[0]):
            ts_idx = random.choice(range(self.sampler.n_quantiles))
            if self.sampler.enable_optimistic_sampling:
                lower_bound.append(
                    min(predictions_per_quantile[i, ts_idx], y_pred[i, 0])
                )
            else:
                lower_bound.append(predictions_per_quantile[i, ts_idx])
        lower_bound = np.array(lower_bound)

        return lower_bound

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        if isinstance(self.sampler, UCBSampler):
            self._update_with_ucb(sampled_idx, sampled_performance)
        elif isinstance(self.sampler, ThompsonSampler):
            self._update_with_thompson(sampled_idx, sampled_performance)

    def _update_with_ucb(self, sampled_idx: int, sampled_performance: float):
        if (
            self.predictions_per_interval[0][sampled_idx, 0]
            <= sampled_performance
            <= self.predictions_per_interval[0][sampled_idx, 1]
        ):
            breach = 0
        else:
            breach = 1
        self.sampler.update_interval_width(breach=breach)

    def _update_with_thompson(self, sampled_idx: int, sampled_performance: float):
        breaches = []
        for predictions in self.predictions_per_interval:
            sampled_predictions = predictions[sampled_idx, :]
            lower_quantile, upper_quantile = (
                sampled_predictions[0],
                sampled_predictions[1],
            )
            if lower_quantile <= sampled_performance <= upper_quantile:
                breach = 0
            else:
                breach = 1
            breaches.append(breach)
        self.sampler.update_interval_width(breaches=breaches)


class SingleFitQuantileConformalSearcher:
    def __init__(
        self,
        quantile_estimator_architecture: Literal["qknn", "qrf"],
        sampler: Union[UCBSampler, ThompsonSampler],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.training_time = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit quantile estimator with option to tune.

        Quantile estimators are fitted based on a specified confidence
        level and return two quantile estimates for the symmetrical
        lower and upper bounds around that level.

        Parameters
        ----------
        X_train :
            Explanatory variables used to train the quantile estimator.
        y_train :
            Target variable used to train the quantile estimator.
        X_val :
            Explanatory variables used to calibrate conformal intervals.
        y_val :
            Target variable used to calibrate conformal intervals.
        confidence_level :
            Confidence level determining quantiles to be predicted
            by the quantile estimator. Quantiles are obtained symmetrically
            around the confidence level (eg. 0.5 confidence level would
            result in a quantile estimator for the 25th and 75th percentiles
            of the target variable).
        tuning_iterations :
            Number of tuning searches to perform (eg. 5 means
            the model will randomly select 5 hyperparameter
            configurations for the quantile estimator to evaluate).
            To skip tuning during fitting, set this to 0.
        random_state :
            Random generation seed.

        Returns
        -------
        estimator :
            Fitted estimator object.
        """
        training_time_tracker = RuntimeTracker()
        training_time_tracker.pause_runtime()
        if isinstance(self.sampler, UCBSampler):
            quantile_intervals = [self.sampler.fetch_interval()]
        elif isinstance(self.sampler, ThompsonSampler):
            quantile_intervals = self.sampler.fetch_intervals()
            if self.sampler.enable_optimistic_sampling:
                pass

        training_time_tracker.resume_runtime()
        if tuning_iterations > 1 and len(X_train) > 10:
            flattened_quantiles = []
            for interval in quantile_intervals:
                flattened_quantiles.append(interval.lower_quantile)
                flattened_quantiles.append(interval.upper_quantile)
            initialization_params = tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                quantiles=flattened_quantiles,
                random_state=random_state,
            )
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                self.quantile_estimator_architecture
            ].copy()

        # TODO HERE
        self.quantile_estimator = initialize_point_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )

        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            self.quantile_estimator.fit(X_train, y_train)

            if isinstance(self.sampler, UCBSampler):
                self.nonconformity_scores_per_interval = []
                for interval in quantile_intervals:
                    val_prediction = self.quantile_estimator.predict(
                        X=X_val,
                        quantiles=[interval.lower_quantile, interval.upper_quantile],
                    )
                    lower_conformal_deviations = list(val_prediction[:, 0] - y_val)
                    upper_conformal_deviations = list(y_val - val_prediction[:, 1])
                    nonconformity_scores = []
                    for lower_deviation, upper_deviation in zip(
                        lower_conformal_deviations, upper_conformal_deviations
                    ):
                        nonconformity_scores.append(
                            max(lower_deviation, upper_deviation)
                        )
                    self.nonconformity_scores_per_interval.append(
                        np.array(nonconformity_scores)
                    )

            elif isinstance(self.sampler, ThompsonSampler):
                self.nonconformity_scores_per_interval = []
                for interval in quantile_intervals:
                    val_prediction = self.quantile_estimator.predict(
                        X=X_val,
                        quantiles=[interval.lower_quantile, interval.upper_quantile],
                    )
                    lower_conformal_deviations = list(val_prediction[:, 0] - y_val)
                    upper_conformal_deviations = list(y_val - val_prediction[:, 1])
                    nonconformity_scores = []
                    for lower_deviation, upper_deviation in zip(
                        lower_conformal_deviations, upper_conformal_deviations
                    ):
                        nonconformity_scores.append(
                            max(lower_deviation, upper_deviation)
                        )
                    self.nonconformity_scores_per_interval.append(
                        np.array(nonconformity_scores)
                    )

            self.conformalize_predictions = True

        else:
            self.quantile_estimator.fit(
                X=np.vstack((X_train, X_val)), y=np.concatenate((y_train, y_val))
            )
            self.conformalize_predictions = False

        self.training_time = training_time_tracker.return_runtime()

        # TODO: TEMP
        scores = []
        for quantile_interval in quantile_intervals:
            predictions = self.quantile_estimator.predict(
                X=X_val,
                quantiles=[
                    quantile_interval.lower_quantile,
                    quantile_interval.upper_quantile,
                ],
            )
            lo_y_pred = predictions[:, 0]
            hi_y_pred = predictions[:, 1]
            lo_score = mean_pinball_loss(
                y_val, lo_y_pred, alpha=quantile_interval.lower_quantile
            )
            hi_score = mean_pinball_loss(
                y_val, hi_y_pred, alpha=quantile_interval.upper_quantile
            )
            score = (lo_score + hi_score) / 2
            scores.append(score)
        self.primary_estimator_error = sum(scores) / len(scores)
        # TODO: END OF TEMP

    def predict(self, X: np.array):
        if isinstance(self.sampler, UCBSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)

    def _predict_with_ucb(self, X: np.array):
        if self.conformalize_predictions:
            interval = self.sampler.fetch_interval()
            score = np.quantile(
                self.nonconformity_scores_per_interval[0],
                interval.upper_quantile - interval.lower_quantile,
            )
        else:
            score = 0
        interval = self.sampler.fetch_interval()
        prediction = self.quantile_estimator.predict(
            X=X, quantiles=[interval.lower_quantile, interval.upper_quantile]
        )
        lower_interval_bound = np.array(prediction[:, 0]) - score
        upper_interval_bound = np.array(prediction[:, 1]) + score

        self.predictions_per_interval = [prediction]

        lower_bound = lower_interval_bound + self.sampler.beta * (
            upper_interval_bound - lower_interval_bound
        )

        self.sampler.update_exploration_step()

        return lower_bound

    def _predict_with_thompson(self, X):
        self.predictions_per_interval = []
        if self.conformalize_predictions:
            for nonconformity_scores, interval in zip(
                self.nonconformity_scores_per_interval, self.sampler.fetch_intervals()
            ):
                score = np.quantile(
                    nonconformity_scores,
                    interval.upper_quantile - interval.lower_quantile,
                )
                scores = [-score, score]
                predictions = self.quantile_estimator.predict(
                    X=X, quantiles=[interval.lower_quantile, interval.upper_quantile]
                )
                adjusted_predictions = predictions + np.array(scores).reshape(-1, 1).T
                self.predictions_per_interval.append(adjusted_predictions)
        else:
            for interval in self.sampler.fetch_intervals():
                predictions = self.quantile_estimator.predict(
                    X=X, quantiles=[interval.lower_quantile, interval.upper_quantile]
                )
                self.predictions_per_interval.append(predictions)

        if self.sampler.enable_optimistic_sampling:
            median_predictions = np.array(
                self.quantile_estimator.predict(X=X, quantiles=[0.5])[:, 0]
            ).reshape(-1, 1)

        predictions_per_quantile = np.hstack(self.predictions_per_interval)
        lower_bound = []
        for i in range(predictions_per_quantile.shape[0]):
            ts_idx = random.choice(range(self.sampler.n_quantiles))
            if self.sampler.enable_optimistic_sampling:
                lower_bound.append(
                    min(
                        predictions_per_quantile[i, ts_idx],
                        median_predictions[i, 0],
                    )
                )
            else:
                lower_bound.append(predictions_per_quantile[i, ts_idx])
        lower_bound = np.array(lower_bound)

        return lower_bound

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        if isinstance(self.sampler, UCBSampler):
            self._update_with_ucb(sampled_idx, sampled_performance)
        elif isinstance(self.sampler, ThompsonSampler):
            self._update_with_thompson(sampled_idx, sampled_performance)

    def _update_with_ucb(self, sampled_idx, sampled_performance):
        predictions_per_interval = self.predictions_per_interval[0]
        if (
            predictions_per_interval[sampled_idx, 0]
            <= sampled_performance
            <= predictions_per_interval[sampled_idx, 1]
        ):
            breach = 0
        else:
            breach = 1
        self.sampler.update_interval_width(breach=breach)

    def _update_with_thompson(self, sampled_idx, sampled_performance):
        breaches = []
        for predictions in self.predictions_per_interval:
            sampled_predictions = predictions[sampled_idx, :]
            lower_quantile, upper_quantile = (
                sampled_predictions[0],
                sampled_predictions[1],
            )
            if lower_quantile <= sampled_performance <= upper_quantile:
                breach = 0
            else:
                breach = 1
            breaches.append(breach)
        self.sampler.update_interval_width(breaches=breaches)


# TODO


class MultiFitQuantileConformalSearcher:
    """
    Quantile conformal regression.

    Fits quantile estimators on X and y data and applies non-conformity
    adjustments to validate quantile estimates.

    The class contains tuning, fitting and prediction methods.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        sampler: Union[UCBSampler, ThompsonSampler],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.training_time = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit quantile estimator with option to tune.

        Quantile estimators are fitted based on a specified confidence
        level and return two quantile estimates for the symmetrical
        lower and upper bounds around that level.

        Parameters
        ----------
        X_train :
            Explanatory variables used to train the quantile estimator.
        y_train :
            Target variable used to train the quantile estimator.
        X_val :
            Explanatory variables used to calibrate conformal intervals.
        y_val :
            Target variable used to calibrate conformal intervals.
        confidence_level :
            Confidence level determining quantiles to be predicted
            by the quantile estimator. Quantiles are obtained symmetrically
            around the confidence level (eg. 0.5 confidence level would
            result in a quantile estimator for the 25th and 75th percentiles
            of the target variable).
        tuning_iterations :
            Number of tuning searches to perform (eg. 5 means
            the model will randomly select 5 hyperparameter
            configurations for the quantile estimator to evaluate).
            To skip tuning during fitting, set this to 0.
        random_state :
            Random generation seed.

        Returns
        -------
        estimator :
            Fitted estimator object.
        """
        training_time_tracker = RuntimeTracker()
        training_time_tracker.pause_runtime()
        if isinstance(self.sampler, UCBSampler):
            quantile_intervals = [self.sampler.fetch_interval()]
        elif isinstance(self.sampler, ThompsonSampler):
            quantile_intervals = self.sampler.fetch_intervals()
            if self.sampler.enable_optimistic_sampling:
                training_time_tracker.resume_runtime()
                median_estimator_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                    self.quantile_estimator_architecture
                ].copy()
                self.median_estimator = initialize_quantile_estimator(
                    estimator_architecture=self.quantile_estimator_architecture,
                    initialization_params=median_estimator_params,
                    pinball_loss_alpha=[0.5],
                    random_state=random_state,
                )
                self.median_estimator.fit(
                    np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))
                )
                training_time_tracker.pause_runtime()

        training_time_tracker.resume_runtime()
        if tuning_iterations > 1 and len(X_train) > 10:
            params_per_interval = []
            for interval in quantile_intervals:
                initialization_params = tune(
                    X=X_train,
                    y=y_train,
                    estimator_architecture=self.quantile_estimator_architecture,
                    n_searches=tuning_iterations,
                    quantiles=[interval.lower_quantile, interval.upper_quantile],
                    random_state=random_state,
                )
                params_per_interval.append(initialization_params)
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                self.quantile_estimator_architecture
            ].copy()
            params_per_interval = [initialization_params] * len(quantile_intervals)

        self.estimators_per_interval = []
        for interval in quantile_intervals:
            quantile_estimator = initialize_quantile_estimator(
                estimator_architecture=self.quantile_estimator_architecture,
                initialization_params=initialization_params,
                pinball_loss_alpha=[interval.lower_quantile, interval.upper_quantile],
                random_state=random_state,
            )
            self.estimators_per_interval.append(quantile_estimator)

        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            for estimator in self.estimators_per_interval:
                estimator.fit(X_train, y_train)

            if isinstance(self.sampler, UCBSampler):
                self.nonconformity_scores_per_interval = []
                val_prediction = self.estimators_per_interval[0].predict(X_val)
                lower_conformal_deviations = list(val_prediction[:, 0] - y_val)
                upper_conformal_deviations = list(y_val - val_prediction[:, -1])
                nonconformity_scores = []
                for lower_deviation, upper_deviation in zip(
                    lower_conformal_deviations, upper_conformal_deviations
                ):
                    nonconformity_scores.append(max(lower_deviation, upper_deviation))
                self.nonconformity_scores_per_interval.append(
                    np.array(nonconformity_scores)
                )

            elif isinstance(self.sampler, ThompsonSampler):
                self.nonconformity_scores_per_interval = []
                for estimator in self.estimators_per_interval:
                    val_prediction = estimator.predict(X_val)
                    lower_conformal_deviations = list(val_prediction[:, 0] - y_val)
                    upper_conformal_deviations = list(y_val - val_prediction[:, 1])
                    nonconformity_scores = []
                    for lower_deviation, upper_deviation in zip(
                        lower_conformal_deviations, upper_conformal_deviations
                    ):
                        nonconformity_scores.append(
                            max(lower_deviation, upper_deviation)
                        )
                    self.nonconformity_scores_per_interval.append(
                        np.array(nonconformity_scores)
                    )

            self.conformalize_predictions = True

        else:
            for estimator in self.estimators_per_interval:
                estimator.fit(
                    np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))
                )

            self.conformalize_predictions = False

        self.training_time = training_time_tracker.return_runtime()

        # TODO: TEMP
        scores = []
        for quantile_interval, estimator in zip(
            quantile_intervals, self.estimators_per_interval
        ):
            predictions = estimator.predict(X_val)
            lo_y_pred = predictions[:, 0]
            hi_y_pred = predictions[:, 1]
            lo_score = mean_pinball_loss(
                y_val, lo_y_pred, alpha=quantile_interval.lower_quantile
            )
            hi_score = mean_pinball_loss(
                y_val, hi_y_pred, alpha=quantile_interval.upper_quantile
            )
            score = (lo_score + hi_score) / 2
            scores.append(score)
        self.primary_estimator_error = sum(scores) / len(scores)
        # TODO: END OF TEMP

    def predict(self, X: np.array):
        if isinstance(self.sampler, UCBSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)

    def _predict_with_ucb(self, X: np.array):
        if self.conformalize_predictions:
            interval = self.sampler.fetch_interval()
            score = np.quantile(
                self.nonconformity_scores_per_interval[0],
                interval.upper_quantile - interval.lower_quantile,
            )
        else:
            score = 0
        prediction = self.estimators_per_interval[0].predict(X)
        lower_interval_bound = np.array(prediction[:, 0]) - score
        upper_interval_bound = np.array(prediction[:, 1]) + score

        self.predictions_per_interval = [prediction]

        lower_bound = lower_interval_bound + self.sampler.beta * (
            upper_interval_bound - lower_interval_bound
        )

        self.sampler.update_exploration_step()

        return lower_bound

    def _predict_with_thompson(self, X):
        self.predictions_per_interval = []
        if self.conformalize_predictions:
            for nonconformity_scores, estimator in zip(
                self.nonconformity_scores_per_interval, self.estimators_per_interval
            ):
                score = np.quantile(
                    nonconformity_scores,
                    estimator.quantiles[1] - estimator.quantiles[0],
                )
                scores = [-score, score]
                predictions = estimator.predict(X)
                adjusted_predictions = predictions + np.array(scores).reshape(-1, 1).T
                self.predictions_per_interval.append(adjusted_predictions)
        else:
            for estimator in self.estimators_per_interval:
                predictions = estimator.predict(X)
                self.predictions_per_interval.append(predictions)

        if self.sampler.enable_optimistic_sampling:
            median_predictions = np.array(
                self.median_estimator.predict(X)[:, 0]
            ).reshape(-1, 1)

        predictions_per_quantile = np.hstack(self.predictions_per_interval)
        lower_bound = []
        for i in range(predictions_per_quantile.shape[0]):
            ts_idx = random.choice(range(self.sampler.n_quantiles))
            if self.sampler.enable_optimistic_sampling:
                lower_bound.append(
                    min(
                        predictions_per_quantile[i, ts_idx],
                        median_predictions[i, 0],
                    )
                )
            else:
                lower_bound.append(predictions_per_quantile[i, ts_idx])
        lower_bound = np.array(lower_bound)

        return lower_bound

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        if isinstance(self.sampler, UCBSampler):
            self._update_with_ucb(sampled_idx, sampled_performance)
        elif isinstance(self.sampler, ThompsonSampler):
            self._update_with_thompson(sampled_idx, sampled_performance)

    def _update_with_ucb(self, sampled_idx, sampled_performance):
        predictions_per_interval = self.predictions_per_interval[0]
        if (
            predictions_per_interval[sampled_idx, 0]
            <= sampled_performance
            <= predictions_per_interval[sampled_idx, 1]
        ):
            breach = 0
        else:
            breach = 1
        self.sampler.update_interval_width(breach=breach)

    def _update_with_thompson(self, sampled_idx, sampled_performance):
        breaches = []
        for predictions in self.predictions_per_interval:
            sampled_predictions = predictions[sampled_idx, :]
            lower_quantile, upper_quantile = (
                sampled_predictions[0],
                sampled_predictions[1],
            )
            if lower_quantile <= sampled_performance <= upper_quantile:
                breach = 0
            else:
                breach = 1
            breaches.append(breach)
        self.sampler.update_interval_width(breaches=breaches)
