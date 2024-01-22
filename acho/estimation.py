import logging
from typing import Dict, Optional, List, Tuple

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
    QUANTILE_ESTIMATOR_ARCHITECTURES,
)
from acho.optimization import RuntimeTracker
from acho.quantile_wrappers import QuantileGBM
from acho.utils import get_tuning_configurations, get_perceptron_layers

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

SEARCH_MODEL_DEFAULT_CONFIGURATIONS: Dict[str, Dict] = {
    DNN_NAME: {
        "solver": "adam",
        "learning_rate_init": 0.001,
        "alpha": 0.1,
        "hidden_layer_sizes": (32, 16),
    },
    RF_NAME: {
        "n_estimators": 150,
        "max_features": 0.8,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
    },
    KNN_NAME: {"n_neighbors": 2},
    GBM_NAME: {
        "learning_rate": 0.2,
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_depth": 3,
    },
    GP_NAME: {"kernel": RBF()},
    KR_NAME: {"alpha": 0.1},
    QRF_NAME: {"n_estimators": 100},
    QGBM_NAME: {
        "learning_rate": 0.2,
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_depth": 3,
    },
}


def initialize_point_estimator(
    estimator_architecture: str,
    initialization_params: Dict,
    random_state: Optional[int] = None,
):
    """
    Initialize a point estimator class from an input dictionary.

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
    elif estimator_architecture == GP_NAME:
        initialized_model = GaussianProcessRegressor(
            **initialization_params, random_state=random_state
        )
    elif estimator_architecture == KR_NAME:
        initialized_model = KernelRidge(**initialization_params)
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
    Initialize a quantile estimator class from an input dictionary.

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
    if estimator_architecture == QRF_NAME:
        initialized_model = RandomForestQuantileRegressor(
            **initialization_params,
            default_quantiles=pinball_loss_alpha,
            random_state=random_state,
        )
    elif estimator_architecture == QGBM_NAME:
        initialized_model = QuantileGBM(
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
    scored_configurations: List[Dict], scores: List[float]
) -> Tuple[List[Dict], List[float]]:
    # TODO: Refactor so it's more efficient or contained.
    #  This is a very convoluted function that does something
    #  very simple.
    aggregated_scores = {}
    fold_counts = {}

    for configuration, score in zip(scored_configurations, scores):
        tuplified_configuration = tuple(configuration.items())
        if tuplified_configuration not in aggregated_scores:
            aggregated_scores[tuplified_configuration] = score
            fold_counts[tuplified_configuration] = 1
        else:
            aggregated_scores[tuplified_configuration] += score
            fold_counts[tuplified_configuration] += 1

    for tuplified_configuration in aggregated_scores:
        aggregated_scores[tuplified_configuration] /= fold_counts[
            tuplified_configuration
        ]

    aggregated_configurations = [
        dict(list(tuplified_configuration))
        for tuplified_configuration in list(aggregated_scores.keys())
    ]
    aggregated_scores = list(aggregated_scores.values())

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
    Cross validates a specified estimator on a passed X, y dataset.

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
            y_pred = model.predict(X_val)

            try:
                if estimator_architecture in QUANTILE_ESTIMATOR_ARCHITECTURES:
                    if quantiles is None:
                        raise ValueError(
                            "'quantiles' cannot be None if passing a quantile regression estimator."
                        )
                    else:
                        # Then evaluate on pinball loss:
                        lo_y_pred = model.predict(X_val)[:, 0]
                        hi_y_pred = model.predict(X_val)[:, 1]
                        lo_score = mean_pinball_loss(
                            Y_val, lo_y_pred, alpha=quantiles[0]
                        )
                        hi_score = mean_pinball_loss(
                            Y_val, hi_y_pred, alpha=quantiles[1]
                        )
                        score = (lo_score + hi_score) / 2
                else:
                    # Then evaluate on MSE:
                    score = metrics.mean_squared_error(Y_val, y_pred)

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


class LocallyWeightedConformalRegression:
    """
    Carry out locally weighted conformal regression.

    Fits sequential estimators on X and y data to form point and
    variability predictions for y.

    The class contains tuning, fitting and prediction methods.
    """

    def __init__(
        self,
        point_estimator_architecture: str,
        demeaning_estimator_architecture: str,
        variance_estimator_architecture: str,
    ):
        self.point_estimator_architecture = point_estimator_architecture
        self.demeaning_estimator_architecture = demeaning_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture

        self.training_time = None

    def _tune_component_estimator(
        self,
        X: np.array,
        y: np.array,
        estimator_architecture: str,
        n_searches: int,
        k_fold_splits: int = 3,
        random_state: Optional[int] = None,
    ) -> Dict:
        """
        Tune specified estimator's hyperparameters.

        Hyperparameters are selected randomly as part of the
        tuning process and a final optimal hyperparameter
        configuration is returned.

        Parameters
        ----------
        X :
            Explanatory variables.
        y :
            Target variable.
        estimator_architecture :
            String name for the type of estimator to tune.
        n_searches :
            Number of tuning searches to perform (eg. 5 means
            the model will randomly select 5 hyperparameter
            configurations for the estimator to evaluate).
        k_fold_splits :
            Number of cross validation data splits.
        random_state :
            Random generation seed.

        Returns
        -------
        best_configuration :
            Best performing hyperparameter configuration
            in tuning.
        """
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
            quantiles=None,
            random_state=random_state,
        )
        best_configuration = scored_configurations[scores.index(max(scores))]

        return best_configuration

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
        if tuning_iterations > 1:
            initialization_params = self._tune_component_estimator(
                X=X,
                y=y,
                estimator_architecture=estimator_architecture,
                n_searches=tuning_iterations,
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
        self.training_time_tracker.resume_runtime()
        estimator.fit(X, y)
        self.training_time_tracker.pause_runtime()

        return estimator

    def fit(
        self,
        X_pe: np.array,
        y_pe: np.array,
        X_ve: np.array,
        y_ve: np.array,
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
        self.training_time_tracker = RuntimeTracker()
        self.training_time_tracker.pause_runtime()

        self.pe_estimator = self._fit_component_estimator(
            X=X_pe,
            y=y_pe,
            estimator_architecture=self.point_estimator_architecture,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )
        pe_residuals = y_ve - self.pe_estimator.predict(X_ve)

        de_estimator = self._fit_component_estimator(
            X=X_ve,
            y=pe_residuals,
            estimator_architecture=self.demeaning_estimator_architecture,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )
        demeaned_pe_residuals = abs(pe_residuals - de_estimator.predict(X_ve))

        self.ve_estimator = self._fit_component_estimator(
            X=X_ve,
            y=demeaned_pe_residuals,
            estimator_architecture=self.variance_estimator_architecture,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )

        var_pred = self.ve_estimator.predict(X_val)
        var_pred = np.array([1 if x <= 0 else x for x in var_pred])

        self.nonconformity_scores = (
            abs(np.array(y_val) - self.pe_estimator.predict(X_val)) / var_pred
        )
        self.training_time = self.training_time_tracker.return_runtime()

    def predict(self, X: np.array, confidence_level: float):
        """
        Predict conformal interval bounds for specified X examples.

        Must be called after a relevant conformal estimator has
        been trained.

        Parameters
        ----------
        X :
            Explanatory variables to return targets for.
        confidence_level :
            Confidence level used to generate intervals.

        Returns
        -------
        lower_interval_bound :
            Lower bound(s) of conformal interval for specified
            X example(s).
        upper_interval_bound :
            Upper bound(s) of conformal interval for specified
            X example(s).
        """
        score_quantile = np.quantile(self.nonconformity_scores, confidence_level)

        y_pred = np.array(self.pe_estimator.predict(X))

        var_pred = self.ve_estimator.predict(X)
        var_pred = np.array([max(x, 0) for x in var_pred])
        scaled_score = score_quantile * var_pred

        lower_interval_bound = y_pred - scaled_score
        upper_interval_bound = y_pred + scaled_score

        return lower_interval_bound, upper_interval_bound


class QuantileConformalRegression:
    """
    Carry out quantile conformal regression.

    Fits quantile estimators on X and y data and applies non-conformity
    adjustments to validate quantile estimates.

    The class contains tuning, fitting and prediction methods.
    """

    def __init__(self, quantile_estimator_architecture: str):
        self.quantile_estimator_architecture = quantile_estimator_architecture

        self.training_time = None

    def _tune(
        self,
        X: np.array,
        y: np.array,
        estimator_architecture: str,
        n_searches: int,
        confidence_level: float,
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
            quantiles=[
                ((1 - confidence_level) / 2),
                confidence_level + ((1 - confidence_level) / 2),
            ],
            random_state=random_state,
        )
        best_configuration = scored_configurations[scores.index(max(scores))]

        return best_configuration

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        confidence_level: float,
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
        if tuning_iterations > 1:
            initialization_params = self._tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                confidence_level=confidence_level,
                random_state=random_state,
            )
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                self.quantile_estimator_architecture
            ].copy()

        self.quantile_estimator = initialize_quantile_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            pinball_loss_alpha=[
                ((1 - confidence_level) / 2),
                confidence_level + ((1 - confidence_level) / 2),
            ],
            random_state=random_state,
        )
        training_time_tracker = RuntimeTracker()
        self.quantile_estimator.fit(X_train, y_train)
        self.training_time = training_time_tracker.return_runtime()

        lower_conformal_deviations = list(
            self.quantile_estimator.predict(X_val)[:, 0] - y_val
        )
        upper_conformal_deviations = list(
            y_val - self.quantile_estimator.predict(X_val)[:, 1]
        )
        nonconformity_scores = []
        for lower_deviation, upper_deviation in zip(
            lower_conformal_deviations, upper_conformal_deviations
        ):
            nonconformity_scores.append(max(lower_deviation, upper_deviation))
        self.nonconformity_scores = np.array(nonconformity_scores)

    def predict(self, X: np.array, confidence_level: float):
        """
        Predict conformal interval bounds for specified X examples.

        Must be called after a relevant quantile estimator has
        been trained. Intervals will be generated based on a passed
        confidence level, which should ideally be the same confidence
        level specified in training, but may differ (though this is
        less desirable and there should rarely be a valid reason).

        Parameters
        ----------
        X :
            Explanatory variables to return targets for.
        confidence_level :
            Confidence level used to generate intervals.

        Returns
        -------
        lower_interval_bound :
            Lower bound(s) of conformal interval for specified
            X example(s).
        upper_interval_bound :
            Upper bound(s) of conformal interval for specified
            X example(s).
        """
        score_quantile = np.quantile(self.nonconformity_scores, confidence_level)
        lower_interval_bound = (
            np.array(self.quantile_estimator.predict(X)[:, 0]) - score_quantile
        )
        upper_interval_bound = (
            np.array(self.quantile_estimator.predict(X)[:, 1]) + score_quantile
        )

        return lower_interval_bound, upper_interval_bound
