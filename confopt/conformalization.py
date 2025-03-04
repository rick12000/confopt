import logging
import numpy as np
from typing import Optional, Tuple
from sklearn.metrics import mean_squared_error

from confopt.preprocessing import train_val_split
from confopt.tracking import RuntimeTracker
from confopt.estimation import (
    initialize_point_estimator,
    tune,
    SEARCH_MODEL_DEFAULT_CONFIGURATIONS,
)

logger = logging.getLogger(__name__)


class LocallyWeightedConformalEstimator:
    """
    Base conformal estimator that fits point and variance estimators
    and produces conformal intervals.
    """

    def __init__(
        self,
        point_estimator_architecture: str,
        variance_estimator_architecture: str,
    ):
        self.point_estimator_architecture = point_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture

        self.pe_estimator = None
        self.ve_estimator = None
        self.nonconformity_scores = None
        self.training_time = None
        self.primary_estimator_error = None

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

        # Performance metric
        self.primary_estimator_error = mean_squared_error(
            self.pe_estimator.predict(X=X_val), y_val
        )

    def predict_interval(
        self, X: np.array, alpha: float, beta: float = 1.0
    ) -> Tuple[np.array, np.array]:
        """
        Predict conformal intervals for a given confidence level.

        Parameters
        ----------
        X : np.array
            Input features
        alpha : float
            Confidence level (between 0 and 1)
        beta : float
            Scaling factor for the interval width

        Returns
        -------
        Tuple[np.array, np.array]
            Lower and upper bounds of the confidence interval
        """
        if self.pe_estimator is None or self.ve_estimator is None:
            raise ValueError("Estimators must be fitted before prediction")

        y_pred = np.array(self.pe_estimator.predict(X)).reshape(-1, 1)
        var_pred = self.ve_estimator.predict(X)
        var_pred = np.array([max(x, 0) for x in var_pred]).reshape(-1, 1)

        score_quantile = np.quantile(self.nonconformity_scores, 1 - alpha)
        scaled_score = score_quantile * var_pred

        lower_bound = y_pred - beta * scaled_score
        upper_bound = y_pred + beta * scaled_score

        return lower_bound, upper_bound
