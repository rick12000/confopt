import logging
import numpy as np
from typing import Optional, Tuple, List, Literal
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from pydantic import BaseModel

from confopt.preprocessing import train_val_split
from confopt.tracking import RuntimeTracker
from confopt.estimation import (
    initialize_point_estimator,
    initialize_quantile_estimator,
    tune,
    SEARCH_MODEL_DEFAULT_CONFIGURATIONS,
)

logger = logging.getLogger(__name__)


class MedianEstimator:
    """
    Simple wrapper for a median estimator used in optimistic sampling.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.median_estimator = None

    def fit(
        self,
        X: np.array,
        y: np.array,
        random_state: Optional[int] = None,
    ):
        """
        Fit a median (50th percentile) estimator.
        """
        initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
            self.quantile_estimator_architecture
        ].copy()

        self.median_estimator = initialize_quantile_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            pinball_loss_alpha=[0.5],
            random_state=random_state,
        )
        self.median_estimator.fit(X, y)

    def predict(self, X: np.array):
        """
        Predict median values.
        """
        if self.median_estimator is None:
            raise ValueError("Median estimator is not initialized")
        return np.array(self.median_estimator.predict(X)[:, 0])


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


class QuantileInterval(BaseModel):
    lower_quantile: float
    upper_quantile: float


class SingleFitQuantileConformalEstimator:
    """
    Single-fit quantile conformal estimator.

    Uses a single model that can predict multiple quantiles with a single fit.
    Can predict any quantile after fitting once.
    """

    def __init__(
        self,
        quantile_estimator_architecture: Literal["qknn", "qrf"],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.quantile_estimator = None
        self.nonconformity_scores = {}  # Store scores by interval
        self.conformalize_predictions = False
        self.training_time = None
        self.primary_estimator_error = None
        self.fitted_quantiles = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        intervals: List[QuantileInterval],
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit the single-fit quantile estimator for multiple intervals with one model.
        """
        training_time_tracker = RuntimeTracker()

        # Extract unique quantiles from all intervals
        all_quantiles = set()
        for interval in intervals:
            all_quantiles.add(interval.lower_quantile)
            all_quantiles.add(interval.upper_quantile)

        # Convert to sorted list
        self.fitted_quantiles = sorted(list(all_quantiles))

        # Tune model parameters if requested
        if tuning_iterations > 1 and len(X_train) > 10:
            initialization_params = tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                quantiles=self.fitted_quantiles,
                random_state=random_state,
            )
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                self.quantile_estimator_architecture
            ].copy()

        # Initialize and fit a single quantile estimator
        self.quantile_estimator = initialize_point_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )

        # Fit the model and calculate nonconformity scores if enough data
        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            self.quantile_estimator.fit(X_train, y_train)

            # Calculate nonconformity scores for each interval on validation data
            for interval in intervals:
                quantiles = [interval.lower_quantile, interval.upper_quantile]
                val_prediction = self.quantile_estimator.predict(
                    X=X_val,
                    quantiles=quantiles,
                )
                lower_conformal_deviations = val_prediction[:, 0] - y_val
                upper_conformal_deviations = y_val - val_prediction[:, 1]
                self.nonconformity_scores[self._interval_key(interval)] = np.maximum(
                    lower_conformal_deviations, upper_conformal_deviations
                )

            self.conformalize_predictions = True
        else:
            self.quantile_estimator.fit(
                X=np.vstack((X_train, X_val)), y=np.concatenate((y_train, y_val))
            )
            self.conformalize_predictions = False

        self.training_time = training_time_tracker.return_runtime()

        # Calculate performance metrics
        scores = []
        for interval in intervals:
            quantiles = [interval.lower_quantile, interval.upper_quantile]
            predictions = self.quantile_estimator.predict(
                X=X_val,
                quantiles=quantiles,
            )
            lo_y_pred = predictions[:, 0]
            hi_y_pred = predictions[:, 1]
            lo_score = mean_pinball_loss(
                y_val, lo_y_pred, alpha=interval.lower_quantile
            )
            hi_score = mean_pinball_loss(
                y_val, hi_y_pred, alpha=interval.upper_quantile
            )
            scores.append((lo_score + hi_score) / 2)

        self.primary_estimator_error = np.mean(scores)

    def _interval_key(self, interval: QuantileInterval) -> str:
        """Create a unique key for an interval to use in the nonconformity scores dictionary."""
        return f"{interval.lower_quantile}_{interval.upper_quantile}"

    def predict_interval(self, X: np.array, interval: QuantileInterval):
        """
        Predict conformal intervals for a specific interval.
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before prediction")

        quantiles = [interval.lower_quantile, interval.upper_quantile]
        prediction = self.quantile_estimator.predict(X=X, quantiles=quantiles)

        if self.conformalize_predictions:
            # Calculate conformity adjustment based on validation scores
            interval_key = self._interval_key(interval)
            if interval_key in self.nonconformity_scores:
                score = np.quantile(
                    self.nonconformity_scores[interval_key],
                    interval.upper_quantile - interval.lower_quantile,
                )
            else:
                # If we don't have exact scores for this interval, use the closest one
                closest_interval = self._find_closest_interval(interval)
                closest_key = self._interval_key(closest_interval)
                score = np.quantile(
                    self.nonconformity_scores[closest_key],
                    interval.upper_quantile - interval.lower_quantile,
                )
        else:
            score = 0

        lower_interval_bound = np.array(prediction[:, 0]) - score
        upper_interval_bound = np.array(prediction[:, 1]) + score

        return lower_interval_bound, upper_interval_bound

    def _find_closest_interval(
        self, target_interval: QuantileInterval
    ) -> QuantileInterval:
        """Find the closest interval in the nonconformity scores dictionary."""
        if not self.nonconformity_scores:
            return target_interval

        best_distance = float("inf")
        closest_interval = target_interval

        for interval_key in self.nonconformity_scores:
            lower, upper = map(float, interval_key.split("_"))
            current_interval = QuantileInterval(
                lower_quantile=lower, upper_quantile=upper
            )

            # Calculate distance between intervals
            distance = abs(
                current_interval.lower_quantile - target_interval.lower_quantile
            ) + abs(current_interval.upper_quantile - target_interval.upper_quantile)

            if distance < best_distance:
                best_distance = distance
                closest_interval = current_interval

        return closest_interval


class MultiFitQuantileConformalEstimator:
    """
    Multi-fit quantile conformal estimator for a single interval.

    Uses a dedicated quantile estimator for a specific interval.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        interval: QuantileInterval,
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.interval = interval
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.quantile_estimator = None
        self.nonconformity_scores = None
        self.conformalize_predictions = False
        self.training_time = None
        self.primary_estimator_error = None

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
        Fit a dedicated quantile estimator for this interval.
        """
        training_time_tracker = RuntimeTracker()

        # Prepare quantiles for this specific interval
        quantiles = [self.interval.lower_quantile, self.interval.upper_quantile]

        # Tune model parameters if requested
        if tuning_iterations > 1 and len(X_train) > 10:
            initialization_params = tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                quantiles=quantiles,
                random_state=random_state,
            )
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                self.quantile_estimator_architecture
            ].copy()

        # Initialize and fit the quantile estimator
        self.quantile_estimator = initialize_quantile_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            pinball_loss_alpha=quantiles,
            random_state=random_state,
        )

        # Fit the model and calculate nonconformity scores if enough data
        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            self.quantile_estimator.fit(X_train, y_train)

            # Calculate nonconformity scores on validation data
            val_prediction = self.quantile_estimator.predict(X_val)
            lower_conformal_deviations = val_prediction[:, 0] - y_val
            upper_conformal_deviations = y_val - val_prediction[:, 1]
            self.nonconformity_scores = np.maximum(
                lower_conformal_deviations, upper_conformal_deviations
            )
            self.conformalize_predictions = True
        else:
            self.quantile_estimator.fit(
                np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))
            )
            self.conformalize_predictions = False

        self.training_time = training_time_tracker.return_runtime()

        # Calculate performance metrics
        predictions = self.quantile_estimator.predict(X_val)
        lo_y_pred = predictions[:, 0]
        hi_y_pred = predictions[:, 1]
        lo_score = mean_pinball_loss(
            y_val, lo_y_pred, alpha=self.interval.lower_quantile
        )
        hi_score = mean_pinball_loss(
            y_val, hi_y_pred, alpha=self.interval.upper_quantile
        )
        self.primary_estimator_error = (lo_score + hi_score) / 2

    def predict_interval(self, X: np.array):
        """
        Predict conformal intervals.
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before prediction")

        prediction = self.quantile_estimator.predict(X)

        if self.conformalize_predictions:
            # Calculate conformity adjustment based on validation scores
            score = np.quantile(
                self.nonconformity_scores,
                self.interval.upper_quantile - self.interval.lower_quantile,
            )
            lower_interval_bound = np.array(prediction[:, 0]) - score
            upper_interval_bound = np.array(prediction[:, 1]) + score
        else:
            lower_interval_bound = np.array(prediction[:, 0])
            upper_interval_bound = np.array(prediction[:, 1])

        return lower_interval_bound, upper_interval_bound
