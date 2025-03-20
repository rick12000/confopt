import logging
import numpy as np
from typing import Optional, Tuple, List, Literal
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from confopt.data_classes import QuantileInterval
from confopt.preprocessing import train_val_split
from confopt.estimation import (
    initialize_point_estimator,
    initialize_quantile_estimator,
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

    def _tune_fit_component_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator_architecture: str,
        tuning_iterations: int,
        min_obs_for_tuning: int = 15,
        random_state: Optional[int] = None,
    ):
        """
        Fit component estimator with option to tune.
        """
        if tuning_iterations > 1 and len(X) > min_obs_for_tuning:
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
        min_obs_for_tuning: int = 15,
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

        self.pe_estimator = self._tune_fit_component_estimator(
            X=X_pe,
            y=y_pe,
            estimator_architecture=self.point_estimator_architecture,
            tuning_iterations=tuning_iterations,
            min_obs_for_tuning=min_obs_for_tuning,
            random_state=random_state,
        )

        abs_pe_residuals = abs(y_ve - self.pe_estimator.predict(X_ve))

        self.ve_estimator = self._tune_fit_component_estimator(
            X=X_ve,
            y=abs_pe_residuals,
            estimator_architecture=self.variance_estimator_architecture,
            tuning_iterations=tuning_iterations,
            min_obs_for_tuning=min_obs_for_tuning,
            random_state=random_state,
        )
        var_pred = self.ve_estimator.predict(X_val)
        var_pred = np.array([1 if x <= 0 else x for x in var_pred])

        self.nonconformity_scores = (
            abs(y_val - self.pe_estimator.predict(X_val)) / var_pred
        )

        # TODO: TEMP: Performance metric storage:
        self.primary_estimator_error = mean_squared_error(
            self.pe_estimator.predict(X=X_val), y_val
        )

    def predict_interval(self, X: np.array, alpha: float) -> Tuple[np.array, np.array]:
        """
        Predict conformal intervals for a given confidence level.

        Parameters
        ----------
        X : np.array
            Input features
        alpha : float
            Confidence level (between 0 and 1)

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

        lower_bound = y_pred - scaled_score
        upper_bound = y_pred + scaled_score

        return lower_bound, upper_bound


class SingleFitQuantileConformalEstimator:
    """
    Single-fit quantile conformal estimator.

    Uses a single model that can predict multiple quantiles with a single fit.
    Can predict any quantile after fitting once.
    """

    def __init__(
        self,
        quantile_estimator_architecture: Literal["qknn", "qrf"],
        intervals: List[QuantileInterval],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.intervals = intervals

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 15,
        random_state: Optional[int] = None,
    ):
        """
        Fit the single-fit quantile estimator for multiple intervals with one model.
        """
        # Tune model parameters if requested
        if tuning_iterations > 1 and len(X_train) > min_obs_for_tuning:
            all_quantiles = []
            for interval in self.intervals:
                all_quantiles.append(interval.lower_quantile)
                all_quantiles.append(interval.upper_quantile)

            # TODO: Tune with pinball loss or as point estimator?
            initialization_params = tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                quantiles=all_quantiles,
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

        # Initialize nonconformity scores list
        self.nonconformity_scores = []

        # Fit the model and calculate nonconformity scores if enough data
        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            self.quantile_estimator.fit(X_train, y_train)

            # Calculate nonconformity scores for each interval on validation data
            for interval in self.intervals:
                quantiles = [interval.lower_quantile, interval.upper_quantile]
                val_prediction = self.quantile_estimator.predict(
                    X=X_val,
                    quantiles=quantiles,
                )
                lower_conformal_deviations = val_prediction[:, 0] - y_val
                upper_conformal_deviations = y_val - val_prediction[:, 1]
                # Store deviations for this interval
                self.nonconformity_scores.append(
                    np.maximum(lower_conformal_deviations, upper_conformal_deviations)
                )

            self.conformalize_predictions = True
        else:
            self.quantile_estimator.fit(
                X=np.vstack((X_train, X_val)), y=np.concatenate((y_train, y_val))
            )
            # Initialize empty nonconformity scores for each interval
            self.nonconformity_scores = [np.array([]) for _ in self.intervals]
            self.conformalize_predictions = False

        # TODO: TEMP: Calculate performance metrics
        scores = []
        for interval in self.intervals:
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

    def predict_interval(self, X: np.array, interval: QuantileInterval):
        """
        Predict conformal intervals for a specific interval.
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before prediction")

        # Find the interval in the list of intervals
        interval_index = None
        for i, fitted_interval in enumerate(self.intervals):
            if (
                fitted_interval.lower_quantile == interval.lower_quantile
                and fitted_interval.upper_quantile == interval.upper_quantile
            ):
                interval_index = i
                break

        if interval_index is None:
            raise ValueError(f"Interval {interval} not found in fitted intervals")

        quantiles = [interval.lower_quantile, interval.upper_quantile]
        prediction = self.quantile_estimator.predict(X=X, quantiles=quantiles)

        if (
            self.conformalize_predictions
            and len(self.nonconformity_scores[interval_index]) > 0
        ):
            # Calculate conformity adjustment based on validation scores for this interval
            score = np.quantile(
                self.nonconformity_scores[interval_index],
                interval.upper_quantile - interval.lower_quantile,
            )
            lower_interval_bound = np.array(prediction[:, 0]) - score
            upper_interval_bound = np.array(prediction[:, 1]) + score
        else:
            # No conformalization
            lower_interval_bound = np.array(prediction[:, 0])
            upper_interval_bound = np.array(prediction[:, 1])

        return lower_interval_bound, upper_interval_bound


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

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 15,
        random_state: Optional[int] = None,
    ):
        """
        Fit a dedicated quantile estimator for this interval.
        """

        # Prepare quantiles for this specific interval
        quantiles = [self.interval.lower_quantile, self.interval.upper_quantile]

        # Tune model parameters if requested
        if tuning_iterations > 1 and len(X_train) > min_obs_for_tuning:
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
            lower_interval = np.array(prediction[:, 0]) - score
            upper_interval = np.array(prediction[:, 1]) + score
        else:
            lower_interval = np.array(prediction[:, 0])
            upper_interval = np.array(prediction[:, 1])

        return lower_interval, upper_interval
