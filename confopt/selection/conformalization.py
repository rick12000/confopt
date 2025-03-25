import logging
import numpy as np
from typing import Optional, Tuple, List
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from confopt.data_classes import ConformalBounds
from confopt.utils.preprocessing import train_val_split
from confopt.selection.estimation import (
    initialize_estimator,
    PointTuner,
    QuantileTuner,
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
        alphas: Optional[List[float]] = None,
    ):
        self.point_estimator_architecture = point_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture
        self.alphas = alphas or []
        self.pe_estimator = None
        self.ve_estimator = None
        self.nonconformity_scores = None
        self.primary_estimator_error = None

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
            # Initialize tuner when needed, don't keep as instance attribute
            tuner = PointTuner(random_state=random_state)
            initialization_params = tuner.tune(
                X=X,
                y=y,
                estimator_architecture=estimator_architecture,
                n_searches=tuning_iterations,
            )
        else:
            # Use an empty dict to get the default estimator as-is
            initialization_params = {}

        estimator = initialize_estimator(
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

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        """
        Predict conformal intervals for all alphas.

        Parameters
        ----------
        X : np.array
            Input features

        Returns
        -------
        List[ConformalInterval]
            List of conformal intervals for each alpha
        """
        if self.pe_estimator is None or self.ve_estimator is None:
            raise ValueError("Estimators must be fitted before prediction")

        y_pred = np.array(self.pe_estimator.predict(X)).reshape(-1, 1)
        var_pred = self.ve_estimator.predict(X)
        var_pred = np.array([max(x, 0) for x in var_pred]).reshape(-1, 1)

        results = []
        for alpha in self.alphas:
            score_quantile = np.quantile(self.nonconformity_scores, 1 - alpha)
            scaled_score = score_quantile * var_pred

            lower_bound = y_pred - scaled_score
            upper_bound = y_pred + scaled_score
            results.append(
                ConformalBounds(lower_bounds=lower_bound, upper_bounds=upper_bound)
            )

        return results

    def calculate_beta(self, X: np.array, y_true: float) -> float:
        """
        Calculate beta value as the percentile rank of the current observation's
        nonconformity score compared to validation set nonconformity scores.

        Parameters
        ----------
        X : np.array
            Input feature vector for a single observation
        y_true : float
            Actual observed value

        Returns
        -------
        float
            Beta value (percentile rank from 0 to 1)
        """
        if self.pe_estimator is None or self.ve_estimator is None:
            raise ValueError("Estimators must be fitted before calculating beta")

        # Calculate prediction and variance
        X = X.reshape(1, -1) if X.ndim == 1 else X  # Ensure 2D
        y_pred = self.pe_estimator.predict(X)[0]
        var_pred = max(1e-6, self.ve_estimator.predict(X)[0])  # Avoid division by zero

        # Calculate nonconformity score for this observation
        nonconformity = abs(y_true - y_pred) / var_pred

        # Calculate beta as percentile rank
        beta = np.mean(self.nonconformity_scores >= nonconformity)

        return beta


class QuantileConformalEstimator:
    """
    Unified quantile conformal estimator that works with both single-fit and multi-fit quantile estimators.

    Uses a single model to predict multiple quantiles for specified alphas.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        alphas: List[float],
        n_pre_conformal_trials: int = 20,
        upper_quantile_cap: Optional[float] = None,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.alphas = alphas
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.upper_quantile_cap = upper_quantile_cap

        self.quantile_estimator = None
        self.nonconformity_scores = None
        self.all_quantiles = None
        self.conformalize_predictions = False
        self.primary_estimator_error = None

    def _alpha_to_quantiles(self, alpha: float) -> Tuple[float, float]:
        """Convert alpha to lower and upper quantiles"""
        lower_quantile = alpha / 2
        upper_quantile = (
            self.upper_quantile_cap
            if self.upper_quantile_cap is not None
            else 1 - lower_quantile
        )
        return lower_quantile, upper_quantile

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
        Fit the quantile estimator for all specified alphas.
        """
        # Prepare all quantiles needed for all alphas
        all_quantiles = []
        for alpha in self.alphas:
            lower_quantile, upper_quantile = self._alpha_to_quantiles(alpha)
            all_quantiles.append(lower_quantile)
            all_quantiles.append(upper_quantile)
        all_quantiles = sorted(list(set(all_quantiles)))  # Remove duplicates and sort

        # Create a mapping from quantile values to their indices for O(1) lookups
        self.quantile_indices = {q: i for i, q in enumerate(all_quantiles)}

        # Tune model parameters if requested
        if tuning_iterations > 1 and len(X_train) > min_obs_for_tuning:
            # Initialize tuner with required quantiles when needed, don't keep as instance attribute
            tuner = QuantileTuner(random_state=random_state, quantiles=all_quantiles)
            initialization_params = tuner.tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
            )
        else:
            # Use an empty dict to get the default estimator as-is
            initialization_params = {}

        # Initialize the quantile estimator
        self.quantile_estimator = initialize_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )

        # Initialize nonconformity scores for each alpha
        self.nonconformity_scores = [np.array([]) for _ in self.alphas]

        # Fit the model and calculate nonconformity scores if enough data
        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            # Pass quantiles and upper_quantile_cap to fit
            self.quantile_estimator.fit(X_train, y_train, quantiles=all_quantiles)

            # Calculate nonconformity scores for each alpha on validation data
            for i, alpha in enumerate(self.alphas):
                lower_quantile, upper_quantile = self._alpha_to_quantiles(alpha)

                # Get the indices of lower and upper quantiles using dictionary lookup
                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                # Get predictions
                val_prediction = self.quantile_estimator.predict(X_val)

                lower_conformal_deviations = val_prediction[:, lower_idx] - y_val
                upper_conformal_deviations = y_val - val_prediction[:, upper_idx]

                # Store deviations for this alpha
                self.nonconformity_scores[i] = np.maximum(
                    lower_conformal_deviations, upper_conformal_deviations
                )

            self.conformalize_predictions = True
        else:
            # For small datasets, use all data without conformalization
            self.quantile_estimator.fit(
                X=np.vstack((X_train, X_val)),
                y=np.concatenate((y_train, y_val)),
                quantiles=all_quantiles,
            )
            self.conformalize_predictions = False

        # Store all_quantiles for later lookup
        self.all_quantiles = all_quantiles

        # Calculate performance metrics
        scores = []
        for alpha in self.alphas:
            lower_quantile, upper_quantile = self._alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            predictions = self.quantile_estimator.predict(X_val)

            lo_y_pred = predictions[:, lower_idx]
            hi_y_pred = predictions[:, upper_idx]

            lo_score = mean_pinball_loss(y_val, lo_y_pred, alpha=lower_quantile)
            hi_score = mean_pinball_loss(y_val, hi_y_pred, alpha=upper_quantile)
            scores.append((lo_score + hi_score) / 2)

        self.primary_estimator_error = np.mean(scores)

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        """
        Predict conformal intervals for all alphas.

        Parameters
        ----------

        X : np.array
            Input features

        Returns
        -------
        List[ConformalInterval]
            List of conformal intervals for each alpha
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before prediction")

        results = []
        prediction = self.quantile_estimator.predict(X)

        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = self._alpha_to_quantiles(alpha)

            # Get the indices of lower and upper quantiles using dictionary lookup
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            # Apply conformalization if possible
            if self.conformalize_predictions and len(self.nonconformity_scores[i]) > 0:
                # Calculate conformity adjustment based on validation scores for this interval
                score = np.quantile(
                    self.nonconformity_scores[i],
                    1 - alpha,
                )
                lower_interval_bound = np.array(prediction[:, lower_idx]) - score
                upper_interval_bound = np.array(prediction[:, upper_idx]) + score
            else:
                # No conformalization
                lower_interval_bound = np.array(prediction[:, lower_idx])
                upper_interval_bound = np.array(prediction[:, upper_idx])

            results.append(
                ConformalBounds(
                    lower_bounds=lower_interval_bound, upper_bounds=upper_interval_bound
                )
            )

        return results

    def calculate_beta(self, X: np.array, y_true: float, alpha_idx: int = 0) -> float:
        """
        Calculate beta value as the percentile rank of the current observation's
        nonconformity score compared to validation set nonconformity scores.

        Parameters
        ----------
        X : np.array
            Input feature vector for a single observation
        y_true : float
            Actual observed value
        alpha_idx : int, optional
            Index of alpha to use for nonconformity calculation (default: 0)

        Returns
        -------
        float
            Beta value (percentile rank from 0 to 1)
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before calculating beta")

        if (
            not self.conformalize_predictions
            or len(self.nonconformity_scores[alpha_idx]) == 0
        ):
            return 0.5  # Default value when conformalization is not possible

        # Ensure X is properly shaped
        X = X.reshape(1, -1) if X.ndim == 1 else X

        # Get the alpha and corresponding quantiles
        alpha = self.alphas[alpha_idx]
        lower_quantile, upper_quantile = self._alpha_to_quantiles(alpha)
        lower_idx = self.quantile_indices[lower_quantile]
        upper_idx = self.quantile_indices[upper_quantile]

        # Get predictions for this point
        prediction = self.quantile_estimator.predict(X)
        lower_bound = prediction[0, lower_idx]
        upper_bound = prediction[0, upper_idx]

        # Calculate nonconformity score (maximum of lower and upper deviations)
        lower_deviation = lower_bound - y_true
        upper_deviation = y_true - upper_bound
        nonconformity = max(lower_deviation, upper_deviation)

        # Calculate beta as percentile rank compared to validation nonconformities
        beta = np.mean(self.nonconformity_scores[alpha_idx] >= nonconformity)

        return beta
