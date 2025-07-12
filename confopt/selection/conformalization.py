import logging
import numpy as np
from typing import Optional, Tuple, List
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from confopt.wrapping import ConformalBounds
from confopt.utils.preprocessing import train_val_split
from confopt.selection.estimation import (
    initialize_estimator,
    PointTuner,
    QuantileTuner,
)
from confopt.selection.estimator_configuration import ESTIMATOR_REGISTRY
from copy import deepcopy

logger = logging.getLogger(__name__)


class LocallyWeightedConformalEstimator:
    """Locally weighted conformal predictor with adaptive variance modeling.

    Implements a two-stage conformal prediction approach that combines point estimation
    with variance estimation to create locally adaptive prediction intervals. The method
    estimates both the conditional mean and conditional variance separately, then uses
    the variance estimates to scale nonconformity scores for improved efficiency.

    The estimator follows split conformal prediction principles, using separate training
    sets for the point estimator, variance estimator, and conformal calibration. This
    ensures proper finite-sample coverage guarantees while adapting interval widths
    to local prediction uncertainty.

    Args:
        point_estimator_architecture: Architecture identifier for the point estimator.
            Must be registered in ESTIMATOR_REGISTRY.
        variance_estimator_architecture: Architecture identifier for the variance estimator.
            Must be registered in ESTIMATOR_REGISTRY.
        alphas: List of miscoverage levels (1-alpha gives coverage probability).
            Must be in (0, 1) range.

    Attributes:
        pe_estimator: Fitted point estimator for conditional mean prediction.
        ve_estimator: Fitted variance estimator for conditional variance prediction.
        nonconformity_scores: Calibration scores from validation set.
        primary_estimator_error: MSE of point estimator on validation set.
        best_pe_config: Best hyperparameters found for point estimator.
        best_ve_config: Best hyperparameters found for variance estimator.

    Mathematical Framework:
        Given training data (X_train, y_train) and validation data (X_val, y_val):
        1. Split training data: (X_pe, y_pe) for point, (X_ve, y_ve) for variance
        2. Fit point estimator: μ̂(x) = E[Y|X=x]
        3. Compute residuals: r_i = |y_i - μ̂(X_i)|
        4. Fit variance estimator: σ̂²(x) = E[r²|X=x] using residuals
        5. Compute nonconformity: R_i = |y_val_i - μ̂(X_val_i)| / max(σ̂(X_val_i), ε)
        6. For new prediction at x: [μ̂(x) ± q_{1-α}(R) × σ̂(x)]

    Performance Characteristics:
        - Computational complexity: O(n_train + n_val) for training each estimator
        - Memory usage: O(n_val) for storing nonconformity scores
        - Prediction time: O(1) per prediction point
        - Adaptation: Intervals adapt to local variance estimates
    """

    def __init__(
        self,
        point_estimator_architecture: str,
        variance_estimator_architecture: str,
        alphas: List[float],
    ):
        self.point_estimator_architecture = point_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture
        self.alphas = alphas
        self.updated_alphas = alphas.copy()
        self.pe_estimator = None
        self.ve_estimator = None
        self.nonconformity_scores = None
        self.primary_estimator_error = None
        self.best_pe_config = None
        self.best_ve_config = None

    def _tune_fit_component_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator_architecture: str,
        tuning_iterations: int,
        min_obs_for_tuning: int = 30,
        random_state: Optional[int] = None,
        last_best_params: Optional[dict] = None,
    ):
        """Tune and fit a component estimator with hyperparameter optimization.

        Performs hyperparameter search when sufficient data is available, otherwise
        uses default or previously best configurations. Incorporates warm-starting
        from previous best parameters to improve convergence.

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            estimator_architecture: Architecture identifier from ESTIMATOR_REGISTRY.
            tuning_iterations: Number of hyperparameter search iterations.
            min_obs_for_tuning: Minimum samples required to trigger tuning.
            random_state: Random seed for reproducible results.
            last_best_params: Previously optimal parameters for warm-starting.

        Returns:
            Tuple containing:
                - Fitted estimator instance
                - Best hyperparameters found or used

        Implementation Details:
            - Uses forced configurations to ensure robust baselines
            - Incorporates last_best_params and defaults as starting points
            - Falls back to default parameters when data is insufficient
            - Leverages PointTuner for automated hyperparameter search
        """
        forced_param_configurations = []

        if last_best_params is not None:
            forced_param_configurations.append(last_best_params)

        estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]
        default_params = deepcopy(estimator_config.default_params)
        if default_params:
            forced_param_configurations.append(default_params)

        if tuning_iterations > 1 and len(X) > min_obs_for_tuning:
            tuner = PointTuner(random_state=random_state)
            initialization_params = tuner.tune(
                X=X,
                y=y,
                estimator_architecture=estimator_architecture,
                n_searches=tuning_iterations,
                forced_param_configurations=forced_param_configurations,
            )
        else:
            initialization_params = (
                forced_param_configurations[0] if forced_param_configurations else None
            )

        estimator = initialize_estimator(
            estimator_architecture=estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )
        estimator.fit(X, y)

        return estimator, initialization_params

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 30,
        random_state: Optional[int] = None,
        best_pe_config: Optional[dict] = None,
        best_ve_config: Optional[dict] = None,
    ):
        """Fit the locally weighted conformal estimator using split conformal prediction.

        Implements the three-stage fitting process: point estimation, variance estimation,
        and conformal calibration. Uses data splitting to ensure proper coverage guarantees
        while optimizing both estimators independently.

        Args:
            X_train: Training features, shape (n_train, n_features).
            y_train: Training targets, shape (n_train,).
            X_val: Validation features for conformal calibration, shape (n_val, n_features).
            y_val: Validation targets for conformal calibration, shape (n_val,).
            tuning_iterations: Hyperparameter search iterations (0 disables tuning).
            min_obs_for_tuning: Minimum samples required for hyperparameter tuning.
            random_state: Random seed for reproducible splits and initialization.
            best_pe_config: Warm-start parameters for point estimator.
            best_ve_config: Warm-start parameters for variance estimator.

        Implementation Process:
            1. Split training data into point estimation and variance estimation sets
            2. Fit point estimator on point estimation subset
            3. Compute absolute residuals on variance estimation subset
            4. Fit variance estimator on residuals
            5. Compute nonconformity scores on validation set
            6. Store scores for conformal adjustment during prediction

        Side Effects:
            - Updates pe_estimator, ve_estimator, nonconformity_scores
            - Updates best_pe_config, best_ve_config for future warm-starting
            - Syncs internal alpha state from updated_alphas
        """
        self._fetch_alphas()
        (X_pe, y_pe, X_ve, y_ve,) = train_val_split(
            X_train,
            y_train,
            train_split=0.75,
            normalize=False,
            random_state=random_state,
        )

        self.pe_estimator, self.best_pe_config = self._tune_fit_component_estimator(
            X=X_pe,
            y=y_pe,
            estimator_architecture=self.point_estimator_architecture,
            tuning_iterations=tuning_iterations,
            min_obs_for_tuning=min_obs_for_tuning,
            random_state=random_state,
            last_best_params=best_pe_config,
        )
        abs_pe_residuals = abs(y_ve - self.pe_estimator.predict(X_ve))

        self.ve_estimator, self.best_ve_config = self._tune_fit_component_estimator(
            X=X_ve,
            y=abs_pe_residuals,
            estimator_architecture=self.variance_estimator_architecture,
            tuning_iterations=tuning_iterations,
            min_obs_for_tuning=min_obs_for_tuning,
            random_state=random_state,
            last_best_params=best_ve_config,
        )
        var_pred = self.ve_estimator.predict(X_val)
        var_pred = np.array([0.001 if x <= 0 else x for x in var_pred])

        self.nonconformity_scores = (
            abs(y_val - self.pe_estimator.predict(X_val)) / var_pred
        )

        self.primary_estimator_error = mean_squared_error(
            self.pe_estimator.predict(X=X_val), y_val
        )

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        """Generate conformal prediction intervals for new observations.

        Produces prediction intervals with finite-sample coverage guarantees by
        combining point predictions, variance estimates, and conformal adjustments
        calibrated on the validation set.

        Args:
            X: Input features for prediction, shape (n_predict, n_features).

        Returns:
            List of ConformalBounds objects, one per alpha level, each containing:
                - lower_bounds: Lower interval bounds, shape (n_predict,)
                - upper_bounds: Upper interval bounds, shape (n_predict,)

        Raises:
            ValueError: If estimators have not been fitted.

        Mathematical Details:
            For each alpha level α and prediction point x:
            1. Compute point prediction: μ̂(x)
            2. Compute variance prediction: σ̂²(x)
            3. Get conformal quantile: q = quantile(nonconformity_scores, 1-α)
            4. Return interval: [μ̂(x) - q×σ̂(x), μ̂(x) + q×σ̂(x)]

        Coverage Guarantee:
            With probability 1-α, the true value will fall within the interval,
            assuming exchangeability of validation and test data.
        """
        if self.pe_estimator is None or self.ve_estimator is None:
            raise ValueError("Estimators must be fitted before prediction")

        y_pred = np.array(self.pe_estimator.predict(X)).reshape(-1, 1)
        var_pred = self.ve_estimator.predict(X)
        var_pred = np.array([max(x, 0) for x in var_pred]).reshape(-1, 1)

        intervals = []
        for alpha in self.alphas:
            non_conformity_score_quantile = np.quantile(
                self.nonconformity_scores, 1 - alpha
            )
            scaled_score = non_conformity_score_quantile * var_pred

            lower_bounds = y_pred - scaled_score
            upper_bounds = y_pred + scaled_score
            intervals.append(
                ConformalBounds(lower_bounds=lower_bounds, upper_bounds=upper_bounds)
            )

        return intervals

    def calculate_betas(self, X: np.array, y_true: float) -> float:
        """Calculate empirical p-values (beta values) for conformity assessment.

        Computes the empirical p-value representing the fraction of calibration
        nonconformity scores that are greater than or equal to the nonconformity
        score of a new observation. Used for conformity testing and coverage
        assessment.

        Args:
            X: Input features for single prediction, shape (n_features,).
            y_true: True target value for conformity assessment.

        Returns:
            List of beta values (empirical p-values), one per alpha level.
            Each beta ∈ [0, 1] represents the empirical quantile of the
            nonconformity score in the calibration distribution.

        Raises:
            ValueError: If estimators have not been fitted.

        Mathematical Details:
            1. Compute nonconformity: R = |y_true - μ̂(x)| / max(σ̂(x), ε)
            2. Calculate beta: β = mean(R_cal >= R) where R_cal are calibration scores
            3. Return same beta for all alphas (locally weighted approach)

        Usage:
            Beta values close to 0 indicate the observation is an outlier
            relative to the calibration distribution. Beta values close to 1
            indicate the observation is typical of the calibration distribution.
        """
        if self.pe_estimator is None or self.ve_estimator is None:
            raise ValueError("Estimators must be fitted before calculating beta")

        X = X.reshape(1, -1)
        y_pred = self.pe_estimator.predict(X)[0]
        var_pred = max(0.001, self.ve_estimator.predict(X)[0])

        nonconformity = abs(y_true - y_pred) / var_pred

        beta = np.mean(self.nonconformity_scores >= nonconformity)
        betas = [beta] * len(self.alphas)

        return betas

    def update_alphas(self, new_alphas: List[float]):
        """Update coverage levels without refitting the estimator.

        Provides an efficient mechanism to change target coverage levels without
        requiring re-training of the underlying estimators or recalibration of
        nonconformity scores. Changes take effect on the next prediction call.

        Args:
            new_alphas: New miscoverage levels (1-alpha gives coverage).
                Must be in (0, 1) range.

        Design Rationale:
            The locally weighted approach uses the same nonconformity scores
            for all alpha levels, making alpha updates computationally free.
            This enables efficient dynamic coverage adjustment in response to
            changing requirements or feedback.
        """
        self.updated_alphas = new_alphas.copy()

    def _fetch_alphas(self) -> List[float]:
        """Fetch the latest updated alphas and sync internal alpha state.

        Returns:
            The current alphas to be used for fitting and prediction.

        Implementation Details:
            Provides an abstraction layer for alpha updates that maintains
            state consistency between update_alphas calls and internal usage.
            Ensures that alpha changes are properly propagated throughout
            the estimator without breaking encapsulation.
        """
        if self.updated_alphas != self.alphas:
            self.alphas = self.updated_alphas.copy()
        return self.alphas


def alpha_to_quantiles(
    alpha: float, upper_quantile_cap: Optional[float] = None
) -> Tuple[float, float]:
    """Convert alpha level to symmetric quantile pair with optional upper bound.

    Transforms a miscoverage level alpha into corresponding lower and upper
    quantiles for symmetric prediction intervals, with support for capped
    upper quantiles to handle extreme coverage requirements.

    Args:
        alpha: Miscoverage level in (0, 1). Coverage = 1 - alpha.
        upper_quantile_cap: Optional upper bound for the upper quantile.
            Useful when dealing with limited training data or extreme alphas.

    Returns:
        Tuple of (lower_quantile, upper_quantile) where:
            - lower_quantile = alpha / 2
            - upper_quantile = min(1 - alpha/2, upper_quantile_cap)

    Raises:
        ValueError: If upper_quantile_cap results in upper_quantile < lower_quantile.

    Mathematical Details:
        For symmetric intervals with coverage 1-α:
        - Lower quantile: α/2 (captures α/2 probability in left tail)
        - Upper quantile: 1-α/2 (captures α/2 probability in right tail)

        When upper_quantile_cap is applied, intervals become asymmetric
        but maintain the desired coverage level through conformal adjustment.
    """
    lower_quantile = alpha / 2
    upper_quantile = 1 - lower_quantile
    if upper_quantile_cap is not None:
        upper_quantile = min(upper_quantile, upper_quantile_cap)
        if upper_quantile < lower_quantile:
            raise ValueError(
                f"Upper quantile cap {upper_quantile_cap} resulted in an upper quantile "
                f"{upper_quantile} that is smaller than the lower quantile {lower_quantile} "
                f"for alpha {alpha}."
            )

    return lower_quantile, upper_quantile


class QuantileConformalEstimator:
    """Quantile-based conformal predictor with direct quantile estimation.

    Implements conformal prediction using quantile regression as the base learner.
    This approach directly estimates the required prediction quantiles and applies
    conformal adjustments to achieve finite-sample coverage guarantees. The method
    is particularly effective when the underlying quantile estimator can capture
    conditional quantiles accurately.

    The estimator supports both conformalized and non-conformalized modes:
    - Conformalized: Uses split conformal prediction with proper calibration
    - Non-conformalized: Direct quantile predictions (when data is limited)

    Args:
        quantile_estimator_architecture: Architecture identifier for quantile estimator.
            Must be registered in ESTIMATOR_REGISTRY and support quantile fitting.
        alphas: List of miscoverage levels (1-alpha gives coverage probability).
            Must be in (0, 1) range.
        n_pre_conformal_trials: Minimum samples required for conformal calibration.
            Below this threshold, uses direct quantile prediction.

    Attributes:
        quantile_estimator: Fitted quantile regression model.
        nonconformity_scores: Calibration scores per alpha level (if conformalized).
        all_quantiles: Sorted list of all required quantiles.
        quantile_indices: Mapping from quantile values to prediction array indices.
        conformalize_predictions: Boolean flag indicating if conformal adjustment is used.
        primary_estimator_error: Mean pinball loss across all quantiles.
        upper_quantile_cap: Maximum allowed upper quantile value.

    Mathematical Framework:
        For each alpha level α:
        1. Estimate quantiles: q̂_α/2(x), q̂_1-α/2(x)
        2. If conformalized: compute nonconformity R_i = max(q̂_α/2(x_i) - y_i, y_i - q̂_1-α/2(x_i))
        3. Get conformal adjustment: C = quantile(R_cal, 1-α)
        4. Final intervals: [q̂_α/2(x) - C, q̂_1-α/2(x) + C]

        If not conformalized: [q̂_α/2(x), q̂_1-α/2(x)]

    Performance Characteristics:
        - Computational complexity: O(|quantiles| × n_train) for training
        - Memory usage: O(|alphas| × n_val) for nonconformity scores
        - Prediction time: O(|quantiles|) per prediction point
        - Accuracy: Depends on base quantile estimator quality
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        alphas: List[float],
        n_pre_conformal_trials: int = 32,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.alphas = alphas
        self.updated_alphas = alphas.copy()
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.quantile_estimator = None
        self.nonconformity_scores = None
        self.all_quantiles = None
        self.quantile_indices = None
        self.conformalize_predictions = False
        self.primary_estimator_error = None
        self.last_best_params = None
        self.upper_quantile_cap = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 30,
        upper_quantile_cap: Optional[float] = None,
        random_state: Optional[int] = None,
        last_best_params: Optional[dict] = None,
    ):
        """Fit the quantile conformal estimator with optional hyperparameter tuning.

        Trains a quantile regression model on all required quantiles and optionally
        applies conformal calibration for finite-sample coverage guarantees. The
        method automatically determines whether to use conformal adjustment based
        on available data volume.

        Args:
            X_train: Training features, shape (n_train, n_features).
            y_train: Training targets, shape (n_train,).
            X_val: Validation features for conformal calibration, shape (n_val, n_features).
            y_val: Validation targets for conformal calibration, shape (n_val,).
            tuning_iterations: Hyperparameter search iterations (0 disables tuning).
            min_obs_for_tuning: Minimum samples required for hyperparameter tuning.
            upper_quantile_cap: Maximum allowed upper quantile value.
            random_state: Random seed for reproducible initialization.
            last_best_params: Warm-start parameters from previous fitting.

        Implementation Process:
            1. Sync alpha state and compute required quantiles
            2. Build quantile index mapping for efficient access
            3. Configure hyperparameter search with forced configurations
            4. Fit quantile estimator using QuantileTuner if appropriate
            5. If sufficient data: compute conformal nonconformity scores
            6. Otherwise: use direct quantile predictions
            7. Evaluate performance using mean pinball loss

        Conformal vs Non-Conformal Decision:
            - Conformal: len(X_train) + len(X_val) > n_pre_conformal_trials
            - Non-conformal: Insufficient data for proper split conformal prediction

        Side Effects:
            - Updates quantile_estimator, nonconformity_scores, conformalize_predictions
            - Sets quantile_indices, upper_quantile_cap, last_best_params
            - Computes primary_estimator_error for performance tracking
        """
        current_alphas = self._fetch_alphas()
        self.upper_quantile_cap = upper_quantile_cap

        all_quantiles = []
        for alpha in current_alphas:
            lower_quantile, upper_quantile = alpha_to_quantiles(
                alpha, upper_quantile_cap
            )
            all_quantiles.append(lower_quantile)
            all_quantiles.append(upper_quantile)
        all_quantiles = sorted(all_quantiles)

        self.quantile_indices = {q: i for i, q in enumerate(all_quantiles)}

        forced_param_configurations = []

        if last_best_params is not None:
            forced_param_configurations.append(last_best_params)

        estimator_config = ESTIMATOR_REGISTRY[self.quantile_estimator_architecture]
        default_params = deepcopy(estimator_config.default_params)
        if default_params:
            forced_param_configurations.append(default_params)

        if tuning_iterations > 1 and len(X_train) > min_obs_for_tuning:
            tuner = QuantileTuner(random_state=random_state, quantiles=all_quantiles)
            initialization_params = tuner.tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                forced_param_configurations=forced_param_configurations,
            )
            self.last_best_params = initialization_params
        else:
            initialization_params = (
                forced_param_configurations[0] if forced_param_configurations else None
            )
            self.last_best_params = last_best_params

        self.quantile_estimator = initialize_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )

        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            self.nonconformity_scores = [np.array([]) for _ in current_alphas]
            self.quantile_estimator.fit(X_train, y_train, quantiles=all_quantiles)

            for i, alpha in enumerate(current_alphas):
                lower_quantile, upper_quantile = alpha_to_quantiles(
                    alpha, upper_quantile_cap
                )

                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                val_prediction = self.quantile_estimator.predict(X_val)

                lower_conformal_deviations = val_prediction[:, lower_idx] - y_val
                upper_conformal_deviations = y_val - val_prediction[:, upper_idx]

                self.nonconformity_scores[i] = np.maximum(
                    lower_conformal_deviations, upper_conformal_deviations
                )
            self.conformalize_predictions = True
        else:
            self.quantile_estimator.fit(
                X=np.vstack((X_train, X_val)),
                y=np.concatenate((y_train, y_val)),
                quantiles=all_quantiles,
            )
            self.conformalize_predictions = False

        scores = []
        for alpha in current_alphas:
            lower_quantile, upper_quantile = alpha_to_quantiles(
                alpha, upper_quantile_cap
            )
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            predictions = self.quantile_estimator.predict(X_val)

            lo_y_pred = predictions[:, lower_idx]
            hi_y_pred = predictions[:, upper_idx]

            lo_score = mean_pinball_loss(y_val, lo_y_pred, alpha=lower_quantile)
            hi_score = mean_pinball_loss(y_val, hi_y_pred, alpha=upper_quantile)
            scores.extend([lo_score, hi_score])

        self.primary_estimator_error = np.mean(scores)

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        """Generate conformal prediction intervals using quantile estimates.

        Produces prediction intervals with finite-sample coverage guarantees by
        combining quantile regression predictions with conformal adjustments
        (when enabled) or using direct quantile predictions.

        Args:
            X: Input features for prediction, shape (n_predict, n_features).

        Returns:
            List of ConformalBounds objects, one per alpha level, each containing:
                - lower_bounds: Lower interval bounds, shape (n_predict,)
                - upper_bounds: Upper interval bounds, shape (n_predict,)

        Raises:
            ValueError: If quantile estimator has not been fitted.

        Mathematical Details:
            For each alpha level α and prediction point x:

            If conformalized:
            1. Get quantile predictions: q̂_α/2(x), q̂_1-α/2(x)
            2. Get conformal adjustment: C = quantile(nonconformity_scores, 1-α)
            3. Return interval: [q̂_α/2(x) - C, q̂_1-α/2(x) + C]

            If not conformalized:
            1. Return direct quantiles: [q̂_α/2(x), q̂_1-α/2(x)]

        Coverage Guarantee:
            With probability 1-α, the true value will fall within the interval,
            assuming exchangeability of calibration and test data (conformalized mode)
            or correct conditional quantile specification (non-conformalized mode).
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before prediction")

        intervals = []
        prediction = self.quantile_estimator.predict(X)

        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(
                alpha, self.upper_quantile_cap
            )

            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            if self.conformalize_predictions:
                score = np.quantile(
                    self.nonconformity_scores[i],
                    1 - alpha,
                    interpolation="linear",
                )
                lower_interval_bound = np.array(prediction[:, lower_idx]) - score
                upper_interval_bound = np.array(prediction[:, upper_idx]) + score
            else:
                lower_interval_bound = np.array(prediction[:, lower_idx])
                upper_interval_bound = np.array(prediction[:, upper_idx])

            intervals.append(
                ConformalBounds(
                    lower_bounds=lower_interval_bound, upper_bounds=upper_interval_bound
                )
            )

        return intervals

    def calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        """Calculate empirical p-values (beta values) for conformity assessment.

        Computes alpha-specific empirical p-values representing the fraction of
        calibration nonconformity scores that are greater than or equal to the
        nonconformity score of a new observation. Used for conformity testing
        and coverage assessment in the quantile-based framework.

        Args:
            X: Input features for single prediction, shape (n_features,).
            y_true: True target value for conformity assessment.

        Returns:
            List of beta values (empirical p-values), one per alpha level.
            Each beta ∈ [0, 1] represents the empirical quantile of the
            nonconformity score in the corresponding calibration distribution.

        Raises:
            ValueError: If quantile estimator has not been fitted.

        Mathematical Details:
            For each alpha level α:
            1. Get quantile predictions: q̂_α/2(x), q̂_1-α/2(x)
            2. Compute nonconformity: R = max(q̂_α/2(x) - y_true, y_true - q̂_1-α/2(x))
            3. Calculate beta: β = mean(R_cal_α >= R) using alpha-specific calibration scores

        Usage:
            Unlike the locally weighted approach, this method produces different
            beta values for each alpha level, reflecting the alpha-specific
            nature of the quantile-based nonconformity scores.
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before calculating beta")

        X = X.reshape(1, -1)

        betas = []
        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(
                alpha, self.upper_quantile_cap
            )
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            prediction = self.quantile_estimator.predict(X)
            lower_bound = prediction[0, lower_idx]
            upper_bound = prediction[0, upper_idx]

            lower_deviation = lower_bound - y_true
            upper_deviation = y_true - upper_bound
            nonconformity = max(lower_deviation, upper_deviation)

            beta = np.mean(self.nonconformity_scores[i] >= nonconformity)

            betas.append(beta)

        return betas

    def update_alphas(self, new_alphas: List[float]):
        """Update coverage levels with quantile recomputation awareness.

        Updates target coverage levels for the quantile-based estimator. Note that
        unlike the locally weighted approach, changing alphas in the quantile-based
        method may require refitting if new quantiles are needed that weren't
        computed during initial training.

        Args:
            new_alphas: New miscoverage levels (1-alpha gives coverage).
                Must be in (0, 1) range.

        Important:
            If new_alphas require quantiles not computed during fit(), the estimator
            may need to be refitted. The current implementation provides a state
            abstraction but optimal performance requires consistent alpha sets
            across fit() and predict() calls.

        Design Consideration:
            For maximum efficiency, determine the complete set of required alphas
            before calling fit() to ensure all necessary quantiles are estimated
            in a single training pass.
        """
        self.updated_alphas = new_alphas.copy()

    def _fetch_alphas(self) -> List[float]:
        """Fetch the latest updated alphas and sync internal alpha state.

        Returns:
            The current alphas to be used for fitting and prediction.

        Implementation Details:
            Provides an abstraction layer for alpha updates that maintains
            state consistency between update_alphas calls and internal usage.
            Critical for quantile-based estimation where alpha changes affect
            the required quantile set.
        """
        if self.updated_alphas != self.alphas:
            self.alphas = self.updated_alphas.copy()
        return self.alphas
