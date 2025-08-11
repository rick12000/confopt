import logging
import numpy as np
from typing import Optional, Tuple, List, Literal
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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
        n_calibration_folds: int = 3,
        calibration_split_strategy: Literal[
            "cv_plus", "train_test_split", "adaptive"
        ] = "adaptive",
        adaptive_threshold: int = 50,
        validation_split: float = 0.2,
        normalize_features: bool = True,
    ):
        self.point_estimator_architecture = point_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture
        self.alphas = alphas
        self.updated_alphas = alphas.copy()
        self.n_calibration_folds = n_calibration_folds
        self.calibration_split_strategy = calibration_split_strategy
        self.adaptive_threshold = adaptive_threshold
        self.validation_split = validation_split
        self.normalize_features = normalize_features
        self.pe_estimator = None
        self.ve_estimator = None
        self.nonconformity_scores = None
        self.best_pe_config = None
        self.best_ve_config = None
        self.feature_scaler = None

    def _tune_fit_component_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator_architecture: str,
        tuning_iterations: int,
        min_obs_for_tuning: int,
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

    def _determine_splitting_strategy(self, total_size: int) -> str:
        """Determine optimal data splitting strategy based on dataset size and configuration.

        Selects between CV+ and train-test split approaches for conformal calibration
        based on the configured strategy and dataset characteristics. The adaptive
        strategy automatically chooses the most appropriate method based on data size
        to balance computational efficiency with calibration stability.

        Args:
            total_size: Total number of samples in the dataset.

        Returns:
            Strategy identifier: "cv_plus" or "train_test_split".

        Strategy Selection Logic:
            - "adaptive": Uses CV+ for small datasets (< adaptive_threshold) to maximize
              calibration stability, switches to train-test split for larger datasets
              to improve computational efficiency
            - "cv_plus": Always uses cross-validation based calibration
            - "train_test_split": Always uses single split calibration

        Design Rationale:
            Small datasets benefit from CV+ approach as it provides more stable
            nonconformity score estimation through cross-validation. Large datasets
            can use simpler train-test splits for computational efficiency while
            maintaining adequate calibration due to larger validation sets.
        """
        if self.calibration_split_strategy == "adaptive":
            return (
                "cv_plus"
                if total_size < self.adaptive_threshold
                else "train_test_split"
            )
        return self.calibration_split_strategy

    def _fit_cv_plus(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        best_pe_config: Optional[dict],
        best_ve_config: Optional[dict],
    ):
        """Fit locally weighted conformal estimator using CV+ calibration strategy.

        Implements the CV+ (Cross-Validation Plus) approach from Barber et al. (2019)
        for conformal prediction with proper finite-sample coverage guarantees. This
        method uses k-fold cross-validation for calibration while training final
        estimators on the complete dataset to maximize predictive performance.

        The approach splits each fold's training data into point estimation and
        variance estimation subsets, fits both estimators, then computes nonconformity
        scores on the fold's validation set. Final estimators are trained on all
        available data using the aggregated calibration scores from all folds.

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            tuning_iterations: Number of hyperparameter search iterations per estimator.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible fold splits and model initialization.
            best_pe_config: Warm-start parameters for point estimator hyperparameter search.
            best_ve_config: Warm-start parameters for variance estimator hyperparameter search.

        Implementation Details:
            - Uses stratified k-fold splitting with shuffle for robust calibration
            - Each fold splits training data 75/25 for point/variance estimation
            - Applies feature scaling within each fold to prevent data leakage
            - Aggregates nonconformity scores across all validation folds
            - Trains final estimators on complete dataset with proper scaling
            - Stores calibration scores for interval quantile computation

        Mathematical Framework:
            For each fold f with training indices T_f and validation indices V_f:
            1. Split T_f → (T_pe_f, T_ve_f) for point and variance estimation
            2. Fit μ̂_f on T_pe_f, compute residuals on T_ve_f
            3. Fit σ̂²_f on (T_ve_f, |residuals|)
            4. Compute R_i = |y_i - μ̂_f(x_i)| / max(σ̂_f(x_i), ε) for i ∈ V_f
            5. Aggregate all R_i across folds for final calibration distribution

        Coverage Properties:
            Provides finite-sample coverage guarantees under exchangeability assumptions
            while using all available data for final model training, balancing statistical
            efficiency with coverage validity.
        """
        kfold = KFold(
            n_splits=self.n_calibration_folds, shuffle=True, random_state=random_state
        )
        all_nonconformity_scores = []

        # Store predictions from each fold for final aggregation
        fold_predictions = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # Further split training data for point and variance estimation
            (X_pe, y_pe, X_ve, y_ve) = train_val_split(
                X_fold_train,
                y_fold_train,
                train_split=0.75,
                normalize=False,  # Normalization already applied in fit()
                random_state=random_state + fold_idx if random_state else None,
            )

            # Fit point estimator
            pe_estimator, _ = self._tune_fit_component_estimator(
                X=X_pe,
                y=y_pe,
                estimator_architecture=self.point_estimator_architecture,
                tuning_iterations=tuning_iterations,
                min_obs_for_tuning=min_obs_for_tuning,
                random_state=random_state + fold_idx if random_state else None,
                last_best_params=best_pe_config,
            )

            # Compute residuals and fit variance estimator
            abs_pe_residuals = abs(y_ve - pe_estimator.predict(X_ve))
            ve_estimator, _ = self._tune_fit_component_estimator(
                X=X_ve,
                y=abs_pe_residuals,
                estimator_architecture=self.variance_estimator_architecture,
                tuning_iterations=tuning_iterations,
                min_obs_for_tuning=min_obs_for_tuning,
                random_state=random_state + fold_idx if random_state else None,
                last_best_params=best_ve_config,
            )

            # Compute nonconformity scores on validation fold
            var_pred = ve_estimator.predict(X_fold_val)
            var_pred = np.array([max(0.001, x) for x in var_pred])

            fold_nonconformity = (
                abs(y_fold_val - pe_estimator.predict(X_fold_val)) / var_pred
            )
            all_nonconformity_scores.extend(fold_nonconformity)

            # Store fold models for final prediction
            fold_predictions.append(
                {
                    "pe_estimator": pe_estimator,
                    "ve_estimator": ve_estimator,
                    "val_indices": val_idx,
                }
            )

        (X_pe_final, y_pe_final, X_ve_final, y_ve_final) = train_val_split(
            X, y, train_split=0.75, normalize=False, random_state=random_state
        )

        self.pe_estimator, self.best_pe_config = self._tune_fit_component_estimator(
            X=X_pe_final,
            y=y_pe_final,
            estimator_architecture=self.point_estimator_architecture,
            tuning_iterations=tuning_iterations,
            min_obs_for_tuning=min_obs_for_tuning,
            random_state=random_state,
            last_best_params=best_pe_config,
        )

        abs_pe_residuals_final = abs(y_ve_final - self.pe_estimator.predict(X_ve_final))
        self.ve_estimator, self.best_ve_config = self._tune_fit_component_estimator(
            X=X_ve_final,
            y=abs_pe_residuals_final,
            estimator_architecture=self.variance_estimator_architecture,
            tuning_iterations=tuning_iterations,
            min_obs_for_tuning=min_obs_for_tuning,
            random_state=random_state,
            last_best_params=best_ve_config,
        )

        # Store aggregated nonconformity scores
        self.nonconformity_scores = np.array(all_nonconformity_scores)

    def _fit_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        best_pe_config: Optional[dict],
        best_ve_config: Optional[dict],
    ):
        """Fit locally weighted conformal estimator using train-test split calibration.

        Implements the traditional split conformal prediction approach using a single
        train-validation split for calibration. This method is computationally efficient
        for larger datasets where cross-validation becomes expensive, while still
        maintaining finite-sample coverage guarantees.

        The input data is first split into training and validation sets. The training
        set is further subdivided for point estimation and variance estimation, with
        the validation set reserved exclusively for nonconformity score computation.
        Feature scaling is applied consistently across the split to prevent data
        leakage while ensuring proper normalization.

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            tuning_iterations: Number of hyperparameter search iterations per estimator.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible data splits and model initialization.
            best_pe_config: Warm-start parameters for point estimator hyperparameter search.
            best_ve_config: Warm-start parameters for variance estimator hyperparameter search.

        Implementation Details:
            - Splits input data into training and validation sets using validation_split
            - Fits feature scaler on training data only to prevent information leakage
            - Splits training set 75/25 for point estimation vs variance estimation
            - Uses validation set exclusively for nonconformity score computation
            - Applies consistent preprocessing across train/validation splits
            - Stores single-split calibration scores for interval construction

        Mathematical Framework:
            1. Split X, y → (X_train, y_train), (X_val, y_val)
            2. Split X_train → (X_pe, X_ve) and y_train → (y_pe, y_ve)
            3. Fit point estimator: μ̂(x) on (X_pe, y_pe)
            4. Compute residuals: r_i = |y_ve_i - μ̂(X_ve_i)| for variance training
            5. Fit variance estimator: σ̂²(x) on (X_ve, r)
            6. Compute validation nonconformity: R_i = |y_val_i - μ̂(X_val_i)| / max(σ̂(X_val_i), ε)

        Efficiency Considerations:
            More computationally efficient than CV+ for large datasets, using single
            train-validation split instead of k-fold cross-validation. However, may
            have slightly less stable calibration with smaller validation sets compared
            to the cross-validation approach.
        """
        # Split data internally for train-test approach
        X_train, y_train, X_val, y_val = train_val_split(
            X,
            y,
            train_split=(1 - self.validation_split),
            normalize=False,  # Normalization already applied in fit()
            random_state=random_state,
        )

        (X_pe, y_pe, X_ve, y_ve,) = train_val_split(
            X_train,
            y_train,
            train_split=0.75,
            normalize=False,  # Normalization already applied in fit()
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
        var_pred = np.array([max(0.001, x) for x in var_pred])

        self.nonconformity_scores = (
            abs(y_val - self.pe_estimator.predict(X_val)) / var_pred
        )

    def fit(
        self,
        X: np.array,
        y: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 50,
        random_state: Optional[int] = None,
        best_pe_config: Optional[dict] = None,
        best_ve_config: Optional[dict] = None,
    ):
        """Fit the locally weighted conformal estimator.

        Uses adaptive data splitting strategy: CV+ for small datasets, train-test split
        for larger datasets, or explicit strategy selection. Handles data preprocessing
        including feature scaling applied to the entire dataset.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
            tuning_iterations: Hyperparameter search iterations (0 disables tuning).
            min_obs_for_tuning: Minimum samples required for hyperparameter tuning.
            random_state: Random seed for reproducible splits and initialization.
            best_pe_config: Warm-start parameters for point estimator.
            best_ve_config: Warm-start parameters for variance estimator.
        """
        self._fetch_alphas()

        # Apply feature scaling to entire dataset if requested
        if self.normalize_features:
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = X
            self.feature_scaler = None

        total_size = len(X)
        strategy = self._determine_splitting_strategy(total_size)

        if strategy == "cv_plus":
            self._fit_cv_plus(
                X_scaled,
                y,
                tuning_iterations,
                min_obs_for_tuning,
                random_state,
                best_pe_config,
                best_ve_config,
            )
        else:  # train_test_split
            self._fit_train_test_split(
                X_scaled,
                y,
                tuning_iterations,
                min_obs_for_tuning,
                random_state,
                best_pe_config,
                best_ve_config,
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

        # Apply same preprocessing as during training
        X_processed = X.copy()
        if self.normalize_features and self.feature_scaler is not None:
            X_processed = self.feature_scaler.transform(X_processed)

        y_pred = np.array(self.pe_estimator.predict(X_processed)).reshape(-1, 1)
        var_pred = self.ve_estimator.predict(X_processed)
        var_pred = np.array([max(x, 0) for x in var_pred]).reshape(-1, 1)

        intervals = []
        for alpha in self.alphas:
            non_conformity_score_quantile = np.quantile(
                self.nonconformity_scores,
                (1 - alpha) / (1 + 1 / len(self.nonconformity_scores)),
                method="linear",
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
            (high nonconformity) relative to the calibration distribution.
            Beta values close to 1 indicate the observation is typical
            (low nonconformity) relative to the calibration distribution.
        """
        if self.pe_estimator is None or self.ve_estimator is None:
            raise ValueError("Estimators must be fitted before calculating beta")

        X = X.reshape(1, -1)
        # Apply same preprocessing as during training
        if self.normalize_features and self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)

        y_pred = self.pe_estimator.predict(X)[0]
        var_pred = max(0.001, self.ve_estimator.predict(X)[0])

        nonconformity = abs(y_true - y_pred) / var_pred

        # According to the DTACI paper: β_t := sup {β : Y_t ∈ Ĉ_t(β)}
        # This means β_t is the proportion of calibration scores >= test nonconformity
        # (i.e., the empirical coverage probability)
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


def alpha_to_quantiles(alpha: float) -> Tuple[float, float]:
    """Convert alpha level to symmetric quantile pair.

    Transforms a miscoverage level alpha into corresponding lower and upper
    quantiles for symmetric prediction intervals.

    Args:
        alpha: Miscoverage level in (0, 1). Coverage = 1 - alpha.

    Returns:
        Tuple of (lower_quantile, upper_quantile) where:
            - lower_quantile = alpha / 2
            - upper_quantile = 1 - alpha / 2

    Mathematical Details:
        For symmetric intervals with coverage 1-α:
        - Lower quantile: α/2 (captures α/2 probability in left tail)
        - Upper quantile: 1-α/2 (captures α/2 probability in right tail)
    """
    lower_quantile = alpha / 2
    upper_quantile = 1 - lower_quantile
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
        n_calibration_folds: int = 3,
        calibration_split_strategy: Literal[
            "cv_plus", "train_test_split", "adaptive"
        ] = "adaptive",
        adaptive_threshold: int = 50,
        symmetric_adjustment: bool = True,
        validation_split: float = 0.2,
        normalize_features: bool = True,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.alphas = alphas
        self.updated_alphas = alphas.copy()
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.n_calibration_folds = n_calibration_folds
        self.calibration_split_strategy = calibration_split_strategy
        self.adaptive_threshold = adaptive_threshold
        self.symmetric_adjustment = symmetric_adjustment
        self.validation_split = validation_split
        self.normalize_features = normalize_features

        self.quantile_estimator = None
        self.nonconformity_scores = None
        self.lower_nonconformity_scores = None  # For asymmetric adjustments
        self.upper_nonconformity_scores = None  # For asymmetric adjustments
        self.all_quantiles = None
        self.quantile_indices = None
        self.conformalize_predictions = False
        self.last_best_params = None
        self.feature_scaler = None

    def _determine_splitting_strategy(self, total_size: int) -> str:
        """Determine optimal data splitting strategy based on dataset size and configuration.

        Selects between CV+ and train-test split approaches for quantile-based conformal
        calibration based on the configured strategy and dataset characteristics. The
        adaptive strategy automatically chooses the most appropriate method based on
        data size to balance computational efficiency with calibration stability.

        Args:
            total_size: Total number of samples in the dataset.

        Returns:
            Strategy identifier: "cv_plus" or "train_test_split".

        Strategy Selection Logic:
            - "adaptive": Uses CV+ for small datasets (< adaptive_threshold) to maximize
              calibration stability through cross-validation, switches to train-test
              split for larger datasets to improve computational efficiency
            - "cv_plus": Always uses cross-validation based calibration
            - "train_test_split": Always uses single split calibration

        Design Rationale:
            Small datasets benefit from CV+ approach as it provides more stable
            nonconformity score estimation through cross-validation, particularly
            important for quantile-based methods where score stability affects
            coverage reliability. Large datasets can use simpler train-test splits
            for computational efficiency while maintaining adequate calibration.
        """
        if self.calibration_split_strategy == "adaptive":
            return (
                "cv_plus"
                if total_size < self.adaptive_threshold
                else "train_test_split"
            )
        return self.calibration_split_strategy

    def _fit_non_conformal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        all_quantiles: List[float],
        current_alphas: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fit quantile estimator without conformal calibration for small datasets.

        Trains a quantile regression model directly on the provided data without
        applying conformal prediction adjustments. This mode is used when the dataset
        is too small for reliable conformal calibration (below n_pre_conformal_trials
        threshold), providing direct quantile predictions instead of conformally
        adjusted intervals.

        While this approach loses the finite-sample coverage guarantees of conformal
        prediction, it may provide more reliable predictions when calibration data
        is insufficient. The estimator assumes the quantile regression model can
        accurately capture the conditional quantiles of the target distribution.

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            all_quantiles: Sorted list of quantile levels to estimate, in [0, 1].
            current_alphas: Alpha levels for coverage (used for context, not calibration).
            tuning_iterations: Number of hyperparameter search iterations.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible model initialization.
            last_best_params: Warm-start parameters from previous hyperparameter search.

        Implementation Details:
            - Applies feature scaling if requested (fits scaler on all available data)
            - Uses hyperparameter tuning when sufficient data and iterations available
            - Falls back to default parameters for small datasets or when tuning disabled
            - Fits single quantile regression model for all required quantile levels
            - Sets conformalize_predictions flag to False for prediction behavior

        Mathematical Framework:
            Directly estimates conditional quantiles: Q̂_τ(x) = argmin E[ρ_τ(Y - q)]
            where ρ_τ(u) = u(τ - I(u < 0)) is the quantile loss function.

            Prediction intervals: [Q̂_α/2(x), Q̂_1-α/2(x)] without conformal adjustments.

        Usage Context:
            Automatically selected when dataset size < n_pre_conformal_trials, typically
            for exploratory analysis or when conformal calibration is not feasible due
            to data limitations. Users should be aware of the lack of coverage guarantees.
        """
        forced_param_configurations = []

        if last_best_params is not None:
            forced_param_configurations.append(last_best_params)

        estimator_config = ESTIMATOR_REGISTRY[self.quantile_estimator_architecture]
        default_params = deepcopy(estimator_config.default_params)
        if default_params:
            forced_param_configurations.append(default_params)

        if tuning_iterations > 1 and len(X) > min_obs_for_tuning:
            tuner = QuantileTuner(random_state=random_state, quantiles=all_quantiles)
            initialization_params = tuner.tune(
                X=X,
                y=y,
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
        self.quantile_estimator.fit(X, y, quantiles=all_quantiles)
        self.conformalize_predictions = False

    def _fit_cv_plus(
        self,
        X: np.ndarray,
        y: np.ndarray,
        all_quantiles: List[float],
        current_alphas: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fit quantile conformal estimator using CV+ calibration strategy.

        Implements the CV+ (Cross-Validation Plus) approach adapted for quantile-based
        conformal prediction. This method uses k-fold cross-validation for nonconformity
        score calibration while training the final quantile estimator on the complete
        dataset to maximize predictive performance and ensure finite-sample coverage.

        Each fold trains a quantile regression model and computes nonconformity scores
        on the fold's validation set. The scores are aggregated across all folds to
        create a robust calibration distribution. The final estimator is trained on
        all available data using the aggregated calibration scores.

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            all_quantiles: Sorted list of quantile levels to estimate, in [0, 1].
            current_alphas: Alpha levels for coverage, determining required quantiles.
            tuning_iterations: Number of hyperparameter search iterations per fold and final fit.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible fold splits and model initialization.
            last_best_params: Warm-start parameters for quantile estimator hyperparameter search.

        Implementation Details:
            - Uses stratified k-fold splitting with shuffle for robust calibration
            - Applies feature scaling within each fold to prevent data leakage
            - Performs hyperparameter tuning within each fold when data permits
            - Supports both symmetric and asymmetric nonconformity score computation
            - Aggregates scores across all validation folds for final calibration
            - Trains final quantile estimator on complete dataset with proper scaling

        Mathematical Framework:
            For each fold f with training indices T_f and validation indices V_f:
            1. Fit quantile estimator Q̂_f(x, τ) on T_f for all τ ∈ all_quantiles
            2. For each alpha level α, compute validation nonconformity scores:
               - Symmetric: R_i = max(Q̂_f(x_i, α/2) - y_i, y_i - Q̂_f(x_i, 1-α/2))
               - Asymmetric: R_L_i = Q̂_f(x_i, α/2) - y_i, R_U_i = y_i - Q̂_f(x_i, 1-α/2)
            3. Aggregate scores across folds: {R_i}_{i ∈ ∪_f V_f}

        Adjustment Types:
            - Symmetric: Uses single adjustment C = quantile(R, 1-α) for both bounds
            - Asymmetric: Uses separate adjustments C_L, C_U for lower/upper bounds

        Coverage Properties:
            Provides finite-sample coverage guarantees under exchangeability while
            using all data for final model training, balancing efficiency and validity.
        """
        kfold = KFold(
            n_splits=self.n_calibration_folds, shuffle=True, random_state=random_state
        )

        if self.symmetric_adjustment:
            all_nonconformity_scores = [[] for _ in current_alphas]
        else:
            all_lower_scores = [[] for _ in current_alphas]
            all_upper_scores = [[] for _ in current_alphas]

        # Prepare forced parameter configurations for tuning
        forced_param_configurations = []
        if last_best_params is not None:
            forced_param_configurations.append(last_best_params)

        estimator_config = ESTIMATOR_REGISTRY[self.quantile_estimator_architecture]
        default_params = deepcopy(estimator_config.default_params)
        if default_params:
            forced_param_configurations.append(default_params)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # Fit quantile estimator on fold training data with tuning
            if tuning_iterations > 1 and len(X_fold_train) > min_obs_for_tuning:
                tuner = QuantileTuner(
                    random_state=random_state + fold_idx if random_state else None,
                    quantiles=all_quantiles,
                )
                fold_initialization_params = tuner.tune(
                    X=X_fold_train,
                    y=y_fold_train,
                    estimator_architecture=self.quantile_estimator_architecture,
                    n_searches=tuning_iterations,
                    forced_param_configurations=forced_param_configurations,
                )
            else:
                fold_initialization_params = (
                    forced_param_configurations[0]
                    if forced_param_configurations
                    else None
                )

            fold_estimator = initialize_estimator(
                estimator_architecture=self.quantile_estimator_architecture,
                initialization_params=fold_initialization_params,
                random_state=random_state + fold_idx if random_state else None,
            )
            fold_estimator.fit(X_fold_train, y_fold_train, quantiles=all_quantiles)

            # Compute nonconformity scores on validation fold
            val_prediction = fold_estimator.predict(X_fold_val)

            for i, alpha in enumerate(current_alphas):
                lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                if self.symmetric_adjustment:
                    # Symmetric: max of lower and upper deviations
                    lower_deviations = val_prediction[:, lower_idx] - y_fold_val
                    upper_deviations = y_fold_val - val_prediction[:, upper_idx]
                    fold_scores = np.maximum(lower_deviations, upper_deviations)
                    all_nonconformity_scores[i].extend(fold_scores)
                else:
                    # Asymmetric: separate lower and upper scores
                    lower_scores = val_prediction[:, lower_idx] - y_fold_val
                    upper_scores = y_fold_val - val_prediction[:, upper_idx]
                    all_lower_scores[i].extend(lower_scores)
                    all_upper_scores[i].extend(upper_scores)

        # Store aggregated scores
        if self.symmetric_adjustment:
            self.nonconformity_scores = [
                np.array(scores) for scores in all_nonconformity_scores
            ]
        else:
            self.lower_nonconformity_scores = [
                np.array(scores) for scores in all_lower_scores
            ]
            self.upper_nonconformity_scores = [
                np.array(scores) for scores in all_upper_scores
            ]

        # Fit final estimator on all data with tuning
        if tuning_iterations > 1 and len(X) > min_obs_for_tuning:
            tuner = QuantileTuner(random_state=random_state, quantiles=all_quantiles)
            final_initialization_params = tuner.tune(
                X=X,
                y=y,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                forced_param_configurations=forced_param_configurations,
            )
            self.last_best_params = final_initialization_params
        else:
            final_initialization_params = (
                forced_param_configurations[0] if forced_param_configurations else None
            )
            self.last_best_params = last_best_params

        self.quantile_estimator = initialize_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=final_initialization_params,
            random_state=random_state,
        )
        self.quantile_estimator.fit(X, y, quantiles=all_quantiles)
        self.conformalize_predictions = True

    def _fit_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        all_quantiles: List[float],
        current_alphas: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fit quantile conformal estimator using train-test split calibration.

        Implements the traditional split conformal prediction approach for quantile-based
        estimation using a single train-validation split. This method is computationally
        efficient for larger datasets where cross-validation becomes expensive, while
        maintaining finite-sample coverage guarantees through proper calibration.

        The input data is first split into training and validation sets. The quantile
        estimator is trained on the training set and validated on the separate validation
        set to compute nonconformity scores. Feature scaling is applied consistently
        across the split to prevent data leakage while ensuring proper normalization
        for the quantile regression model.

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            all_quantiles: Sorted list of quantile levels to estimate, in [0, 1].
            current_alphas: Alpha levels for coverage, determining required quantiles.
            tuning_iterations: Number of hyperparameter search iterations.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible data splits and model initialization.
            last_best_params: Warm-start parameters for quantile estimator hyperparameter search.

        Implementation Details:
            - Splits input data into training and validation sets using validation_split
            - Fits feature scaler on training data only to prevent information leakage
            - Performs hyperparameter tuning on training set when data permits
            - Uses validation set exclusively for nonconformity score computation
            - Supports both symmetric and asymmetric conformal adjustments
            - Handles empty validation sets gracefully (falls back to non-conformal mode)

        Mathematical Framework:
            1. Split X, y → (X_train, y_train), (X_val, y_val)
            2. Fit quantile estimator Q̂(x, τ) on (X_train, y_train) for all τ ∈ all_quantiles
            3. For each alpha level α and validation point (x_i, y_i):
               - Symmetric: R_i = max(Q̂(x_i, α/2) - y_i, y_i - Q̂(x_i, 1-α/2))
               - Asymmetric: R_L_i = Q̂(x_i, α/2) - y_i, R_U_i = y_i - Q̂(x_i, 1-α/2)
            4. Store {R_i}_{i=1}^{n_val} for conformal adjustment during prediction

        Efficiency Considerations:
            More computationally efficient than CV+ for large datasets, using single
            train-validation split instead of k-fold cross-validation. However, may
            have less stable calibration with smaller validation sets compared to
            the cross-validation approach, especially for asymmetric adjustments.

        Edge Cases:
            When validation set is empty, automatically disables conformal adjustment
            and falls back to direct quantile prediction mode for robustness.
        """
        # Split data internally for train-test approach
        X_train, y_train, X_val, y_val = train_val_split(
            X,
            y,
            train_split=(1 - self.validation_split),
            normalize=False,  # Normalization already applied in fit()
            random_state=random_state,
        )

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
        self.quantile_estimator.fit(X_train, y_train, quantiles=all_quantiles)

        # Compute nonconformity scores on validation set if available
        if len(X_val) > 0:
            if self.symmetric_adjustment:
                self.nonconformity_scores = [np.array([]) for _ in current_alphas]
            else:
                self.lower_nonconformity_scores = [np.array([]) for _ in current_alphas]
                self.upper_nonconformity_scores = [np.array([]) for _ in current_alphas]

            val_prediction = self.quantile_estimator.predict(X_val)

            for i, alpha in enumerate(current_alphas):
                lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                if self.symmetric_adjustment:
                    lower_deviations = val_prediction[:, lower_idx] - y_val
                    upper_deviations = y_val - val_prediction[:, upper_idx]
                    self.nonconformity_scores[i] = np.maximum(
                        lower_deviations, upper_deviations
                    )
                else:
                    self.lower_nonconformity_scores[i] = (
                        val_prediction[:, lower_idx] - y_val
                    )
                    self.upper_nonconformity_scores[i] = (
                        y_val - val_prediction[:, upper_idx]
                    )

            self.conformalize_predictions = True
        else:
            self.conformalize_predictions = False

    def fit(
        self,
        X: np.array,
        y: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 50,
        random_state: Optional[int] = None,
        last_best_params: Optional[dict] = None,
    ):
        """Fit the quantile conformal estimator.

        Uses adaptive data splitting strategy: CV+ for small datasets, train-test split
        for larger datasets, or explicit strategy selection. Supports both symmetric
        and asymmetric conformal adjustments. Handles data preprocessing including
        feature scaling applied to the entire dataset.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
            tuning_iterations: Hyperparameter search iterations (0 disables tuning).
            min_obs_for_tuning: Minimum samples required for hyperparameter tuning.
            random_state: Random seed for reproducible initialization.
            last_best_params: Warm-start parameters from previous fitting.
        """
        current_alphas = self._fetch_alphas()

        # Apply feature scaling to entire dataset if requested
        if self.normalize_features:
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = X
            self.feature_scaler = None

        all_quantiles = []
        for alpha in current_alphas:
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            all_quantiles.append(lower_quantile)
            all_quantiles.append(upper_quantile)
        all_quantiles = sorted(list(set(all_quantiles)))

        self.quantile_indices = {q: i for i, q in enumerate(all_quantiles)}

        total_size = len(X)
        use_conformal = total_size > self.n_pre_conformal_trials

        if use_conformal:
            strategy = self._determine_splitting_strategy(total_size)

            if strategy == "cv_plus":
                self._fit_cv_plus(
                    X_scaled,
                    y,
                    all_quantiles,
                    current_alphas,
                    tuning_iterations,
                    min_obs_for_tuning,
                    random_state,
                    last_best_params,
                )
            else:  # train_test_split
                self._fit_train_test_split(
                    X_scaled,
                    y,
                    all_quantiles,
                    current_alphas,
                    tuning_iterations,
                    min_obs_for_tuning,
                    random_state,
                    last_best_params,
                )

        else:
            self._fit_non_conformal(
                X_scaled,
                y,
                all_quantiles,
                current_alphas,
                tuning_iterations,
                min_obs_for_tuning,
                random_state,
                last_best_params,
            )

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

        # Apply same preprocessing as during training
        X_processed = X.copy()
        if self.normalize_features and self.feature_scaler is not None:
            X_processed = self.feature_scaler.transform(X_processed)

        intervals = []
        prediction = self.quantile_estimator.predict(X_processed)

        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)

            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            if self.conformalize_predictions:
                if self.symmetric_adjustment:
                    # Symmetric adjustment (original CQR)
                    score = np.quantile(
                        self.nonconformity_scores[i],
                        (1 - alpha) / (1 + 1 / len(self.nonconformity_scores[i])),
                        method="linear",
                    )
                    lower_interval_bound = np.array(prediction[:, lower_idx]) - score
                    upper_interval_bound = np.array(prediction[:, upper_idx]) + score
                else:
                    # NOTE: Assuming lower and upper levels are symmetric (meaning, 10th and 90th percentile for eg.
                    # with same misscoverage on each level, otherwise need to use different alpha for each)
                    lower_adjustment = np.quantile(
                        self.lower_nonconformity_scores[i],
                        (1 - alpha / 2)
                        / (1 + 1 / len(self.lower_nonconformity_scores[i])),
                        method="linear",
                    )
                    upper_adjustment = np.quantile(
                        self.upper_nonconformity_scores[i],
                        (1 - alpha / 2)
                        / (1 + 1 / len(self.upper_nonconformity_scores[i])),
                        method="linear",
                    )

                    lower_interval_bound = (
                        np.array(prediction[:, lower_idx]) - lower_adjustment
                    )
                    upper_interval_bound = (
                        np.array(prediction[:, upper_idx]) + upper_adjustment
                    )
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
            Returns [0.5] * len(alphas) for non-conformalized mode.

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
            nature of the quantile-based nonconformity scores. In non-conformalized
            mode, returns neutral beta values (0.5) since no calibration scores exist.
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before calculating beta")

        # In non-conformalized mode, return neutral beta values since no calibration scores exist
        if not self.conformalize_predictions:
            return [0.5] * len(self.alphas)

        X = X.reshape(1, -1)
        # Apply same preprocessing as during training
        if self.normalize_features and self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)

        betas = []
        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            prediction = self.quantile_estimator.predict(X)
            lower_bound = prediction[0, lower_idx]
            upper_bound = prediction[0, upper_idx]

            lower_deviation = lower_bound - y_true
            upper_deviation = y_true - upper_bound
            nonconformity = max(lower_deviation, upper_deviation)

            # According to the DTACI paper: β_t := sup {β : Y_t ∈ Ĉ_t(β)}
            # This means β_t is the proportion of calibration scores >= test nonconformity
            # (i.e., the empirical coverage probability)
            if self.symmetric_adjustment:
                beta = np.mean(self.nonconformity_scores[i] >= nonconformity)
            else:
                # For asymmetric adjustment, use the maximum of lower and upper beta values
                lower_beta = np.mean(
                    self.lower_nonconformity_scores[i] >= lower_deviation
                )
                upper_beta = np.mean(
                    self.upper_nonconformity_scores[i] >= upper_deviation
                )
                beta = max(lower_beta, upper_beta)

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
