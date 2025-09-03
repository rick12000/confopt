import logging
import numpy as np
from typing import Optional, Tuple, List, Literal
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from confopt.wrapping import ConformalBounds
from confopt.utils.preprocessing import train_val_split
from confopt.selection.estimation import (
    initialize_estimator,
    QuantileTuner,
)
from confopt.selection.estimator_configuration import ESTIMATOR_REGISTRY
from copy import deepcopy

logger = logging.getLogger(__name__)


def set_calibration_split(n_observations: int) -> float:
    """Set to 20%, but limit to between 4 and 8 observations
    since we tend to only need at most 4 quantiles for conformal search"""
    candidate_split = 0.2
    if candidate_split * n_observations < 4:
        return 4 / n_observations
    else:
        return candidate_split


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
    """CV+ quantile-based conformal predictor with theoretical coverage guarantees.

    Implements the CV+ method from Barber et al. (2019) using quantile regression
    as the base learner. This approach provides 1-2α - √(2/n) coverage guarantees
    under exchangeability assumptions by using K-fold cross-validation for
    conformal calibration while storing all fold estimators for prediction.

    The estimator supports both conformalized and non-conformalized modes:
    - Conformalized: Uses CV+ conformal prediction with theoretical guarantees
    - Non-conformalized: Direct quantile predictions (when data is limited)

    Args:
        quantile_estimator_architecture: Architecture identifier for quantile estimator.
            Must be registered in ESTIMATOR_REGISTRY and support quantile fitting.
        alphas: List of miscoverage levels (1-alpha gives coverage probability).
            Must be in (0, 1) range.
        n_pre_conformal_trials: Minimum samples required for conformal calibration.
            Below this threshold, uses direct quantile prediction.

    Attributes:
        fold_estimators: List of K fitted quantile regression models from CV+.
        nonconformity_scores: Calibration scores per alpha level from CV+.
        all_quantiles: Sorted list of all required quantiles.
        quantile_indices: Mapping from quantile values to prediction array indices.
        conformalize_predictions: Boolean flag indicating if conformal adjustment is used.

    Mathematical Framework (CV+):
        For each fold k and alpha level α:
        1. Fit quantile estimator Q̂_{-S_k}(x, τ) on fold k training data
        2. Compute nonconformity R_i = max(Q̂_{-S_k}(x_i, α/2) - y_i, y_i - Q̂_{-S_k}(x_i, 1-α/2))
        3. For prediction at x, construct interval:
           [q_{n,α}{Q̂_{-S_{k(i)}}(x) - R_i}, q_{n,1-α}{Q̂_{-S_{k(i)}}(x) + R_i}]

    Coverage Properties:
        Provides 1-2α - √(2/n) coverage under exchangeability assumptions.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        alphas: List[float],
        n_pre_conformal_trials: int = 32,
        n_calibration_folds: int = 3,
        calibration_split_strategy: Literal[
            "cv", "train_test_split", "adaptive"
        ] = "adaptive",
        adaptive_threshold: int = 50,
        normalize_features: bool = True,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.alphas = alphas
        self.updated_alphas = self.alphas.copy()
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.n_calibration_folds = n_calibration_folds
        self.calibration_split_strategy = calibration_split_strategy
        self.adaptive_threshold = adaptive_threshold
        self.normalize_features = normalize_features

        self.quantile_estimator = None
        self.nonconformity_scores = None
        self.all_quantiles = None
        self.quantile_indices = None
        self.conformalize_predictions = False
        self.last_best_params = None
        self.feature_scaler = None
        self.fold_estimators = []  # Store K-fold estimators for CV+

    def _determine_splitting_strategy(self, total_size: int) -> str:
        """Determine optimal data splitting strategy based on dataset size and configuration.

        Selects between cross-validation (CV) and train-test split approaches for quantile-based conformal
        calibration based on the configured strategy and dataset characteristics. The
        adaptive strategy automatically chooses the most appropriate method based on
        data size to balance computational efficiency with calibration stability.

        Args:
            total_size: Total number of samples in the dataset.

        Returns:
            Strategy identifier: "cv" or "train_test_split".

        Strategy Selection Logic:
            - "adaptive": Uses CV for small datasets (< adaptive_threshold) to improve
              calibration stability with fewer folds, and switches to train-test split
              for larger datasets to improve computational efficiency
            - "cv": Always uses cross-validation-based calibration (CV, not CV+)
            - "train_test_split": Always uses single split calibration

        Design Rationale:
            Small datasets benefit from CV-based calibration which provides more stable
            nonconformity score estimation than a single split. Note: CV (not CV+)
            offers weaker distribution-free guarantees than CV+ but is effective with
            fewer folds. Large datasets can use simpler train-test splits for
            computational efficiency while maintaining adequate calibration.
        """
        if self.calibration_split_strategy == "adaptive":
            return "cv" if total_size < self.adaptive_threshold else "train_test_split"
        return self.calibration_split_strategy

    def _fit_non_conformal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        all_quantiles: List[float],
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

        # Store single estimator for compatibility with CV+ framework
        self.fold_estimators = [self.quantile_estimator]
        self.conformalize_predictions = False

    def _fit_cv_plus(
        self,
        X: np.ndarray,
        y: np.ndarray,
        all_quantiles: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fit quantile conformal estimator using CV+ method.

        Implements the CV+ method from Barber et al. (2019) for quantile regression.
        For each fold k, trains quantile estimator on fold k's training data and computes
        nonconformity scores on fold k's validation data. Stores all K fold estimators
        for use in prediction intervals, providing theoretical coverage guarantees of
        1-2α - √(2/n).

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            all_quantiles: Sorted list of quantile levels to estimate, in [0, 1].
            tuning_iterations: Number of hyperparameter search iterations per fold.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible fold splits and model initialization.
            last_best_params: Warm-start parameters for quantile estimator hyperparameter search.

        Mathematical Framework (CV+):
            For each fold k with training indices T_k and validation indices V_k:
            1. Fit quantile estimator Q̂_{-S_k}(x, τ) on T_k for all τ ∈ all_quantiles
            2. For validation point i ∈ V_k, compute nonconformity:
               R_i = max(Q̂_{-S_k}(x_i, α/2) - y_i, y_i - Q̂_{-S_k}(x_i, 1-α/2))
            3. For prediction at x_{n+1}, construct interval:
               [q_{n,α}{Q̂_{-S_{k(i)}}(x_{n+1}) - R_i}, q_{n,1-α}{Q̂_{-S_{k(i)}}(x_{n+1}) + R_i}]
               where k(i) identifies fold containing point i.

        Coverage Properties:
            Provides 1-2α - √(2/n) coverage guarantee under exchangeability.
        """
        kfold = KFold(
            n_splits=self.n_calibration_folds, shuffle=True, random_state=random_state
        )

        # Store nonconformity scores and fold estimators for CV+
        fold_nonconformity_scores = [[] for _ in self.alphas]
        self.fold_estimators = []

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
                    random_state=random_state if random_state else None,
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
                random_state=random_state if random_state else None,
            )
            fold_estimator.fit(X_fold_train, y_fold_train, quantiles=all_quantiles)

            # Store fold estimator for CV+
            self.fold_estimators.append(fold_estimator)

            # Compute nonconformity scores on validation fold
            val_prediction = fold_estimator.predict(X_fold_val)

            for i, alpha in enumerate(self.alphas):
                lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                # Symmetric nonconformity scores (CQR approach)
                lower_deviations = val_prediction[:, lower_idx] - y_fold_val
                upper_deviations = y_fold_val - val_prediction[:, upper_idx]
                fold_scores = np.maximum(lower_deviations, upper_deviations)
                fold_nonconformity_scores[i].append(fold_scores)

        # Store nonconformity scores as list of lists (one per alpha, containing fold arrays)
        self.nonconformity_scores = fold_nonconformity_scores

        # For CV+, we don't fit a final estimator on all data
        # Instead, we use the fold estimators for prediction
        self.last_best_params = last_best_params
        self.conformalize_predictions = True

    def _fit_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        all_quantiles: List[float],
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
            tuning_iterations: Number of hyperparameter search iterations.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible data splits and model initialization.
            last_best_params: Warm-start parameters for quantile estimator hyperparameter search.

        Implementation Details:
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
            More computationally efficient than CV-based calibration for large datasets,
            using a single train-validation split instead of k-fold cross-validation.
            However, it may have less stable calibration with smaller validation sets
            compared to the cross-validation approach, especially for asymmetric
            adjustments.

        Edge Cases:
            When validation set is empty, automatically disables conformal adjustment
            and falls back to direct quantile prediction mode for robustness.
        """
        # Split data internally for train-test approach
        X_train, y_train, X_val, y_val = train_val_split(
            X,
            y,
            train_split=(1 - set_calibration_split(len(X))),
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

        quantile_estimator = initialize_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )
        quantile_estimator.fit(X_train, y_train, quantiles=all_quantiles)

        # Compute nonconformity scores on validation set if available
        if len(X_val) > 0:
            # Store single fold estimator for split conformal
            self.fold_estimators = [quantile_estimator]

            val_prediction = quantile_estimator.predict(X_val)
            fold_nonconformity_scores = [[] for _ in self.alphas]

            for i, alpha in enumerate(self.alphas):
                lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                # Symmetric nonconformity scores
                lower_deviations = val_prediction[:, lower_idx] - y_val
                upper_deviations = y_val - val_prediction[:, upper_idx]
                fold_scores = np.maximum(lower_deviations, upper_deviations)
                fold_nonconformity_scores[i].append(fold_scores)

            # Store as list of lists for consistency with CV+ structure
            self.nonconformity_scores = fold_nonconformity_scores

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

        Uses an adaptive data splitting strategy: CV (not CV+) for small datasets,
        train-test split for larger datasets, or explicit strategy selection. Supports
        both symmetric and asymmetric conformal adjustments. Handles data preprocessing
        including feature scaling applied to the entire dataset.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
            tuning_iterations: Hyperparameter search iterations (0 disables tuning).
            min_obs_for_tuning: Minimum samples required for hyperparameter tuning.
            random_state: Random seed for reproducible initialization.
            last_best_params: Warm-start parameters from previous fitting.
        """
        # Apply feature scaling to entire dataset if requested
        if self.normalize_features:
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = X
            self.feature_scaler = None

        all_quantiles = []
        for alpha in self.alphas:
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            all_quantiles.append(lower_quantile)
            all_quantiles.append(upper_quantile)
        all_quantiles = sorted(list(set(all_quantiles)))

        self.quantile_indices = {q: i for i, q in enumerate(all_quantiles)}

        total_size = len(X)
        use_conformal = total_size > self.n_pre_conformal_trials

        if use_conformal:
            strategy = self._determine_splitting_strategy(total_size)

            if strategy == "cv":
                self._fit_cv_plus(
                    X_scaled,
                    y,
                    all_quantiles,
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
                tuning_iterations,
                min_obs_for_tuning,
                random_state,
                last_best_params,
            )

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        """Generate conformal prediction intervals using CV+ method.

        Produces prediction intervals with finite-sample coverage guarantees using
        the CV+ method from Barber et al. (2019). For each prediction point,
        constructs intervals using quantiles of {Q̂_{-S_{k(i)}}(x) ± R_i} where
        Q̂_{-S_{k(i)}} is the fold estimator and R_i are the nonconformity scores.

        Args:
            X: Input features for prediction, shape (n_predict, n_features).

        Returns:
            List of ConformalBounds objects, one per alpha level, each containing:
                - lower_bounds: Lower interval bounds, shape (n_predict,)
                - upper_bounds: Upper interval bounds, shape (n_predict,)

        Raises:
            ValueError: If fold estimators have not been fitted.

        Mathematical Details (CV+):
            For each alpha level α and prediction point x:
            1. Compute {Q̂_{-S_{k(i)}}(x) + R_i} and {Q̂_{-S_{k(i)}}(x) - R_i}
               for all validation points i with their corresponding fold estimators
            2. Return interval: [q_{n,α}{Q̂_{-S_{k(i)}}(x) - R_i}, q_{n,1-α}{Q̂_{-S_{k(i)}}(x) + R_i}]

        Coverage Guarantee:
            Provides 1-2α - √(2/n) coverage under exchangeability assumptions.
        """
        if not self.fold_estimators:
            raise ValueError("Fold estimators must be fitted before prediction")

        # Apply same preprocessing as during training
        X_processed = X.copy()
        if self.normalize_features and self.feature_scaler is not None:
            X_processed = self.feature_scaler.transform(X_processed)

        intervals = []
        n_predict = X_processed.shape[0]

        # For CV+, we need to construct intervals using fold estimators
        for i, (alpha, alpha_adjusted) in enumerate(
            zip(self.alphas, self.updated_alphas)
        ):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            if self.conformalize_predictions:
                # CV+ method: for each validation point i and corresponding fold k(i),
                # compute Q̂_{-S_{k(i)}}(x) ± R_i, then take quantiles
                lower_values = []
                upper_values = []

                # Iterate through each fold and its nonconformity scores for this alpha
                for fold_idx, fold_scores in enumerate(self.nonconformity_scores[i]):
                    # Get predictions from the corresponding fold estimator
                    fold_pred = self.fold_estimators[fold_idx].predict(X_processed)

                    # Add to CV+ collections for each score in this fold
                    for score in fold_scores:
                        lower_values.extend(fold_pred[:, lower_idx] - score)
                        upper_values.extend(fold_pred[:, upper_idx] + score)

                # Reshape to group by prediction point
                n_scores = sum(
                    len(fold_scores) for fold_scores in self.nonconformity_scores[i]
                )
                lower_values = np.array(lower_values).reshape(n_scores, n_predict)
                upper_values = np.array(upper_values).reshape(n_scores, n_predict)

                # Compute CV+ interval bounds for each prediction point
                lower_bounds = []
                upper_bounds = []

                for pred_idx in range(n_predict):
                    lower_bound = np.quantile(
                        lower_values[:, pred_idx],
                        alpha_adjusted / (1 + 1 / n_scores),
                        method="linear",
                    )
                    upper_bound = np.quantile(
                        upper_values[:, pred_idx],
                        (1 - alpha_adjusted) / (1 + 1 / n_scores),
                        method="linear",
                    )

                    lower_bounds.append(lower_bound)
                    upper_bounds.append(upper_bound)

                lower_interval_bound = np.array(lower_bounds)
                upper_interval_bound = np.array(upper_bounds)
            else:
                # Non-conformalized: use first fold estimator (or any single estimator)
                prediction = self.fold_estimators[0].predict(X_processed)
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
        if self.fold_estimators == []:
            raise ValueError("Estimator must be fitted before calculating beta")

        # In non-conformalized mode, return neutral beta values since no calibration scores exist
        if not self.conformalize_predictions:
            return [0.5] * len(self.alphas)

        X_processed = X.reshape(1, -1)
        # Apply same preprocessing as during training
        if self.normalize_features and self.feature_scaler is not None:
            X_processed = self.feature_scaler.transform(X_processed)

        betas = []
        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            # Compute average prediction across all fold estimators
            all_predictions = []
            for fold_estimator in self.fold_estimators:
                fold_pred = fold_estimator.predict(X_processed)
                all_predictions.append(fold_pred)

            avg_prediction = np.mean(all_predictions, axis=0)
            lower_bound = avg_prediction[0, lower_idx]
            upper_bound = avg_prediction[0, upper_idx]

            lower_deviation = lower_bound - y_true
            upper_deviation = y_true - upper_bound
            nonconformity = max(lower_deviation, upper_deviation)

            # Calculate beta using calibration scores from all folds for this alpha
            all_fold_scores = []
            for fold_scores in self.nonconformity_scores[i]:
                all_fold_scores.extend(fold_scores)
            beta = np.mean(np.array(all_fold_scores) >= nonconformity)

            betas.append(beta)

        return betas

    def update_alphas(self, new_alphas: List[float]):
        """Update coverage levels for CV+ quantile conformal estimator.

        Updates target coverage levels for the CV+ quantile-based estimator.
        Since CV+ uses the same fold estimators and nonconformity scores for
        all alpha levels, this operation is computationally efficient.

        Args:
            new_alphas: New miscoverage levels (1-alpha gives coverage).
                Must be in (0, 1) range.

        Important:
            If new_alphas require quantiles not computed during fit(), the estimator
            may need to be refitted. For maximum efficiency, determine the complete
            set of required alphas before calling fit().
        """
        self.updated_alphas = new_alphas.copy()
