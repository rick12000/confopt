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
    """Determines the calibration split ratio based on dataset size.

    Ensures a minimum of 4 observations for calibration while defaulting to 20%.

    Args:
        n_observations: Total number of observations in the dataset.

    Returns:
        Calibration split ratio between 0 and 1.
    """
    candidate_split = 0.2
    if candidate_split * n_observations < 4:
        return 4 / n_observations
    else:
        return candidate_split


def alpha_to_quantiles(alpha: float) -> Tuple[float, float]:
    """Converts miscoverage level to corresponding quantile bounds.

    Creates symmetric quantile bounds for two-sided prediction intervals.

    Args:
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage intervals).

    Returns:
        Tuple of (lower_quantile, upper_quantile) values.
    """
    lower_quantile = alpha / 2
    upper_quantile = 1 - lower_quantile
    return lower_quantile, upper_quantile


class QuantileConformalEstimator:
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
        """Conformal quantile regression estimator with adaptive calibration strategies.

        Implements conformal prediction to create statistically valid prediction intervals
        using quantile regression models.

        Args:
           quantile_estimator_architecture: Architecture name from ESTIMATOR_REGISTRY (e.g., 'qgbm', 'qrf', 'qknn').
           alphas: List of miscoverage levels for prediction intervals (e.g., [0.1] for 90% coverage).
           n_pre_conformal_trials: Minimum observations needed before using conformal prediction.
           n_calibration_folds: Number of folds for cross-validation calibration.
           calibration_split_strategy: Strategy for data splitting during calibration.
           adaptive_threshold: Observation threshold for adaptive strategy switching.
           normalize_features: Whether to standardize input features using StandardScaler.
        """
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.alphas = alphas
        self.updated_alphas = self.alphas.copy()
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.n_calibration_folds = n_calibration_folds
        self.calibration_split_strategy = calibration_split_strategy
        self.adaptive_threshold = adaptive_threshold
        self.normalize_features = normalize_features

        self.quantile_estimator = None
        self.fold_scores_per_alpha = None
        self.flattened_quantiles = None
        self.quantile_indices = None
        self.conformalize_predictions = False
        self.last_best_params = None
        self.feature_scaler = None
        self.fold_estimators = []

    def _determine_splitting_strategy(self, n_observations: int) -> str:
        """Selects the optimal data splitting strategy based on dataset size.

        Uses cross-validation for small datasets and train-test split for larger ones when adaptive.

        Args:
            n_observations: Total number of observations in the training dataset.

        Returns:
            Strategy name: 'cv', 'train_test_split', or the fixed calibration_split_strategy.
        """
        if self.calibration_split_strategy == "adaptive":
            return (
                "cv" if n_observations < self.adaptive_threshold else "train_test_split"
            )
        return self.calibration_split_strategy

    def _fit_non_conformal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        flattened_quantiles: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fits a standard quantile estimator without conformal calibration.

        Used when dataset size is below n_pre_conformal_trials threshold.

        Args:
            X: Input feature matrix, shape (n_samples, n_features).
            y: Target values array, shape (n_samples,).
            flattened_quantiles: Sorted list of unique quantile levels derived from alphas.
            tuning_iterations: Number of hyperparameter optimization iterations using QuantileTuner.
            min_obs_for_tuning: Minimum observations required for hyperparameter tuning.
            random_state: Random seed for reproducible estimator initialization.
            last_best_params: Previously optimized parameters from estimator_configuration to warm-start.
        """
        forced_param_configurations = []

        if last_best_params is not None:
            forced_param_configurations.append(last_best_params)

        estimator_config = ESTIMATOR_REGISTRY[self.quantile_estimator_architecture]
        default_params = deepcopy(estimator_config.default_params)
        if default_params:
            forced_param_configurations.append(default_params)

        if tuning_iterations > 1 and len(X) > min_obs_for_tuning:
            tuner = QuantileTuner(
                random_state=random_state, quantiles=flattened_quantiles
            )
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
        self.quantile_estimator.fit(X, y, quantiles=flattened_quantiles)

        self.fold_estimators = [self.quantile_estimator]
        self.conformalize_predictions = False

    def _fit_cv_plus(
        self,
        X: np.ndarray,
        y: np.ndarray,
        flattened_quantiles: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fits conformal estimator using cross-validation for calibration.

        Trains separate models on each fold and computes nonconformity scores for conformal adjustment.

        Args:
            X: Input feature matrix, shape (n_samples, n_features).
            y: Target values array, shape (n_samples,).
            flattened_quantiles: Sorted list of unique quantile levels derived from alphas.
            tuning_iterations: Number of hyperparameter optimization iterations per fold.
            min_obs_for_tuning: Minimum observations required for hyperparameter tuning per fold.
            random_state: Random seed for KFold splitting and estimator initialization.
            last_best_params: Previously optimized parameters to warm-start each fold.
        """
        kfold = KFold(
            n_splits=self.n_calibration_folds, shuffle=True, random_state=random_state
        )

        fold_scores_per_alpha = [[] for _ in self.alphas]
        self.fold_estimators = []

        forced_param_configurations = []
        if last_best_params is not None:
            forced_param_configurations.append(last_best_params)

        estimator_config = ESTIMATOR_REGISTRY[self.quantile_estimator_architecture]
        default_params = deepcopy(estimator_config.default_params)
        if default_params:
            forced_param_configurations.append(default_params)

        for _, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            if tuning_iterations > 1 and len(X_fold_train) > min_obs_for_tuning:
                tuner = QuantileTuner(
                    random_state=random_state if random_state else None,
                    quantiles=flattened_quantiles,
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
            fold_estimator.fit(
                X_fold_train, y_fold_train, quantiles=flattened_quantiles
            )

            self.fold_estimators.append(fold_estimator)

            val_prediction = fold_estimator.predict(X_fold_val)

            for i, alpha in enumerate(self.alphas):
                lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                lower_deviations = val_prediction[:, lower_idx] - y_fold_val
                upper_deviations = y_fold_val - val_prediction[:, upper_idx]
                fold_scores = np.maximum(lower_deviations, upper_deviations)
                fold_scores_per_alpha[i].append(fold_scores)

        self.fold_scores_per_alpha = fold_scores_per_alpha

        self.last_best_params = last_best_params
        self.conformalize_predictions = True

    def _fit_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        flattened_quantiles: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fits conformal estimator using train-test split for calibration.

        Trains on training portion and computes nonconformity scores on validation portion.

        Args:
            X: Input feature matrix, shape (n_samples, n_features).
            y: Target values array, shape (n_samples,).
            flattened_quantiles: Sorted list of unique quantile levels derived from alphas.
            tuning_iterations: Number of hyperparameter optimization iterations.
            min_obs_for_tuning: Minimum observations required for hyperparameter tuning.
            random_state: Random seed for train_val_split and estimator initialization.
            last_best_params: Previously optimized parameters to warm-start training.
        """
        X_train, y_train, X_val, y_val = train_val_split(
            X,
            y,
            train_split=(1 - set_calibration_split(len(X))),
            normalize=False,
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
            tuner = QuantileTuner(
                random_state=random_state, quantiles=flattened_quantiles
            )
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
        quantile_estimator.fit(X_train, y_train, quantiles=flattened_quantiles)

        self.fold_estimators = [quantile_estimator]

        val_prediction = quantile_estimator.predict(X_val)
        fold_scores_per_alpha = [[] for _ in self.alphas]

        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            lower_deviations = val_prediction[:, lower_idx] - y_val
            upper_deviations = y_val - val_prediction[:, upper_idx]
            fold_scores = np.maximum(lower_deviations, upper_deviations)
            fold_scores_per_alpha[i].append(fold_scores)

        self.fold_scores_per_alpha = fold_scores_per_alpha
        self.conformalize_predictions = True

    def fit(
        self,
        X: np.array,
        y: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 50,
        random_state: Optional[int] = None,
        last_best_params: Optional[dict] = None,
    ):
        """Trains the conformal quantile estimator on the provided data.

        Automatically selects between conformal and non-conformal approaches based on dataset size.

        Args:
            X: Input feature matrix, shape (n_samples, n_features).
            y: Target values array, shape (n_samples,).
            tuning_iterations: Number of hyperparameter optimization iterations using QuantileTuner.
            min_obs_for_tuning: Minimum observations required to enable hyperparameter tuning.
            random_state: Random seed for reproducible results across folds and estimators.
            last_best_params: Previously optimized parameters from ESTIMATOR_REGISTRY to warm-start tuning.
        """
        if self.normalize_features:
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = X
            self.feature_scaler = None

        flattened_quantiles = []
        for alpha in self.alphas:
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            flattened_quantiles.append(lower_quantile)
            flattened_quantiles.append(upper_quantile)
        flattened_quantiles = sorted(list(set(flattened_quantiles)))

        self.quantile_indices = {q: i for i, q in enumerate(flattened_quantiles)}

        n_observations = len(X)
        use_conformal = n_observations > self.n_pre_conformal_trials

        if use_conformal:
            strategy = self._determine_splitting_strategy(n_observations)

            if strategy == "cv":
                self._fit_cv_plus(
                    X=X_scaled,
                    y=y,
                    flattened_quantiles=flattened_quantiles,
                    tuning_iterations=tuning_iterations,
                    min_obs_for_tuning=min_obs_for_tuning,
                    random_state=random_state,
                    last_best_params=last_best_params,
                )
            else:
                self._fit_train_test_split(
                    X=X_scaled,
                    y=y,
                    flattened_quantiles=flattened_quantiles,
                    tuning_iterations=tuning_iterations,
                    min_obs_for_tuning=min_obs_for_tuning,
                    random_state=random_state,
                    last_best_params=last_best_params,
                )

        else:
            self._fit_non_conformal(
                X=X_scaled,
                y=y,
                flattened_quantiles=flattened_quantiles,
                tuning_iterations=tuning_iterations,
                min_obs_for_tuning=min_obs_for_tuning,
                random_state=random_state,
                last_best_params=last_best_params,
            )

    def _preprocess_features(self, X: np.array) -> np.array:
        """Applies feature preprocessing transformations to input data.

        Normalizes features using fitted StandardScaler if enabled during initialization.

        Args:
            X: Raw input feature matrix, shape (n_samples, n_features).

        Returns:
            Preprocessed feature array with same shape, standardized if normalize_features=True.
        """
        X_processed = X.copy()
        if self.normalize_features and self.feature_scaler is not None:
            X_processed = self.feature_scaler.transform(X=X_processed)

        return X_processed

    def _get_quantile_indices(self, alpha: float) -> Tuple[int, int]:
        """Retrieves array indices for lower and upper quantiles corresponding to alpha.

        Maps miscoverage level to quantile positions in the prediction array.

        Args:
            alpha: Miscoverage level for the prediction interval.

        Returns:
            Tuple of (lower_index, upper_index) for quantile array positions.
        """
        lower_quantile, upper_quantile = alpha_to_quantiles(alpha=alpha)

        return (
            self.quantile_indices[lower_quantile],
            self.quantile_indices[upper_quantile],
        )

    def _compute_conformal_bounds(
        self,
        X: np.array,
        fold_nonconformity_scores: List[np.array],
        alpha_adjusted: float,
        lower_idx: int,
        upper_idx: int,
    ) -> Tuple[np.array, np.array]:
        """Computes conformal prediction bounds using calibrated nonconformity scores.

        Combines predictions from multiple folds with nonconformity scores to create
        statistically valid prediction intervals.

        Args:
            X: Input features for prediction, shape (n_samples, n_features).
            fold_nonconformity_scores: List of nonconformity score arrays from each calibration fold.
            alpha_adjusted: Adjusted miscoverage level for the prediction interval from adaptive mechanisms.
            lower_idx: Index of lower quantile in flattened_quantiles prediction array.
            upper_idx: Index of upper quantile in flattened_quantiles prediction array.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays for prediction intervals, shape (n_samples,).
        """
        fold_preds = [estimator.predict(X=X) for estimator in self.fold_estimators]

        flattened_lower_values = np.concatenate(
            [
                pred[:, lower_idx] - scores.reshape(-1, 1)
                for pred, scores in zip(fold_preds, fold_nonconformity_scores)
            ]
        )

        flattened_upper_values = np.concatenate(
            [
                pred[:, upper_idx] + scores.reshape(-1, 1)
                for pred, scores in zip(fold_preds, fold_nonconformity_scores)
            ]
        )

        flattened_scores = np.concatenate(fold_nonconformity_scores)
        n_scores = len(flattened_scores)
        lower_quantile = alpha_adjusted / (1 + 1 / n_scores)
        upper_quantile = (1 - alpha_adjusted) / (1 + 1 / n_scores)

        lower_bound = np.quantile(
            a=flattened_lower_values, q=lower_quantile, axis=0, method="linear"
        )
        upper_bound = np.quantile(
            a=flattened_upper_values, q=upper_quantile, axis=0, method="linear"
        )

        return lower_bound, upper_bound

    def _compute_nonconformal_bounds(
        self, X_processed: np.array, lower_idx: int, upper_idx: int
    ) -> Tuple[np.array, np.array]:
        """Computes standard quantile bounds without conformal calibration.

        Returns raw quantile predictions from the single trained estimator.

        Args:
            X_processed: Preprocessed input features, shape (n_samples, n_features).
            lower_idx: Index of lower quantile in flattened_quantiles prediction array.
            upper_idx: Index of upper quantile in flattened_quantiles prediction array.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays from quantile predictions, shape (n_samples,).
        """
        prediction = self.fold_estimators[0].predict(X=X_processed)

        return prediction[:, lower_idx], prediction[:, upper_idx]

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        """Generates prediction intervals for new input data.

        Creates statistically valid prediction intervals using either conformal
        or standard quantile bounds depending on the fitted model type.

        Args:
            X: Input feature matrix for prediction, shape (n_samples, n_features).

        Returns:
            List of ConformalBounds objects with lower_bounds and upper_bounds arrays, one per miscoverage level.

        Raises:
            ValueError: If the estimator has not been fitted yet.
        """
        if not self.fold_estimators:
            raise ValueError("Fold estimators must be fitted before prediction")

        X_processed = self._preprocess_features(X=X)
        intervals = []

        for i, (alpha, alpha_adjusted) in enumerate(
            zip(self.alphas, self.updated_alphas)
        ):
            lower_idx, upper_idx = self._get_quantile_indices(alpha=alpha)

            if self.conformalize_predictions:
                fold_scores = self.fold_scores_per_alpha[i]
                lower_bound, upper_bound = self._compute_conformal_bounds(
                    X=X_processed,
                    fold_nonconformity_scores=fold_scores,
                    alpha_adjusted=alpha_adjusted,
                    lower_idx=lower_idx,
                    upper_idx=upper_idx,
                )
            else:
                lower_bound, upper_bound = self._compute_nonconformal_bounds(
                    X_processed=X_processed, lower_idx=lower_idx, upper_idx=upper_idx
                )

            intervals.append(
                ConformalBounds(lower_bounds=lower_bound, upper_bounds=upper_bound)
            )

        return intervals

    def calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        """Calculates beta values indicating empirical coverage probability.

        Computes the fraction of calibration nonconformity scores that exceed the
        nonconformity of the given observation, used for adaptive alpha adjustment.

        Args:
            X: Single observation features to evaluate, shape (n_features,).
            y_true: True target value for the observation.

        Returns:
            List of beta values (empirical coverage probabilities), one per miscoverage level.

        Raises:
            ValueError: If the estimator has not been fitted yet.
        """
        if self.fold_estimators == []:
            raise ValueError("Estimator must be fitted before calculating beta")

        if not self.conformalize_predictions:
            return [0.5] * len(self.alphas)

        X_processed = X.reshape(1, -1)
        if self.normalize_features and self.feature_scaler is not None:
            X_processed = self.feature_scaler.transform(X_processed)

        betas = []
        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

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

            flattened_scores = []
            for fold_scores in self.fold_scores_per_alpha[i]:
                flattened_scores.extend(fold_scores)
            beta = np.mean(np.array(flattened_scores) >= nonconformity)

            betas.append(beta)

        return betas

    def update_alphas(self, new_alphas: List[float]):
        """Updates the miscoverage levels for prediction intervals.

        Allows dynamic adjustment of coverage levels without refitting the model.

        Args:
            new_alphas: New list of miscoverage levels to use for predictions.
        """
        self.updated_alphas = new_alphas.copy()
