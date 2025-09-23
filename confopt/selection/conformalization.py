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
    candidate_split = 0.2
    if candidate_split * n_observations < 4:
        return 4 / n_observations
    else:
        return candidate_split


def alpha_to_quantiles(alpha: float) -> Tuple[float, float]:
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
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.alphas = alphas
        self.updated_alphas = self.alphas.copy()
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.n_calibration_folds = n_calibration_folds
        self.calibration_split_strategy = calibration_split_strategy
        self.adaptive_threshold = adaptive_threshold
        self.normalize_features = normalize_features

        self.quantile_estimator = None
        self.fold_nonconformity_scores = None
        self.flattened_quantiles = None
        self.quantile_indices = None
        self.conformalize_predictions = False
        self.last_best_params = None
        self.feature_scaler = None
        self.fold_estimators = []

    def _determine_splitting_strategy(self, n_observations: int) -> str:
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
        kfold = KFold(
            n_splits=self.n_calibration_folds, shuffle=True, random_state=random_state
        )

        fold_nonconformity_scores = [[] for _ in self.alphas]
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
                fold_nonconformity_scores[i].append(fold_scores)

        self.fold_nonconformity_scores = fold_nonconformity_scores

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
        fold_nonconformity_scores = [[] for _ in self.alphas]

        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            lower_deviations = val_prediction[:, lower_idx] - y_val
            upper_deviations = y_val - val_prediction[:, upper_idx]
            fold_scores = np.maximum(lower_deviations, upper_deviations)
            fold_nonconformity_scores[i].append(fold_scores)

        self.fold_nonconformity_scores = fold_nonconformity_scores
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

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        if not self.fold_estimators:
            raise ValueError("Fold estimators must be fitted before prediction")

        X_processed = X.copy()
        if self.normalize_features and self.feature_scaler is not None:
            X_processed = self.feature_scaler.transform(X_processed)

        intervals = []
        n_candidates = len(X_processed)
        for i, (alpha, alpha_adjusted) in enumerate(
            zip(self.alphas, self.updated_alphas)
        ):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            if self.conformalize_predictions:
                flattened_nonconformity_scores = []
                for fold_nonconformity_scores in self.fold_nonconformity_scores[i]:
                    flattened_nonconformity_scores.extend(fold_nonconformity_scores)
                flattened_nonconformity_scores = np.array(
                    flattened_nonconformity_scores
                )
                n_scores = len(flattened_nonconformity_scores)

                lower_values = np.empty((n_scores, n_candidates))
                upper_values = np.empty((n_scores, n_candidates))

                nonconformity_score_idx = 0
                for fold_idx, fold_nonconformity_scores in enumerate(
                    self.fold_nonconformity_scores[i]
                ):
                    fold_pred = self.fold_estimators[fold_idx].predict(X_processed)
                    n_fold_nonconformity_scores = len(fold_nonconformity_scores)

                    fold_lower_pred = fold_pred[:, lower_idx]
                    fold_upper_pred = fold_pred[:, upper_idx]
                    fold_scores_array = np.array(fold_nonconformity_scores).reshape(
                        -1, 1
                    )

                    lower_values[
                        nonconformity_score_idx : nonconformity_score_idx
                        + n_fold_nonconformity_scores
                    ] = (fold_lower_pred - fold_scores_array)
                    upper_values[
                        nonconformity_score_idx : nonconformity_score_idx
                        + n_fold_nonconformity_scores
                    ] = (fold_upper_pred + fold_scores_array)

                    nonconformity_score_idx += n_fold_nonconformity_scores

                lower_conformal_quantile = alpha_adjusted / (1 + 1 / n_scores)
                upper_conformal_quantile = (1 - alpha_adjusted) / (1 + 1 / n_scores)
                lower_interval_bound = np.quantile(
                    lower_values, lower_conformal_quantile, axis=0, method="linear"
                )
                upper_interval_bound = np.quantile(
                    upper_values, upper_conformal_quantile, axis=0, method="linear"
                )

            else:
                prediction = self.fold_estimators[0].predict(X_processed)
                lower_interval_bound = prediction[:, lower_idx]
                upper_interval_bound = prediction[:, upper_idx]

            intervals.append(
                ConformalBounds(
                    lower_bounds=lower_interval_bound, upper_bounds=upper_interval_bound
                )
            )

        return intervals

    def calculate_betas(self, X: np.array, y_true: float) -> list[float]:
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
            for fold_scores in self.fold_nonconformity_scores[i]:
                flattened_scores.extend(fold_scores)
            beta = np.mean(np.array(flattened_scores) >= nonconformity)

            betas.append(beta)

        return betas

    def update_alphas(self, new_alphas: List[float]):
        self.updated_alphas = new_alphas.copy()
