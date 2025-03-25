import logging
from typing import List, Optional, Tuple, Literal
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def calculate_quantile_error(
    estimator, X: np.ndarray, y: np.ndarray, quantiles: List[float]
) -> List[float]:
    y_pred = estimator.predict(X)

    errors = []
    for i, q in enumerate(quantiles):
        q_pred = y_pred[:, i]
        errors.append(mean_pinball_loss(y, q_pred, alpha=q))

    return errors


class BaseEnsembleEstimator:
    def __init__(
        self,
        estimators: List[BaseEstimator],
        cv: int = 3,
        weighting_strategy: Literal["uniform", "meta_learner"] = "uniform",
        random_state: Optional[int] = None,
    ):
        if len(estimators) < 2:
            raise ValueError("At least two estimators are required")

        self.estimators = estimators
        self.cv = cv
        self.weighting_strategy = weighting_strategy
        self.random_state = random_state
        self.weights = None
        self.meta_learner = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Base fit method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Base predict method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement predict method")


class PointEnsembleEstimator(BaseEnsembleEstimator):
    def _collect_cv_predictions(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        cv_errors = []
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        need_predictions = self.weighting_strategy == "meta_learner"
        all_val_indices = np.array([], dtype=int) if need_predictions else None
        all_val_predictions = (
            np.zeros((len(y), len(self.estimators))) if need_predictions else None
        )
        all_val_targets = np.array([]) if need_predictions else None

        for i, estimator in enumerate(self.estimators):
            fold_errors = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                est_clone = deepcopy(estimator)
                est_clone.fit(X_train, y_train)

                # Calculate error and store it
                y_pred = est_clone.predict(X_val)
                error = mean_squared_error(y_val, y_pred)
                fold_errors.append(error)

                # For meta_learner, collect predictions
                if need_predictions:
                    if i == 0:
                        if fold_idx == 0:
                            all_val_indices = val_idx
                            all_val_targets = y_val
                        else:
                            all_val_indices = np.concatenate([all_val_indices, val_idx])
                            all_val_targets = np.concatenate([all_val_targets, y_val])

                    all_val_predictions[val_idx, i] = y_pred.reshape(-1)

            cv_errors.append(np.mean(fold_errors))

        return cv_errors, all_val_indices, all_val_targets, all_val_predictions

    def _compute_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        (
            cv_errors,
            all_val_indices,
            all_val_targets,
            all_val_predictions,
        ) = self._collect_cv_predictions(X, y)

        if self.weighting_strategy == "uniform":
            weights = np.ones(len(self.estimators))
        elif self.weighting_strategy == "meta_learner":
            sorted_indices = np.argsort(all_val_indices)
            all_val_predictions = all_val_predictions[all_val_indices[sorted_indices]]
            all_val_targets = all_val_targets[sorted_indices]

            self.meta_learner = LinearRegression(fit_intercept=False, positive=True)
            self.meta_learner.fit(all_val_predictions, all_val_targets)
            weights = self.meta_learner.coef_
            weights = np.maximum(weights, 1e-6)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

        return weights / np.sum(weights)

    def fit(self, X: np.ndarray, y: np.ndarray):
        for estimator in self.estimators:
            estimator.fit(X, y)

        self.weights = self._compute_weights(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])

        if self.weighting_strategy == "meta_learner" and self.meta_learner is not None:
            return self.meta_learner.predict(predictions.T)
        else:
            return np.tensordot(self.weights, predictions, axes=([0], [0]))


class QuantileEnsembleEstimator(BaseEnsembleEstimator):
    def _compute_quantile_weights(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float]
    ) -> List[np.ndarray]:
        """Shared method to compute quantile-specific weights for both quantile estimator types"""
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        n_quantiles = len(quantiles)

        quantile_cv_errors = [[] for _ in range(n_quantiles)]
        all_val_indices = None
        all_val_targets = None

        if self.weighting_strategy == "meta_learner":
            all_val_indices = np.array([], dtype=int)
            all_val_targets = np.array([])
            all_val_predictions_by_quantile = [
                np.zeros((len(y), len(self.estimators))) for _ in range(n_quantiles)
            ]

        for i, estimator in enumerate(self.estimators):
            fold_errors_by_quantile = [[] for _ in range(n_quantiles)]

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                est_clone = deepcopy(estimator)
                est_clone.fit(X_train, y_train, quantiles=quantiles)

                errors = calculate_quantile_error(est_clone, X_val, y_val, quantiles)
                for q_idx, error in enumerate(errors):
                    fold_errors_by_quantile[q_idx].append(error)

                if self.weighting_strategy == "meta_learner":
                    val_preds = est_clone.predict(X_val)

                    if i == 0:
                        if fold_idx == 0:
                            all_val_indices = val_idx
                            all_val_targets = y_val
                        else:
                            all_val_indices = np.concatenate([all_val_indices, val_idx])
                            all_val_targets = np.concatenate([all_val_targets, y_val])

                    for q_idx in range(n_quantiles):
                        all_val_predictions_by_quantile[q_idx][val_idx, i] = val_preds[
                            :, q_idx
                        ]

            for q_idx in range(n_quantiles):
                quantile_cv_errors[q_idx].append(
                    np.mean(fold_errors_by_quantile[q_idx])
                )

        quantile_weights = []

        for q_idx in range(n_quantiles):
            if self.weighting_strategy == "uniform":
                # Skip using q_errors for uniform weights
                weights = np.ones(len(self.estimators))
            elif self.weighting_strategy == "meta_learner":
                sorted_indices = np.argsort(all_val_indices)
                sorted_predictions = all_val_predictions_by_quantile[q_idx][
                    all_val_indices[sorted_indices]
                ]
                sorted_targets = all_val_targets[sorted_indices]

                meta_learner = LinearRegression(fit_intercept=False, positive=True)
                meta_learner.fit(sorted_predictions, sorted_targets)
                weights = meta_learner.coef_
                weights = np.maximum(weights, 1e-6)
            else:
                raise ValueError(
                    f"Unknown weighting strategy: {self.weighting_strategy}"
                )

            quantile_weights.append(weights / np.sum(weights))

        return quantile_weights

    def fit(self, X: np.ndarray, y: np.ndarray, quantiles: List[float]):
        self.quantiles = quantiles
        if not quantiles or not all(0 <= q <= 1 for q in quantiles):
            raise ValueError(
                "Valid quantiles must be provided (values between 0 and 1)"
            )

        for estimator in self.estimators:
            estimator.fit(X, y, quantiles=quantiles)

        # Use quantile-specific weights computation
        self.quantile_weights = self._compute_quantile_weights(X, y, quantiles)
        # Average weights across quantiles for backward compatibility
        self.weights = np.mean(self.quantile_weights, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_quantiles = len(self.quantiles)
        weighted_predictions = np.zeros((n_samples, n_quantiles))

        for q_idx in range(n_quantiles):
            quantile_preds = np.zeros(n_samples)

            for i, estimator in enumerate(self.estimators):
                preds = estimator.predict(X)[:, q_idx]
                quantile_preds += self.quantile_weights[q_idx][i] * preds

            weighted_predictions[:, q_idx] = quantile_preds

        return weighted_predictions
