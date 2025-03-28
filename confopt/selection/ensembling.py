import logging
from typing import List, Optional, Tuple, Literal, Union
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import mean_pinball_loss
from sklearn.linear_model import LinearRegression
from confopt.selection.quantile_estimation import (
    BaseMultiFitQuantileEstimator,
    BaseSingleFitQuantileEstimator,
)
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def calculate_quantile_error(
    y_pred: np.ndarray, y: np.ndarray, quantiles: List[float]
) -> List[float]:
    return [
        mean_pinball_loss(y, y_pred[:, i], alpha=q) for i, q in enumerate(quantiles)
    ]


class BaseEnsembleEstimator(ABC):
    def __init__(
        self,
        estimators: List[
            Union[
                BaseEstimator,
                BaseMultiFitQuantileEstimator,
                BaseSingleFitQuantileEstimator,
            ]
        ],
        cv: int = 3,
        weighting_strategy: Literal["uniform", "linear_stack"] = "linear_stack",
        random_state: Optional[int] = None,
    ):
        if len(estimators) < 2:
            raise ValueError("At least two estimators are required")

        self.estimators = estimators
        self.cv = cv
        self.weighting_strategy = weighting_strategy
        self.random_state = random_state

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class PointEnsembleEstimator(BaseEnsembleEstimator):
    def __init__(
        self,
        estimators: List[BaseEstimator],
        cv: int = 3,
        weighting_strategy: Literal["uniform", "linear_stack"] = "linear_stack",
        random_state: Optional[int] = None,
    ):
        super().__init__(estimators, cv, weighting_strategy, random_state)

    def _get_stacking_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        val_indices = np.array([], dtype=int)
        val_targets = np.array([])
        val_predictions = np.zeros((len(y), len(self.estimators)))

        for i, estimator in enumerate(self.estimators):
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = deepcopy(estimator)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                if i == 0:
                    if fold_idx == 0:
                        val_indices = val_idx
                        val_targets = y_val
                    else:
                        val_indices = np.concatenate([val_indices, val_idx])
                        val_targets = np.concatenate([val_targets, y_val])

                val_predictions[val_idx, i] = y_pred.reshape(-1)

        return val_indices, val_targets, val_predictions

    def _compute_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weighting_strategy == "uniform":
            return np.ones(len(self.estimators)) / len(self.estimators)

        elif self.weighting_strategy == "linear_stack":
            (
                val_indices,
                val_targets,
                val_predictions,
            ) = self._get_stacking_training_data(X, y)
            sorted_indices = np.argsort(val_indices)
            val_predictions = val_predictions[val_indices[sorted_indices]]
            val_targets = val_targets[sorted_indices]

            self.stacker = LinearRegression(fit_intercept=False, positive=True)
            self.stacker.fit(val_predictions, val_targets)
            weights = np.maximum(self.stacker.coef_, 1e-6)

            return weights / np.sum(weights)

        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        for estimator in self.estimators:
            estimator.fit(X, y)

        self.weights = self._compute_weights(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        # TODO: Reintroduce if using more complex stacker architectures
        # and want to predict from predictions rather than apply weights:
        #     return self.stacker.predict(predictions.T)
        return np.tensordot(self.weights, predictions, axes=([0], [0]))


class QuantileEnsembleEstimator(BaseEnsembleEstimator):
    def __init__(
        self,
        estimators: List[
            Union[BaseMultiFitQuantileEstimator, BaseSingleFitQuantileEstimator]
        ],
        cv: int = 3,
        weighting_strategy: Literal["uniform", "linear_stack"] = "linear_stack",
        random_state: Optional[int] = None,
    ):
        super().__init__(estimators, cv, weighting_strategy, random_state)

    def _get_stacking_training_data(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float]
    ) -> Tuple:
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        n_quantiles = len(quantiles)

        val_predictions_by_quantile = [
            np.zeros((len(y), len(self.estimators))) for _ in range(n_quantiles)
        ]
        val_indices = np.array([], dtype=int)
        val_targets = np.array([])

        for i, estimator in enumerate(self.estimators):
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = deepcopy(estimator)
                model.fit(X_train, y_train, quantiles=quantiles)
                y_pred = model.predict(X_val)

                if i == 0:
                    if fold_idx == 0:
                        val_indices = val_idx
                        val_targets = y_val
                    else:
                        val_indices = np.concatenate([val_indices, val_idx])
                        val_targets = np.concatenate([val_targets, y_val])

                for q_idx in range(n_quantiles):
                    val_predictions_by_quantile[q_idx][val_idx, i] = y_pred[:, q_idx]

        return val_indices, val_targets, val_predictions_by_quantile

    def _compute_quantile_weights(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float]
    ) -> List[np.ndarray]:
        if self.weighting_strategy == "uniform":
            return [
                np.ones(len(self.estimators)) / len(self.estimators)
                for _ in range(len(quantiles))
            ]
        elif self.weighting_strategy == "linear_stack":
            (
                val_indices,
                val_targets,
                val_predictions_by_quantile,
            ) = self._get_stacking_training_data(X, y, quantiles)

            weights_per_quantile = []
            sorted_indices = np.argsort(val_indices)
            sorted_targets = val_targets[sorted_indices]

            for q_idx in range(len(quantiles)):
                sorted_predictions = val_predictions_by_quantile[q_idx][
                    val_indices[sorted_indices]
                ]

                meta_learner = LinearRegression(fit_intercept=False, positive=True)
                meta_learner.fit(sorted_predictions, sorted_targets)
                weights = np.maximum(meta_learner.coef_, 1e-6)
                weights_per_quantile.append(weights / np.sum(weights))

            return weights_per_quantile
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

    def fit(self, X: np.ndarray, y: np.ndarray, quantiles: List[float]):
        self.quantiles = quantiles
        if not quantiles or not all(0 <= q <= 1 for q in quantiles):
            raise ValueError(
                "Valid quantiles must be provided (values between 0 and 1)"
            )

        for estimator in self.estimators:
            estimator.fit(X, y, quantiles=quantiles)

        self.quantile_weights = self._compute_quantile_weights(X, y, quantiles)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_quantiles = len(self.quantiles)
        weighted_predictions = np.zeros((n_samples, n_quantiles))
        for q_idx in range(n_quantiles):
            ensembled_preds = np.zeros(n_samples)

            for i, estimator in enumerate(self.estimators):
                preds = estimator.predict(X)[:, q_idx]
                ensembled_preds += self.quantile_weights[q_idx][i] * preds

            weighted_predictions[:, q_idx] = ensembled_preds

        return weighted_predictions
