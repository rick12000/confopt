"""Ensemble estimators for combining multiple point and quantile predictors.

This module provides ensemble methods that combine predictions from multiple base
estimators to improve predictive performance and robustness. Ensembles use cross-
validation based stacking with linear regression meta-learners to optimally weight
individual estimator contributions.
"""

import logging
from typing import List, Optional, Tuple, Literal, Union
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import mean_pinball_loss
from sklearn.linear_model import LinearRegression
from confopt.selection.estimators.quantile_estimation import (
    BaseMultiFitQuantileEstimator,
    BaseSingleFitQuantileEstimator,
)
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def calculate_quantile_error(
    y_pred: np.ndarray, y: np.ndarray, quantiles: List[float]
) -> List[float]:
    """Calculate pinball loss for quantile predictions.

    Computes the pinball loss (quantile loss) for each quantile prediction,
    which is the standard metric for evaluating quantile regression models.

    Args:
        y_pred: Predicted quantile values with shape (n_samples, n_quantiles).
        y: True target values with shape (n_samples,).
        quantiles: Quantile levels corresponding to prediction columns.

    Returns:
        List of pinball losses for each quantile level.
    """
    return [
        mean_pinball_loss(y, y_pred[:, i], alpha=q) for i, q in enumerate(quantiles)
    ]


class BaseEnsembleEstimator(ABC):
    """Abstract base class for ensemble estimators.

    Provides common initialization and interface for combining multiple estimators
    using either uniform averaging or cross-validation based linear stacking. The
    stacking approach trains a linear meta-learner on out-of-fold predictions to
    learn optimal weights for each base estimator.

    Args:
        estimators: List of base estimators to ensemble. Must be scikit-learn
            compatible estimators or quantile estimators with fit/predict methods.
        cv: Number of cross-validation folds for stacking meta-learner training.
        weighting_strategy: Method for combining estimator predictions. "uniform"
            applies equal weights, "linear_stack" learns optimal weights via
            cross-validation and linear regression.
        random_state: Seed for reproducible cross-validation splits.

    Raises:
        ValueError: If fewer than 2 estimators provided.
    """

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
    """Ensemble estimator for point (single-value) predictions.

    Combines multiple regression estimators using either uniform weighting or
    learned weights from cross-validation stacking. The stacking approach trains
    a constrained linear regression meta-learner on out-of-fold predictions to
    determine optimal combination weights.

    Args:
        estimators: List of scikit-learn compatible regression estimators.
        cv: Number of cross-validation folds for weight learning.
        weighting_strategy: Combination method - "uniform" for equal weights,
            "linear_stack" for learned weights via constrained linear regression.
        random_state: Seed for reproducible cross-validation splits.
    """

    def __init__(
        self,
        estimators: List[BaseEstimator],
        cv: int = 3,
        weighting_strategy: Literal["uniform", "linear_stack"] = "linear_stack",
        random_state: Optional[int] = None,
    ):
        super().__init__(estimators, cv, weighting_strategy, random_state)

    def _get_stacking_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Generate out-of-fold predictions for stacking meta-learner training.

        Uses k-fold cross-validation to generate unbiased predictions from each
        base estimator. Each estimator is trained on k-1 folds and predicts on
        the held-out fold, ensuring no data leakage for meta-learner training.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Tuple containing:
                - val_indices: Indices of validation samples.
                - val_targets: True targets for validation samples.
                - val_predictions: Out-of-fold predictions with shape
                  (n_samples, n_estimators).
        """
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
        """Compute ensemble weights based on the selected weighting strategy.

        For uniform weighting, assigns equal weights to all estimators. For linear
        stacking, learns optimal weights by training a constrained linear regression
        on out-of-fold predictions. Weights are constrained to be non-negative and
        sum to 1.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Array of ensemble weights with shape (n_estimators,).

        Raises:
            ValueError: If unknown weighting strategy specified.
        """
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
        """Fit all base estimators and compute ensemble weights.

        Trains each base estimator on the full training data, then computes
        optimal ensemble weights using the specified weighting strategy. For
        linear stacking, this involves cross-validation to generate out-of-fold
        predictions for meta-learner training.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
        """
        for estimator in self.estimators:
            estimator.fit(X, y)

        self.weights = self._compute_weights(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions by weighting individual estimator outputs.

        Combines predictions from all base estimators using the learned or uniform
        weights. Uses tensor dot product for efficient weighted averaging.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Ensemble predictions with shape (n_samples,).
        """
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        # TODO: Reintroduce if using more complex stacker architectures
        # and want to predict from predictions rather than apply weights:
        #     return self.stacker.predict(predictions.T)
        return np.tensordot(self.weights, predictions, axes=([0], [0]))


class QuantileEnsembleEstimator(BaseEnsembleEstimator):
    """Ensemble estimator for quantile regression predictions.

    Combines multiple quantile regression estimators using either uniform weighting
    or learned weights from cross-validation stacking. Supports separate weight
    learning for each quantile level, allowing the ensemble to adapt differently
    across the prediction distribution.

    Args:
        estimators: List of quantile regression estimators (BaseMultiFitQuantileEstimator
            or BaseSingleFitQuantileEstimator instances).
        cv: Number of cross-validation folds for weight learning.
        weighting_strategy: Combination method - "uniform" for equal weights,
            "linear_stack" for quantile-specific learned weights.
        random_state: Seed for reproducible cross-validation splits.
    """

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
        """Generate out-of-fold quantile predictions for stacking meta-learner training.

        Uses k-fold cross-validation to generate unbiased quantile predictions from
        each base estimator. Each estimator is trained on k-1 folds and predicts
        quantiles on the held-out fold, with predictions organized by quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels to predict.

        Returns:
            Tuple containing:
                - val_indices: Indices of validation samples.
                - val_targets: True targets for validation samples.
                - val_predictions_by_quantile: List of prediction arrays, one per
                  quantile level, each with shape (n_samples, n_estimators).
        """
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
        """Compute ensemble weights for each quantile level.

        For uniform weighting, assigns equal weights across all quantiles. For linear
        stacking, learns separate optimal weights for each quantile using constrained
        linear regression on out-of-fold predictions. This allows the ensemble to
        weight estimators differently across the prediction distribution.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels to compute weights for.

        Returns:
            List of weight arrays, one per quantile, each with shape (n_estimators,).

        Raises:
            ValueError: If unknown weighting strategy specified.
        """
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
        """Fit all base quantile estimators and compute quantile-specific weights.

        Trains each base quantile estimator on the full training data for the
        specified quantile levels, then computes optimal ensemble weights using
        the selected weighting strategy. For linear stacking, this involves
        cross-validation to generate out-of-fold predictions for meta-learner training.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels to predict, with values in [0, 1].

        Raises:
            ValueError: If quantiles list is empty or contains invalid values.
        """
        self.quantiles = quantiles
        if not quantiles or not all(0 <= q <= 1 for q in quantiles):
            raise ValueError(
                "Valid quantiles must be provided (values between 0 and 1)"
            )

        for estimator in self.estimators:
            estimator.fit(X, y, quantiles=quantiles)

        self.quantile_weights = self._compute_quantile_weights(X, y, quantiles)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble quantile predictions using quantile-specific weights.

        Combines quantile predictions from all base estimators using the learned
        or uniform weights. Each quantile level uses its own set of weights,
        allowing the ensemble to adapt differently across the prediction distribution.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Ensemble quantile predictions with shape (n_samples, n_quantiles).
        """
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
