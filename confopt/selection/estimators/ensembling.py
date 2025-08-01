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
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
from confopt.selection.estimators.quantile_estimation import (
    BaseMultiFitQuantileEstimator,
    BaseSingleFitQuantileEstimator,
)
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Compute quantile loss for a specific quantile level.

    Args:
        y_true: True target values
        y_pred: Predicted values
        quantile: Quantile level in [0, 1]

    Returns:
        Mean quantile loss
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


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
    stacking approach trains a Lasso meta-learner on out-of-fold predictions to
    learn optimal weights for each base estimator.

    Args:
        estimators: List of base estimators to ensemble. Must be scikit-learn
            compatible estimators or quantile estimators with fit/predict methods.
        cv: Number of cross-validation folds for stacking meta-learner training.
        weighting_strategy: Method for combining estimator predictions. "uniform"
            applies equal weights, "linear_stack" learns optimal weights via
            cross-validation and Lasso regression.
        random_state: Seed for reproducible cross-validation splits.
        alpha: Regularization strength for Lasso regression. Higher values
            produce more sparse solutions, allowing bad estimators to be
            completely turned off with zero weights.

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
        cv: int = 5,
        weighting_strategy: Literal["uniform", "linear_stack"] = "linear_stack",
        random_state: Optional[int] = None,
        alpha: float = 0.01,
    ):
        if len(estimators) < 2:
            raise ValueError("At least two estimators are required")

        self.estimators = estimators
        self.cv = cv
        self.weighting_strategy = weighting_strategy
        self.random_state = random_state
        self.alpha = alpha

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class PointEnsembleEstimator(BaseEnsembleEstimator):
    """Ensemble estimator for point (single-value) predictions.

    Combines multiple regression estimators using either uniform weighting or
    learned weights from cross-validation stacking. The stacking approach trains
    a constrained Lasso meta-learner on out-of-fold predictions to determine
    optimal combination weights, allowing bad estimators to be turned off.

    Args:
        estimators: List of scikit-learn compatible regression estimators.
        cv: Number of cross-validation folds for weight learning.
        weighting_strategy: Combination method - "uniform" for equal weights,
            "linear_stack" for learned weights via constrained Lasso regression.
        random_state: Seed for reproducible cross-validation splits.
        alpha: Regularization strength for Lasso regression.
    """

    def __init__(
        self,
        estimators: List[BaseEstimator],
        cv: int = 5,
        weighting_strategy: Literal["uniform", "linear_stack"] = "linear_stack",
        random_state: Optional[int] = None,
        alpha: float = 0.01,
    ):
        super().__init__(estimators, cv, weighting_strategy, random_state, alpha)

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
        cv_splits = list(kf.split(X))

        val_indices = np.array([], dtype=int)
        val_targets = np.array([])
        val_predictions = np.zeros((len(y), len(self.estimators)))

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if fold_idx == 0:
                val_indices = val_idx
                val_targets = y_val
            else:
                val_indices = np.concatenate([val_indices, val_idx])
                val_targets = np.concatenate([val_targets, y_val])

            for i, estimator in enumerate(self.estimators):
                model = deepcopy(estimator)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                val_predictions[val_idx, i] = y_pred.reshape(-1)

        return val_indices, val_targets, val_predictions

    def _compute_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute ensemble weights based on the selected weighting strategy.

        For uniform weighting, assigns equal weights to all estimators. For linear
        stacking, learns optimal weights by training a constrained Lasso regression
        on out-of-fold predictions. Weights are constrained to be non-negative and
        sum to 1, with Lasso regularization allowing bad estimators to be zeroed out.

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

            self.stacker = Lasso(alpha=self.alpha, fit_intercept=False, positive=True)
            self.stacker.fit(val_predictions, val_targets)
            weights = np.maximum(self.stacker.coef_, 0.0)

            # Handle case where all weights are zero
            if np.sum(weights) == 0:
                logger.warning(
                    "All Lasso weights are zero, falling back to uniform weighting"
                )
                weights = np.ones(len(self.estimators))

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
    across the prediction distribution and turn off bad estimators per quantile.

    Args:
        estimators: List of quantile regression estimators (BaseMultiFitQuantileEstimator
            or BaseSingleFitQuantileEstimator instances).
        cv: Number of cross-validation folds for weight learning.
        weighting_strategy: Combination method - "uniform" for equal weights,
            "joint_shared" for joint optimization with shared weights across quantiles,
            "joint_separate" for joint optimization with separate weights per quantile.
        regularization_target: Regularization target - "uniform" biases toward equal weights,
            "best_component" biases toward the best performing individual estimator.
        random_state: Seed for reproducible cross-validation splits.
        alpha: Regularization strength for optimization.
    """

    def __init__(
        self,
        estimators: List[
            Union[BaseMultiFitQuantileEstimator, BaseSingleFitQuantileEstimator]
        ],
        cv: int = 5,
        weighting_strategy: Literal[
            "uniform", "joint_shared", "joint_separate"
        ] = "joint_shared",
        regularization_target: Literal["uniform", "best_component"] = "uniform",
        random_state: Optional[int] = None,
        alpha: float = 0.001,
    ):
        super().__init__(estimators, cv, weighting_strategy, random_state, alpha)
        self.regularization_target = regularization_target

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
        cv_splits = list(kf.split(X))
        n_quantiles = len(quantiles)

        val_predictions_by_quantile = [
            np.zeros((len(y), len(self.estimators))) for _ in range(n_quantiles)
        ]
        val_indices = np.array([], dtype=int)
        val_targets = np.array([])

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if fold_idx == 0:
                val_indices = val_idx
                val_targets = y_val
            else:
                val_indices = np.concatenate([val_indices, val_idx])
                val_targets = np.concatenate([val_targets, y_val])

            for i, estimator in enumerate(self.estimators):
                model = deepcopy(estimator)
                model.fit(X_train, y_train, quantiles=quantiles)
                y_pred = model.predict(X_val)

                for q_idx in range(n_quantiles):
                    val_predictions_by_quantile[q_idx][val_idx, i] = y_pred[:, q_idx]

        return val_indices, val_targets, val_predictions_by_quantile

    def _optimize_weights(self, objective_func, n_estimators: int) -> np.ndarray:
        """Single solver weight optimization using SLSQP.

        Args:
            objective_func: Objective function to minimize
            n_estimators: Number of estimators

        Returns:
            Optimal weights array
        """
        initial_weights = np.ones(n_estimators) / n_estimators
        bounds = [(0, 1)] * n_estimators
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        try:
            result = minimize(
                objective_func,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

            if result.success:
                weights = np.maximum(result.x, 0.0)
                return weights / np.sum(weights)
            else:
                logger.warning(f"SLSQP optimization failed: {result.message}")
        except Exception as e:
            logger.warning(f"Weight optimization failed: {e}")

        logger.warning("Using uniform weights as fallback")
        return initial_weights

    def _compute_joint_shared_weights(
        self,
        val_predictions_by_quantile: List[np.ndarray],
        val_targets: np.ndarray,
        quantiles: List[float],
    ) -> np.ndarray:
        """Optimize single set of weights jointly across all quantiles.

        Args:
            val_predictions_by_quantile: List of prediction arrays per quantile
            val_targets: True target values
            quantiles: List of quantile levels

        Returns:
            Optimal shared weights array
        """
        n_estimators = val_predictions_by_quantile[0].shape[1]

        # Determine regularization target
        if self.regularization_target == "best_component":
            # Identify best individual estimator based on CV performance
            estimator_losses = []
            for est_idx in range(n_estimators):
                total_loss = 0.0
                for q_idx, quantile in enumerate(quantiles):
                    pred = val_predictions_by_quantile[q_idx][:, est_idx]
                    loss = quantile_loss(val_targets, pred, quantile)
                    total_loss += loss
                estimator_losses.append(total_loss / len(quantiles))

            best_estimator_idx = np.argmin(estimator_losses)
            target_weights = np.zeros(n_estimators)
            target_weights[best_estimator_idx] = 1.0
            logger.info(
                f"Best estimator: {best_estimator_idx} (loss: {estimator_losses[best_estimator_idx]:.4f})"
            )
        else:
            # Uniform regularization target
            target_weights = np.ones(n_estimators) / n_estimators

        def multi_quantile_objective(weights):
            weights = np.maximum(weights, 1e-8)
            weights = weights / np.sum(weights)

            total_loss = 0.0
            for i, quantile in enumerate(quantiles):
                ensemble_pred = np.dot(val_predictions_by_quantile[i], weights)
                loss = quantile_loss(val_targets, ensemble_pred, quantile)
                total_loss += loss

            avg_loss = total_loss / len(quantiles)
            regularization_penalty = self.alpha * np.sum(
                np.abs(weights - target_weights)
            )

            return avg_loss + regularization_penalty

        return self._optimize_weights(multi_quantile_objective, n_estimators)

    def _compute_joint_separate_weights(
        self,
        val_predictions_by_quantile: List[np.ndarray],
        val_targets: np.ndarray,
        quantiles: List[float],
    ) -> List[np.ndarray]:
        """Optimize separate weights for each quantile independently.

        Args:
            val_predictions_by_quantile: List of prediction arrays per quantile
            val_targets: True target values
            quantiles: List of quantile levels

        Returns:
            List of optimal weights arrays, one per quantile
        """
        weights_per_quantile = []

        # For separate weights, determine regularization target once for consistency
        n_estimators = val_predictions_by_quantile[0].shape[1]

        if self.regularization_target == "best_component":
            # Identify best individual estimator based on overall CV performance
            estimator_losses = []
            for est_idx in range(n_estimators):
                total_loss = 0.0
                for q_idx, quantile in enumerate(quantiles):
                    pred = val_predictions_by_quantile[q_idx][:, est_idx]
                    loss = quantile_loss(val_targets, pred, quantile)
                    total_loss += loss
                estimator_losses.append(total_loss / len(quantiles))

            best_estimator_idx = np.argmin(estimator_losses)
            target_weights = np.zeros(n_estimators)
            target_weights[best_estimator_idx] = 1.0
            logger.info(f"Best estimator for separate weights: {best_estimator_idx}")
        else:
            # Uniform regularization target
            target_weights = np.ones(n_estimators) / n_estimators

        for i, quantile in enumerate(quantiles):
            predictions = val_predictions_by_quantile[i]

            def single_quantile_objective(weights):
                weights = np.maximum(weights, 1e-8)
                weights = weights / np.sum(weights)

                ensemble_pred = np.dot(predictions, weights)
                loss = quantile_loss(val_targets, ensemble_pred, quantile)
                regularization_penalty = self.alpha * np.sum(
                    np.abs(weights - target_weights)
                )

                return loss + regularization_penalty

            optimal_weights = self._optimize_weights(
                single_quantile_objective, n_estimators
            )
            weights_per_quantile.append(optimal_weights)

        return weights_per_quantile

    def _compute_quantile_weights(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Compute ensemble weights using the specified strategy.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels to compute weights for.

        Returns:
            For uniform and joint_shared: Single weight array with shape (n_estimators,).
            For joint_separate: List of weight arrays, one per quantile.

        Raises:
            ValueError: If unknown weighting strategy specified.
        """
        if self.weighting_strategy == "uniform":
            return np.ones(len(self.estimators)) / len(self.estimators)

        # Get cross-validation predictions for optimization
        (
            val_indices,
            val_targets,
            val_predictions_by_quantile,
        ) = self._get_stacking_training_data(X, y, quantiles)

        # Sort by validation indices for consistent ordering
        sorted_indices = np.argsort(val_indices)
        sorted_targets = val_targets[sorted_indices]
        sorted_predictions_by_quantile = [
            pred_array[sorted_indices] for pred_array in val_predictions_by_quantile
        ]

        if self.weighting_strategy == "joint_shared":
            return self._compute_joint_shared_weights(
                sorted_predictions_by_quantile, sorted_targets, quantiles
            )
        elif self.weighting_strategy == "joint_separate":
            return self._compute_joint_separate_weights(
                sorted_predictions_by_quantile, sorted_targets, quantiles
            )
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
        """Generate ensemble quantile predictions using learned weights.

        Combines quantile predictions from all base estimators using either shared
        weights (uniform/joint_shared) or separate weights per quantile (joint_separate).

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Ensemble quantile predictions with shape (n_samples, n_quantiles).
        """
        n_samples = X.shape[0]
        n_quantiles = len(self.quantiles)

        # Get predictions from all base estimators
        base_predictions = []
        for estimator in self.estimators:
            base_predictions.append(estimator.predict(X))

        # Stack predictions: [n_estimators, n_samples, n_quantiles]
        base_predictions = np.array(base_predictions)

        # Combine using appropriate weighting scheme
        weighted_predictions = np.zeros((n_samples, n_quantiles))

        if isinstance(self.quantile_weights, np.ndarray):
            # Shared weights across all quantiles (uniform or joint_shared)
            for q_idx in range(n_quantiles):
                for i, weight in enumerate(self.quantile_weights):
                    weighted_predictions[:, q_idx] += (
                        weight * base_predictions[i, :, q_idx]
                    )
        else:
            # Separate weights per quantile (joint_separate)
            for q_idx in range(n_quantiles):
                weights = self.quantile_weights[q_idx]
                for i, weight in enumerate(weights):
                    weighted_predictions[:, q_idx] += (
                        weight * base_predictions[i, :, q_idx]
                    )

        return weighted_predictions
