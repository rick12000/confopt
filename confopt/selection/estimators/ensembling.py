import logging
from typing import List, Optional, Tuple, Literal, Union
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from confopt.selection.estimators.quantile_estimation import (
    BaseMultiFitQuantileEstimator,
    BaseSingleFitQuantileEstimator,
)
from abc import ABC, abstractmethod
from sklearn.linear_model import Lasso
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Compute the quantile loss (pinball loss) for quantile regression evaluation."""
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


class QuantileLassoMeta:
    """Quantile Lasso meta-learner that optimizes pinball loss with L1 regularization.

    Custom implementation for ensemble meta-learning that directly optimizes
    quantile loss (pinball loss) instead of mean squared error. Uses scipy
    optimization for more robust convergence.

    Args:
        alpha: L1 regularization strength. Higher values promote sparsity.
        quantile: Quantile level in [0, 1] to optimize for.
        max_iter: Maximum iterations for optimization.
        tol: Convergence tolerance for parameter changes.
        positive: If True, constrain weights to be non-negative.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        quantile: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
        positive: bool = True,
    ):
        self.alpha = alpha
        self.quantile = quantile
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.coef_ = None

    def _quantile_loss_objective(
        self, weights: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Compute quantile loss + L1 penalty."""
        y_pred = X @ weights
        errors = y - y_pred
        quantile_loss = np.mean(
            np.maximum(self.quantile * errors, (self.quantile - 1) * errors)
        )
        l1_penalty = self.alpha * np.sum(np.abs(weights))
        return quantile_loss + l1_penalty

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileLassoMeta":
        """Fit quantile lasso using scipy optimization.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target values with shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        n_features = X.shape[1]

        # Initialize with uniform weights
        initial_weights = np.ones(n_features) / n_features

        # Set up constraints
        bounds = [
            (0, None) if self.positive else (None, None) for _ in range(n_features)
        ]

        # Equality constraint: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Optimize
        result = minimize(
            fun=self._quantile_loss_objective,
            x0=initial_weights,
            args=(X, y),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        if result.success:
            self.coef_ = result.x
        else:
            logger.warning("Quantile Lasso optimization failed, using uniform weights")
            self.coef_ = np.ones(n_features) / n_features

        # Ensure weights are normalized and non-negative if required
        if self.positive:
            self.coef_ = np.maximum(self.coef_, 0)

        if np.sum(self.coef_) > 0:
            self.coef_ = self.coef_ / np.sum(self.coef_)
        else:
            self.coef_ = np.ones(n_features) / n_features

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using fitted coefficients.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Predictions with shape (n_samples,).
        """
        if self.coef_ is None:
            raise ValueError("Must call fit before predict")
        return X @ self.coef_


class BaseEnsembleEstimator(ABC):
    """Abstract base class for ensemble estimators."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
        """Fit the ensemble to training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the fitted ensemble."""


class QuantileEnsembleEstimator(BaseEnsembleEstimator):
    """Ensemble estimator for quantile regression combining multiple quantile predictors.

    Implements ensemble methods that combine predictions from multiple quantile estimators
    to improve uncertainty quantification and prediction accuracy. Uses separate weights
    for each quantile level, allowing different estimators to specialize in different
    quantile regions. Supports both uniform weighting and linear stacking strategies
    with cross-validation for optimal weight computation.

    Weighting Strategies:
        - Uniform: Equal weights for all base estimators, providing simple averaging
          that reduces variance through ensemble diversity without optimization overhead.
        - Linear Stack: Quantile Lasso-based weight optimization using cross-validation to
          minimize quantile loss (pinball loss). Automatically selects the best-performing
          estimators and handles multicollinearity through L1 regularization, with separate
          quantile-specific optimization for each quantile level.

    Args:
        estimators: List of quantile estimators to combine. Must be instances of
            BaseMultiFitQuantileEstimator or BaseSingleFitQuantileEstimator. Requires
            at least 2 estimators for meaningful ensemble benefits.
        cv: Number of cross-validation folds for weight computation in linear stacking.
            Higher values provide more robust weight estimates but increase computation.
            Typical range: 3-10 folds.
        weighting_strategy: Strategy for combining base estimator predictions.
            "uniform" uses equal weights, "linear_stack" optimizes weights via quantile Lasso.
        random_state: Seed for reproducible cross-validation splits and quantile Lasso fitting.
            Ensures deterministic ensemble behavior across runs.
        alpha: L1 regularization strength for quantile Lasso weight optimization. Higher values
            increase sparsity in ensemble weights. Range: [0.0, 1.0] with 0.0 being
            unregularized and higher values promoting sparser solutions.

    Attributes:
        quantiles: List of quantile levels fitted during training.
        quantile_weights: Learned weights for combining base estimator predictions.
            Shape (n_quantiles, n_estimators) with separate weights per quantile level.
        stacker: Fitted quantile Lasso models used for linear stacking weight computation.

    Raises:
        ValueError: If fewer than 2 estimators provided or invalid parameter values.

    Examples:
        Basic uniform ensemble:
        >>> estimators = [QuantileGBM(), QuantileForest(), QuantileKNN()]
        >>> ensemble = QuantileEnsembleEstimator(estimators)
        >>> ensemble.fit(X_train, y_train, quantiles=[0.1, 0.5, 0.9])
        >>> predictions = ensemble.predict(X_test)

        Linear stacking with regularization:
        >>> ensemble = QuantileEnsembleEstimator(
        ...     estimators, weighting_strategy="linear_stack", alpha=0.01
        ... )
        >>> ensemble.fit(X_train, y_train, quantiles=np.linspace(0.05, 0.95, 19))
    """

    def __init__(
        self,
        estimators: List[
            Union[BaseMultiFitQuantileEstimator, BaseSingleFitQuantileEstimator]
        ],
        cv: int = 5,
        weighting_strategy: Literal["uniform", "linear_stack"] = "uniform",
        random_state: Optional[int] = None,
        alpha: float = 0.0,
    ):
        if len(estimators) < 2:
            raise ValueError("At least 2 estimators required for ensemble")

        self.estimators = estimators
        self.cv = cv
        self.weighting_strategy = weighting_strategy
        self.random_state = random_state
        self.alpha = alpha

        self.quantiles = None
        self.quantile_weights = None
        self.stacker = None

    def _get_stacking_training_data(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate cross-validation training data for linear stacking weight optimization.

        Creates validation predictions using k-fold cross-validation to avoid overfitting
        in weight computation. Each base estimator is trained on k-1 folds and predicts
        on the held-out fold, generating unbiased predictions for Lasso weight fitting.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels to fit models for.

        Returns:
            Tuple containing:
                - val_indices: Validation sample indices with shape (n_validation_samples,).
                - val_targets: Validation targets with shape (n_validation_samples,).
                - val_predictions: Validation predictions with shape
                  (n_validation_samples, n_estimators * n_quantiles).
        """
        cv_strategy = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        val_indices = []
        val_targets = []
        val_predictions = []

        for train_idx, val_idx in cv_strategy.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_predictions = []

            for estimator in self.estimators:
                estimator_copy = deepcopy(estimator)
                estimator_copy.fit(X_train_fold, y_train_fold, quantiles)
                pred = estimator_copy.predict(X_val_fold)
                fold_predictions.append(pred)

            fold_predictions_reshaped = []
            for pred in fold_predictions:
                fold_predictions_reshaped.append(pred)
            fold_predictions = np.concatenate(fold_predictions_reshaped, axis=1)

            val_indices.extend(val_idx)
            val_targets.extend(y_val_fold)
            val_predictions.append(fold_predictions)

        val_indices = np.array(val_indices)
        val_targets = np.array(val_targets)
        val_predictions = np.vstack(val_predictions)

        return val_indices, val_targets, val_predictions

    def _compute_linear_stack_weights(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float]
    ) -> np.ndarray:
        """Compute optimal ensemble weights using quantile Lasso regression on validation predictions.

        Implements linear stacking by fitting separate quantile Lasso regression models for each
        quantile level to minimize quantile loss (pinball loss) on cross-validation predictions.
        L1 regularization promotes sparse solutions, automatically selecting the most
        relevant base estimators while handling multicollinearity. Uses custom quantile Lasso
        that optimizes pinball loss instead of mean squared error.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels for weight optimization.

        Returns:
            Optimal ensemble weights with shape (n_quantiles, n_estimators).
        """
        val_indices, val_targets, val_predictions = self._get_stacking_training_data(
            X, y, quantiles
        )

        sorted_indices = np.argsort(val_indices)
        val_predictions_sorted = val_predictions[sorted_indices]
        val_targets_sorted = val_targets[sorted_indices]

        n_estimators = len(self.estimators)
        n_quantiles = len(quantiles)

        weights_per_quantile = []

        for q_idx in range(n_quantiles):
            quantile_predictions = []
            for est_idx in range(n_estimators):
                col_idx = est_idx * n_quantiles + q_idx
                quantile_predictions.append(val_predictions_sorted[:, col_idx])

            quantile_pred_matrix = np.column_stack(quantile_predictions)

            quantile_stacker = QuantileLassoMeta(
                alpha=self.alpha, quantile=quantiles[q_idx], positive=True
            )
            quantile_stacker.fit(quantile_pred_matrix, val_targets_sorted)
            quantile_weights = quantile_stacker.coef_

            if np.sum(quantile_weights) == 0:
                logger.warning(
                    f"All QuantileLasso weights are zero for quantile {q_idx}, falling back to uniform weighting"
                )
                quantile_weights = np.ones(len(self.estimators))

            quantile_weights = quantile_weights / np.sum(quantile_weights)
            weights_per_quantile.append(quantile_weights)

        return np.array(weights_per_quantile)

    def _compute_quantile_weights(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float]
    ) -> np.ndarray:
        """Compute ensemble weights based on the specified weighting strategy.

        Dispatches to the appropriate weight computation method based on the weighting_strategy
        parameter. Supports uniform weighting for simple averaging and linear stacking for
        optimized weight computation via Lasso regression.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels for weight computation.

        Returns:
            Ensemble weights with shape (n_quantiles, n_estimators).

        Raises:
            ValueError: If unknown weighting strategy specified.
        """
        if self.weighting_strategy == "uniform":
            n_estimators = len(self.estimators)
            n_quantiles = len(quantiles)
            return np.ones((n_quantiles, n_estimators)) / n_estimators
        elif self.weighting_strategy == "linear_stack":
            return self._compute_linear_stack_weights(X, y, quantiles)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

    def fit(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float]
    ) -> "QuantileEnsembleEstimator":
        """Fit the quantile ensemble to training data.

        Trains all base estimators on the provided data and computes separate ensemble
        weights for each quantile level according to the specified weighting strategy.
        For linear stacking, performs cross-validation to generate unbiased validation
        predictions for weight optimization.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels in [0, 1] to fit models for.

        Returns:
            Self for method chaining.
        """
        self.quantiles = quantiles

        for estimator in self.estimators:
            estimator.fit(X, y, quantiles)

        self.quantile_weights = self._compute_quantile_weights(X, y, quantiles)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble quantile predictions by combining base estimator outputs.

        Combines predictions from all fitted base estimators using quantile-specific
        weights learned during training. Each quantile level uses its own set of weights
        for more flexible combination that allows estimators to specialize in different
        quantile regions.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Ensemble quantile predictions with shape (n_samples, n_quantiles).
            Each column corresponds to one quantile level in the same order as
            specified during fitting.

        Raises:
            ValueError: If called before fitting the ensemble.
        """
        if self.quantiles is None:
            raise ValueError("Must call fit before predict")

        predictions = []
        for estimator in self.estimators:
            pred = estimator.predict(X)
            predictions.append(pred)

        predictions = np.array(
            predictions
        )  # Shape: (n_estimators, n_samples, n_quantiles)
        n_samples = predictions.shape[1]
        n_quantiles = len(self.quantiles)

        ensemble_predictions = np.zeros((n_samples, n_quantiles))
        for q_idx in range(n_quantiles):
            quantile_weights = self.quantile_weights[q_idx]  # Shape: (n_estimators,)
            quantile_preds = predictions[
                :, :, q_idx
            ]  # Shape: (n_estimators, n_samples)
            ensemble_predictions[:, q_idx] = np.dot(quantile_weights, quantile_preds)

        return ensemble_predictions


class PointEnsembleEstimator(BaseEnsembleEstimator):
    """Ensemble estimator for point prediction combining multiple regression models.

    Implements ensemble methods that combine predictions from multiple regression estimators
    to improve prediction accuracy through variance reduction. Supports uniform weighting
    for simple averaging and linear stacking with cross-validation for optimal weight
    computation.

    Weighting Strategies:
        - Uniform: Equal weights for all base estimators, providing simple averaging
          that reduces variance through model diversity without optimization overhead.
        - Linear Stack: Lasso-based weight optimization using cross-validation to
          minimize mean squared error. Automatically selects best-performing estimators
          and handles multicollinearity through L1 regularization.

    Args:
        estimators: List of regression estimators to combine. Must be scikit-learn
            compatible estimators with fit/predict methods. Requires at least 2
            estimators for meaningful ensemble benefits.
        cv: Number of cross-validation folds for weight computation in linear stacking.
            Higher values provide more robust weight estimates but increase computation.
            Typical range: 3-10 folds.
        weighting_strategy: Strategy for combining base estimator predictions.
            "uniform" uses equal weights, "linear_stack" optimizes weights via Lasso.
        random_state: Seed for reproducible cross-validation splits and Lasso fitting.
            Ensures deterministic ensemble behavior across runs.
        alpha: L1 regularization strength for Lasso weight optimization. Higher values
            increase sparsity in ensemble weights, promoting simpler combinations.
            Range: [0.0, 1.0] with 0.0 being unregularized.

    Attributes:
        weights: Learned weights for combining base estimator predictions with
            shape (n_estimators,). Weights sum to 1.0 for proper averaging.
        stacker: Fitted Lasso model used for linear stacking weight computation.

    Raises:
        ValueError: If fewer than 2 estimators provided or invalid parameter values.

    Examples:
        Basic uniform ensemble:
        >>> estimators = [RandomForestRegressor(), GradientBoostingRegressor(), SVR()]
        >>> ensemble = PointEnsembleEstimator(estimators)
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)

        Linear stacking with regularization:
        >>> ensemble = PointEnsembleEstimator(
        ...     estimators, weighting_strategy="linear_stack", alpha=0.01
        ... )
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        estimators: List[BaseEstimator],
        cv: int = 5,
        weighting_strategy: Literal["uniform", "linear_stack"] = "uniform",
        random_state: Optional[int] = None,
        alpha: float = 0.0,
    ):
        if len(estimators) < 2:
            raise ValueError("At least 2 estimators required for ensemble")

        self.estimators = estimators
        self.cv = cv
        self.weighting_strategy = weighting_strategy
        self.random_state = random_state
        self.alpha = alpha

        self.weights = None
        self.stacker = None

    def _get_stacking_training_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate cross-validation training data for linear stacking weight optimization."""
        cv_strategy = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        val_indices = []
        val_targets = []
        val_predictions = []

        for train_idx, val_idx in cv_strategy.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_predictions = []

            for estimator in self.estimators:
                estimator_copy = deepcopy(estimator)
                estimator_copy.fit(X_train_fold, y_train_fold)
                pred = estimator_copy.predict(X_val_fold)
                fold_predictions.append(pred)

            fold_predictions = np.column_stack(fold_predictions)

            val_indices.extend(val_idx)
            val_targets.extend(y_val_fold)
            val_predictions.append(fold_predictions)

        val_indices = np.array(val_indices)
        val_targets = np.array(val_targets)
        val_predictions = np.vstack(val_predictions)

        return val_indices, val_targets, val_predictions

    def _compute_linear_stack_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute optimal ensemble weights using Lasso regression on validation predictions."""
        val_indices, val_targets, val_predictions = self._get_stacking_training_data(
            X, y
        )

        sorted_indices = np.argsort(val_indices)
        val_predictions_sorted = val_predictions[sorted_indices]
        val_targets_sorted = val_targets[sorted_indices]

        self.stacker = Lasso(alpha=self.alpha, fit_intercept=False, positive=True)
        self.stacker.fit(val_predictions_sorted, val_targets_sorted)
        weights = self.stacker.coef_

        if np.sum(weights) == 0:
            logger.warning(
                "All Lasso weights are zero, falling back to uniform weighting"
            )
            weights = np.ones(len(self.estimators))

        return weights / np.sum(weights)

    def _compute_point_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute ensemble weights based on the specified weighting strategy."""
        if self.weighting_strategy == "uniform":
            n_estimators = len(self.estimators)
            return np.ones(n_estimators) / n_estimators
        elif self.weighting_strategy == "linear_stack":
            return self._compute_linear_stack_weights(X, y)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PointEnsembleEstimator":
        """Fit the point ensemble to training data."""
        for estimator in self.estimators:
            estimator.fit(X, y)

        self.weights = self._compute_point_weights(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble point predictions by combining base estimator outputs."""
        if self.weights is None:
            raise ValueError("Must call fit before predict")

        predictions = []
        for estimator in self.estimators:
            pred = estimator.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        ensemble_predictions = np.dot(self.weights, predictions)

        return ensemble_predictions
