import logging
from typing import List, Optional
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from confopt.quantile_wrappers import (
    BaseSingleFitQuantileEstimator,
    BaseQuantileEstimator,
)

logger = logging.getLogger(__name__)


class BaseEnsembleEstimator:
    """
    Base class for ensembling estimators.

    This abstract class provides the foundation for creating ensemble estimators
    that combine predictions from multiple models with weighted averaging based
    on cross-validation performance.
    """

    def __init__(
        self,
        estimators: List[BaseEstimator] = None,
        cv: int = 3,
        weighting_strategy: str = "inverse_error",
        random_state: Optional[int] = None,
    ):
        """
        Initialize the base ensemble estimator.

        Parameters
        ----------
        estimators : list of estimator instances, optional
            List of pre-initialized estimators to include in the ensemble.
        cv : int, default=3
            Number of cross-validation folds for computing weights.
        weighting_strategy : str, default="inverse_error"
            Strategy for computing weights:
            - "inverse_error": weights are inverse of CV errors
            - "uniform": equal weights for all estimators
            - "rank": weights based on rank of estimators (best gets highest weight)
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.estimators = estimators if estimators is not None else []
        self.cv = cv
        self.weighting_strategy = weighting_strategy
        self.random_state = random_state
        self.weights = None
        self.fitted = False

    def add_estimator(self, estimator: BaseEstimator) -> None:
        """
        Add a single estimator to the ensemble.

        Parameters
        ----------
        estimator : estimator instance
            The estimator to add to the ensemble.
        """
        self.estimators.append(estimator)
        self.fitted = False  # Reset fitted status when adding new estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEnsembleEstimator":
        """
        Fit all estimators and compute weights based on CV performance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        if len(self.estimators) == 0:
            raise ValueError("No estimators have been added to the ensemble.")

        # Fit each estimator on the full dataset
        for i, estimator in enumerate(self.estimators):
            logger.info(f"Fitting estimator {i + 1}/{len(self.estimators)}")
            estimator.fit(X, y)

        # Compute weights based on cross-validation performance
        self.weights = self._compute_weights(X, y)
        self.fitted = True
        return self

    def _compute_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute weights for each estimator based on cross-validation performance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        weights : array-like of shape (n_estimators,)
            Weights for each estimator.
        """
        cv_errors = []
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # Calculate cross-validation error for each estimator
        for i, estimator in enumerate(self.estimators):
            fold_errors = []
            logger.info(
                f"Computing CV errors for estimator {i + 1}/{len(self.estimators)}"
            )

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Use deepcopy instead of clone for custom estimators
                est_clone = deepcopy(estimator)
                est_clone.fit(X_train, y_train)

                # Calculate error on validation set (to be implemented in subclasses)
                error = self._calculate_error(est_clone, X_val, y_val)
                fold_errors.append(error)

            # Use mean error across folds
            cv_errors.append(np.mean(fold_errors))

        # Convert errors to weights based on strategy
        if self.weighting_strategy == "uniform":
            weights = np.ones(len(self.estimators))
        elif self.weighting_strategy == "inverse_error":
            # Prevent division by zero
            errors = np.array(cv_errors)
            if np.any(errors == 0):
                errors[errors == 0] = np.min(errors[errors > 0]) / 100
            weights = 1.0 / errors
        elif self.weighting_strategy == "rank":
            # Rank estimators (lower error is better, so we use negative errors for sorting)
            ranks = np.argsort(np.argsort(-np.array(cv_errors)))
            weights = 1.0 / (ranks + 1)  # +1 to avoid division by zero
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

        # Normalize weights
        weights = weights / np.sum(weights)

        return weights

    def _calculate_error(
        self, estimator: BaseEstimator, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Calculate error for an estimator on validation data.
        To be implemented by subclasses.

        Parameters
        ----------
        estimator : estimator instance
            Fitted estimator to evaluate.
        X : array-like
            Validation features.
        y : array-like
            Validation targets.

        Returns
        -------

        error : float
            Error measure.
        """
        raise NotImplementedError("Subclasses must implement _calculate_error method")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ensemble.
        To be implemented by subclasses.

        Parameters
        ----------
        X : array-like
            Features.

        Returns
        -------

        y_pred : array-like
            Predictions.
        """
        raise NotImplementedError("Subclasses must implement predict method")


class PointEnsembleEstimator(BaseEnsembleEstimator):
    """
    Ensemble estimator for point predictions.

    This class combines multiple point estimators, weighting their predictions
    based on cross-validation performance.
    """

    def _calculate_error(
        self, estimator: BaseEstimator, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Calculate mean squared error for point estimators.

        Parameters
        ----------
        estimator : estimator instance
            Fitted estimator to evaluate.
        X : array-like
            Validation features.
        y : array-like
            Validation targets.

        Returns
        -------

        error : float
            Mean squared error.
        """
        y_pred = estimator.predict(X)
        return mean_squared_error(y, y_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using weighted average of estimator predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        Returns
        -------

        y_pred : array-like of shape (n_samples,)
            Weighted average predictions.
        """
        if not self.fitted:
            raise RuntimeError("Ensemble is not fitted. Call fit first.")

        # Get predictions from each estimator
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])

        # Apply weights to predictions
        weighted_predictions = np.tensordot(self.weights, predictions, axes=([0], [0]))

        return weighted_predictions


class SingleFitQuantileEnsembleEstimator(
    BaseEnsembleEstimator, BaseSingleFitQuantileEstimator
):
    """
    Ensemble estimator for single-fit quantile predictions that follows the
    BaseSingleFitQuantileEstimator interface.

    This class combines multiple BaseSingleFitQuantileEstimator instances and weights
    their predictions based on cross-validation performance.
    """

    def __init__(
        self,
        estimators: List[BaseSingleFitQuantileEstimator] = None,
        cv: int = 3,
        weighting_strategy: str = "inverse_error",
        random_state: Optional[int] = None,
    ):
        """
        Initialize the single-fit quantile ensemble estimator.

        Parameters
        ----------
        estimators : list of BaseSingleFitQuantileEstimator instances, optional
            List of pre-initialized quantile estimators to include in the ensemble.
        cv : int, default=3
            Number of cross-validation folds for computing weights.
        weighting_strategy : str, default="inverse_error"
            Strategy for computing weights.
        random_state : int, optional
            Random seed for reproducibility.
        """
        BaseEnsembleEstimator.__init__(
            self,
            estimators=estimators,
            cv=cv,
            weighting_strategy=weighting_strategy,
            random_state=random_state,
        )
        BaseSingleFitQuantileEstimator.__init__(self)

        # Validate that all estimators are BaseSingleFitQuantileEstimator instances
        if estimators is not None:
            for estimator in estimators:
                if not isinstance(estimator, BaseSingleFitQuantileEstimator):
                    raise TypeError(
                        "All estimators must be BaseSingleFitQuantileEstimator instances"
                    )

    def _calculate_error(
        self, estimator: BaseSingleFitQuantileEstimator, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Calculate mean pinball loss across all quantiles.

        Parameters
        ----------
        estimator : BaseSingleFitQuantileEstimator instance
            Fitted estimator to evaluate.
        X : array-like
            Validation features.
        y : array-like
            Validation targets.

        Returns
        -------

        error : float
            Mean pinball loss averaged across all quantiles.
        """
        # For consistency with fit/predict, use a standard set of quantiles for evaluation
        quantiles = [0.1, 0.5, 0.9]  # Example quantiles - could be parameterized
        predictions = estimator.predict(X, quantiles)

        errors = []
        for i, q in enumerate(quantiles):
            q_pred = predictions[:, i]
            q_error = mean_pinball_loss(y, q_pred, alpha=q)
            errors.append(q_error)

        return np.mean(errors)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit all estimators and compute weights based on CV performance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------

        self : object
            Returns self.
        """
        BaseEnsembleEstimator.fit(self, X, y)
        return self

    def _get_submodel_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get aggregated predictions from all estimators in the ensemble.
        For the SingleFitQuantileEnsembleEstimator, we'll use a representative
        set of quantiles for visualization/analysis purposes.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for prediction.

        Returns
        -------

        np.ndarray
            Array of predictions for visualization/analysis.
        """
        # This is a simplified implementation - just return some representative predictions
        # from one of the estimators
        if len(self.estimators) > 0:
            estimator = self.estimators[0]
            return estimator._get_submodel_predictions(X)
        else:
            return np.array([])

    def predict(self, X: np.ndarray, quantiles: List[float]) -> np.ndarray:
        """
        Predict quantiles using weighted average of estimator predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.
        quantiles : list of float
            List of quantiles to predict (values between 0 and 1).

        Returns
        -------

        y_pred : array-like of shape (n_samples, len(quantiles))
            Weighted average quantile predictions.
        """
        if not self.fitted:
            raise RuntimeError("Ensemble is not fitted. Call fit first.")

        # Initialize predictions array
        n_samples = X.shape[0]
        n_quantiles = len(quantiles)
        weighted_predictions = np.zeros((n_samples, n_quantiles))

        for i, estimator in enumerate(self.estimators):
            preds = estimator.predict(X, quantiles)
            weighted_predictions += self.weights[i] * preds

        return weighted_predictions


class MultiFitQuantileEnsembleEstimator(BaseEnsembleEstimator, BaseQuantileEstimator):
    """
    Ensemble estimator for multi-fit quantile predictions that follows the
    BaseQuantileEstimator interface.

    This class combines multiple BaseQuantileEstimator instances and weights
    their predictions based on cross-validation performance.
    """

    def __init__(
        self,
        estimators: List[BaseQuantileEstimator] = None,
        quantiles: List[float] = None,
        cv: int = 3,
        weighting_strategy: str = "inverse_error",
        random_state: Optional[int] = None,
    ):
        """
        Initialize the multi-fit quantile ensemble estimator.

        Parameters
        ----------
        estimators : list of BaseQuantileEstimator instances, optional
            List of pre-initialized quantile estimators to include in the ensemble.
        quantiles : list of float, required
            List of quantiles to predict (values between 0 and 1).
        cv : int, default=3
            Number of cross-validation folds for computing weights.
        weighting_strategy : str, default="inverse_error"
            Strategy for computing weights.
        random_state : int, optional
            Random seed for reproducibility.
        """
        if quantiles is None:
            raise ValueError("quantiles must be provided")

        BaseEnsembleEstimator.__init__(
            self,
            estimators=estimators,
            cv=cv,
            weighting_strategy=weighting_strategy,
            random_state=random_state,
        )

        # Initialize BaseQuantileEstimator with a dummy model (not actually used)
        # since we're overriding the core methods
        BaseQuantileEstimator.__init__(
            self, quantiles=quantiles, model_class=None, model_params={}
        )

        # Validate that all estimators are BaseQuantileEstimator instances
        if estimators is not None:
            for estimator in estimators:
                if not isinstance(estimator, BaseQuantileEstimator):
                    raise TypeError(
                        "All estimators must be BaseQuantileEstimator instances"
                    )

    def _calculate_error(
        self, estimator: BaseQuantileEstimator, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Calculate mean pinball loss across all quantiles.

        Parameters
        ----------
        estimator : BaseQuantileEstimator instance
            Fitted estimator to evaluate.
        X : array-like
            Validation features.
        y : array-like
            Validation targets.

        Returns
        -------

        error : float
            Mean pinball loss averaged across all quantiles.
        """
        predictions = estimator.predict(X)

        errors = []
        for i, q in enumerate(estimator.quantiles):
            q_pred = predictions[:, i]
            q_error = mean_pinball_loss(y, q_pred, alpha=q)
            errors.append(q_error)

        return np.mean(errors)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit all estimators and compute weights based on CV performance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------

        self : object
            Returns self.
        """
        BaseEnsembleEstimator.fit(self, X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict quantiles using weighted average of estimator predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        Returns
        -------

        y_pred : array-like of shape (n_samples, len(self.quantiles))
            Weighted average quantile predictions.
        """
        if not self.fitted:
            raise RuntimeError("Ensemble is not fitted. Call fit first.")

        # Initialize predictions array
        n_samples = X.shape[0]
        n_quantiles = len(self.quantiles)
        weighted_predictions = np.zeros((n_samples, n_quantiles))

        # Check that all estimators have the same quantiles
        for estimator in self.estimators:
            if estimator.quantiles != self.quantiles:
                raise ValueError(
                    f"All estimators must have the same quantiles. Expected {self.quantiles}, "
                    f"got {estimator.quantiles}"
                )

        for i, estimator in enumerate(self.estimators):
            preds = estimator.predict(X)
            weighted_predictions += self.weights[i] * preds

        return weighted_predictions
