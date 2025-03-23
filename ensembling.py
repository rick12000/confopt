import logging
from typing import List, Optional
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from sklearn.linear_model import LinearRegression
from confopt.quantile_wrappers import (
    BaseSingleFitQuantileEstimator,
    BaseMultiFitQuantileEstimator,
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
        **kwargs,
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
            - "meta_learner": uses linear regression to learn optimal weights from CV predictions
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs :
            Additional parameters, including component-specific parameters in the form
            component_<index>.<param_name>.
        """
        self.estimators = estimators if estimators is not None else []
        self.cv = cv
        self.weighting_strategy = weighting_strategy
        self.random_state = random_state
        self.weights = None
        self.fitted = False
        self.meta_learner = None

        # Apply any component-specific parameters from kwargs
        if kwargs and self.estimators:
            self.set_params(**kwargs)

    def add_estimator(self, estimator: BaseEstimator, **params) -> None:
        """
        Add a single estimator to the ensemble.

        Parameters
        ----------
        estimator : estimator instance
            The estimator to add to the ensemble.
        **params : dict
            Additional parameters to set on the estimator.
        """
        if params and hasattr(estimator, "set_params"):
            estimator.set_params(**params)

        self.estimators.append(estimator)
        self.fitted = False  # Reset fitted status when adding new estimator

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Supports component-specific parameter setting using the format:
        component_<index>.<param_name>

        Parameters
        ----------
        **params : dict
            Estimator parameters, including component parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        component_params = {}
        ensemble_params = {}

        # Separate ensemble parameters from component parameters
        for key, value in params.items():
            if key.startswith("component_"):
                # Parse component index and parameter name
                try:
                    parts = key.split(".")
                    if len(parts) != 2:
                        raise ValueError(f"Invalid component parameter format: {key}")

                    comp_idx_str = parts[0].split("_")[1]
                    if not comp_idx_str.isdigit():
                        raise ValueError(
                            f"Component index must be a number: {comp_idx_str}"
                        )

                    comp_idx = int(comp_idx_str)
                    comp_param = parts[1]

                    if comp_idx not in component_params:
                        component_params[comp_idx] = {}
                    component_params[comp_idx][comp_param] = value
                except (IndexError, ValueError) as e:
                    logger.warning(f"Skipping invalid component parameter {key}: {e}")
            else:
                ensemble_params[key] = value

        # Set parameters on the ensemble itself
        for key, value in ensemble_params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key} for {self}")
            setattr(self, key, value)

        # Set parameters on components
        for comp_idx, params in component_params.items():
            if comp_idx >= len(self.estimators):
                logger.warning(
                    f"Component index {comp_idx} out of range (0 - {len(self.estimators) - 1}), skipping"
                )
                continue

            if hasattr(self.estimators[comp_idx], "set_params"):
                self.estimators[comp_idx].set_params(**params)
            else:
                logger.warning(f"Component {comp_idx} does not support set_params")

        # Reset fitted status when parameters change
        self.fitted = False
        return self

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            "cv": self.cv,
            "weighting_strategy": self.weighting_strategy,
            "random_state": self.random_state,
        }

        # Add component parameters if deep=True
        if deep:
            for i, estimator in enumerate(self.estimators):
                if hasattr(estimator, "get_params"):
                    comp_params = estimator.get_params(deep=True)
                    for param_name, param_value in comp_params.items():
                        params[f"component_{i}.{param_name}"] = param_value

        return params

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEnsembleEstimator":
        """
        Base fit method for regular estimators. Quantile-based ensemble classes
        should override this method to include quantile parameters.

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
        Base compute_weights method for regular estimators. Quantile-based ensemble classes
        should override this method to include quantile parameters.

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

        # For meta_learner strategy, we need to collect predictions on validation folds
        if self.weighting_strategy == "meta_learner":
            all_val_indices = np.array([], dtype=int)
            all_val_predictions = np.zeros((len(y), len(self.estimators)))
            all_val_targets = np.array([])

        # Calculate cross-validation error for each estimator
        for i, estimator in enumerate(self.estimators):
            fold_errors = []
            logger.info(
                f"Computing CV errors for estimator {i + 1}/{len(self.estimators)}"
            )

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Use deepcopy instead of clone for custom estimators
                est_clone = deepcopy(estimator)
                est_clone.fit(X_train, y_train)

                # Calculate error on validation set (to be implemented in subclasses)
                error = self._calculate_error(est_clone, X_val, y_val)
                fold_errors.append(error)

                # For meta_learner, collect validation predictions
                if self.weighting_strategy == "meta_learner":
                    val_preds = est_clone.predict(X_val).reshape(-1)

                    # For the first estimator in each fold, store the validation indices and targets
                    if i == 0:
                        if fold_idx == 0:
                            all_val_indices = val_idx
                            all_val_targets = y_val
                        else:
                            all_val_indices = np.concatenate([all_val_indices, val_idx])
                            all_val_targets = np.concatenate([all_val_targets, y_val])

                    # Store predictions for this estimator
                    all_val_predictions[val_idx, i] = val_preds

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
        elif self.weighting_strategy == "meta_learner":
            # Sort predictions by the original indices to align with targets
            sorted_indices = np.argsort(all_val_indices)
            sorted_predictions = all_val_predictions[all_val_indices[sorted_indices]]
            sorted_targets = all_val_targets[sorted_indices]

            # Fit linear regression to learn optimal weights
            self.meta_learner = LinearRegression(fit_intercept=False, positive=True)
            self.meta_learner.fit(sorted_predictions, sorted_targets)
            weights = self.meta_learner.coef_

            # If any weights are negative (shouldn't happen with positive=True), set to small positive value
            weights = np.maximum(weights, 1e-6)
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

        For meta_learner strategy, this method continues to use the learned weights
        but can also apply the linear regression directly.

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

        if self.weighting_strategy == "meta_learner" and self.meta_learner is not None:
            # Transpose predictions to shape (n_samples, n_estimators)
            predictions = predictions.T
            # Use meta_learner for prediction
            return self.meta_learner.predict(predictions)
        else:
            # Apply weights to predictions using traditional method
            weighted_predictions = np.tensordot(
                self.weights, predictions, axes=([0], [0])
            )
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

    def fit(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float] = None
    ) -> "SingleFitQuantileEnsembleEstimator":
        """
        Fit the single-fit quantile ensemble estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        quantiles : list of float, optional
            List of quantiles to predict (values between 0 and 1).
            Must be provided here.

        Returns
        -------
        self : object
            Returns self.
        """
        if len(self.estimators) == 0:
            raise ValueError("No estimators have been added to the ensemble.")

        # Validate and store quantiles
        self.quantiles = quantiles
        if self.quantiles is None or len(self.quantiles) == 0:
            raise ValueError("Quantiles must be provided in fit method")

        if not all(0 <= q <= 1 for q in self.quantiles):
            raise ValueError("All quantiles must be between 0 and 1")

        # Fit each estimator on the full dataset with the quantiles
        for i, estimator in enumerate(self.estimators):
            logger.info(f"Fitting estimator {i + 1}/{len(self.estimators)}")
            estimator.fit(X, y, quantiles=self.quantiles)

        # Compute weights based on cross-validation performance
        self.weights = self._compute_weights(X, y)
        self.fitted = True
        return self

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
        # Predict all quantiles
        y_pred = estimator.predict(X)

        # Calculate pinball loss for each quantile separately
        errors = []
        for i, q in enumerate(estimator.quantiles):
            q_pred = y_pred[:, i]
            q_error = mean_pinball_loss(y, q_pred, alpha=q)
            errors.append(q_error)

        # Return average error across all quantiles
        return np.mean(errors)

    def _compute_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute weights based on cross-validation performance.

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

        # For meta_learner strategy, we need to collect predictions on validation folds
        if self.weighting_strategy == "meta_learner":
            all_val_indices = np.array([], dtype=int)
            all_val_predictions = np.zeros((len(y), len(self.estimators)))
            all_val_targets = np.array([])

        # Calculate cross-validation error for each estimator
        for i, estimator in enumerate(self.estimators):
            fold_errors = []
            logger.info(
                f"Computing CV errors for estimator {i + 1}/{len(self.estimators)}"
            )

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Use deepcopy instead of clone for custom estimators
                est_clone = deepcopy(estimator)
                # Include quantiles in the fit call
                est_clone.fit(X_train, y_train, quantiles=self.quantiles)

                # Calculate error on validation set
                error = self._calculate_error(est_clone, X_val, y_val)
                fold_errors.append(error)

                # For meta_learner, collect validation predictions
                if self.weighting_strategy == "meta_learner":
                    # Get the median prediction (closest to 0.5)
                    median_idx = min(
                        range(len(self.quantiles)),
                        key=lambda i: abs(self.quantiles[i] - 0.5),
                    )
                    val_preds = est_clone.predict(X_val)[:, median_idx]

                    # For the first estimator in each fold, store the validation indices and targets
                    if i == 0:
                        if fold_idx == 0:
                            all_val_indices = val_idx
                            all_val_targets = y_val
                        else:
                            all_val_indices = np.concatenate([all_val_indices, val_idx])
                            all_val_targets = np.concatenate([all_val_targets, y_val])

                    # Store predictions for this estimator
                    all_val_predictions[val_idx, i] = val_preds

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
        elif self.weighting_strategy == "meta_learner":
            # Sort predictions by the original indices to align with targets
            sorted_indices = np.argsort(all_val_indices)
            sorted_predictions = all_val_predictions[all_val_indices[sorted_indices]]
            sorted_targets = all_val_targets[sorted_indices]

            # Fit linear regression to learn optimal weights
            self.meta_learner = LinearRegression(fit_intercept=False, positive=True)
            self.meta_learner.fit(sorted_predictions, sorted_targets)
            weights = self.meta_learner.coef_

            # If any weights are negative (shouldn't happen with positive=True), set to small positive value
            weights = np.maximum(weights, 1e-6)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

        # Normalize weights
        weights = weights / np.sum(weights)

        return weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict quantiles using weighted average of estimator predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        Returns
        -------
        y_pred : array-like of shape (n_samples, len(quantiles))
            Weighted average quantile predictions.
        """
        if not self.fitted:
            raise RuntimeError("Ensemble is not fitted. Call fit first.")

        # Initialize predictions array
        n_samples = X.shape[0]
        n_quantiles = len(self.quantiles)
        weighted_predictions = np.zeros((n_samples, n_quantiles))

        for i, estimator in enumerate(self.estimators):
            preds = estimator.predict(X)
            weighted_predictions += self.weights[i] * preds

        return weighted_predictions


class MultiFitQuantileEnsembleEstimator(
    BaseEnsembleEstimator, BaseMultiFitQuantileEstimator
):
    """
    Ensemble estimator for multi-fit quantile predictions that follows the
    BaseQuantileEstimator interface.

    This class combines multiple BaseQuantileEstimator instances and weights
    their predictions based on cross-validation performance.
    """

    def __init__(
        self,
        estimators: List[BaseMultiFitQuantileEstimator] = None,
        cv: int = 3,
        weighting_strategy: str = "inverse_error",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the multi-fit quantile ensemble estimator.

        Parameters
        ----------
        estimators : list of BaseQuantileEstimator instances, optional
            List of pre-initialized quantile estimators to include in the ensemble.
        cv : int, default=3
            Number of cross-validation folds for computing weights.
        weighting_strategy : str, default="inverse_error"
            Strategy for computing weights.
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs :
            Additional parameters, including component-specific parameters in the form
            component_<index>.<param_name>.
        """
        self.estimators = estimators if estimators is not None else []
        self.cv = cv
        self.weighting_strategy = weighting_strategy
        self.random_state = random_state
        self.weights = None
        self.fitted = False
        self.quantile_weights = None

        # Apply any component-specific parameters from kwargs
        if kwargs and self.estimators:
            self.set_params(**kwargs)

    def fit(
        self, X: np.ndarray, y: np.ndarray, quantiles: List[float] = None
    ) -> "MultiFitQuantileEnsembleEstimator":
        """
        Fit the multi-fit quantile ensemble estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        quantiles : list of float, optional
            List of quantiles to predict (values between 0 and 1).
            Must be provided here.

        Returns
        -------
        self : object
            Returns self.
        """
        if len(self.estimators) == 0:
            raise ValueError("No estimators have been added to the ensemble.")

        # Validate and store quantiles
        self.quantiles = quantiles
        if self.quantiles is None or len(self.quantiles) == 0:
            raise ValueError("Quantiles must be provided in fit method")

        if not all(0 <= q <= 1 for q in self.quantiles):
            raise ValueError("All quantiles must be between 0 and 1")

        # Fit each estimator on the full dataset with the quantiles
        for i, estimator in enumerate(self.estimators):
            logger.info(f"Fitting estimator {i + 1}/{len(self.estimators)}")
            estimator.fit(X, y, quantiles=self.quantiles)

        # Compute weights based on cross-validation performance
        self.weights = self._compute_weights(X, y)
        self.fitted = True
        return self

    def _calculate_error(
        self, estimator: BaseMultiFitQuantileEstimator, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Calculate mean pinball loss for a specific quantile.

        Parameters
        ----------
        estimator : BaseQuantileEstimator instance
            Fitted estimator to evaluate.
        X : array-like
            Validation features.
        y : array-like
            Validation targets.
        quantile_idx : int
            Index of the quantile to evaluate.

        Returns
        -------
        error : float
            Mean pinball loss for the specified quantile.
        """
        predictions = estimator.predict(X)

        # Calculate error for each quantile separately
        errors = []
        for i, q in enumerate(estimator.quantiles):
            q_pred = predictions[:, i]
            q_error = mean_pinball_loss(y, q_pred, alpha=q)
            errors.append(q_error)

        return errors

    def _compute_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute separate weights for each quantile based on cross-validation performance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        weights : array-like of shape (n_estimators,)
            Combined weights for all estimators (for compatibility with base class).
        """
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # Get number of quantiles from the first estimator
        n_quantiles = len(self.estimators[0].quantiles)

        # Store errors for each quantile separately
        quantile_cv_errors = [[] for _ in range(n_quantiles)]

        # For meta_learner strategy, collect predictions for each quantile
        if self.weighting_strategy == "meta_learner":
            all_val_indices = np.array([], dtype=int)
            all_val_targets = np.array([])
            all_val_predictions_by_quantile = [
                np.zeros((len(y), len(self.estimators))) for _ in range(n_quantiles)
            ]

        # Calculate cross-validation error for each estimator
        for i, estimator in enumerate(self.estimators):
            logger.info(
                f"Computing CV errors for estimator {i + 1}/{len(self.estimators)}"
            )

            # Initialize errors for each fold and quantile
            fold_errors_by_quantile = [[] for _ in range(n_quantiles)]

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Use deepcopy instead of clone for custom estimators
                est_clone = deepcopy(estimator)
                est_clone.fit(X_train, y_train)

                # Calculate error on validation set for each quantile
                errors = self._calculate_error(est_clone, X_val, y_val)

                # Store errors by quantile
                for q_idx, error in enumerate(errors):
                    fold_errors_by_quantile[q_idx].append(error)

                # For meta_learner, collect validation predictions for each quantile
                if self.weighting_strategy == "meta_learner":
                    val_preds = est_clone.predict(X_val)

                    # For the first estimator in each fold, store validation indices and targets
                    if i == 0:
                        if fold_idx == 0:
                            all_val_indices = val_idx
                            all_val_targets = y_val
                        else:
                            all_val_indices = np.concatenate([all_val_indices, val_idx])
                            all_val_targets = np.concatenate([all_val_targets, y_val])

                    # Store predictions for each quantile
                    for q_idx in range(n_quantiles):
                        all_val_predictions_by_quantile[q_idx][val_idx, i] = val_preds[
                            :, q_idx
                        ]

            # Average errors across folds for each quantile
            for q_idx in range(n_quantiles):
                quantile_cv_errors[q_idx].append(
                    np.mean(fold_errors_by_quantile[q_idx])
                )

        # Calculate separate weights for each quantile
        self.quantile_weights = []

        for q_idx in range(n_quantiles):
            q_errors = np.array(quantile_cv_errors[q_idx])

            if self.weighting_strategy == "uniform":
                weights = np.ones(len(self.estimators))
            elif self.weighting_strategy == "inverse_error":
                # Prevent division by zero
                if np.any(q_errors == 0):
                    q_errors[q_errors == 0] = np.min(q_errors[q_errors > 0]) / 100
                weights = 1.0 / q_errors
            elif self.weighting_strategy == "rank":
                # Rank estimators (lower error is better)
                ranks = np.argsort(np.argsort(-np.array(q_errors)))
                weights = 1.0 / (ranks + 1)  # +1 to avoid division by zero
            elif self.weighting_strategy == "meta_learner":
                # Process predictions for this quantile
                sorted_indices = np.argsort(all_val_indices)
                sorted_predictions = all_val_predictions_by_quantile[q_idx][
                    all_val_indices[sorted_indices]
                ]
                sorted_targets = all_val_targets[sorted_indices]

                # Fit a separate meta learner for each quantile
                meta_learner = LinearRegression(fit_intercept=False, positive=True)
                meta_learner.fit(sorted_predictions, sorted_targets)
                weights = meta_learner.coef_
                weights = np.maximum(weights, 1e-6)  # Ensure positive weights
            else:
                raise ValueError(
                    f"Unknown weighting strategy: {self.weighting_strategy}"
                )

            # Normalize weights for this quantile
            weights = weights / np.sum(weights)
            self.quantile_weights.append(weights)

        # Return average weights across quantiles for compatibility with base class
        return np.mean(self.quantile_weights, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict quantiles using weighted average of estimator predictions,
        with separate weights for each quantile.

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

        # Get predictions from all estimators
        n_samples = X.shape[0]
        n_quantiles = len(self.estimators[0].quantiles)

        # Initialize the weighted predictions array
        weighted_predictions = np.zeros((n_samples, n_quantiles))

        # Apply appropriate weights for each quantile
        for q_idx in range(n_quantiles):
            # Initialize predictions for this quantile
            quantile_preds = np.zeros(n_samples)

            # Get predictions from each estimator for this quantile and apply weights
            for i, estimator in enumerate(self.estimators):
                preds = estimator.predict(X)[
                    :, q_idx
                ]  # Get predictions for this quantile
                quantile_preds += self.quantile_weights[q_idx][i] * preds

            # Store the weighted predictions for this quantile
            weighted_predictions[:, q_idx] = quantile_preds

        return weighted_predictions
