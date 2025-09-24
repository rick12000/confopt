"""Hyperparameter tuning framework for quantile and point estimation models.

This module provides automated hyperparameter optimization infrastructure for both
quantile regression and standard point estimation models. It implements random search
with cross-validation, supporting various split strategies and evaluation metrics.
The framework integrates with the estimator registry system for unified model
configuration and supports warm-start optimization with forced parameter configurations.
"""

import logging
from typing import Dict, Optional, List, Union, Tuple, Any, Literal
from copy import deepcopy
import inspect

from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from sklearn.model_selection import KFold

from confopt.selection.estimator_configuration import (
    ESTIMATOR_REGISTRY,
    EstimatorConfig,
)
from confopt.selection.estimators.quantile_estimation import (
    BaseSingleFitQuantileEstimator,
    BaseMultiFitQuantileEstimator,
)
from confopt.selection.estimators.ensembling import QuantileEnsembleEstimator
from confopt.utils.configurations.sampling import get_tuning_configurations

logger = logging.getLogger(__name__)


def initialize_estimator(
    estimator_architecture: str,
    initialization_params: Dict = None,
    random_state: Optional[int] = None,
):
    """Initialize an estimator instance from registry with given configuration.

    Creates estimator instances using configurations from the global estimator registry,
    with support for parameter overrides and ensemble component initialization. Handles
    random state propagation and special processing for ensemble estimators requiring
    fresh sub-estimator instances.

    Args:
        estimator_architecture: Registered estimator name from ESTIMATOR_REGISTRY.
        initialization_params: Parameter overrides for default configuration.
            Missing parameters use registry defaults.
        random_state: Seed for reproducible estimator initialization. Automatically
            propagated to estimators supporting random_state parameter.

    Returns:
        Initialized estimator instance ready for fitting.

    Raises:
        KeyError: If estimator_architecture not found in registry.
        TypeError: If initialization_params contain invalid parameters.
    """
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

    # Start with default parameters
    params = deepcopy(estimator_config.default_params)

    # If additional parameters are provided, update the defaults
    if initialization_params:
        params.update(initialization_params)

    # Check if random_state is a valid parameter for the estimator class
    if random_state is not None:
        estimator_class = estimator_config.estimator_class
        init_signature = inspect.signature(estimator_class.__init__)
        if "random_state" in init_signature.parameters:
            params["random_state"] = random_state

    # Special handling for ensemble estimators
    if (
        estimator_config.is_ensemble_estimator()
        and estimator_config.ensemble_components
    ):
        # For ensemble models, initialize fresh sub-estimators from component configurations
        fresh_estimators = []
        for component in estimator_config.ensemble_components:
            component_class = component["class"]
            component_params = deepcopy(component["params"])

            # Set random state if supported by this component
            if random_state is not None:
                component_init_signature = inspect.signature(component_class.__init__)
                if "random_state" in component_init_signature.parameters:
                    component_params["random_state"] = random_state

            # Create a fresh instance
            fresh_estimator = component_class(**component_params)
            fresh_estimators.append(fresh_estimator)

        # Add the fresh estimators to the parameters
        params["estimators"] = fresh_estimators

    # Create and return the estimator instance
    return estimator_config.estimator_class(**params)


def average_scores_across_folds(
    scored_configurations: List[List[Dict]], scores: List[float]
) -> Tuple[List[Dict], List[float]]:
    """Aggregate cross-validation scores by averaging across identical configurations.

    Combines scores from multiple folds for configurations that appear multiple times,
    computing mean performance across all evaluations. Used internally to consolidate
    cross-validation results before selecting optimal hyperparameters.

    Args:
        scored_configurations: List of parameter dictionaries from cross-validation.
        scores: Corresponding performance scores for each configuration.

    Returns:
        Tuple of (unique_configurations, averaged_scores) with consolidated results.
    """
    aggregated_scores = []
    fold_counts = []
    aggregated_configurations = []
    for configuration, score in zip(scored_configurations, scores):
        if configuration in aggregated_configurations:
            index = aggregated_configurations.index(configuration)
            aggregated_scores[index] += score
            fold_counts[index] += 1
        else:
            aggregated_configurations.append(configuration)
            aggregated_scores.append(score)
            fold_counts.append(1)
    for i in range(len(aggregated_scores)):
        aggregated_scores[i] /= fold_counts[i]
    return aggregated_configurations, aggregated_scores


class RandomTuner:
    """Base class for hyperparameter optimization using random search with cross-validation.

    Implements random hyperparameter search with flexible cross-validation strategies
    for model selection. Supports warm-start configurations, multiple split types,
    and robust error handling during evaluation. Subclasses implement model-specific
    fitting and evaluation logic for different learning tasks.

    The tuning process randomly samples from parameter spaces defined in estimator
    configurations, evaluates each configuration via cross-validation, and returns
    the configuration with optimal performance.

    Args:
        random_state: Seed for reproducible parameter sampling and data splitting.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def tune(
        self,
        X: np.array,
        y: np.array,
        estimator_architecture: str,
        n_searches: int,
        train_split: float = 0.8,
        split_type: Literal["k_fold", "ordinal_split"] = "k_fold",
        forced_param_configurations: Optional[List[Dict]] = None,
    ) -> Dict:
        """Perform hyperparameter optimization via random search with cross-validation.

        Randomly samples parameter configurations from the estimator's parameter space,
        evaluates each via cross-validation, and returns the best-performing configuration.
        Supports warm-start configurations that are evaluated before random sampling.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target values with shape (n_samples,).
            estimator_architecture: Registered estimator name for optimization.
            n_searches: Total number of configurations to evaluate.
            train_split: Fraction of data for training in ordinal splits, or determines
                K-fold count via 1/(1-train_split) for k_fold splits.
            split_type: Cross-validation strategy. "k_fold" for random splits,
                "ordinal_split" for single time-ordered split.
            forced_param_configurations: Pre-specified configurations evaluated first.
                Remaining slots filled with random sampling.

        Returns:
            Best parameter configuration dictionary based on cross-validation performance.
        """
        estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

        # Handle warm start configurations
        forced_param_configurations = forced_param_configurations or []

        # Determine configurations to evaluate
        n_random_configs = max(0, n_searches - len(forced_param_configurations))
        if len(forced_param_configurations) >= n_searches:
            tuning_configurations = forced_param_configurations[:n_searches]
        else:
            # Generate random configurations for the remaining slots
            random_configs = get_tuning_configurations(
                parameter_grid=estimator_config.estimator_parameter_space,
                n_configurations=n_random_configs,
                random_state=self.random_state,
                sampling_method="uniform",
            )
            # Combine warm start and random configurations
            tuning_configurations = forced_param_configurations + random_configs

        logger.info(f"Tuning configurations: {tuning_configurations}")

        scored_configurations, scores = self._score_configurations(
            configurations=tuning_configurations,
            estimator_config=estimator_config,
            X=X,
            y=y,
            train_split=train_split,
            split_type=split_type,
        )

        # Find the configuration with the minimum score
        best_idx = scores.index(min(scores))
        best_configuration = scored_configurations[best_idx]

        logger.info(f"Best configuration: {best_configuration}")
        return best_configuration

    def _create_fold_indices(
        self,
        X: np.array,
        train_split: float,
        split_type: Literal["k_fold", "ordinal_split"],
    ) -> List[Tuple[np.array, np.array]]:
        """Generate cross-validation fold indices based on split strategy.

        Creates train/test index pairs for cross-validation. Supports K-fold random
        splitting and ordinal time-series splits for temporal data.

        Args:
            X: Feature matrix to determine data size.
            train_split: Training fraction for ordinal splits or K-fold determination.
            split_type: Split strategy specification.

        Returns:
            List of (train_indices, test_indices) tuples for cross-validation.
        """
        if split_type == "ordinal_split":
            # Single train-test split
            split_index = int(len(X) * train_split)
            train_indices = np.arange(split_index)
            test_indices = np.arange(split_index, len(X))
            return [(train_indices, test_indices)]
        else:  # "k_fold"
            # Reverse-engineer the number of folds based on train_split
            k_fold_splits = round(1 / (1 - train_split))
            kf = KFold(
                n_splits=k_fold_splits, random_state=self.random_state, shuffle=True
            )
            return list(kf.split(X))

    def _score_configurations(
        self,
        configurations: List[Dict],
        estimator_config: EstimatorConfig,
        X: np.array,
        y: np.array,
        train_split: float = 0.66,
        split_type: Literal["k_fold", "ordinal_split"] = "k_fold",
    ) -> Tuple[List[Dict], List[float]]:
        """Evaluate parameter configurations via cross-validation.

        Fits and evaluates each configuration across all cross-validation folds,
        computing average performance scores. Handles training failures gracefully
        by excluding failed configurations from results.

        Args:
            configurations: List of parameter dictionaries to evaluate.
            estimator_config: Configuration object containing estimator metadata.
            X: Feature matrix for model training and evaluation.
            y: Target values for model training and evaluation.
            train_split: Training data fraction for split generation.
            split_type: Cross-validation split strategy.

        Returns:
            Tuple of (valid_configurations, average_scores) for successful evaluations.
        """
        # Initialize data structures to store results
        config_scores = {i: [] for i in range(len(configurations))}
        fold_indices = self._create_fold_indices(X, train_split, split_type)

        # For each configuration, evaluate across all folds
        for config_idx, configuration in enumerate(configurations):
            for train_index, test_index in fold_indices:
                X_train, X_val = X[train_index, :], X[test_index, :]
                Y_train, Y_val = y[train_index], y[test_index]

                model = initialize_estimator(
                    estimator_architecture=estimator_config.estimator_name,
                    initialization_params=configuration,
                    random_state=self.random_state,
                )

                try:
                    self._fit_model(model, X_train, Y_train)
                    score = self._evaluate_model(model, X_val, Y_val)
                    config_scores[config_idx].append(score)
                except Exception as e:
                    logger.warning(
                        f"Configuration {config_idx} failed on a fold. Error: {e}"
                    )
                    config_scores[config_idx].append(np.nan)

        # Compute average scores for each configuration
        scored_configurations = []
        scores = []
        for config_idx, configuration in enumerate(configurations):
            fold_scores = config_scores[config_idx]
            valid_scores = [s for s in fold_scores if not np.isnan(s)]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                scored_configurations.append(configuration)
                scores.append(avg_score)

        return scored_configurations, scores

    def _fit_model(self, model, X_train: np.array, Y_train: np.array) -> None:
        """Fit estimator to training data.

        Args:
            model: Estimator instance to train.
            X_train: Training feature matrix.
            Y_train: Training target values.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _fit_model")

    def _evaluate_model(self, model, X_val: np.array, Y_val: np.array) -> float:
        """Evaluate fitted model on validation data.

        Args:
            model: Fitted estimator instance.
            X_val: Validation feature matrix.
            Y_val: Validation target values.

        Returns:
            Performance score (lower is better).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _evaluate_model")


class PointTuner(RandomTuner):
    """Hyperparameter tuner for point estimation models using MSE evaluation.

    Specializes RandomTuner for standard regression tasks where models predict
    single-valued outputs. Uses mean squared error as the optimization metric
    for selecting the best hyperparameter configuration.
    """

    def _fit_model(
        self, model: BaseEstimator, X_train: np.array, Y_train: np.array
    ) -> None:
        """Fit point estimation model to training data.

        Args:
            model: Scikit-learn compatible estimator.
            X_train: Training feature matrix.
            Y_train: Training target values.
        """
        model.fit(X_train, Y_train)

    def _evaluate_model(self, model: Any, X_val: np.array, Y_val: np.array) -> float:
        """Evaluate point estimation model using mean squared error.

        Args:
            model: Fitted estimator instance.
            X_val: Validation feature matrix.
            Y_val: Validation target values.

        Returns:
            Mean squared error (lower is better).
        """
        y_pred = model.predict(X=X_val)
        return mean_squared_error(Y_val, y_pred)


class QuantileTuner(RandomTuner):
    """Hyperparameter tuner for quantile regression models using pinball loss evaluation.

    Specializes RandomTuner for quantile regression tasks where models predict
    multiple quantile levels simultaneously. Uses average pinball loss across
    all quantiles as the optimization metric for hyperparameter selection.

    Args:
        quantiles: List of quantile levels to predict (values in [0,1]).
        random_state: Seed for reproducible optimization.
    """

    def __init__(self, quantiles: List[float], random_state: Optional[int] = None):
        super().__init__(random_state)
        self.quantiles = quantiles

    def _fit_model(
        self,
        model: Union[
            QuantileEnsembleEstimator,
            BaseMultiFitQuantileEstimator,
            BaseSingleFitQuantileEstimator,
        ],
        X_train: np.array,
        Y_train: np.array,
    ) -> None:
        """Fit quantile regression model to training data.

        Args:
            model: Quantile regression estimator supporting multi-quantile fitting.
            X_train: Training feature matrix.
            Y_train: Training target values.
        """
        model.fit(X_train, Y_train, quantiles=self.quantiles)

    def _evaluate_model(
        self,
        model: Union[
            QuantileEnsembleEstimator,
            BaseMultiFitQuantileEstimator,
            BaseSingleFitQuantileEstimator,
        ],
        X_val: np.array,
        Y_val: np.array,
    ) -> float:
        """Evaluate quantile regression model using average pinball loss.

        Computes pinball loss for each quantile level and returns the average
        as the overall performance metric for hyperparameter optimization.

        Args:
            model: Fitted quantile regression estimator.
            X_val: Validation feature matrix.
            Y_val: Validation target values.

        Returns:
            Average pinball loss across all quantiles (lower is better).
        """
        prediction = model.predict(X_val)
        scores_list = []
        for i, quantile in enumerate(self.quantiles):
            y_pred = prediction[:, i]
            quantile_score = mean_pinball_loss(Y_val, y_pred, alpha=quantile)
            scores_list.append(quantile_score)
        return sum(scores_list) / len(scores_list)
