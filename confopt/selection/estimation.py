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
from confopt.utils.encoding import get_tuning_configurations

logger = logging.getLogger(__name__)


def initialize_estimator(
    estimator_architecture: str,
    initialization_params: Dict = None,
    random_state: Optional[int] = None,
):
    """Initialize an estimator with given parameters or default parameters."""
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
    # TODO: Not the nicest way to do this
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
        """Create fold indices based on split type."""
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
        train_split: float = 0.8,
        split_type: Literal["k_fold", "ordinal_split"] = "k_fold",
    ) -> Tuple[List[Dict], List[float]]:
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
        raise NotImplementedError("Subclasses must implement _fit_model")

    def _evaluate_model(self, model, X_val: np.array, Y_val: np.array) -> float:
        raise NotImplementedError("Subclasses must implement _evaluate_model")


class PointTuner(RandomTuner):
    def _fit_model(
        self, model: BaseEstimator, X_train: np.array, Y_train: np.array
    ) -> None:
        model.fit(X_train, Y_train)

    def _evaluate_model(self, model: Any, X_val: np.array, Y_val: np.array) -> float:
        y_pred = model.predict(X=X_val)
        return mean_squared_error(Y_val, y_pred)


class QuantileTuner(RandomTuner):
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
        prediction = model.predict(X_val)
        scores_list = []
        for i, quantile in enumerate(self.quantiles):
            y_pred = prediction[:, i]
            quantile_score = mean_pinball_loss(Y_val, y_pred, alpha=quantile)
            scores_list.append(quantile_score)
        return sum(scores_list) / len(scores_list)
