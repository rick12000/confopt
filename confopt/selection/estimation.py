import logging
from typing import Dict, Optional, List, Union, Tuple, Any
from copy import deepcopy

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
    initialization_params_copy = deepcopy(initialization_params)
    if initialization_params_copy is not None:
        initialization_params_copy["random_state"] = random_state

    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]
    estimator = deepcopy(estimator_config.estimator_instance)
    if initialization_params_copy:
        for param_name, param_value in initialization_params_copy.items():
            if hasattr(estimator, param_name):
                setattr(estimator, param_name, param_value)
            else:
                logger.warning(
                    f"Estimator {estimator_architecture} does not have attribute {param_name}"
                )
    return estimator


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
        k_fold_splits: int = 3,
        forced_param_configurations: Optional[List[Dict]] = None,
    ) -> Dict:
        estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

        # Handle warm start configurations
        if forced_param_configurations is None:
            forced_param_configurations = []

        # Determine how many random configurations to generate
        n_random_configs = max(0, n_searches - len(forced_param_configurations))

        # If we have more warm start configs than needed, truncate the list
        if len(forced_param_configurations) > n_searches:
            tuning_configurations = forced_param_configurations
        else:
            # Generate random configurations for the remaining slots
            random_configs = get_tuning_configurations(
                parameter_grid=estimator_config.estimator_parameter_space,
                n_configurations=n_random_configs,
                random_state=self.random_state,
            )
            # Combine warm start and random configurations
            tuning_configurations = forced_param_configurations + random_configs

        scored_configurations, scores = self._cross_validate_configurations(
            configurations=tuning_configurations,
            estimator_config=estimator_config,
            X=X,
            y=y,
            k_fold_splits=k_fold_splits,
        )
        best_configuration = scored_configurations[scores.index(min(scores))]
        return best_configuration

    def _cross_validate_configurations(
        self,
        configurations: List[Dict],
        estimator_config: EstimatorConfig,
        X: np.array,
        y: np.array,
        k_fold_splits: int = 3,
    ) -> Tuple[List[Dict], List[float]]:
        scored_configurations, scores = [], []
        kf = KFold(n_splits=k_fold_splits, random_state=self.random_state, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_val = X[train_index, :], X[test_index, :]
            Y_train, Y_val = y[train_index], y[test_index]
            for configuration in configurations:
                model = initialize_estimator(
                    estimator_architecture=estimator_config.estimator_name,
                    initialization_params=configuration,
                    random_state=self.random_state,
                )
                try:
                    self._fit_model(model, X_train, Y_train)
                    score = self._evaluate_model(model, X_val, Y_val)
                    scored_configurations.append(configuration)
                    scores.append(score)
                except Exception as e:
                    logger.warning(
                        "Scoring failed and result was not appended. "
                        f"Caught exception: {e}"
                    )
                    continue
        (
            cross_fold_scored_configurations,
            cross_fold_scores,
        ) = average_scores_across_folds(
            scored_configurations=scored_configurations, scores=scores
        )
        return cross_fold_scored_configurations, cross_fold_scores

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
