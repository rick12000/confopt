import logging
from typing import Dict, Optional, List, Tuple
import copy

import numpy as np
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from sklearn.model_selection import KFold

from confopt.config import ESTIMATOR_REGISTRY, EstimatorConfig
from confopt.quantile_wrappers import BaseSingleFitQuantileEstimator
from confopt.utils import get_tuning_configurations

logger = logging.getLogger(__name__)


def initialize_estimator(
    estimator_architecture: str,
    initialization_params: Dict = None,
    random_state: Optional[int] = None,
):
    """
    Initialize an estimator by creating a deep copy of the default estimator
    and updating it with the provided parameters.
    """
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

    # Create a deep copy of the default estimator
    estimator = copy.deepcopy(estimator_config.estimator_instance)

    # Add random_state if provided and the estimator supports it
    if random_state is not None and hasattr(estimator, "random_state"):
        initialization_params["random_state"] = random_state

    # Apply all parameters
    if initialization_params:
        # Directly set attributes if set_params is not available
        for param_name, param_value in initialization_params.items():
            if hasattr(estimator, param_name):
                setattr(estimator, param_name, param_value)
            else:
                logger.warning(
                    f"Estimator {estimator_architecture} does not have attribute {param_name}"
                )

    return estimator


def average_scores_across_folds(
    scored_configurations: List[List[Tuple[str, float]]], scores: List[float]
) -> Tuple[List[List[Tuple[str, float]]], List[float]]:
    # Use a list to store aggregated scores and fold counts
    aggregated_scores = []
    fold_counts = []
    aggregated_configurations = []

    for configuration, score in zip(scored_configurations, scores):
        # Check if the configuration already exists in the aggregated_configurations list
        if configuration in aggregated_configurations:
            index = aggregated_configurations.index(configuration)
            aggregated_scores[index] += score
            fold_counts[index] += 1
        else:
            aggregated_configurations.append(configuration)
            aggregated_scores.append(score)
            fold_counts.append(1)

    # Calculate the average scores
    for i in range(len(aggregated_scores)):
        aggregated_scores[i] /= fold_counts[i]

    return aggregated_configurations, aggregated_scores


def cross_validate_configurations(
    configurations: List[Dict],
    estimator_config: EstimatorConfig,
    X: np.array,
    y: np.array,
    k_fold_splits: int = 3,
    quantiles: Optional[List[float]] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[Dict], List[float]]:
    scored_configurations, scores = [], []
    kf = KFold(n_splits=k_fold_splits, random_state=random_state, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index, :], X[test_index, :]
        Y_train, Y_val = y[train_index], y[test_index]

        for configuration in configurations:
            logger.debug(
                f"Evaluating search model parameter configuration: {configuration}"
            )

            # Initialize the estimator with the configuration
            model = initialize_estimator(
                estimator_architecture=estimator_config.estimator_name,
                initialization_params=configuration,
                random_state=random_state,
            )

            try:
                is_quantile_model = estimator_config.is_quantile_estimator()
                # For multi-fit quantile estimators, pass quantiles to fit
                if is_quantile_model:
                    model.fit(X_train, Y_train, quantiles=quantiles)
                else:
                    model.fit(X_train, Y_train)

                # Evaluate the model
                if is_quantile_model:
                    # Then evaluate on pinball loss:
                    prediction = model.predict(X_val)
                    lo_y_pred = prediction[:, 0]
                    hi_y_pred = prediction[:, 1]
                    lo_score = mean_pinball_loss(Y_val, lo_y_pred, alpha=quantiles[0])
                    hi_score = mean_pinball_loss(Y_val, hi_y_pred, alpha=quantiles[1])
                    score = (lo_score + hi_score) / 2
                elif isinstance(model, BaseSingleFitQuantileEstimator):
                    prediction = model.predict(X_val, quantiles=quantiles)
                    scores_list = []
                    for i, quantile in enumerate(quantiles):
                        y_pred = prediction[:, i]
                        quantile_score = mean_pinball_loss(
                            Y_val, y_pred, alpha=quantile
                        )
                        scores_list.append(quantile_score)
                    score = sum(scores_list) / len(scores_list)
                else:
                    # Then evaluate on MSE:
                    y_pred = model.predict(X=X_val)
                    score = mean_squared_error(Y_val, y_pred)

                scored_configurations.append(configuration)
                scores.append(score)

            except Exception as e:
                logger.warning(
                    "Scoring failed and result was not appended. "
                    f"Caught exception: {e}"
                )
                continue

    cross_fold_scored_configurations, cross_fold_scores = average_scores_across_folds(
        scored_configurations=scored_configurations, scores=scores
    )

    return cross_fold_scored_configurations, cross_fold_scores


def tune(
    X: np.array,
    y: np.array,
    estimator_architecture: str,
    n_searches: int,
    k_fold_splits: int = 3,
    quantiles: Optional[List[float]] = None,
    random_state: Optional[int] = None,
) -> Dict:
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]
    # Generate configurations using the tuning space
    tuning_configurations = get_tuning_configurations(
        parameter_grid=estimator_config.estimator_parameter_space,
        n_configurations=n_searches,
        random_state=random_state,
    )

    scored_configurations, scores = cross_validate_configurations(
        configurations=tuning_configurations,
        estimator_config=estimator_config,
        X=X,
        y=y,
        k_fold_splits=k_fold_splits,
        quantiles=quantiles,
        random_state=random_state,
    )

    best_configuration = scored_configurations[scores.index(min(scores))]
    return best_configuration
