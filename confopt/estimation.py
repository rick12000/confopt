import logging
from typing import Dict, Optional, List, Tuple
import copy

import numpy as np
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from sklearn.model_selection import KFold

from confopt.config import (
    ESTIMATOR_REGISTRY,
)
from confopt.quantile_wrappers import BaseSingleFitQuantileEstimator
from confopt.utils import get_tuning_configurations

logger = logging.getLogger(__name__)


def tune(
    X: np.array,
    y: np.array,
    estimator_architecture: str,
    n_searches: int,
    quantiles: Optional[List[float]] = None,
    k_fold_splits: int = 3,
    random_state: Optional[int] = None,
) -> Dict:

    # Get tuning configurations based on estimator type
    tuning_configurations = get_tuning_configurations_for_architecture(
        estimator_architecture=estimator_architecture,
        n_searches=n_searches,
        random_state=random_state,
    )

    scored_configurations, scores = cross_validate_configurations(
        configurations=tuning_configurations,
        estimator_architecture=estimator_architecture,
        X=X,
        y=y,
        k_fold_splits=k_fold_splits,
        quantiles=quantiles,
        random_state=random_state,
    )

    best_configuration = scored_configurations[scores.index(min(scores))]
    return best_configuration


def get_tuning_configurations_for_architecture(
    estimator_architecture: str,
    n_searches: int,
    random_state: Optional[int] = None,
) -> List[Dict]:
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

    # Generate configurations using the tuning space
    configurations = get_tuning_configurations(
        parameter_grid=estimator_config.tuning_space,
        n_configurations=n_searches,
        random_state=random_state,
    )

    # Empty dict represents using the default estimator as-is
    configurations.append({})

    return configurations


def initialize_estimator(
    estimator_architecture: str,
    initialization_params: Dict = None,
    quantiles: Optional[List[float]] = None,
    random_state: Optional[int] = None,
):
    """
    Initialize an estimator by creating a deep copy of the default estimator
    and updating it with the provided parameters.
    """
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

    # Create a deep copy of the default estimator
    estimator = copy.deepcopy(estimator_config.default_estimator)

    # Apply any parameter updates
    if initialization_params:
        # For ensemble estimators, apply parameters to the ensemble and components
        if estimator_config.is_ensemble():
            for param_name, param_value in initialization_params.items():
                if param_name.startswith("component_"):
                    # Parse component index and parameter name
                    parts = param_name.split(".")
                    comp_idx = int(parts[0].split("_")[1])
                    comp_param = parts[1]

                    # Set parameter on the specific component
                    if hasattr(estimator.estimators[comp_idx], "set_params"):
                        estimator.estimators[comp_idx].set_params(
                            **{comp_param: param_value}
                        )
                else:
                    # Set parameter on the ensemble itself
                    if hasattr(estimator, "set_params"):
                        estimator.set_params(**{param_name: param_value})
        else:
            # For non-ensemble estimators, set parameters directly
            if hasattr(estimator, "set_params"):
                estimator.set_params(**initialization_params)

    # Handle quantiles for multi-fit quantile estimators
    if estimator_config.needs_multiple_fits() and quantiles is not None:
        if hasattr(estimator, "set_params"):
            estimator.set_params(quantiles=quantiles)

    # Set random state if applicable and provided
    if (
        random_state is not None
        and hasattr(estimator, "set_params")
        and hasattr(estimator, "random_state")
    ):
        estimator.set_params(random_state=random_state)

    return estimator


def initialize_point_estimator(
    estimator_architecture: str,
    initialization_params: Dict = None,
    random_state: Optional[int] = None,
):
    return initialize_estimator(
        estimator_architecture=estimator_architecture,
        initialization_params=initialization_params,
        random_state=random_state,
    )


def initialize_quantile_estimator(
    estimator_architecture: str,
    initialization_params: Dict = None,
    pinball_loss_alpha: List[float] = None,
    random_state: Optional[int] = None,
):
    return initialize_estimator(
        estimator_architecture=estimator_architecture,
        initialization_params=initialization_params,
        quantiles=pinball_loss_alpha,
        random_state=random_state,
    )


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
    estimator_architecture: str,
    X: np.array,
    y: np.array,
    k_fold_splits: int = 3,
    quantiles: Optional[List[float]] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[Dict], List[float]]:
    scored_configurations, scores = [], []
    kf = KFold(n_splits=k_fold_splits, random_state=random_state, shuffle=True)
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index, :], X[test_index, :]
        Y_train, Y_val = y[train_index], y[test_index]

        for configuration in configurations:
            logger.debug(
                f"Evaluating search model parameter configuration: {configuration}"
            )

            is_quantile = estimator_config.needs_multiple_fits()

            # Initialize the estimator with the configuration
            model = initialize_estimator(
                estimator_architecture=estimator_architecture,
                initialization_params=configuration,
                quantiles=quantiles if is_quantile else None,
                random_state=random_state,
            )

            model.fit(X_train, Y_train)

            try:
                if is_quantile:
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
                    "Scoring failed and result was not appended."
                    f"Caught exception: {e}"
                )
                continue

    cross_fold_scored_configurations, cross_fold_scores = average_scores_across_folds(
        scored_configurations=scored_configurations, scores=scores
    )

    return cross_fold_scored_configurations, cross_fold_scores
