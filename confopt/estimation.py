import logging
from typing import Dict, Optional, List, Tuple

import numpy as np
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from sklearn.model_selection import KFold

from confopt.data_classes import CategoricalRange, IntRange, FloatRange

from confopt.config import (
    ESTIMATOR_REGISTRY,
    EstimatorType,
    MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
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
    """
    Tune hyperparameters for an estimator.
    For ensemble estimators, tunes the full ensemble to find optimal component parameters.
    """
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

    # Special handling for ensemble models
    if estimator_config.estimator_type in [
        EstimatorType.ENSEMBLE_POINT,
        EstimatorType.ENSEMBLE_QUANTILE_SINGLE_FIT,
        EstimatorType.ENSEMBLE_QUANTILE_MULTI_FIT,
    ]:
        return tune_ensemble(
            X=X,
            y=y,
            estimator_architecture=estimator_architecture,
            n_searches=n_searches,
            quantiles=quantiles,
            k_fold_splits=k_fold_splits,
            random_state=random_state,
        )

    # Regular tuning for non-ensemble estimators
    tuning_configurations = get_tuning_configurations(
        parameter_grid=ESTIMATOR_REGISTRY[estimator_architecture].tuning_space,
        n_configurations=n_searches,
        random_state=random_state,
    )
    tuning_configurations.append(
        ESTIMATOR_REGISTRY[estimator_architecture].default_config.copy()
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


def tune_ensemble(
    X: np.array,
    y: np.array,
    estimator_architecture: str,
    n_searches: int,
    quantiles: Optional[List[float]] = None,
    k_fold_splits: int = 3,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Tune an ensemble estimator by searching across component parameter combinations.
    """
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]
    component_names = estimator_config.component_estimators

    if not component_names:
        raise ValueError(
            f"No component estimators defined for {estimator_architecture}"
        )

    # Collect parameter spaces for each component
    component_params = {}
    for component_name in component_names:
        component_config = ESTIMATOR_REGISTRY[component_name]
        component_params[component_name] = component_config.tuning_space

    # Ensemble-specific parameters
    ensemble_params = {
        "weighting_strategy": estimator_config.tuning_space.get(
            "weighting_strategy",
            CategoricalRange(
                choices=["inverse_error", "rank", "uniform", "meta_learner"]
            ),
        )
    }

    # Generate combined parameter configurations
    ensemble_configurations = []
    rng = np.random.RandomState(random_state)

    # Add default configuration
    default_config = {"weighting_strategy": "inverse_error", "cv": 3}

    # Add default component parameters
    for component_name in component_names:
        component_defaults = ESTIMATOR_REGISTRY[component_name].default_config
        for param, value in component_defaults.items():
            default_config[f"{component_name}_{param}"] = value

    ensemble_configurations.append(default_config)

    # Generate random configurations
    for _ in range(n_searches):
        config = {
            "weighting_strategy": rng.choice(
                ensemble_params["weighting_strategy"].choices
            ),
            "cv": 3,
        }  # CV is fixed

        # Generate parameters for each component
        for component_name, param_space in component_params.items():
            for param_name, param_range in param_space.items():
                # Sample from the parameter range
                if isinstance(param_range, IntRange):
                    value = rng.randint(
                        param_range.min_value, param_range.max_value + 1
                    )
                elif isinstance(param_range, FloatRange):
                    if param_range.log_scale:
                        log_min = np.log(param_range.min_value)
                        log_max = np.log(param_range.max_value)
                        value = np.exp(rng.uniform(log_min, log_max))
                    else:
                        value = rng.uniform(
                            param_range.min_value, param_range.max_value
                        )
                elif isinstance(param_range, CategoricalRange):
                    value = rng.choice(param_range.choices)
                else:
                    raise ValueError(
                        f"Unknown parameter range type: {type(param_range)}"
                    )

                # Add to config with component name prefix
                config[f"{component_name}_{param_name}"] = value

        ensemble_configurations.append(config)

    # Cross-validate all configurations
    scored_configurations, scores = cross_validate_configurations(
        configurations=ensemble_configurations,
        estimator_architecture=estimator_architecture,
        X=X,
        y=y,
        k_fold_splits=k_fold_splits,
        quantiles=quantiles,
        random_state=random_state,
    )

    best_configuration = scored_configurations[scores.index(min(scores))]
    return best_configuration


def initialize_estimator(
    estimator_architecture: str,
    initialization_params: Dict,
    quantiles: Optional[List[float]] = None,
    random_state: Optional[int] = None,
):
    """
    Initialize an estimator based on its architecture.
    """
    # Get the estimator configuration from the registry
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]
    estimator_class = estimator_config.estimator_class
    estimator_type = estimator_config.estimator_type

    # Make a working copy of params
    params = initialization_params.copy()

    # Handle random state
    if random_state is not None and "random_state" in estimator_config.default_config:
        params["random_state"] = random_state

    # Initialize based on estimator type
    if estimator_type in [EstimatorType.POINT, EstimatorType.SINGLE_FIT_QUANTILE]:
        # For simple estimators, just initialize with the parameters
        return estimator_class(**params)

    elif estimator_type == EstimatorType.MULTI_FIT_QUANTILE:
        # For multi-fit quantile estimators, add quantiles parameter
        if quantiles is None:
            raise ValueError(f"Quantiles must be provided for {estimator_architecture}")
        params["quantiles"] = quantiles
        return estimator_class(**params)

    elif estimator_type in [
        EstimatorType.ENSEMBLE_POINT,
        EstimatorType.ENSEMBLE_QUANTILE_SINGLE_FIT,
        EstimatorType.ENSEMBLE_QUANTILE_MULTI_FIT,
    ]:
        # Extract ensemble-specific parameters
        ensemble_params = {
            "cv": params.pop("cv", 3),  # Default to 3 if not specified
            "weighting_strategy": params.pop("weighting_strategy", "inverse_error"),
            "random_state": random_state,
        }

        # Initialize ensemble
        ensemble = estimator_class(**ensemble_params)

        # Initialize each component with parameters extracted from the combined params
        for component_name in estimator_config.component_estimators:
            comp_params = {}
            prefix = f"{component_name}_"
            prefix_len = len(prefix)

            # Extract parameters for this component
            for key in list(params.keys()):
                if key.startswith(prefix):
                    comp_params[key[prefix_len:]] = params.pop(key)

            # For multi-fit quantile ensemble, pass quantiles to components
            is_quantile_component = ESTIMATOR_REGISTRY[
                component_name
            ].estimator_type in [
                EstimatorType.MULTI_FIT_QUANTILE,
                EstimatorType.ENSEMBLE_QUANTILE_MULTI_FIT,
            ]

            comp_estimator = initialize_estimator(
                estimator_architecture=component_name,
                initialization_params=comp_params,
                quantiles=quantiles if is_quantile_component else None,
                random_state=random_state,
            )

            # Add to ensemble
            ensemble.add_estimator(comp_estimator)

        return ensemble

    else:
        raise ValueError(f"Unknown estimator type for {estimator_architecture}")


def initialize_point_estimator(
    estimator_architecture: str,
    initialization_params: Dict,
    random_state: Optional[int] = None,
):
    """
    Initialize a point estimator.
    Compatibility wrapper for the unified initialize_estimator function.
    """
    return initialize_estimator(
        estimator_architecture=estimator_architecture,
        initialization_params=initialization_params,
        random_state=random_state,
    )


def initialize_quantile_estimator(
    estimator_architecture: str,
    initialization_params: Dict,
    pinball_loss_alpha: List[float],
    random_state: Optional[int] = None,
):
    """
    Initialize a quantile estimator.
    Compatibility wrapper for the unified initialize_estimator function.
    """
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
    """
    Cross validate a specified estimator on a passed X, y dataset.

    Cross validation loops through a list of passed hyperparameter
    configurations for the previously specified estimator and returns
    an average score across folds for each.

    Parameters
    ----------
    configurations :
        List of estimator parameter configurations, where each
        configuration contains all parameter values necessary
        to create an estimator instance.
    estimator_architecture :
        String name for the type of estimator to cross validate.
    X :
        Explanatory variables to train estimator on.
    y :
        Target variable to train estimator on.
    k_fold_splits :
        Number of cross validation data splits.
    quantiles :
        If the estimator to cross validate is a quantile estimator,
        specify the quantiles it should estimate as a list in this
        variable (eg. [0.25, 0.75] will cross validate an estimator
        predicting the 25th and 75th percentiles of the target variable).
    random_state :
        Random generation seed.

    Returns
    -------
    cross_fold_scored_configurations :
        List of cross validated configurations.
    cross_fold_scores :
        List of corresponding cross validation scores (averaged across
        folds).
    """
    scored_configurations, scores = [], []
    kf = KFold(n_splits=k_fold_splits, random_state=random_state, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index, :], X[test_index, :]
        Y_train, Y_val = y[train_index], y[test_index]

        for configuration in configurations:
            logger.debug(
                f"Evaluating search model parameter configuration: {configuration}"
            )

            is_quantile = (
                estimator_architecture in MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES
            )

            if is_quantile:
                if quantiles is None:
                    raise ValueError(
                        "'quantiles' cannot be None if passing a quantile regression estimator."
                    )
                model = initialize_estimator(
                    estimator_architecture=estimator_architecture,
                    initialization_params=configuration,
                    quantiles=quantiles,
                    random_state=random_state,
                )
            else:
                model = initialize_estimator(
                    estimator_architecture=estimator_architecture,
                    initialization_params=configuration,
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
