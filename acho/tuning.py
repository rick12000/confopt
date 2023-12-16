import logging
import random
from copy import deepcopy
from typing import Optional, Dict, Any

import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from acho.config import (
    NON_NORMALIZING_MODEL_TYPES,
    METRIC_PROPORTIONALITY_LOOKUP,
    KNN_NAME,
)
from acho.estimation import (
    QuantileConformalRegression,
    LocallyWeightedConformalRegression,
)
from acho.optimization import derive_optimal_tuning_count, RuntimeTracker
from acho.preprocessing import train_val_split, remove_iqr_outliers
from acho.utils import get_tuning_configurations, tabularize_configurations

logger = logging.getLogger(__name__)


def is_interval_breach(
    performance_lower_bounds: np.array,
    performance_higher_bounds: np.array,
    bound_idx: int,
    realization: float,
) -> bool:
    if (realization > performance_higher_bounds[bound_idx]) or (
        realization < performance_lower_bounds[bound_idx]
    ):
        return True
    else:
        return False


def get_best_configuration_idx(
    performance_bounds: np.array, optimization_direction: str
) -> int:
    performance_lower_bounds, performance_higher_bounds = performance_bounds
    if optimization_direction == "inverse":
        best_idx = np.argmin(performance_lower_bounds)

    elif optimization_direction == "direct":
        best_idx = np.argmax(performance_higher_bounds)
    else:
        raise ValueError(
            f"{optimization_direction} is not a valid loss direction instruction."
        )

    return best_idx


def score_predictions(
    y_obs: np.array, y_pred: np.array, scoring_function: str
) -> float:
    if scoring_function == "accuracy_score":
        score = accuracy_score(y_true=y_obs, y_pred=y_pred)
    elif scoring_function == "log_loss":
        score = log_loss(y_true=y_obs, y_pred=y_pred)
    elif scoring_function == "mean_squared_error":
        score = mean_squared_error(y_true=y_obs, y_pred=y_pred)
    else:
        raise ValueError(f"{scoring_function} is not a recognized scoring function.")

    return score


def process_and_split_estimation_data(
    searched_configurations: np.array,
    searched_performances: np.array,
    train_split: float,
    filter_outliers: Optional[bool] = False,
    outlier_scope: Optional[str] = "top_and_bottom",
    random_state: Optional[int] = None,
):
    X = searched_configurations.copy()
    y = searched_performances.copy()
    logger.debug(f"Minimum accuracy/loss in searcher's sampled data: {y.min()}")
    logger.debug(f"Maximum accuracy/loss in searcher's sampled data: {y.max()}")

    if filter_outliers:
        X, y = remove_iqr_outliers(X=X, y=y, scope=outlier_scope)

    X_train, y_train, X_val, y_val = train_val_split(
        X=X,
        y=y,
        train_split=train_split,
        normalize=False,
        ordinal=False,
        random_state=random_state,
    )

    return X_train, y_train, X_val, y_val


def normalize_estimation_data(
    training_searched_configurations: np.array,
    validation_searched_configurations: np.array,
    searchable_configurations: np.array,
):
    scaler = StandardScaler()
    scaler.fit(training_searched_configurations)
    normalized_searchable_configurations = scaler.transform(searchable_configurations)
    normalized_training_searched_configurations = scaler.transform(
        training_searched_configurations
    )
    normalized_validation_searched_configurations = scaler.transform(
        validation_searched_configurations
    )

    return (
        normalized_training_searched_configurations,
        normalized_validation_searched_configurations,
        normalized_searchable_configurations,
    )


def update_adaptive_interval(
    true_confidence_level: float,
    last_confidence_level: float,
    breach: bool,
    learning_rate: float,
) -> float:
    updated_confidence_level = 1 - (
        (1 - last_confidence_level)
        + learning_rate * ((1 - true_confidence_level) - breach)
    )
    updated_confidence_level = min(max(0.01, updated_confidence_level), 0.99)
    logger.debug(
        f"Updated confidence level of {last_confidence_level} to {updated_confidence_level}."
    )

    return updated_confidence_level


def update_model_parameters(
    model_instance: Any, configuration: Dict, random_state: int = None
):
    updated_model_instance = deepcopy(model_instance)
    for tuning_attr_name, tuning_attr in configuration.items():
        setattr(updated_model_instance, tuning_attr_name, tuning_attr)
    if hasattr(updated_model_instance, "random_state"):
        setattr(updated_model_instance, "random_state", random_state)
    return updated_model_instance


class ConformalSearcher:
    def __init__(
        self,
        model: Any,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        search_space: Dict,
        prediction_type: str,
        custom_loss_function: Optional[str] = None,
    ):

        if (
            hasattr(model, "fit")
            and hasattr(model, "predict")
            and callable(model.fit)
            and callable(model.predict)
        ):
            self.model = model
        else:
            raise ValueError(
                "Model to tune must be wrapped in class with 'fit' and 'predict' methods."
            )

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.search_space = search_space
        self.prediction_type = prediction_type

        self.custom_loss_function = (
            self._set_default_evaluation_metric()
            if custom_loss_function is None
            else custom_loss_function
        )
        self.tuning_configurations = self._get_tuning_configurations()

    def _set_default_evaluation_metric(self) -> str:
        if self.prediction_type == "regression":
            custom_loss_function = "mean_squared_error"
        elif self.prediction_type == "classification":
            custom_loss_function = "accuracy_score"
        else:
            raise ValueError(
                f"Unable to auto-allocate evaluation metric for {self.prediction_type} prediction type."
            )
        return custom_loss_function

    def _get_tuning_configurations(self):
        logger.info("Creating hyperparameter space...")
        tuning_configurations = get_tuning_configurations(
            parameter_grid=self.search_space, n_configurations=1000, random_state=1234
        )
        return tuning_configurations

    def _evaluate_configuration_performance(
        self, configuration: Dict, random_state: Optional[int] = None
    ):
        logger.debug(f"Evaluating model with configuration: {configuration}")

        updated_model = update_model_parameters(
            model_instance=self.model,
            configuration=configuration,
            random_state=random_state,
        )
        updated_model.fit(self.X_train, self.y_train)

        if self.custom_loss_function in ["log_loss"]:
            y_pred = updated_model.predict_proba(self.X_val)
        else:
            y_pred = updated_model.predict(self.X_val)

        final_loss = score_predictions(
            y_obs=self.y_val, y_pred=y_pred, scoring_function=self.custom_loss_function
        )

        return final_loss

    def _random_search(
        self,
        min_training_iterations: int,
        max_runtime: int,
        verbose: bool = True,
        random_state: Optional[int] = None,
    ):
        random.seed(random_state)
        np.random.seed(random_state)

        searched_configurations = []
        searched_performances = []

        skipped_configuration_counter = 0
        runtime_per_search = 0

        shuffled_tuning_configurations = self.tuning_configurations.copy()
        random.seed(random_state)
        random.shuffle(shuffled_tuning_configurations)
        randomly_sampled_configurations = shuffled_tuning_configurations[
            : min(min_training_iterations, len(self.tuning_configurations))
        ]

        model_training_timer = RuntimeTracker()
        model_training_timer.pause_runtime()
        if verbose:
            randomly_sampled_configurations = tqdm(
                randomly_sampled_configurations, desc="Random searches: "
            )
        for config_idx, hyperparameter_configuration in enumerate(
            randomly_sampled_configurations
        ):
            model_training_timer.resume_runtime()
            validation_performance = self._evaluate_configuration_performance(
                configuration=hyperparameter_configuration, random_state=random_state
            )
            model_training_timer.pause_runtime()

            if np.isnan(validation_performance):
                skipped_configuration_counter += 1
                logger.debug(
                    "Obtained non-numerical performance, skipping configuration."
                )
                continue

            searched_configurations.append(hyperparameter_configuration.copy())
            searched_performances.append(validation_performance)

            runtime_per_search = (
                runtime_per_search + model_training_timer.return_runtime()
            ) / (config_idx - skipped_configuration_counter + 1)

            logger.info(f"RS Iteration: {config_idx}")
            logger.info(
                f"RS Iteration's validation performance: {validation_performance}"
            )

            if self.search_timer.return_runtime() > max_runtime:
                raise RuntimeError(
                    "ACHO preliminary random search exceeded total runtime budget. "
                    "Retry with larger runtime budget or set iteration-capped budget instead."
                )

        return searched_configurations, searched_performances, runtime_per_search

    def search(
        self,
        conformal_model_type: str,
        confidence_level: float,
        n_random_searches: int = 20,
        interval_type: str = "quantile_regression",
        conformal_variance_model_type: str = KNN_NAME,
        conformal_retraining_frequency: int = 1,
        enable_adaptive_intervals: bool = True,
        conformal_learning_rate: float = 0.1,
        runtime_budget: int = 600,
        verbose: bool = True,
        random_state: Optional[int] = None,
    ):

        self.random_state = random_state
        self.search_timer = RuntimeTracker()

        (
            self.searched_configurations,
            self.searched_performances,
            runtime_per_search,
        ) = self._random_search(
            min_training_iterations=n_random_searches,
            max_runtime=runtime_budget,
            verbose=verbose,
            random_state=random_state,
        )

        starting_search_estimator_tunings = 30

        best_lw_pe_config = None
        best_lw_de_config = None
        best_lw_ve_config = None

        best_cqr_config = None

        search_idx_range = range(len(self.tuning_configurations) - n_random_searches)
        for config_idx in search_idx_range:
            if verbose:
                print(
                    f"Conformal searches: {config_idx + 1}"
                    + " | "
                    + f"Budget consumed: {int(self.search_timer.return_runtime())}s/{runtime_budget}s",
                    end="\r",
                )
            searchable_configurations = [
                configuration
                for configuration in self.tuning_configurations
                if configuration not in self.searched_configurations
            ]
            tabularized_searchable_configurations = tabularize_configurations(
                configurations=searchable_configurations
            ).to_numpy()
            tabularized_searched_configurations = tabularize_configurations(
                configurations=self.searched_configurations.copy()
            ).to_numpy()

            validation_split = ConformalSearcher._set_conformal_validation_split(
                tabularized_searched_configurations
            )
            remove_outliers = (
                True
                if self.custom_loss_function == "log_loss"
                or self.prediction_type == "regression"
                else False
            )
            outlier_scope = "top_only"
            (
                X_train_conformal,
                y_train_conformal,
                X_val_conformal,
                y_val_conformal,
            ) = process_and_split_estimation_data(
                searched_configurations=tabularized_searched_configurations,
                searched_performances=np.array(self.searched_performances),
                train_split=(1 - validation_split),
                filter_outliers=remove_outliers,
                outlier_scope=outlier_scope,
                random_state=random_state,
            )

            if (
                conformal_model_type.lower() not in NON_NORMALIZING_MODEL_TYPES
                or conformal_variance_model_type.lower()
                not in NON_NORMALIZING_MODEL_TYPES
            ):
                (
                    X_train_conformal,
                    X_val_conformal,
                    tabularized_searchable_configurations,
                ) = normalize_estimation_data(
                    training_searched_configurations=X_train_conformal,
                    validation_searched_configurations=X_val_conformal,
                    searchable_configurations=tabularized_searchable_configurations,
                )

            is_retraining_interval_passed = (
                config_idx % conformal_retraining_frequency == 0
            )
            if config_idx == 0 or is_retraining_interval_passed:
                logger.info("Triggering conformal retraining...")
                if config_idx == 0:
                    latest_confidence_level = confidence_level

                if interval_type == "quantile_regression":
                    conformal_regressor = QuantileConformalRegression(
                        quantile_estimator_architecture=conformal_model_type,
                        random_state=random_state,
                    )

                    previous_best_cqr_config = (
                        best_cqr_config
                        if best_cqr_config is None
                        else best_cqr_config.copy()
                    )
                    logger.debug(
                        f"Tune fitting with custom best configuration: {previous_best_cqr_config}"
                    )
                    best_cqr_config = conformal_regressor.tune_fit(
                        X_train=X_train_conformal,
                        y_train=y_train_conformal,
                        X_val=X_val_conformal,
                        y_val=y_val_conformal,
                        confidence_level=latest_confidence_level,
                        tuning_param_combinations=starting_search_estimator_tunings,
                        custom_best_param_combination=previous_best_cqr_config,
                    )
                    if conformal_regressor.tuning_runtime is not None:
                        hyperreg_model_runtime_per_iter = (
                            conformal_regressor.tuning_runtime
                        )

                elif interval_type == "locally_weighted":
                    logger.debug(
                        "Subsplitting training set into sub training and sub validation sets..."
                    )
                    (
                        HR_X_pe_fitting,
                        HR_y_pe_fitting,
                        HR_X_ve_fitting,
                        HR_y_ve_fitting,
                    ) = train_val_split(
                        X_train_conformal,
                        y_train_conformal,
                        train_split=0.75,
                        normalize=False,
                        random_state=random_state,
                    )
                    logger.debug(
                        f"Obtained sub training set of size {HR_X_pe_fitting.shape} "
                        f"and sub validation set of size {HR_X_ve_fitting.shape}"
                    )

                    conformal_regressor = LocallyWeightedConformalRegression(
                        point_estimator_architecture=conformal_model_type,
                        demeaning_estimator_architecture=conformal_variance_model_type,
                        variance_estimator_architecture=conformal_variance_model_type,
                        random_state=random_state,
                    )

                    previous_best_lw_pe_config = (
                        best_lw_pe_config
                        if best_lw_pe_config is None
                        else best_lw_pe_config.copy()
                    )
                    previous_best_lw_de_config = (
                        best_lw_de_config
                        if best_lw_de_config is None
                        else best_lw_de_config.copy()
                    )
                    previous_best_lw_ve_config = (
                        best_lw_ve_config
                        if best_lw_ve_config is None
                        else best_lw_ve_config.copy()
                    )

                    logger.debug(
                        f"Tune fitting with custom best configurations: {previous_best_lw_pe_config}"
                        f"{previous_best_lw_de_config}"
                        f"{previous_best_lw_ve_config}"
                    )
                    (
                        best_lw_pe_config,
                        best_lw_de_config,
                        best_lw_ve_config,
                    ) = conformal_regressor.tune_fit(
                        X_pe=HR_X_pe_fitting,
                        y_pe=HR_y_pe_fitting,
                        X_ve=HR_X_ve_fitting,
                        y_ve=HR_y_ve_fitting,
                        X_val=X_val_conformal,
                        y_val=y_val_conformal,
                        confidence_level=latest_confidence_level,
                        tuning_param_combinations=starting_search_estimator_tunings,
                        custom_best_pe_param_combination=previous_best_lw_pe_config,
                        custom_best_de_param_combination=previous_best_lw_de_config,
                        custom_best_ve_param_combination=previous_best_lw_ve_config,
                    )
                    if conformal_regressor.tuning_runtime is not None:
                        hyperreg_model_runtime_per_iter = (
                            conformal_regressor.tuning_runtime
                        )

                else:
                    ValueError(f"{interval_type} is not a valid interval type.")

            starting_search_estimator_tunings = derive_optimal_tuning_count(
                baseline_model_runtime=runtime_per_search,
                search_model_runtime=hyperreg_model_runtime_per_iter,
                search_model_retraining_freq=conformal_retraining_frequency,
                search_to_baseline_runtime_ratio=0.3,
            )
            logger.info(
                f"Optimal number of searcher hyperparameters to search: {starting_search_estimator_tunings}"
            )

            (
                parameter_performance_lower_bounds,
                parameter_performance_higher_bounds,
            ) = conformal_regressor.predict(
                X=tabularized_searchable_configurations,
                confidence_level=latest_confidence_level,
            )

            maximal_idx = get_best_configuration_idx(
                performance_bounds=(
                    parameter_performance_lower_bounds,
                    parameter_performance_higher_bounds,
                ),
                optimization_direction=METRIC_PROPORTIONALITY_LOOKUP[
                    self.custom_loss_function
                ],
            )

            maximal_parameter = searchable_configurations[maximal_idx].copy()
            validation_performance = self._evaluate_configuration_performance(
                configuration=maximal_parameter, random_state=random_state
            )
            if np.isnan(validation_performance):
                logger.debug(
                    "Obtained non-numerical performance, skipping configuration."
                )
                continue

            is_last_interval_breached = is_interval_breach(
                performance_lower_bounds=parameter_performance_lower_bounds,
                performance_higher_bounds=parameter_performance_higher_bounds,
                bound_idx=maximal_idx,
                realization=validation_performance,
            )
            if enable_adaptive_intervals:
                latest_confidence_level = update_adaptive_interval(
                    true_confidence_level=confidence_level,
                    last_confidence_level=latest_confidence_level,
                    breach=is_last_interval_breached,
                    learning_rate=conformal_learning_rate,
                )

            self.searched_configurations.append(maximal_parameter.copy())
            self.searched_performances.append(validation_performance)

            logger.info(f"Iteration: {config_idx}")
            logger.info(f"Iteration's validation performance: {validation_performance}")

            if self.search_timer.return_runtime() > runtime_budget:
                break

    def get_best_params(self) -> Dict:
        best_performance_idx = self.searched_performances.index(
            max(self.searched_performances)
        )
        best_params = self.searched_configurations[best_performance_idx]

        return best_params

    def get_best_model(self):
        best_model = update_model_parameters(
            model_instance=self.model,
            configuration=self.get_best_params(),
            random_state=self.random_state,
        )
        return best_model

    def get_best_fitted_model(self):
        best_model = self.get_best_model()
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.hstack((self.y_train, self.y_val))

        best_model.fit(X_full, y_full)

        return best_model

    @staticmethod
    def _set_conformal_validation_split(X: np.array) -> float:
        if len(X) <= 30:
            validation_split = 5 / len(X)
        else:
            validation_split = 0.33
        return validation_split
