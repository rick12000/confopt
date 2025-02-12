import logging
import random
from copy import deepcopy
from typing import Optional, Dict, Any, Tuple, List, get_type_hints, Literal

import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
import inspect

from confopt.optimization import derive_optimal_tuning_count, RuntimeTracker
from confopt.preprocessing import train_val_split, remove_iqr_outliers
from confopt.utils import get_tuning_configurations, tabularize_configurations

logger = logging.getLogger(__name__)


def update_model_parameters(
    model_instance: Any, configuration: Dict, random_state: int = None
):
    """
    Updates the attributes of an initialized model object.

    Only attributes which are specified in the 'configuration'
    dictionary input of this function will be overridden.

    Parameters
    ----------
    model_instance :
        An instance of a prediction model.
    configuration :
        A dictionary whose keys represent the attributes of
        the model instance that need to be overridden and whose
        values represent what they should be overridden to.
        Keys must match model instance attribute names.
    random_state :
        Random generation seed.

    Returns
    -------
    updated_model_instance :
        Model instance with updated attributes.
    """
    updated_model_instance = deepcopy(model_instance)
    for tuning_attr_name, tuning_attr in configuration.items():
        setattr(updated_model_instance, tuning_attr_name, tuning_attr)
    if hasattr(updated_model_instance, "random_state"):
        setattr(updated_model_instance, "random_state", random_state)
    return updated_model_instance


def score_predictions(
    y_obs: np.array, y_pred: np.array, scoring_function: str
) -> float:
    """
    Score a model's predictions against observed realizations.

    Parameters
    ----------
    y_obs :
        Observed target variable realizations.
    y_pred :
        Model predicted target variable values.
    scoring_function :
        Type of scoring function to use. Can be one of
        either:
            - 'accuracy_score'
            - 'log_loss'
            - 'mean_squared_error'

    Returns
    -------
    score :
        Scored model predictions.
    """
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
    filter_outliers: bool = False,
    outlier_scope: str = "top_and_bottom",
    random_state: Optional[int] = None,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Preprocess configuration data used to train conformal search estimators.

    Data is split into training and validation sets, with optional
    outlier filtering.

    Parameters
    ----------
    searched_configurations :
        Parameter configurations selected for search as part
        of conformal hyperparameter optimization framework.
    searched_performances :
        Validation performance of each parameter configuration.
    train_split :
        Proportion of overall configurations that should be allocated
        to the training set.
    filter_outliers :
        Whether to remove outliers from the input configuration
        data based on performance.
    outlier_scope :
        Determines which outliers are removed. Takes:
            - 'top_only': Only upper threshold outliers are removed.
            - 'bottom_only': Only lower threshold outliers are removed.
            - 'top_and_bottom': All outliers are removed.
    random_state :
        Random generation seed.

    Returns
    -------
    X_train :
        Training portion of configurations.
    y_train :
        Training portion of configuration performances.
    X_val :
        Validation portion of configurations.
    y_val :
        Validation portion of configuration performances.
    """
    X = searched_configurations.copy()
    y = searched_performances.copy()
    logger.debug(f"Minimum performance in searcher data: {y.min()}")
    logger.debug(f"Maximum performance in searcher data: {y.max()}")

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
    """
    Normalize configuration data used to train conformal search estimators.

    Parameters
    ----------
    training_searched_configurations :
        Training portion of parameter configurations selected for
        search as part of conformal optimization framework.
    validation_searched_configurations :
        Validation portion of parameter configurations selected for
        search as part of conformal optimization framework.
    searchable_configurations :
        Larger range of parameter configurations that remain
        un-searched (i.e. whose validation performance has not
        yet been evaluated).

    Returns
    -------
    normalized_training_searched_configurations :
        Normalized training portion of searched parameter
        configurations.
    normalized_validation_searched_configurations :
        Normalized validation portion of searched parameter
        configurations.
    normalized_searchable_configurations :
        Normalized un-searched parameter configurations.
    """
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


def get_best_performance_idx(
    metric_optimization: str, searched_performances: List[float]
) -> int:
    if metric_optimization == "direct":
        best_performance_idx = searched_performances.index(max(searched_performances))
    elif metric_optimization == "inverse":
        best_performance_idx = searched_performances.index(min(searched_performances))
    else:
        raise ValueError()

    return best_performance_idx


def update_adaptive_confidence_level(
    true_confidence_level: float,
    last_confidence_level: float,
    breach: bool,
    learning_rate: float,
) -> float:
    """
    Update adaptive confidence level based on breach events.

    The confidence level is increased or decreased based on
    a specified learning rate and whether the last used interval
    was breached or not.

    Parameters
    ----------
    true_confidence_level :
        Global confidence level specified at the beginning of
        conformal hyperparameter search.
    last_confidence_level :
        Confidence level as of the last used interval.
    learning_rate :
        Learning rate dictating the magnitude of the confidence
        level update.

    Returns
    -------
    updated_confidence_level :
        Updated confidence level.
    """
    updated_confidence_level = 1 - (
        (1 - last_confidence_level)
        + learning_rate * ((1 - true_confidence_level) - breach)
    )
    updated_confidence_level = min(max(0.01, updated_confidence_level), 0.99)
    logger.debug(
        f"Updated confidence level of {last_confidence_level} to {updated_confidence_level}."
    )

    return updated_confidence_level


class ObjectiveConformalSearcher:
    """
    Conformal hyperparameter searcher.

    Tunes a desired model by inferentially searching a
    specified hyperparameter space using conformal estimators.
    """

    def __init__(
        self,
        objective_function: callable,
        search_space: Dict,
        metric_optimization: Literal["direct", "inverse"],
    ):
        """
        Create a conformal searcher instance.

        Parameters
        ----------
        # TODO
        search_space :
            Dictionary mapping parameter names to possible parameter
            values they can take.
        """

        self.objective_function = objective_function
        self._check_objective_function()
        self.search_space = search_space
        self.metric_optimization = metric_optimization

        self.tuning_configurations = self._get_tuning_configurations()

        self.searched_configurations = []
        self.searched_performances = []
        self.searched_timestamps = []

    def _check_objective_function(self):
        signature = inspect.signature(self.objective_function)
        args = list(signature.parameters.values())

        if len(args) != 1:
            raise ValueError("Objective function must take exactly one argument.")

        first_arg = args[0]
        if first_arg.name != "configuration":
            raise ValueError(
                "The objective function must take exactly one argument named 'configuration'."
            )

        type_hints = get_type_hints(self.objective_function)
        if "configuration" in type_hints and type_hints["configuration"] is not Dict:
            raise TypeError(
                "The 'configuration' argument of the objective must be of type Dict."
            )
        if "return" in type_hints and type_hints["return"] not in [
            int,
            float,
            np.number,
        ]:
            raise TypeError(
                "The return type of the objective function must be numeric (int, float, or np.number)."
            )

    def _get_tuning_configurations(self):
        logger.debug("Creating hyperparameter space...")
        tuning_configurations = get_tuning_configurations(
            parameter_grid=self.search_space, n_configurations=10000, random_state=1234
        )
        return tuning_configurations

    def _evaluate_configuration_performance(
        self,
        configuration: Dict,
    ) -> float:
        """
        Evaluate the performance of a specified parameter configuration.

        Parameters
        ----------
        configuration :
            Parameter configuration for the base model being tuned using
            conformal search.

        Returns
        -------
        performance :
            Specified configuration's validation performance.
        """
        logger.debug(f"Evaluating model with configuration: {configuration}")

        performance = self.objective_function(configuration=configuration)

        return performance

    def _random_search(
        self,
        n_searches: int,
        verbose: bool = True,
        max_runtime: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[List, List, List, float]:
        """
        Randomly search a portion of the model's hyperparameter space.

        Parameters
        ----------
        n_searches :
            Number of random searches to perform.
        max_runtime :
            Maximum runtime after which search stops.
        verbose :
            Whether to print updates during code execution.
        random_state :
            Random generation seed.

        Returns
        -------
        searched_configurations :
            List of parameter configurations that were randomly
            selected and searched.
        searched_performances :
            Search performance of each searched configuration,
            consisting of out of sample, validation performance
            of a model trained using the searched configuration.
        searched_timestamps :
            List of timestamps corresponding to each searched
            hyperparameter configuration.
        runtime_per_search :
            Average time taken to train the model being tuned
            across configurations, in seconds.
        """
        random.seed(random_state)
        np.random.seed(random_state)

        searched_configurations = []
        searched_performances = []
        searched_timestamps = []

        skipped_configuration_counter = 0
        runtime_per_search = 0

        shuffled_tuning_configurations = self.tuning_configurations.copy()
        random.seed(random_state)
        random.shuffle(shuffled_tuning_configurations)
        randomly_sampled_configurations = shuffled_tuning_configurations[
            : min(n_searches, len(self.tuning_configurations))
        ]

        model_training_timer = RuntimeTracker()
        model_training_timer.pause_runtime()
        if verbose:
            randomly_sampled_configurations = tqdm(
                randomly_sampled_configurations, desc="Random search: "
            )
        for config_idx, hyperparameter_configuration in enumerate(
            randomly_sampled_configurations
        ):
            model_training_timer.resume_runtime()
            validation_performance = self._evaluate_configuration_performance(
                configuration=hyperparameter_configuration
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
            searched_timestamps.append(datetime.now())

            runtime_per_search = (
                runtime_per_search + model_training_timer.return_runtime()
            ) / (config_idx - skipped_configuration_counter + 1)

            logger.debug(
                f"Random search iter {config_idx} performance: {validation_performance}"
            )

            if max_runtime is not None:
                if self.search_timer.return_runtime() > max_runtime:
                    raise RuntimeError(
                        "confopt preliminary random search exceeded total runtime budget. "
                        "Retry with larger runtime budget or set iteration-capped budget instead."
                    )

        return (
            searched_configurations,
            searched_performances,
            searched_timestamps,
            runtime_per_search,
        )

    @staticmethod
    def _set_conformal_validation_split(X: np.array) -> float:
        if len(X) <= 30:
            validation_split = 5 / len(X)
        else:
            validation_split = 0.33
        return validation_split

    def search(
        self,
        searcher,
        confidence_level: float = 0.8,
        n_random_searches: int = 20,
        conformal_retraining_frequency: int = 1,
        enable_adaptive_intervals: bool = True,
        conformal_learning_rate: float = 0.1,
        verbose: bool = True,
        random_state: Optional[int] = None,
        max_iter: Optional[int] = None,
        runtime_budget: Optional[int] = None,
    ):
        """
        Search model hyperparameter space using conformal estimators.

        Model and hyperparameter space are defined in the initialization
        of this class. This method takes as inputs a limit on the duration
        of search and several overrides for search behaviour.

        Search involves randomly evaluating an initial number of hyperparameter
        configurations, then training a conformal estimator on the relationship
        between configurations and performance to optimally select the next
        best configuration to sample at each subsequent sampling event.
        Upon exceeding the maximum search duration, search results are stored
        in the class instance and accessible via dedicated externalizing methods.

        Parameters
        ----------
        runtime_budget :
            Maximum time budget to allocate to hyperparameter search in seconds.
            After the budget is exceeded, search stops and results are stored in
            the instance for later access.
            An error will be raised if the budget is not sufficient to carry out
            conformal search, in which case it should be raised.
        confidence_level :
            Confidence level used during construction of conformal searchers'
            intervals. The confidence level controls the exploration/exploitation
            tradeoff, with smaller values making search greedier.
            Confidence level must be bound between [0, 1].
        conformal_search_estimator :
            String identifier specifying which type of estimator should be
            used to infer model hyperparameter performance.
            Supported estimators include:
                - 'qgbm' (default): quantile gradient boosted machine.
                - 'qrf': quantile random forest.
                - 'kr': kernel ridge.
                - 'gp': gaussian process.
                - 'gbm': gradient boosted machine.
                - 'knn': k-nearest neighbours.
                - 'rf': random forest.
                - 'dnn': dense neural network.
        n_random_searches :
            Number of initial random searches to perform before switching
            to inferential search. A larger number delays the beginning of
            conformal search, but provides the search estimator with more
            data and more robust patterns. The more parameters are being
            optimized during search, the more random search observations
            are needed before the conformal searcher can extrapolate
            effectively. This value defaults to 20, which is the minimum
            advisable number before the estimator will struggle to train.
        conformal_retraining_frequency :
            Sampling interval after which conformal search estimators should be
            retrained. Eg. an interval of 5, would mean conformal estimators
            are retrained after every 5th sampled/searched parameter configuration.
            A lower retraining frequency is always desirable, but may be increased
            to reduce runtime.
        enable_adaptive_intervals :
            Whether to allow conformal intervals used for configuration sampling
            to change after each sampling event. This allows for better interval
            coverage under covariate shift and is enabled by default.
        conformal_learning_rate :
            Learning rate dictating how rapidly adaptive intervals are updated.
        verbose :
            Whether to print updates during code execution.
        random_state :
            Random generation seed.
        """

        self.random_state = random_state
        self.search_timer = RuntimeTracker()

        (
            searched_configurations,
            searched_performances,
            searched_timestamps,
            runtime_per_search,
        ) = self._random_search(
            n_searches=n_random_searches,
            max_runtime=runtime_budget,
            verbose=verbose,
            random_state=random_state,
        )

        self.searched_configurations.extend(searched_configurations)
        self.searched_performances.extend(searched_performances)
        self.searched_timestamps.extend(searched_timestamps)

        search_model_tuning_count = 0

        search_idx_range = range(len(self.tuning_configurations) - n_random_searches)
        if runtime_budget is not None:
            search_progress_bar = tqdm(total=runtime_budget, desc="Conformal search: ")
        elif max_iter is not None:
            search_progress_bar = tqdm(total=max_iter, desc="Conformal search: ")
        for config_idx in search_idx_range:
            if verbose:
                if runtime_budget is not None:
                    search_progress_bar.update(
                        int(self.search_timer.return_runtime()) - search_progress_bar.n
                    )
                elif max_iter is not None:
                    search_progress_bar.update(1)
            searchable_configurations = [
                configuration
                for configuration in self.tuning_configurations
                if configuration not in self.searched_configurations
            ]
            (
                tabularized_searchable_configurations,
                tabularized_searched_configurations,
            ) = tabularize_configurations(
                searchable_configurations=searchable_configurations,
                searched_configurations=self.searched_configurations.copy(),
            )
            (
                tabularized_searchable_configurations,
                tabularized_searched_configurations,
            ) = (
                tabularized_searchable_configurations.to_numpy(),
                tabularized_searched_configurations.to_numpy(),
            )

            validation_split = (
                ObjectiveConformalSearcher._set_conformal_validation_split(
                    tabularized_searched_configurations
                )
            )
            (
                X_train_conformal,
                y_train_conformal,
                X_val_conformal,
                y_val_conformal,
            ) = process_and_split_estimation_data(
                searched_configurations=tabularized_searched_configurations,
                searched_performances=np.array(self.searched_performances),
                train_split=(1 - validation_split),
                filter_outliers=False,
                random_state=random_state,
            )

            (
                X_train_conformal,
                X_val_conformal,
                tabularized_searchable_configurations,
            ) = normalize_estimation_data(
                training_searched_configurations=X_train_conformal,
                validation_searched_configurations=X_val_conformal,
                searchable_configurations=tabularized_searchable_configurations,
            )

            hit_retraining_interval = config_idx % conformal_retraining_frequency == 0
            if config_idx == 0 or hit_retraining_interval:
                # if config_idx == 0:
                #     latest_confidence_level = confidence_level

                searcher.fit(
                    X_train=X_train_conformal,
                    y_train=y_train_conformal,
                    X_val=X_val_conformal,
                    y_val=y_val_conformal,
                    tuning_iterations=search_model_tuning_count,
                    random_state=random_state,
                )

            hyperreg_model_runtime_per_iter = searcher.training_time
            search_model_tuning_count = derive_optimal_tuning_count(
                baseline_model_runtime=runtime_per_search,
                search_model_runtime=hyperreg_model_runtime_per_iter,
                search_model_retraining_freq=conformal_retraining_frequency,
                search_to_baseline_runtime_ratio=0.3,
            )

            parameter_performance_bounds = searcher.predict(
                X=tabularized_searchable_configurations
            )

            minimal_idx = np.argmin(parameter_performance_bounds)

            minimal_parameter = searchable_configurations[minimal_idx].copy()
            validation_performance = self._evaluate_configuration_performance(
                configuration=minimal_parameter
            )
            logger.debug(
                f"Conformal search iter {config_idx} performance: {validation_performance}"
            )

            if np.isnan(validation_performance):
                continue

            # if (
            #     validation_performance
            #     > parameter_performance_higher_bounds[maximal_idx]
            # ) or (
            #     validation_performance < parameter_performance_lower_bounds[maximal_idx]
            # ):
            #     is_last_interval_breached = True
            # else:
            #     is_last_interval_breached = False

            # if enable_adaptive_intervals:
            #     latest_confidence_level = update_adaptive_confidence_level(
            #         true_confidence_level=confidence_level,
            #         last_confidence_level=latest_confidence_level,
            #         breach=is_last_interval_breached,
            #         learning_rate=conformal_learning_rate,
            #     )

            self.searched_configurations.append(minimal_parameter.copy())
            self.searched_performances.append(validation_performance)
            self.searched_timestamps.append(datetime.now())

            if runtime_budget is not None:
                if self.search_timer.return_runtime() > runtime_budget:
                    if verbose:
                        if runtime_budget is not None:
                            search_progress_bar.update(
                                runtime_budget - search_progress_bar.n
                            )
                        elif max_iter is not None:
                            search_progress_bar.update(1)
                        search_progress_bar.close()
                    break
            elif max_iter is not None:
                if n_random_searches + config_idx + 1 >= max_iter:
                    if verbose:
                        if runtime_budget is not None:
                            search_progress_bar.update(
                                runtime_budget - search_progress_bar.n
                            )
                        elif max_iter is not None:
                            search_progress_bar.update(1)
                        search_progress_bar.close()
                    break

    def get_best_params(self) -> Dict:
        """
        Extract hyperparameters from best performing parameter
        configuration identified during conformal search.

        Returns
        -------
        best_params :
            Best performing model hyperparameters.
        """
        best_performance_idx = get_best_performance_idx(
            metric_optimization=self.metric_optimization,
            searched_performances=self.searched_performances,
        )
        best_params = self.searched_configurations[best_performance_idx]

        return best_params

    def get_best_value(self) -> float:
        """
        Extract validation performance of best performing parameter
        configuration identified during conformal search.

        Returns
        -------
        best_performance :
            Best predictive performance achieved.
        """
        best_performance_idx = get_best_performance_idx(
            metric_optimization=self.metric_optimization,
            searched_performances=self.searched_performances,
        )
        best_performance = self.searched_performances[best_performance_idx]

        return best_performance

    def configure_best_model(self):
        """
        Extract best initialized (but unfitted) model identified
        during conformal search.

        Returns
        -------
        best_model :
            Best model from search.
        """
        best_model = update_model_parameters(
            model_instance=self.model,
            configuration=self.get_best_params(),
            random_state=self.random_state,
        )
        return best_model

    def fit_best_model(self):
        """
        Fit best model identified during conformal search.

        Returns
        -------
        best_fitted_model :
            Best model from search, fit on all available data.
        """
        best_fitted_model = self.configure_best_model()
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.hstack((self.y_train, self.y_val))

        best_fitted_model.fit(X=X_full, y=y_full)

        return best_fitted_model
