import logging
import random
from typing import Optional, Dict, Tuple, get_type_hints, Literal, Union

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
import inspect
import pandas as pd

from confopt.preprocessing import train_val_split, remove_iqr_outliers
from confopt.utils import get_tuning_configurations, tabularize_configurations
from confopt.tracking import Trial, Study, RuntimeTracker, derive_optimal_tuning_count
from confopt.acquisition import (
    LocallyWeightedConformalSearcher,
    MultiFitQuantileConformalSearcher,
    UCBSampler,
)

logger = logging.getLogger(__name__)


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
        n_candidate_configurations: int = 10000,
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
        self.n_candidate_configurations = n_candidate_configurations

        self.tuning_configurations = self._get_tuning_configurations()

        # Pre-tabularize all configurations for efficiency
        self.tabularized_configs_df = self._pre_tabularize_configurations()
        self.tabularized_configs = self.tabularized_configs_df.to_numpy()

        # Create efficient index tracking
        self.available_indices = np.arange(len(self.tuning_configurations))
        self.searched_indices = np.array([], dtype=int)
        self.searched_configs = []
        self.searched_performances = np.array([])

        self.study = Study()

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
            parameter_grid=self.search_space,
            n_configurations=self.n_candidate_configurations,
            random_state=1234,
        )
        return tuning_configurations

    def _pre_tabularize_configurations(self) -> pd.DataFrame:
        """Pre-tabularize all configurations to avoid repeated conversions."""
        # Use tabularize_configurations with empty searched_configurations
        tabularized_configs, _ = tabularize_configurations(
            searchable_configurations=self.tuning_configurations,
            searched_configurations=[],
        )
        return tabularized_configs

    def _random_search(
        self,
        n_searches: int,
        verbose: bool = True,
        max_runtime: Optional[int] = None,
    ) -> list[Trial]:
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
        rs_trials = []
        skipped_configuration_counter = 0

        # Use numpy for faster sampling without replacement
        n_sample = min(n_searches, len(self.available_indices))
        random_indices = np.random.choice(
            self.available_indices, size=n_sample, replace=False
        )

        # Update available indices immediately
        self.available_indices = np.setdiff1d(
            self.available_indices, random_indices, assume_unique=True
        )

        # Store sampled configurations
        randomly_sampled_indices = random_indices.tolist()

        if verbose:
            iterator = tqdm(randomly_sampled_indices, desc="Random search: ")
        else:
            iterator = randomly_sampled_indices

        for config_idx, idx in enumerate(iterator):
            hyperparameter_configuration = self.tuning_configurations[idx]

            training_time_tracker = RuntimeTracker()
            validation_performance = self.objective_function(
                configuration=hyperparameter_configuration
            )
            training_time = training_time_tracker.return_runtime()

            if np.isnan(validation_performance):
                skipped_configuration_counter += 1
                logger.debug(
                    "Obtained non-numerical performance, skipping configuration."
                )
                continue

            # Track this as a searched index
            self.searched_indices = np.append(self.searched_indices, idx)
            self.searched_configs.append(hyperparameter_configuration)
            self.searched_performances = np.append(
                self.searched_performances, validation_performance
            )

            rs_trials.append(
                Trial(
                    iteration=config_idx,
                    timestamp=datetime.now(),
                    configuration=hyperparameter_configuration.copy(),
                    performance=validation_performance,
                    target_model_runtime=training_time,
                    acquisition_source="rs",
                )
            )

            logger.debug(
                f"Random search iter {config_idx} performance: {validation_performance}"
            )

            if max_runtime is not None:
                if self.search_timer.return_runtime() > max_runtime:
                    raise RuntimeError(
                        "confopt preliminary random search exceeded total runtime budget. "
                        "Retry with larger runtime budget or set iteration-capped budget instead."
                    )

        return rs_trials

    @staticmethod
    def _set_conformal_validation_split(X: np.array) -> float:
        if len(X) <= 30:
            validation_split = 4 / len(X)
        else:
            validation_split = 0.20
        return validation_split

    def _dict_to_hashable(self, configuration: dict) -> tuple:
        """Convert a configuration dictionary to a hashable representation efficiently.

        Uses sorted frozensets for better hashing performance and memory usage.
        """
        # For small dictionaries, this is faster than complex transformations
        return frozenset(configuration.items())

    def search(
        self,
        searcher: Union[
            LocallyWeightedConformalSearcher, MultiFitQuantileConformalSearcher
        ],
        n_random_searches: int = 20,
        conformal_retraining_frequency: int = 1,
        searcher_tuning_framework: Optional[Literal["runtime", "ucb", "fixed"]] = None,
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
        self.search_timer = RuntimeTracker()

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        rs_trials = self._random_search(
            n_searches=n_random_searches,
            max_runtime=runtime_budget,
            verbose=verbose,
        )
        self.study.batch_append_trials(trials=rs_trials)

        # Pre-allocate storage for efficiency
        search_model_tuning_count = 0
        scaler = StandardScaler()

        # Setup progress bar
        if verbose:
            if runtime_budget is not None:
                search_progress_bar = tqdm(
                    total=runtime_budget, desc="Conformal search: "
                )
            elif max_iter is not None:
                search_progress_bar = tqdm(
                    total=max_iter - n_random_searches, desc="Conformal search: "
                )

        # Get initial searched configurations in tabular form once
        tabularized_searched_configurations = self.tabularized_configs[
            self.searched_indices
        ]

        # Main search loop
        max_iterations = min(
            len(self.available_indices),
            len(self.tuning_configurations) - n_random_searches,
        )
        for config_idx in range(max_iterations):
            if verbose:
                if runtime_budget is not None:
                    search_progress_bar.update(
                        int(self.search_timer.return_runtime()) - search_progress_bar.n
                    )
                elif max_iter is not None:
                    search_progress_bar.update(1)

            # Check if we've exhausted all configurations
            if len(self.available_indices) == 0:
                logger.info("All configurations have been searched. Stopping early.")
                break

            # Get tabularized searchable configurations more efficiently
            # We can index the pre-tabularized configurations directly
            tabularized_searchable_configurations = self.tabularized_configs[
                self.available_indices
            ]

            # Calculate validation split based on number of searched configurations
            validation_split = (
                ObjectiveConformalSearcher._set_conformal_validation_split(
                    tabularized_searched_configurations
                )
            )

            # Process data and normalize
            (
                X_train_conformal,
                y_train_conformal,
                X_val_conformal,
                y_val_conformal,
            ) = process_and_split_estimation_data(
                searched_configurations=tabularized_searched_configurations,
                searched_performances=self.searched_performances,
                train_split=(1 - validation_split),
                filter_outliers=False,
            )

            # Fit scaler on training data and transform all datasets
            scaler.fit(X_train_conformal)
            X_train_conformal = scaler.transform(X_train_conformal)
            X_val_conformal = scaler.transform(X_val_conformal)
            tabularized_searchable_configurations = scaler.transform(
                tabularized_searchable_configurations
            )

            # Handle model retraining
            hit_retraining_interval = config_idx % conformal_retraining_frequency == 0
            if config_idx == 0 or hit_retraining_interval:
                runtime_tracker = RuntimeTracker()
                searcher.fit(
                    X_train=X_train_conformal,
                    y_train=y_train_conformal,
                    X_val=X_val_conformal,
                    y_val=y_val_conformal,
                    tuning_iterations=search_model_tuning_count,
                )
                searcher_runtime = runtime_tracker.return_runtime()

                if config_idx == 0:
                    first_searcher_runtime = searcher_runtime

            # Determine tuning count if necessary
            if searcher_tuning_framework is not None:
                if searcher_tuning_framework == "runtime":
                    search_model_tuning_count = derive_optimal_tuning_count(
                        target_model_runtime=self.study.get_average_target_model_runtime(),
                        search_model_runtime=first_searcher_runtime,
                        search_model_retraining_freq=conformal_retraining_frequency,
                        search_to_baseline_runtime_ratio=0.3,
                    )
                elif searcher_tuning_framework == "fixed":
                    search_model_tuning_count = 10
            else:
                search_model_tuning_count = 0

            # Get performance predictions for searchable configurations
            parameter_performance_bounds = searcher.predict(
                X=tabularized_searchable_configurations
            )

            # Find minimum performing configuration
            minimal_local_idx = np.argmin(parameter_performance_bounds)
            global_idx = self.available_indices[minimal_local_idx]
            minimal_parameter = self.tuning_configurations[global_idx].copy()

            # Evaluate with objective function
            validation_performance = self.objective_function(
                configuration=minimal_parameter
            )

            # Update intervals if needed
            if hasattr(searcher.sampler, "adapter") or hasattr(
                searcher.sampler, "adapters"
            ):
                searcher.update_interval_width(
                    sampled_idx=minimal_local_idx,
                    sampled_performance=validation_performance,
                )

            logger.debug(
                f"Conformal search iter {config_idx} performance: {validation_performance}"
            )

            if np.isnan(validation_performance):
                continue

            # Handle UCBSampler breach calculation
            if isinstance(searcher.sampler, UCBSampler):
                if (
                    searcher.predictions_per_interval[0][minimal_local_idx][0]
                    <= validation_performance
                    <= searcher.predictions_per_interval[0][minimal_local_idx][1]
                ):
                    breach = 0
                else:
                    breach = 1
            else:
                breach = None

            estimator_error = searcher.primary_estimator_error

            # Update indices efficiently
            # Remove the global index from available indices
            self.available_indices = self.available_indices[
                self.available_indices != global_idx
            ]
            # Add to searched indices
            self.searched_indices = np.append(self.searched_indices, global_idx)
            # Add the configuration and performance to our tracking
            self.searched_configs.append(minimal_parameter)
            self.searched_performances = np.append(
                self.searched_performances, validation_performance
            )
            # Update the tabularized searched configurations for next iteration
            tabularized_searched_configurations = np.vstack(
                [
                    tabularized_searched_configurations,
                    self.tabularized_configs[global_idx : global_idx + 1],
                ]
            )

            # Add trial to study
            self.study.append_trial(
                Trial(
                    iteration=config_idx,
                    timestamp=datetime.now(),
                    configuration=minimal_parameter.copy(),
                    performance=validation_performance,
                    acquisition_source=str(searcher),
                    searcher_runtime=searcher_runtime,
                    breached_interval=breach,
                    primary_estimator_error=estimator_error,
                )
            )

            # Check stopping criteria
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
        return self.study.get_best_configuration()

    def get_best_value(self) -> float:
        """
        Extract validation performance of best performing parameter
        configuration identified during conformal search.

        Returns
        -------
        best_performance :
            Best predictive performance achieved.
        """
        return self.study.get_best_performance()
