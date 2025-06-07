import logging
import random
from typing import Optional, Dict, Tuple, get_type_hints, Literal, Union, List

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
import inspect
from confopt.utils.encoding import ConfigurationEncoder
from confopt.utils.preprocessing import train_val_split, remove_iqr_outliers
from confopt.utils.encoding import get_tuning_configurations
from confopt.utils.tracking import (
    Trial,
    Study,
    RuntimeTracker,
)
from confopt.utils.optimization import BayesianTuner, FixedSurrogateTuner
from confopt.selection.acquisition import (
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,
    LowerBoundSampler,
    PessimisticLowerBoundSampler,
    BaseConformalSearcher,
)
from confopt.wrapping import ParameterRange

logger = logging.getLogger(__name__)


def process_and_split_estimation_data(
    searched_configurations: np.array,
    searched_performances: np.array,
    train_split: float,
    filter_outliers: bool = False,
    outlier_scope: str = "top_and_bottom",
    random_state: Optional[int] = None,
) -> Tuple[np.array, np.array, np.array, np.array]:
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


def check_early_stopping(
    searchable_count,
    current_runtime=None,
    runtime_budget=None,
    current_iter=None,
    max_iter=None,
    n_random_searches=None,
):
    if searchable_count == 0:
        return True, "All configurations have been searched"

    if runtime_budget is not None and current_runtime is not None:
        if current_runtime > runtime_budget:
            return True, f"Runtime budget ({runtime_budget}) exceeded"

    if (
        max_iter is not None
        and current_iter is not None
        and n_random_searches is not None
    ):
        if n_random_searches + current_iter >= max_iter:
            return True, f"Maximum iterations ({max_iter}) reached"

    return False


def create_config_hash(config: Dict) -> tuple:
    """Create a hashable representation of a configuration for fast lookups"""
    # Use a more consistent approach for all values including complex types
    return tuple(
        sorted(
            (k, str(v) if not isinstance(v, (int, float, bool, str)) else v)
            for k, v in config.items()
        )
    )


class ConformalTuner:
    def __init__(
        self,
        objective_function: callable,
        search_space: Dict[str, ParameterRange],
        metric_optimization: Literal["maximize", "minimize"],
        n_candidate_configurations: int = 10000,
        warm_start_configurations: Optional[List[Tuple[Dict, float]]] = None,
        dynamic_sampling: bool = False,
    ):
        self.objective_function = objective_function
        self._check_objective_function()

        self.search_space = search_space
        self.metric_sign = -1 if metric_optimization == "maximize" else 1
        self.n_candidate_configurations = n_candidate_configurations
        self.warm_start_configurations = warm_start_configurations
        self.dynamic_sampling = dynamic_sampling

        # Initialize storage for configurations with more efficient data structures
        self.searchable_configs = []
        self.searched_configs = []
        self.searched_performances = []
        self.searched_configs_set = set()

        # For fast lookup of config positions - critical for performance
        self.searchable_hash_to_idx = (
            {}
        )  # Maps config hash -> index in searchable_configs
        self.tabularized_configs_map = {}  # Maps config hash -> tabularized config

    @staticmethod
    def _set_conformal_validation_split(X: np.array) -> float:
        if len(X) <= 30:
            validation_split = 4 / len(X)
        else:
            validation_split = 0.20
        return validation_split

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

    def _initialize_tuning_resources(self):
        """Initialize resources needed for tuning with optimized performance"""
        # Load warm start configurations
        warm_start_configs = []
        warm_start_performances = []

        if self.warm_start_configurations:
            for config, perf in self.warm_start_configurations:
                warm_start_configs.append(config)
                warm_start_performances.append(perf)

        # Get initial configurations
        # Use a smaller number of initial configurations for dynamic sampling to improve startup speed
        initial_config_count = (
            min(self.n_candidate_configurations, 5000)
            if self.dynamic_sampling
            else self.n_candidate_configurations
        )

        initial_configs = get_tuning_configurations(
            parameter_grid=self.search_space,
            n_configurations=initial_config_count,
            random_state=None,
            warm_start_configs=warm_start_configs,
        )

        # Set up encoder for tabularization - this is a costly operation we want to do only once
        self.encoder = ConfigurationEncoder()
        self.encoder.fit(initial_configs)

        # Initialize data structures
        self.searchable_configs = []
        self.searched_configs = []
        self.searched_performances = []
        self.searched_configs_set = set()
        self.searchable_hash_to_idx = {}  # Reset hash-to-index mapping

        # Pre-allocate hash table with appropriate size for better performance
        self.tabularized_configs_map = {}

        # Pre-compute tabularized versions of configs in batches for better efficiency
        batch_size = 1000
        for start_idx in range(0, len(initial_configs), batch_size):
            batch_configs = initial_configs[start_idx : start_idx + batch_size]
            tabularized_batch = self.encoder.transform(batch_configs).to_numpy()

            for i, config in enumerate(batch_configs):
                config_hash = create_config_hash(config)
                # Skip if already in searched set (should only happen for warm starts)
                if config_hash not in self.searched_configs_set:
                    # Add to searchable configs
                    self.searchable_configs.append(config)
                    # Update the hash-to-index mapping - CRITICAL for performance
                    self.searchable_hash_to_idx[config_hash] = (
                        len(self.searchable_configs) - 1
                    )
                    # Cache the tabularized representation
                    self.tabularized_configs_map[config_hash] = tabularized_batch[i]

        self.study = Study()

        # Process warm starts
        if self.warm_start_configurations:
            self._process_warm_start_configurations()

    def _process_warm_start_configurations(self):
        """Process warm start configurations efficiently"""
        if not self.warm_start_configurations:
            return

        warm_start_trials = []

        # For each warm start config
        for i, (config, performance) in enumerate(self.warm_start_configurations):
            config_hash = create_config_hash(config)

            # Mark as searched
            self.searched_configs.append(config)
            self.searched_performances.append(performance)
            self.searched_configs_set.add(config_hash)

            # Compute tabularized representation if not already cached
            if config_hash not in self.tabularized_configs_map:
                tabularized = self.encoder.transform([config]).to_numpy()[0]
                self.tabularized_configs_map[config_hash] = tabularized

            # Remove from searchable if it's there using hash-based lookup
            if config_hash in self.searchable_hash_to_idx:
                idx_to_remove = self.searchable_hash_to_idx.pop(config_hash)

                # Remove the configuration from searchable configs
                if idx_to_remove < len(self.searchable_configs):
                    self.searchable_configs.pop(idx_to_remove)

                    # Update indices for all configurations after the removed one
                    for hash_key, idx in list(self.searchable_hash_to_idx.items()):
                        if idx > idx_to_remove:
                            self.searchable_hash_to_idx[hash_key] = idx - 1

            # Create trial
            warm_start_trials.append(
                Trial(
                    iteration=i,
                    timestamp=datetime.now(),
                    configuration=config.copy(),
                    performance=performance,
                    acquisition_source="warm_start",
                )
            )

        self.study.batch_append_trials(trials=warm_start_trials)
        logger.debug(
            f"Added {len(warm_start_trials)} warm start configurations to search history"
        )

    def _evaluate_configuration(self, configuration):
        runtime_tracker = RuntimeTracker()
        performance = self.objective_function(configuration=configuration)
        runtime = runtime_tracker.return_runtime()
        return performance, runtime

    def _update_search_state(self, config, performance, config_idx=None):
        """
        Update search state after evaluating a configuration.
        Works directly with the configuration rather than indices.
        - First, adds the configuration to the searched collections
        - Then, efficiently removes it from searchable configurations using hash-based lookup
        """
        # Add to searched collections
        config_hash = create_config_hash(config)
        self.searched_configs.append(config)
        self.searched_performances.append(performance)
        self.searched_configs_set.add(config_hash)

        # Use the hash-to-index mapping for O(1) lookup instead of O(n) search
        if config_hash in self.searchable_hash_to_idx:
            idx_to_remove = self.searchable_hash_to_idx.pop(config_hash)

            # Remove the configuration from searchable configs
            if idx_to_remove < len(self.searchable_configs):
                # Remove configuration at this index
                self.searchable_configs.pop(idx_to_remove)

                # Update indices for all configurations after the removed one
                # This is critical to keep the hash-to-idx mapping accurate
                for hash_key, idx in list(self.searchable_hash_to_idx.items()):
                    if idx > idx_to_remove:
                        self.searchable_hash_to_idx[hash_key] = idx - 1
        else:
            # Rare fallback for exact matches not found via hash
            for idx, searchable_config in enumerate(list(self.searchable_configs)):
                if config == searchable_config:
                    self.searchable_configs.pop(idx)
                    # Update hash-to-idx mapping for all configs after this one
                    for hash_key, idx_val in list(self.searchable_hash_to_idx.items()):
                        if idx_val > idx:
                            self.searchable_hash_to_idx[hash_key] = idx_val - 1
                    break

    def _get_tabularized_searchable(self):
        """Get tabularized representation of all searchable configurations"""
        if not self.searchable_configs:
            # Empty array with correct shape
            if self.tabularized_configs_map:
                sample_shape = next(iter(self.tabularized_configs_map.values())).shape
                return np.zeros((0, sample_shape[0]))
            return np.array([])

        # Get tabularized configs from cache or compute if not available
        tabularized_configs = []
        for config in self.searchable_configs:
            config_hash = create_config_hash(config)
            if config_hash in self.tabularized_configs_map:
                tabularized_configs.append(self.tabularized_configs_map[config_hash])
            else:
                # Should rarely happen in practice
                tabularized = self.encoder.transform([config]).to_numpy()[0]
                self.tabularized_configs_map[config_hash] = tabularized
                tabularized_configs.append(tabularized)

        return np.array(tabularized_configs)

    def _get_tabularized_searched(self):
        """Get tabularized representation of all searched configurations"""
        if not self.searched_configs:
            return np.array([])

        # Get tabularized configs from cache or compute if not available
        tabularized_configs = []
        for config in self.searched_configs:
            config_hash = create_config_hash(config)
            if config_hash in self.tabularized_configs_map:
                tabularized_configs.append(self.tabularized_configs_map[config_hash])
            else:
                # Should rarely happen in practice
                tabularized = self.encoder.transform([config]).to_numpy()[0]
                self.tabularized_configs_map[config_hash] = tabularized
                tabularized_configs.append(tabularized)

        return np.array(tabularized_configs)

    def _sample_new_configurations(self):
        """Generate new configurations for dynamic sampling"""
        # Generate new configurations
        new_configs = get_tuning_configurations(
            parameter_grid=self.search_space,
            n_configurations=self.n_candidate_configurations,
            random_state=None,
            warm_start_configs=self.searched_configs,  # Use all searched configs
        )

        # Clear old data structures completely
        self.searchable_configs = []
        self.searchable_hash_to_idx = {}  # Reset hash-to-index mapping

        # Pre-tabularize configurations in batches for better efficiency
        batch_size = 1000
        tabularized_configs = []

        for start_idx in range(0, len(new_configs), batch_size):
            batch_configs = new_configs[start_idx : start_idx + batch_size]

            # Filter out configurations that have already been searched
            filtered_batch = []
            for config in batch_configs:
                config_hash = create_config_hash(config)
                if config_hash not in self.searched_configs_set:
                    filtered_batch.append(config)

            if filtered_batch:
                # Tabularize filtered batch at once
                tabularized_batch = self.encoder.transform(filtered_batch).to_numpy()

                # Add to searchable and update mappings
                for i, config in enumerate(filtered_batch):
                    config_hash = create_config_hash(config)
                    self.searchable_configs.append(config)
                    # Update hash-to-index mapping
                    self.searchable_hash_to_idx[config_hash] = (
                        len(self.searchable_configs) - 1
                    )
                    # Cache tabularized representation
                    self.tabularized_configs_map[config_hash] = tabularized_batch[i]
                    tabularized_configs.append(tabularized_batch[i])

        # Return the tabularized searchable configurations directly
        if tabularized_configs:
            return np.array(tabularized_configs)
        elif self.tabularized_configs_map:
            # Return empty array with right shape
            sample_shape = next(iter(self.tabularized_configs_map.values())).shape
            return np.zeros((0, sample_shape[0]))
        else:
            return np.array([])

    def _random_search(
        self, n_searches: int, verbose: bool = True, max_runtime: Optional[int] = None
    ) -> list[Trial]:
        rs_trials = []

        # Cap the number of searches based on available configurations
        adj_n_searches = min(n_searches, len(self.searchable_configs))

        # Randomly sample from searchable configurations
        search_idxs = np.random.choice(
            len(self.searchable_configs), size=adj_n_searches, replace=False
        )

        sampled_configs = [self.searchable_configs[idx] for idx in search_idxs]

        # Set up progress bar
        progress_iter = (
            tqdm(sampled_configs, desc="Random search: ")
            if verbose
            else sampled_configs
        )

        for config in progress_iter:
            # Evaluate configuration
            validation_performance, training_time = self._evaluate_configuration(config)

            if np.isnan(validation_performance):
                logger.debug(
                    "Obtained non-numerical performance, forbidding configuration."
                )
                continue

            # Update search state with the config itself
            self._update_search_state(config=config, performance=validation_performance)

            # Create trial
            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=config.copy(),
                performance=validation_performance,
                acquisition_source="rs",
                target_model_runtime=training_time,
            )
            rs_trials.append(trial)

            logger.debug(
                f"Random search iter {len(rs_trials)} performance: {validation_performance}"
            )

            # Check for early stopping
            stop = check_early_stopping(
                searchable_count=len(self.searchable_configs),
                current_runtime=(
                    self.search_timer.return_runtime() if max_runtime else None
                ),
                runtime_budget=max_runtime,
            )
            if stop:
                raise RuntimeError(
                    "confopt preliminary random search exceeded total runtime budget. "
                    "Retry with larger runtime budget or set iteration-capped budget instead."
                )

        return rs_trials

    def _select_next_configuration(
        self, searcher, tabularized_searchable_configurations
    ):
        """Select the next best configuration to evaluate directly"""
        # Get predictions from searcher
        parameter_performance_bounds = searcher.predict(
            X=tabularized_searchable_configurations
        )

        # Find configuration with best predicted performance
        best_idx = np.argmin(parameter_performance_bounds)
        best_config = self.searchable_configs[best_idx]

        return best_config

    def _conformal_search(
        self,
        searcher: BaseConformalSearcher,
        n_random_searches,
        conformal_retraining_frequency,
        tabularized_searched_configurations,
        verbose,
        max_iter,
        runtime_budget,
        searcher_tuning_framework=None,
    ):
        # Setup progress bar
        progress_bar = None
        if verbose:
            if runtime_budget is not None:
                progress_bar = tqdm(total=runtime_budget, desc="Conformal search: ")
            elif max_iter is not None:
                progress_bar = tqdm(
                    total=max_iter - n_random_searches, desc="Conformal search: "
                )

        # Set up scaler for standardization
        scaler = StandardScaler()

        # Calculate maximum iterations
        if self.dynamic_sampling:
            max_iterations = (
                max_iter - n_random_searches if max_iter is not None else float("inf")
            )
        else:
            max_iterations = min(
                len(self.searchable_configs),
                self.n_candidate_configurations - len(self.searched_configs),
            )

        # Initialize searcher tuning optimization
        if searcher_tuning_framework == "reward_cost":
            tuning_optimizer = BayesianTuner(
                max_tuning_count=20,
                max_tuning_interval=15,
                conformal_retraining_frequency=conformal_retraining_frequency,
                min_observations=5,
                exploration_weight=0.1,
                random_state=42,
            )
        elif searcher_tuning_framework == "fixed":
            tuning_optimizer = FixedSurrogateTuner(
                n_tuning_episodes=10,
                tuning_interval=3 * conformal_retraining_frequency,
                conformal_retraining_frequency=conformal_retraining_frequency,
            )
        elif searcher_tuning_framework is None:
            tuning_optimizer = FixedSurrogateTuner(
                n_tuning_episodes=0,
                tuning_interval=conformal_retraining_frequency,
                conformal_retraining_frequency=conformal_retraining_frequency,
            )
        else:
            raise ValueError(
                "searcher_tuning_framework must be either 'reward_cost', 'fixed', or None."
            )

        # Initialize search parameters
        search_model_retuning_frequency = 1
        search_model_tuning_count = 0
        searcher_error_history = []

        # Main search loop
        for search_iter in range(int(max_iterations)):
            # Update progress bar if needed
            if progress_bar:
                if runtime_budget is not None:
                    progress_bar.update(
                        int(self.search_timer.return_runtime()) - progress_bar.n
                    )
                elif max_iter is not None:
                    progress_bar.update(1)

            # For dynamic sampling, generate new configurations at each iteration
            if self.dynamic_sampling:
                tabularized_searchable_configurations = (
                    self._sample_new_configurations()
                )
                if len(tabularized_searchable_configurations) == 0:
                    logger.warning("No more unique configurations to search. Stopping.")
                    break
            else:
                # Use existing searchable configurations
                tabularized_searchable_configurations = (
                    self._get_tabularized_searchable()
                )

            # Prepare data for conformal search
            validation_split = self._set_conformal_validation_split(
                X=tabularized_searched_configurations
            )

            # Split data for training
            X_train, y_train, X_val, y_val = process_and_split_estimation_data(
                searched_configurations=tabularized_searched_configurations,
                searched_performances=np.array(self.searched_performances),
                train_split=(1 - validation_split),
                filter_outliers=False,
            )

            # Apply metric sign for optimization direction
            y_train = y_train * self.metric_sign
            y_val = y_val * self.metric_sign

            # Scale the data
            scaler.fit(X=X_train)
            X_train = scaler.transform(X=X_train)
            X_val = scaler.transform(X=X_val)
            tabularized_searchable_configurations = scaler.transform(
                X=tabularized_searchable_configurations
            )

            # Retrain the searcher if needed
            searcher_runtime = None
            if search_iter == 0 or search_iter % conformal_retraining_frequency == 0:
                if (
                    search_model_retuning_frequency % conformal_retraining_frequency
                    != 0
                ):
                    raise ValueError(
                        "search_model_retuning_frequency must be a multiple of conformal_retraining_frequency."
                    )

                runtime_tracker = RuntimeTracker()
                searcher.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    tuning_iterations=search_model_tuning_count,
                )
                searcher_runtime = runtime_tracker.return_runtime()
                searcher_error_history.append(searcher.primary_estimator_error)

                # Update tuning optimizer if we have multiple iterations
                if len(searcher_error_history) > 1:
                    error_improvement = max(
                        0, searcher_error_history[-2] - searcher_error_history[-1]
                    )
                    try:
                        normalized_searcher_runtime = (
                            searcher_runtime
                            / self.study.get_average_target_model_runtime()
                        )
                    except ZeroDivisionError:
                        normalized_searcher_runtime = 0

                    # Pass the search iteration to update
                    tuning_optimizer.update(
                        arm=(
                            search_model_tuning_count,
                            search_model_retuning_frequency,
                        ),
                        reward=error_improvement,
                        cost=normalized_searcher_runtime,
                        search_iter=search_iter,
                    )

                # Get next tuning parameters
                (
                    search_model_tuning_count,
                    search_model_retuning_frequency,
                ) = tuning_optimizer.select_arm()

            # Select the next configuration to evaluate
            if len(self.searchable_configs) == 0:
                logger.warning("No more configurations to search.")
                break

            config = self._select_next_configuration(
                searcher=searcher,
                tabularized_searchable_configurations=tabularized_searchable_configurations,
            )

            # Evaluate the selected configuration
            validation_performance, _ = self._evaluate_configuration(config)
            logger.debug(
                f"Conformal search iter {search_iter} performance: {validation_performance}"
            )

            if np.isnan(validation_performance):
                continue

            # Update the searcher with the new result
            config_hash = create_config_hash(config)
            tabularized = self.tabularized_configs_map[config_hash]
            transformed_X = scaler.transform(tabularized.reshape(1, -1))
            searcher.update(
                X=transformed_X, y_true=self.metric_sign * validation_performance
            )

            # Calculate breach for logging/tracking
            breach = None
            if isinstance(
                searcher.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)
            ):
                if searcher.last_beta is not None:
                    # Breach is 1 if beta < alpha, 0 otherwise
                    breach = 1 if searcher.last_beta < searcher.sampler.alpha else 0

            estimator_error = searcher.primary_estimator_error

            # Update search state with the config itself
            self._update_search_state(config=config, performance=validation_performance)

            # Create and add trial
            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=config.copy(),
                performance=validation_performance,
                acquisition_source=str(searcher),
                searcher_runtime=searcher_runtime,
                breached_interval=breach,
                primary_estimator_error=estimator_error,
            )
            self.study.append_trial(trial)

            # Update tabularized searched configurations for the next iteration
            tabularized_searched_configurations = self._get_tabularized_searched()

            # Check for early stopping
            stop = check_early_stopping(
                searchable_count=len(self.searchable_configs),
                current_runtime=self.search_timer.return_runtime(),
                runtime_budget=runtime_budget,
                current_iter=search_iter + 1,
                max_iter=max_iter,
                n_random_searches=n_random_searches,
            )
            if stop:
                break

        # Close progress bar if it exists
        if progress_bar:
            if runtime_budget is not None:
                progress_bar.update(n=runtime_budget - progress_bar.n)
            elif max_iter is not None:
                progress_bar.update(
                    n=max(
                        0,
                        max_iter
                        - n_random_searches
                        - len(self.study.trials)
                        + n_random_searches,
                    )
                )
            progress_bar.close()

    def tune(
        self,
        n_random_searches: int = 20,
        conformal_retraining_frequency: int = 1,
        searcher: Optional[
            Union[LocallyWeightedConformalSearcher, QuantileConformalSearcher]
        ] = None,
        searcher_tuning_framework: Optional[Literal["reward_cost", "fixed"]] = None,
        random_state: Optional[int] = None,
        max_iter: Optional[int] = None,
        runtime_budget: Optional[int] = None,
        verbose: bool = True,
        dynamic_sampling: bool = None,
    ):
        # Set random seed if provided
        if random_state is not None:
            random.seed(a=random_state)
            np.random.seed(seed=random_state)

        # Override dynamic_sampling if provided
        if dynamic_sampling is not None:
            self.dynamic_sampling = dynamic_sampling

        # Set up default searcher if not provided
        if searcher is None:
            searcher = QuantileConformalSearcher(
                quantile_estimator_architecture="qrf",
                sampler=LowerBoundSampler(
                    interval_width=0.05,
                    adapter="DtACI",
                    beta_decay="logarithmic_decay",
                    c=1,
                ),
                n_pre_conformal_trials=20,
            )

        # Initialize resources
        self._initialize_tuning_resources()
        self.search_timer = RuntimeTracker()

        # Perform random search
        rs_trials = self._random_search(
            n_searches=n_random_searches,
            max_runtime=runtime_budget,
            verbose=verbose,
        )
        self.study.batch_append_trials(trials=rs_trials)

        # Get tabularized searched configurations
        tabularized_searched_configurations = self._get_tabularized_searched()

        # Perform conformal search
        self._conformal_search(
            searcher=searcher,
            n_random_searches=n_random_searches,
            conformal_retraining_frequency=conformal_retraining_frequency,
            tabularized_searched_configurations=tabularized_searched_configurations,
            verbose=verbose,
            max_iter=max_iter,
            runtime_budget=runtime_budget,
            searcher_tuning_framework=searcher_tuning_framework,
        )

    def get_best_params(self) -> Dict:
        return self.study.get_best_configuration()

    def get_best_value(self) -> float:
        return self.study.get_best_performance()
