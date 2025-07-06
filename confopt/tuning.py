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
        normalize=False,  # False, handled outside of this function
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
):
    if searchable_count == 0:
        return True, "All configurations have been searched"

    if runtime_budget is not None and current_runtime is not None:
        if current_runtime > runtime_budget:
            return True, f"Runtime budget ({runtime_budget}) exceeded"

    if max_iter is not None and current_iter is not None:
        if current_iter >= max_iter:
            return True, f"Maximum iterations ({max_iter}) reached"

    return False, "No stopping condition met"


def create_config_hash(config: Dict) -> str:
    """Create a fast hashable representation of a configuration"""
    items = []
    for k in sorted(config.keys()):
        v = config[k]
        if isinstance(v, (int, float, bool)):
            items.append(f"{k}:{v}")
        else:
            items.append(f"{k}:{str(v)}")
    return "|".join(items)


class BaseConfigurationManager:
    def __init__(
        self,
        search_space: Dict[str, ParameterRange],
        n_candidate_configurations: int,
    ):
        self.search_space = search_space
        self.n_candidate_configurations = n_candidate_configurations
        self.searched_configs = []
        self.searched_performances = []
        self.searched_config_hashes = set()
        self.encoder = None
        self.banned_configurations = []

    def _setup_encoder(self, configs: List[Dict]):
        encoder_training_configs = get_tuning_configurations(
            parameter_grid=self.search_space,
            n_configurations=min(1000, self.n_candidate_configurations),
            random_state=None,
            sampling_method="uniform",
        )
        if configs:
            encoder_training_configs.extend(configs)
        self.encoder = ConfigurationEncoder()
        self.encoder.fit(encoder_training_configs)

    def mark_as_searched(self, config: Dict, performance: float):
        config_hash = create_config_hash(config)
        self.searched_configs.append(config)
        self.searched_performances.append(performance)
        self.searched_config_hashes.add(config_hash)

    def get_tabularized_configs(self, configs: List[Dict]) -> np.array:
        if not configs:
            return np.array([])
        return self.encoder.transform(configs).to_numpy()

    def add_to_banned_configurations(self, config: Dict):
        # Add configuration to banned list if not already present
        config_hash = create_config_hash(config)
        if config_hash not in [
            create_config_hash(c) for c in self.banned_configurations
        ]:
            self.banned_configurations.append(config)


class StaticConfigurationManager(BaseConfigurationManager):
    def __init__(
        self,
        search_space: Dict[str, ParameterRange],
        n_candidate_configurations: int,
    ):
        super().__init__(search_space, n_candidate_configurations)
        self.cached_searchable_configs = []
        self._initialize_static_configs_and_encoder()

    def _initialize_static_configs_and_encoder(self):
        candidate_configurations = get_tuning_configurations(
            parameter_grid=self.search_space,
            n_configurations=self.n_candidate_configurations,
            random_state=None,
            sampling_method="uniform",
        )
        filtered_configs = []
        for config in candidate_configurations:
            config_hash = create_config_hash(config)
            if config_hash not in self.searched_config_hashes:
                filtered_configs.append(config)
        self.cached_searchable_configs = filtered_configs
        self._setup_encoder(self.searched_configs + self.cached_searchable_configs)

    def get_searchable_configurations(self) -> List[Dict]:
        # Remove already searched and banned configs from cache
        banned_hashes = set(create_config_hash(c) for c in self.banned_configurations)
        self.cached_searchable_configs = [
            c
            for c in self.cached_searchable_configs
            if create_config_hash(c) not in self.searched_config_hashes
            and create_config_hash(c) not in banned_hashes
        ]
        return self.cached_searchable_configs.copy()

    def mark_as_searched(self, config: Dict, performance: float):
        super().mark_as_searched(config, performance)
        # Remove from cache if present
        config_hash = create_config_hash(config)
        self.cached_searchable_configs = [
            c
            for c in self.cached_searchable_configs
            if create_config_hash(c) != config_hash
        ]


class DynamicConfigurationManager(BaseConfigurationManager):
    def __init__(
        self,
        search_space: Dict[str, ParameterRange],
        n_candidate_configurations: int,
    ):
        super().__init__(search_space, n_candidate_configurations)
        self._setup_encoder(self.searched_configs)

    def get_searchable_configurations(self) -> List[Dict]:
        candidate_configurations = get_tuning_configurations(
            parameter_grid=self.search_space,
            n_configurations=self.n_candidate_configurations
            + len(self.searched_configs),
            random_state=None,
            sampling_method="uniform",
        )
        banned_hashes = set(create_config_hash(c) for c in self.banned_configurations)
        filtered_configs = []
        for config in candidate_configurations:
            config_hash = create_config_hash(config)
            if (
                config_hash not in self.searched_config_hashes
                and config_hash not in banned_hashes
            ):
                filtered_configs.append(config)
                if len(filtered_configs) >= self.n_candidate_configurations:
                    break
        return filtered_configs


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
        self.warm_start_configurations = warm_start_configurations
        self.n_candidate_configurations = n_candidate_configurations
        self.dynamic_sampling = dynamic_sampling

    @staticmethod
    def _set_conformal_validation_split(X: np.array) -> float:
        return 4 / len(X) if len(X) <= 30 else 0.20

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

    def process_warm_starts(self):
        for idx, (config, performance) in enumerate(self.warm_start_configurations):
            self.config_manager.mark_as_searched(config, performance)
            trial = Trial(
                iteration=idx,
                timestamp=datetime.now(),
                configuration=config.copy(),
                performance=performance,
                acquisition_source="warm_start",
            )
            self.study.append_trial(trial)

    def _initialize_tuning_resources(self):
        self.study = Study()

        if self.dynamic_sampling:
            self.config_manager = DynamicConfigurationManager(
                search_space=self.search_space,
                n_candidate_configurations=self.n_candidate_configurations,
            )
        else:
            self.config_manager = StaticConfigurationManager(
                search_space=self.search_space,
                n_candidate_configurations=self.n_candidate_configurations,
            )

        if self.warm_start_configurations:
            self.process_warm_starts()

    def _evaluate_configuration(self, configuration) -> Tuple[float, float]:
        runtime_tracker = RuntimeTracker()
        performance = self.objective_function(configuration=configuration)
        runtime = runtime_tracker.return_runtime()
        return performance, runtime

    def _random_search(
        self,
        n_searches: int,
        max_runtime: Optional[int] = None,
        max_iter: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Trial]:
        """Perform random search phase"""
        rs_trials = []

        # Get available configurations
        available_configs = self.config_manager.get_searchable_configurations()
        adj_n_searches = min(n_searches, len(available_configs))

        if adj_n_searches == 0:
            logger.warning("No configurations available for random search")
            rs_trials = []

        # Randomly sample configurations
        search_idxs = np.random.choice(
            len(available_configs), size=adj_n_searches, replace=False
        )
        sampled_configs = [available_configs[idx] for idx in search_idxs]

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
                    "Obtained non-numerical performance, skipping configuration."
                )
                self.config_manager.add_to_banned_configurations(config)
                continue

            # Update search state
            self.config_manager.mark_as_searched(config, validation_performance)

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
            searchable_count = len(self.config_manager.get_searchable_configurations())
            current_runtime = None
            if max_runtime and hasattr(self, "search_timer"):
                current_runtime = self.search_timer.return_runtime()

            stop, stop_reason = check_early_stopping(
                searchable_count=searchable_count,
                current_runtime=current_runtime,
                runtime_budget=max_runtime,
                current_iter=len(self.study.trials) + len(rs_trials),
                max_iter=max_iter,
            )
            if stop:
                if "runtime budget" in stop_reason.lower():
                    raise RuntimeError(
                        "confopt preliminary random search exceeded total runtime budget. "
                        "Retry with larger runtime budget or set iteration-capped budget instead."
                    )
                else:
                    logger.info(f"Random search stopping early: {stop_reason}")
                    break

        return rs_trials

    def _select_next_configuration(
        self, searcher, available_configs, tabularized_configs=None
    ):
        """Select the next best configuration to evaluate"""
        if not available_configs:
            return None

        # Use provided tabularized configs or generate them
        if tabularized_configs is None:
            tabularized_configs = self.config_manager.get_tabularized_configs(
                available_configs
            )

        # Get predictions from searcher
        parameter_performance_bounds = searcher.predict(X=tabularized_configs)

        # Find configuration with best predicted performance
        best_idx = np.argmin(parameter_performance_bounds)
        return available_configs[best_idx]

    def _conformal_search(
        self,
        searcher: BaseConformalSearcher,
        n_random_searches,
        conformal_retraining_frequency,
        verbose,
        max_iter,
        runtime_budget,
        searcher_tuning_framework=None,
    ):
        """Perform conformal search phase"""
        # Setup progress bar
        progress_bar = None
        if verbose:
            if runtime_budget is not None:
                progress_bar = tqdm(total=runtime_budget, desc="Conformal search: ")
            elif max_iter is not None:
                progress_bar = tqdm(
                    total=max_iter - len(self.study.trials), desc="Conformal search: "
                )

        # Set up scaler for standardization
        scaler = StandardScaler()

        # Calculate maximum iterations
        if max_iter is not None:
            max_iterations = max_iter - len(self.study.trials)
        else:
            max_iterations = float("inf")

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
        search_model_retuning_frequency = conformal_retraining_frequency  # Must be multiple of conformal_retraining_frequency
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

            # Get available configurations for this iteration
            available_configs = self.config_manager.get_searchable_configurations()

            if not available_configs:
                logger.warning("No more unique configurations to search. Stopping.")
                break

            # Get tabularized representations
            tabularized_searched = self.config_manager.get_tabularized_configs(
                self.config_manager.searched_configs
            )

            # Check if we have enough data for conformal search
            if len(tabularized_searched) < 2:
                logger.warning(
                    f"Insufficient data for conformal search (only {len(tabularized_searched)} samples). Skipping iteration."
                )
                continue

            # Prepare data for conformal search
            validation_split = self._set_conformal_validation_split(
                X=tabularized_searched
            )

            # Split data for training
            X_train, y_train, X_val, y_val = process_and_split_estimation_data(
                searched_configurations=tabularized_searched,
                searched_performances=np.array(
                    self.config_manager.searched_performances
                ),
                train_split=(1 - validation_split),
                filter_outliers=False,
            )

            # Check if we have enough training data
            if len(X_train) == 0:
                logger.warning(
                    "No training data available after split. Skipping iteration."
                )
                continue

            # Apply metric sign for optimization direction
            y_train = y_train * self.metric_sign
            y_val = y_val * self.metric_sign

            # Scale the data
            scaler.fit(X=X_train)
            X_train = scaler.transform(X=X_train)
            X_val = (
                scaler.transform(X=X_val)
                if len(X_val) > 0
                else np.array([]).reshape(0, X_train.shape[1])
            )

            # Transform available configurations
            tabularized_available = self.config_manager.get_tabularized_configs(
                available_configs
            )
            tabularized_available = scaler.transform(X=tabularized_available)

            # Retrain the searcher if needed
            searcher_runtime = None
            estimator_error = None
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
                estimator_error = searcher.primary_estimator_error
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
            next_config = self._select_next_configuration(
                searcher, available_configs, tabularized_available
            )

            if next_config is None:
                logger.warning("No more configurations to search.")
                break

            # Evaluate the selected configuration
            validation_performance, _ = self._evaluate_configuration(next_config)
            logger.debug(
                f"Conformal search iter {search_iter} performance: {validation_performance}"
            )

            if np.isnan(validation_performance):
                self.config_manager.add_to_banned_configurations(next_config)
                continue

            # Calculate breach for logging/tracking
            breach = None
            if isinstance(
                searcher.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)
            ):
                config_tabularized = self.config_manager.get_tabularized_configs(
                    [next_config]
                )
                transformed_X = scaler.transform(config_tabularized)
                breach = searcher.calculate_breach(
                    X=transformed_X, y_true=self.metric_sign * validation_performance
                )

            # Update searcher
            config_tabularized = self.config_manager.get_tabularized_configs(
                [next_config]
            )
            transformed_X = scaler.transform(config_tabularized)
            searcher.update(
                X=transformed_X, y_true=self.metric_sign * validation_performance
            )

            # Update search state
            self.config_manager.mark_as_searched(next_config, validation_performance)

            # Create and add trial
            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=next_config.copy(),
                performance=validation_performance,
                acquisition_source=str(searcher),
                searcher_runtime=searcher_runtime,
                breached_interval=breach,
                primary_estimator_error=estimator_error,
            )
            self.study.append_trial(trial)

            # Check for early stopping
            searchable_count = len(self.config_manager.get_searchable_configurations())
            stop, stop_reason = check_early_stopping(
                searchable_count=searchable_count,
                current_runtime=self.search_timer.return_runtime(),
                runtime_budget=runtime_budget,
                current_iter=len(self.study.trials),
                max_iter=max_iter,
            )
            if stop:
                logger.info(f"Conformal search stopping early: {stop_reason}")
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
            self.config_manager.dynamic_sampling = dynamic_sampling

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

        # Calculate remaining random searches after warm starts
        n_warm_starts = (
            len(self.warm_start_configurations) if self.warm_start_configurations else 0
        )
        remaining_random_searches = max(0, n_random_searches - n_warm_starts)

        logger.debug(
            f"Warm starts: {n_warm_starts}, Required random searches: {n_random_searches}, Remaining: {remaining_random_searches}"
        )

        # Perform random search only if needed
        if remaining_random_searches > 0:
            rs_trials = self._random_search(
                n_searches=remaining_random_searches,
                max_runtime=runtime_budget,
                max_iter=max_iter,
                verbose=verbose,
            )
            self.study.batch_append_trials(trials=rs_trials)

        # Perform conformal search
        self._conformal_search(
            searcher=searcher,
            n_random_searches=n_random_searches,
            conformal_retraining_frequency=conformal_retraining_frequency,
            verbose=verbose,
            max_iter=max_iter,
            runtime_budget=runtime_budget,
            searcher_tuning_framework=searcher_tuning_framework,
        )

    def get_best_params(self) -> Dict:
        return self.study.get_best_configuration()

    def get_best_value(self) -> float:
        return self.study.get_best_performance()
