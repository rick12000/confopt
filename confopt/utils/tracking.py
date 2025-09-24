import logging
import time
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Literal
from confopt.wrapping import ParameterRange
import numpy as np
from confopt.utils.configurations.encoding import ConfigurationEncoder
from confopt.utils.configurations.sampling import get_tuning_configurations
from tqdm import tqdm
from confopt.utils.configurations.utils import create_config_hash


logger = logging.getLogger(__name__)


class RuntimeTracker:
    """
    Tracks wall-clock runtime for iterative search or training processes.

    Used to measure elapsed time for optimization or model training, supporting
    pause/resume semantics for accurate accounting in multi-stage workflows.
    """

    def __init__(self):
        self.start_time = time.time()
        self.runtime = 0

    def _elapsed_runtime(self):
        """
        Returns the elapsed time since the last start or resume.

        Returns:
            Elapsed time in seconds.
        """
        take_time = time.time()
        return abs(take_time - self.start_time)

    def pause_runtime(self):
        """
        Accumulates elapsed time into the runtime counter and pauses tracking.
        """
        self.runtime = self.runtime + self._elapsed_runtime()

    def resume_runtime(self):
        """
        Resumes runtime tracking from the current time.
        """
        self.start_time = time.time()

    def return_runtime(self):
        """
        Returns the total accumulated runtime, including the current interval.

        Returns:
            Total runtime in seconds.
        """
        self.pause_runtime()
        taken_runtime = self.runtime
        self.resume_runtime()
        return taken_runtime


class ProgressBarManager:
    """
    Manages progress bar creation, updates, and closure for search operations.

    Integrates with tqdm to provide runtime- or iteration-based progress feedback
    during optimization or training loops. Used in tuning workflows to visualize
    progress and support user feedback.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.progress_bar = None

    def create_progress_bar(
        self,
        max_runtime: Optional[int] = None,
        max_searches: Optional[int] = None,
        current_trials: int = 0,
        description: str = "Search progress",
    ) -> None:
        """
        Initializes a progress bar based on runtime or iteration constraints.

        Args:
            max_runtime: Maximum allowed runtime in seconds.
            max_searches: Maximum number of iterations.
            current_trials: Number of completed trials (for offsetting
                iteration progress).
            description: Description for the progress bar.
        """
        if self.verbose:
            if max_runtime is not None:
                self.progress_bar = tqdm(total=max_runtime, desc=f"{description}: ")
            elif max_searches is not None:
                remaining_iter = max_searches - current_trials
                if remaining_iter > 0:
                    self.progress_bar = tqdm(
                        total=remaining_iter, desc=f"{description}: "
                    )

    def update_progress(
        self, current_runtime: Optional[float] = None, iteration_count: int = 1
    ) -> None:
        """
        Updates the progress bar based on runtime or iteration increments.

        Args:
            current_runtime: Current elapsed runtime in seconds.
            iteration_count: Number of iterations to increment (if not
                runtime-based).
        """
        if self.progress_bar:
            if current_runtime is not None:
                # Runtime-based progress
                new_progress = int(current_runtime) - self.progress_bar.n
                if new_progress > 0:
                    self.progress_bar.update(new_progress)
            else:
                # Iteration-based progress
                self.progress_bar.update(iteration_count)

    def close_progress_bar(self) -> None:
        """
        Closes and cleans up the progress bar.
        """
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None


class Trial(BaseModel):
    """
    Represents a single experiment trial in a hyperparameter search.

    Captures configuration, performance, timing, and metadata for each evaluation.
    Used for experiment logging, analysis, and reproducibility.
    """

    iteration: int
    timestamp: datetime
    configuration: dict
    tabularized_configuration: list[float]
    performance: float
    acquisition_source: Optional[str] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    searcher_runtime: Optional[float] = None
    target_model_runtime: Optional[float] = None


class Study:
    """
    Aggregates and manages a collection of experiment trials.

    Provides methods for appending, querying, and analyzing trials, including best
    configuration selection and runtime statistics. Used as the main experiment
    log in tuning workflows.
    """

    def __init__(
        self, metric_optimization: Literal["minimize", "maximize"] = "minimize"
    ):
        self.trials: list[Trial] = []
        self.metric_optimization = metric_optimization

    def append_trial(self, trial: Trial):
        """
        Appends a single trial to the study log.

        Args:
            trial: Trial object to append.
        """
        self.trials.append(trial)

    def batch_append_trials(self, trials: list[Trial]):
        """
        Appends multiple trials to the study log.

        Args:
            trials: List of Trial objects to append.
        """
        self.trials.extend(trials)

    def get_searched_configurations(self) -> list[dict]:
        """
        Returns a list of all configurations evaluated in the study.

        Returns:
            List of configuration dictionaries.
        """
        searched_configurations = []
        for trial in self.trials:
            searched_configurations.append(trial.configuration)
        return searched_configurations

    def get_searched_performances(self) -> list[dict]:
        """
        Returns a list of all performance values from the study.

        Returns:
            List of performance values.
        """
        searched_performances = []
        for trial in self.trials:
            searched_performances.append(trial.performance)
        return searched_performances

    def get_best_configuration(self) -> dict:
        """
        Returns the configuration with the best performance according to the
        optimization direction.

        Returns:
            Best configuration dictionary.
        """
        searched_configurations = []
        for trial in self.trials:
            searched_configurations.append((trial.configuration, trial.performance))

        if self.metric_optimization == "minimize":
            best_config, _ = min(searched_configurations, key=lambda x: x[1])
        else:  # maximize
            best_config, _ = max(searched_configurations, key=lambda x: x[1])
        return best_config

    def get_best_performance(self) -> float:
        """
        Returns the best performance value according to the optimization
        direction.

        Returns:
            Best performance value.
        """
        searched_performances = []
        for trial in self.trials:
            searched_performances.append(trial.performance)

        if self.metric_optimization == "minimize":
            return min(searched_performances)
        else:  # maximize
            return max(searched_performances)

    def get_average_target_model_runtime(self) -> float:
        """
        Returns the average runtime of the target model across all trials.

        Returns:
            Average runtime in seconds.
        """
        target_model_runtimes = []
        for trial in self.trials:
            if trial.target_model_runtime is not None:
                target_model_runtimes.append(trial.target_model_runtime)
        return sum(target_model_runtimes) / len(target_model_runtimes)


class BaseConfigurationManager:
    """
    Abstract base class for configuration management in search workflows.

    Handles tracking of searched, banned, and candidate configurations, and
    provides tabularization for model input. Used as a base for static and
    dynamic configuration managers.
    """

    def __init__(
        self,
        search_space: dict[str, ParameterRange],
        n_candidate_configurations: int,
    ) -> None:
        self.search_space = search_space
        self.n_candidate_configurations = n_candidate_configurations
        self.searched_configs = []
        self.searched_performances = []
        self.searched_config_hashes = set()
        self.encoder = None
        self.banned_configurations = []

    def _setup_encoder(self) -> None:
        """
        Initializes the configuration encoder for tabularization.
        """
        self.encoder = ConfigurationEncoder(search_space=self.search_space)

    def mark_as_searched(self, config: dict, performance: float) -> None:
        """
        Marks a configuration as searched and records its performance.

        Args:
            config: Configuration dictionary.
            performance: Observed performance value.
        """
        config_hash = create_config_hash(config)
        self.searched_configs.append(config)
        self.searched_performances.append(performance)
        self.searched_config_hashes.add(config_hash)

    def tabularize_configs(self, configs: list[dict]) -> np.array:
        """
        Converts a list of configuration dictionaries to a tabular numpy array for
        model input.

        Args:
            configs: List of configuration dictionaries.
        Returns:
            Tabularized configuration array.
        """
        if not configs:
            return np.array([])
        return self.encoder.transform(configs).to_numpy()

    def listify_configs(self, configs: list[dict]) -> list[list[float]]:
        """
        Converts a list of configuration dictionaries to lists of numerical values.

        Args:
            configs: List of configuration dictionaries to convert.
        Returns:
            List of lists, where each inner list contains numerical values
            in the same order as DataFrame columns.
        """
        if not configs:
            return []
        if self.encoder is None:
            self._setup_encoder()
        tabularized = self.encoder.transform(configs).to_numpy()
        return [row.tolist() for row in tabularized]

    def add_to_banned_configurations(self, config: dict) -> None:
        """
        Adds a configuration to the banned list if not already present.

        Args:
            config: Configuration dictionary to ban.
        """
        config_hash = create_config_hash(config)
        if config_hash not in [
            create_config_hash(c) for c in self.banned_configurations
        ]:
            self.banned_configurations.append(config)


class StaticConfigurationManager(BaseConfigurationManager):
    """
    Manages a static set of candidate configurations for search.

    Precomputes and caches candidate configurations, filtering out searched and
    banned ones. Used for search strategies where the candidate pool is fixed.

    Optimized with set-based tracking for O(1) operations and intelligent caching.
    """

    def __init__(
        self,
        search_space: dict[str, ParameterRange],
        n_candidate_configurations: int,
    ) -> None:
        super().__init__(search_space, n_candidate_configurations)

        # Core optimization: set-based tracking for O(1) operations
        self.searched_indices = set()
        self.banned_indices = set()

        # Pre-computed data for efficiency
        self.all_candidate_configs = []
        self.config_to_index = {}  # Hash -> index mapping

        # Simple caching
        self._searchable_configs_cache = None
        self._cache_valid = False

        self._initialize_static_configs_and_encoder()

    def _initialize_static_configs_and_encoder(self) -> None:
        """
        Initializes the static candidate configuration pool and encoder.
        """
        # Generate all candidate configurations
        self.all_candidate_configs = get_tuning_configurations(
            parameter_grid=self.search_space,
            n_configurations=self.n_candidate_configurations,
            random_state=None,
            sampling_method="uniform",
        )

        # Setup encoder
        self._setup_encoder()

        # Build hash-to-index mapping for O(1) lookups
        for i, config in enumerate(self.all_candidate_configs):
            config_hash = create_config_hash(config)
            self.config_to_index[config_hash] = i

    def mark_as_searched(self, config: dict, performance: float) -> None:
        """
        Marks a configuration as searched using optimized O(1) operations.

        Args:
            config: Configuration dictionary.
            performance: Observed performance value.
        """
        config_hash = create_config_hash(config)

        # Use index tracking for pre-computed configs
        if config_hash in self.config_to_index:
            idx = self.config_to_index[config_hash]
            self.searched_indices.add(idx)

        # Update base class tracking
        super().mark_as_searched(config, performance)

        # Invalidate cache
        self._cache_valid = False

    def add_to_banned_configurations(self, config: dict) -> None:
        """
        Adds a configuration to the banned list using O(1) operations.

        Args:
            config: Configuration dictionary to ban.
        """
        config_hash = create_config_hash(config)

        # Use index tracking for pre-computed configs
        if config_hash in self.config_to_index:
            idx = self.config_to_index[config_hash]
            self.banned_indices.add(idx)

        # Update base class tracking
        super().add_to_banned_configurations(config)

        # Invalidate cache
        self._cache_valid = False

    def get_searchable_configurations(self) -> list[dict]:
        """
        Returns the list of candidate configurations not yet searched or banned
        using optimized set operations and caching.

        Returns:
            List of configuration dictionaries.
        """
        if self._cache_valid and self._searchable_configs_cache is not None:
            return self._searchable_configs_cache.copy()

        # Use set operations for O(1) filtering
        excluded_indices = self.searched_indices | self.banned_indices
        self._searchable_configs_cache = [
            self.all_candidate_configs[i]
            for i in range(len(self.all_candidate_configs))
            if i not in excluded_indices
        ]
        self._cache_valid = True

        return self._searchable_configs_cache.copy()

    def get_searchable_configurations_count(self) -> int:
        """
        Returns the count of searchable configurations using O(1) set operations.

        Returns:
            Number of searchable configurations remaining.
        """
        excluded_count = len(self.searched_indices | self.banned_indices)
        return len(self.all_candidate_configs) - excluded_count


class DynamicConfigurationManager(BaseConfigurationManager):
    """
    Dynamically generates candidate configurations for each search iteration.

    Used for search strategies where the candidate pool is not fixed and can
    adapt to search history. Integrates with configuration sampling utilities for
    on-the-fly candidate generation.
    """

    def __init__(
        self,
        search_space: dict[str, ParameterRange],
        n_candidate_configurations: int,
    ) -> None:
        super().__init__(search_space, n_candidate_configurations)
        self.current_searchable_configs = []
        self._setup_encoder()

    def get_searchable_configurations(self) -> list[dict]:
        """
        Generates and returns a list of candidate configurations not yet searched
        or banned.

        Returns:
            List of configuration dictionaries.
        """
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

        # Store current searchable configs for count method
        self.current_searchable_configs = filtered_configs
        return filtered_configs

    def get_searchable_configurations_count(self) -> int:
        """
        Returns the count of searchable configurations from the last call to
        get_searchable_configurations().

        Returns:
            Number of searchable configurations remaining.
        """
        return len(self.current_searchable_configs)
