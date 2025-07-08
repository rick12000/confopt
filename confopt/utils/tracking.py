import logging
import time
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Literal
from confopt.wrapping import ParameterRange
import numpy as np
from confopt.utils.configurations.encoding import ConfigurationEncoder
from confopt.utils.configurations.sampling import get_tuning_configurations
from confopt.utils.configurations.utils import create_config_hash
from tqdm import tqdm


logger = logging.getLogger(__name__)


class RuntimeTracker:
    def __init__(self):
        self.start_time = time.time()
        self.runtime = 0

    def _elapsed_runtime(self):
        take_time = time.time()
        return abs(take_time - self.start_time)

    def pause_runtime(self):
        self.runtime = self.runtime + self._elapsed_runtime()

    def resume_runtime(self):
        self.start_time = time.time()

    def return_runtime(self):
        self.pause_runtime()
        taken_runtime = self.runtime
        self.resume_runtime()
        return taken_runtime


class ProgressBarManager:
    """Manages progress bar creation, updates, and closure for search operations"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.progress_bar = None

    def create_progress_bar(
        self,
        max_runtime: Optional[int] = None,
        max_iter: Optional[int] = None,
        current_trials: int = 0,
        description: str = "Search progress",
    ) -> None:
        """Create appropriate progress bar based on constraints"""
        if self.verbose:
            if max_runtime is not None:
                self.progress_bar = tqdm(total=max_runtime, desc=f"{description}: ")
            elif max_iter is not None:
                remaining_iter = max_iter - current_trials
                if remaining_iter > 0:
                    self.progress_bar = tqdm(
                        total=remaining_iter, desc=f"{description}: "
                    )

    def update_progress(
        self, current_runtime: Optional[float] = None, iteration_count: int = 1
    ) -> None:
        """Update progress bar based on available metrics"""
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
        """Close progress bar and cleanup"""
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None


class Trial(BaseModel):
    iteration: int
    timestamp: datetime
    configuration: dict
    performance: float
    acquisition_source: Optional[str] = None
    breached_interval: Optional[bool] = None
    searcher_runtime: Optional[float] = None
    target_model_runtime: Optional[float] = None
    primary_estimator_error: Optional[float] = None


class Study:
    def __init__(
        self, metric_optimization: Literal["minimize", "maximize"] = "minimize"
    ):
        self.trials: list[Trial] = []
        self.metric_optimization = metric_optimization

    def append_trial(self, trial: Trial):
        self.trials.append(trial)

    def batch_append_trials(self, trials: list[Trial]):
        self.trials.extend(trials)

    def get_searched_configurations(self) -> list[dict]:
        searched_configurations = []
        for trial in self.trials:
            searched_configurations.append(trial.configuration)
        return searched_configurations

    def get_searched_performances(self) -> list[dict]:
        searched_performances = []
        for trial in self.trials:
            searched_performances.append(trial.performance)
        return searched_performances

    def get_best_configuration(self) -> dict:
        searched_configurations = []
        for trial in self.trials:
            searched_configurations.append((trial.configuration, trial.performance))

        if self.metric_optimization == "minimize":
            best_config, _ = min(searched_configurations, key=lambda x: x[1])
        else:  # maximize
            best_config, _ = max(searched_configurations, key=lambda x: x[1])
        return best_config

    def get_best_performance(self) -> float:
        searched_performances = []
        for trial in self.trials:
            searched_performances.append(trial.performance)

        if self.metric_optimization == "minimize":
            return min(searched_performances)
        else:  # maximize
            return max(searched_performances)

    def get_average_target_model_runtime(self) -> float:
        target_model_runtimes = []
        for trial in self.trials:
            if trial.target_model_runtime is not None:
                target_model_runtimes.append(trial.target_model_runtime)
        return sum(target_model_runtimes) / len(target_model_runtimes)


class BaseConfigurationManager:
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
        self.encoder = ConfigurationEncoder(search_space=self.search_space)

    def mark_as_searched(self, config: dict, performance: float) -> None:
        config_hash = create_config_hash(config)
        self.searched_configs.append(config)
        self.searched_performances.append(performance)
        self.searched_config_hashes.add(config_hash)

    def tabularize_configs(self, configs: list[dict]) -> np.array:
        if not configs:
            return np.array([])
        return self.encoder.transform(configs).to_numpy()

    def add_to_banned_configurations(self, config: dict) -> None:
        # Add configuration to banned list if not already present
        config_hash = create_config_hash(config)
        if config_hash not in [
            create_config_hash(c) for c in self.banned_configurations
        ]:
            self.banned_configurations.append(config)


class StaticConfigurationManager(BaseConfigurationManager):
    def __init__(
        self,
        search_space: dict[str, ParameterRange],
        n_candidate_configurations: int,
    ) -> None:
        super().__init__(search_space, n_candidate_configurations)
        self.cached_searchable_configs = []
        self._initialize_static_configs_and_encoder()

    def _initialize_static_configs_and_encoder(self) -> None:
        # NOTE: Overfill n_configurations to avoid losing configurations during
        # searched config filtering, then filter down to actual n_configurations at the end:
        candidate_configurations = get_tuning_configurations(
            parameter_grid=self.search_space,
            n_configurations=self.n_candidate_configurations
            + len(self.searched_configs),
            random_state=None,
            sampling_method="uniform",
        )[: self.n_candidate_configurations]
        filtered_configs = []
        for config in candidate_configurations:
            config_hash = create_config_hash(config)
            if config_hash not in self.searched_config_hashes:
                filtered_configs.append(config)
        self.cached_searchable_configs = filtered_configs
        self._setup_encoder()

    def get_searchable_configurations(self) -> list[dict]:
        # Remove already searched and banned configs from cache
        banned_hashes = set(create_config_hash(c) for c in self.banned_configurations)
        self.cached_searchable_configs = [
            c
            for c in self.cached_searchable_configs
            if create_config_hash(c) not in self.searched_config_hashes
            and create_config_hash(c) not in banned_hashes
        ]
        return self.cached_searchable_configs.copy()

    def mark_as_searched(self, config: dict, performance: float) -> None:
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
        search_space: dict[str, ParameterRange],
        n_candidate_configurations: int,
    ) -> None:
        super().__init__(search_space, n_candidate_configurations)
        self._setup_encoder()

    def get_searchable_configurations(self) -> list[dict]:
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
