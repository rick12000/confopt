import logging
import time
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

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
    def __init__(self):
        self.trials: list[Trial] = []

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
        best_config, _ = min(searched_configurations, key=lambda x: x[1])
        return best_config

    def get_best_performance(self) -> float:
        searched_performances = []
        for trial in self.trials:
            searched_performances.append(trial.performance)
        return min(searched_performances)

    def get_average_target_model_runtime(self) -> float:
        target_model_runtimes = []
        for trial in self.trials:
            if trial.target_model_runtime is not None:
                target_model_runtimes.append(trial.target_model_runtime)
        return sum(target_model_runtimes) / len(target_model_runtimes)


def derive_optimal_tuning_count(
    target_model_runtime: float,
    search_model_runtime: float,
    search_model_retraining_freq: int,
    search_to_baseline_runtime_ratio: float,
) -> int:
    """
    Derives the optimal number of tuning evaluations to perform on a search model.

    The number of evaluations will satisfy a specified runtime ratio between
    the search model and the baseline model being optimized by it.

    Parameters
    ----------
    baseline_model_runtime :
        Baseline model training time (per training event).
    search_model_runtime :
        Search model training time (per training event).
    search_model_retraining_freq :
        Search model retraining frequency. Determines how often the
        search model will be retrained and thus re-tuned.
    search_to_baseline_runtime_ratio :
        Desired ratio between the total training time of the search
        model and the baseline model. A ratio > 1 indicates the search
        model is allowed to train for longer than the baseline model
        and vice versa. The number of tuning evaluations will be set
        to ensure the runtime ratio is met (or closely matched).

    Returns
    -------
    search_model_tuning_count :
        Optimal number of search model tuning evaluations, given runtime
        ratio constraint.
    """
    margin_of_error_runtime = 0.0001
    target_model_runtime = max(target_model_runtime, margin_of_error_runtime)
    search_model_runtime = max(search_model_runtime, margin_of_error_runtime)
    search_model_tuning_count = (
        target_model_runtime * search_model_retraining_freq
    ) / (search_model_runtime * (1 / search_to_baseline_runtime_ratio) ** 2)

    # Hard coded number of maximum useful evaluations (arbitrary):
    count_ceiling = 60
    search_model_tuning_count = min(
        count_ceiling, max(1, int(round(search_model_tuning_count)))
    )

    logger.debug(f"Optimal search model param evaluations: {search_model_tuning_count}")

    return search_model_tuning_count
