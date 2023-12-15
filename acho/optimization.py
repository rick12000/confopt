import time
import logging

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


def derive_optimal_tuning_count(
    base_model_runtime: float,
    search_model_runtime: float,
    search_retraining_freq: int,
    search_to_base_runtime_ratio: float,
) -> int:
    logger.debug(
        f"RS runtime per iter to C runtime per iter: {base_model_runtime / search_model_runtime}"
    )

    optimal_n_of_secondary_model_param_combinations = (
        base_model_runtime * search_retraining_freq
    ) / (search_model_runtime * (1 / search_to_base_runtime_ratio) ** 2)
    optimal_n_of_secondary_model_param_combinations = max(
        1, int(round(optimal_n_of_secondary_model_param_combinations))
    )
    n_combinations_ceiling = 60
    optimal_n_of_secondary_model_param_combinations = min(
        n_combinations_ceiling, optimal_n_of_secondary_model_param_combinations
    )

    return optimal_n_of_secondary_model_param_combinations
