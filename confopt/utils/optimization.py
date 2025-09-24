import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DecayingSearcherOptimizer:
    """Searcher optimizer that increases tuning_interval as search progresses.

    This optimizer implements a decaying strategy where the tuning interval
    starts at an initial value and increases over time according to various
    decay rate options. The n_tuning_episodes remains constant throughout
    the search process. The conformal model retrains every iteration (frequency = 1).

    Args:
        n_tuning_episodes (int): Number of tuning episodes to perform at each
            optimization step. Defaults to 10.
        initial_tuning_interval (int): Initial tuning interval to decay from.
            Must be a positive integer. Defaults to 1.
        decay_rate (float): Rate of decay - higher values mean faster increase
            in tuning interval. Defaults to 0.1.
        decay_type (str): Type of decay function. Must be one of 'linear',
            'exponential', or 'logarithmic'. Defaults to 'linear'.
        max_tuning_interval (int): Maximum tuning interval cap to prevent
            excessive intervals. Defaults to 20.

    Attributes:
        current_iter (int): Current search iteration number.

    Note:
        The decay functions are:
        - Linear: interval = initial + decay_rate * iter
        - Exponential: interval = initial * (1 + decay_rate)^iter
        - Logarithmic: interval = initial + decay_rate * log(1 + iter)

        All intervals are rounded to integers.
    """

    def __init__(
        self,
        n_tuning_episodes: int = 10,
        initial_tuning_interval: int = 1,
        decay_rate: float = 0.1,
        decay_type: str = "linear",
        max_tuning_interval: int = 20,
    ):
        self.n_tuning_episodes = n_tuning_episodes
        self.initial_tuning_interval = initial_tuning_interval
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        self.max_tuning_interval = max_tuning_interval
        self.current_iter = 0

        # Validate decay_type
        if decay_type not in ["linear", "exponential", "logarithmic"]:
            raise ValueError(
                "decay_type must be one of 'linear', 'exponential', 'logarithmic'"
            )

    def _calculate_current_interval(self, search_iter: int) -> int:
        """Calculate the current tuning interval based on search iteration.

        Args:
            search_iter (int): Current search iteration number.

        Returns:
            int: Calculated tuning interval, rounded to integer.
        """
        if self.decay_type == "linear":
            # Linear increase: interval = initial + decay_rate * iter
            interval = self.initial_tuning_interval + self.decay_rate * search_iter
        elif self.decay_type == "exponential":
            # Exponential increase: interval = initial * (1 + decay_rate)^iter
            interval = self.initial_tuning_interval * (
                (1 + self.decay_rate) ** search_iter
            )
        elif self.decay_type == "logarithmic":
            # Logarithmic increase: interval = initial + decay_rate * log(1 + iter)
            interval = self.initial_tuning_interval + self.decay_rate * np.log(
                1 + search_iter
            )

        # Cap at maximum interval
        interval = min(interval, self.max_tuning_interval)

        # Round to integer and ensure minimum interval
        interval = max(int(round(interval)), 1)

        return interval

    def update(self, search_iter: Optional[int] = None) -> None:
        """Update the optimizer with search iteration information.

        Args:
            search_iter (int, optional): Current search iteration number. If provided,
                updates the internal iteration counter used for decay calculations.
        """
        if search_iter is not None:
            self.current_iter = search_iter

    def select_arm(self) -> Tuple[int, int]:
        """Select the tuning count and interval based on current decay strategy.

        Returns:
            tuple[int, int]: Tuple containing (n_tuning_episodes, current_tuning_interval).
                The tuning interval is calculated based on the current iteration
                and decay parameters.
        """
        current_interval = self._calculate_current_interval(self.current_iter)
        return (self.n_tuning_episodes, current_interval)


class FixedSearcherOptimizer:
    """Fixed searcher optimizer with constant tuning parameters.

    This optimizer returns fixed tuning parameters regardless of search progress.
    Useful as a baseline or when consistent tuning behavior is desired.
    The conformal model retrains every iteration (frequency = 1).

    Args:
        n_tuning_episodes (int): Number of tuning episodes to perform at each
            optimization step. Defaults to 10.
        tuning_interval (int): Fixed tuning interval to use throughout optimization.
            Defaults to 5.

    Attributes:
        fixed_count (int): Fixed number of tuning episodes.
        fixed_interval (int): Fixed tuning interval.
    """

    def __init__(
        self,
        n_tuning_episodes: int = 10,
        tuning_interval: int = 5,
    ):
        self.fixed_count = n_tuning_episodes
        self.fixed_interval = tuning_interval

    def select_arm(self) -> Tuple[int, int]:
        """Select the fixed tuning count and interval.

        Returns:
            tuple[int, int]: Tuple containing (fixed_count, fixed_interval).
        """
        return self.fixed_count, self.fixed_interval

    def update(self, search_iter: Optional[int] = None) -> None:
        """Update method that accepts search_iter for API compatibility.

        This method does nothing for the fixed optimizer but maintains
        the same interface as other optimizers.

        Args:
            search_iter (int, optional): Current search iteration number.
                Ignored by this optimizer.
        """
