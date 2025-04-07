import logging
import numpy as np
from typing import Tuple, Optional, List
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class PowerLawTuner:
    def __init__(
        self,
        max_tuning_count: int = 20,
        max_tuning_interval: int = 5,
        conformal_retraining_frequency: int = 1,
        min_observations: int = 3,
        cost_weight: float = 0.5,
        random_state: Optional[int] = None,
    ):
        self.max_tuning_count = max_tuning_count
        self.max_tuning_interval = max_tuning_interval
        self.conformal_retraining_frequency = conformal_retraining_frequency
        self.min_observations = min_observations
        self.cost_weight = cost_weight

        # Calculate valid tuning intervals (multiples of conformal_retraining_frequency)
        self.valid_intervals = [
            i
            for i in range(1, max_tuning_interval + 1)
            if i % self.conformal_retraining_frequency == 0
        ]

        # If no valid intervals found, force at least one valid interval
        if not self.valid_intervals:
            self.valid_intervals = [self.conformal_retraining_frequency]
            logger.warning(
                f"No valid tuning intervals found. Using {self.conformal_retraining_frequency}."
            )

        if random_state is not None:
            np.random.seed(random_state)

        # Observation storage
        self.tuning_counts: List[int] = []
        self.rewards: List[float] = []
        self.costs: List[float] = []
        self.search_iters: List[int] = []

        # Model parameters
        self.power_law_params = None
        self.cost_model_params = None
        self.current_iter = 0

    @staticmethod
    def _power_law(x, a, b, c):
        """Power law function: f(x) = a * x^b + c"""
        return a * np.power(x, b) + c

    @staticmethod
    def _exponential_decay(x, a, b, c):
        """Exponential decay function: f(x) = a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c

    def update(
        self,
        arm: Tuple[int, int],
        reward: float,
        cost: float,
        search_iter: Optional[int] = None,
    ) -> None:
        """Update the model with new observation data"""
        # Update current iteration if provided
        if search_iter is not None:
            self.current_iter = search_iter

        # Extract the tuning count from the arm
        tuning_count = arm[0]

        # Store the observation
        self.tuning_counts.append(tuning_count)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.search_iters.append(self.current_iter)

        # Try to fit models if we have enough data
        self._fit_models()

    def _fit_models(self):
        """Fit power law models to the observations"""
        if len(self.tuning_counts) < self.min_observations:
            return

        try:
            # Convert to numpy arrays
            x = np.array(self.tuning_counts)
            y_reward = np.array(self.rewards)
            y_cost = np.array(self.costs)

            # Try to fit power law to rewards
            # If it fails, try exponential decay
            try:
                self.power_law_params, _ = curve_fit(
                    self._power_law,
                    x,
                    y_reward,
                    bounds=(
                        [0, -5, -np.inf],
                        [np.inf, 0, np.inf],
                    ),  # Enforce diminishing returns with b < 0
                    maxfev=1000,
                )
            except RuntimeError:
                try:
                    # Try exponential decay as fallback
                    self.power_law_params, _ = curve_fit(
                        self._exponential_decay,
                        x,
                        y_reward,
                        bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                        maxfev=1000,
                    )
                    # Use exponential decay for predictions
                    self._predict_improvement = self._predict_improvement_exp
                except RuntimeError:
                    # If all fitting attempts fail, use simple average as fallback
                    logger.warning(
                        "Could not fit diminishing returns model to reward data. Using average."
                    )
                    self.power_law_params = None

            # Try to fit model to costs
            try:
                self.cost_model_params, _ = curve_fit(
                    lambda x, a, b: a * x + b,  # Linear cost model
                    x,
                    y_cost,
                    maxfev=1000,
                )
            except RuntimeError:
                logger.warning("Could not fit cost model. Using average.")
                self.cost_model_params = None

        except Exception as e:
            logger.warning(f"Error fitting models: {e}")
            self.power_law_params = None
            self.cost_model_params = None

    def _predict_improvement(self, x):
        """Predict improvement using power law model"""
        if self.power_law_params is None:
            # If no model, return average reward
            return np.mean(self.rewards) * np.ones_like(x)

        return self._power_law(x, *self.power_law_params)

    def _predict_improvement_exp(self, x):
        """Predict improvement using exponential decay model"""
        if self.power_law_params is None:
            # If no model, return average reward
            return np.mean(self.rewards) * np.ones_like(x)

        return self._exponential_decay(x, *self.power_law_params)

    def _predict_cost(self, x):
        """Predict cost based on tuning count"""
        if self.cost_model_params is None:
            # If no model, return average cost
            return np.mean(self.costs) * np.ones_like(x)

        # Linear cost model
        a, b = self.cost_model_params
        return a * x + b

    def _compute_efficiency(self, counts):
        """Compute efficiency (reward/cost) for different tuning counts"""
        improvements = self._predict_improvement(counts)
        costs = self._predict_cost(counts)

        # Avoid division by zero
        costs = np.maximum(costs, 1e-10)

        return improvements / costs

    def select_arm(self) -> Tuple[int, int]:
        """Select the optimal tuning count and interval"""
        if len(self.tuning_counts) < self.min_observations:
            # Not enough data, select random arm
            count = np.random.randint(1, self.max_tuning_count + 1)
            interval = np.random.choice(self.valid_intervals)
            return (count, interval)

        # Generate all possible tuning counts
        counts = np.arange(1, self.max_tuning_count + 1)

        # Compute efficiency for each count
        efficiency = self._compute_efficiency(counts)

        # Select the count with highest efficiency
        best_count_idx = np.argmax(efficiency)
        best_count = counts[best_count_idx]

        # Select a random valid interval
        best_interval = np.random.choice(self.valid_intervals)

        return (best_count, best_interval)


class FixedSurrogateTuner:
    def __init__(
        self,
        n_tuning_episodes: int = 5,
        tuning_interval: int = 1,
        conformal_retraining_frequency: int = 1,
    ):
        self.fixed_count = n_tuning_episodes

        # Ensure tuning interval is a multiple of conformal_retraining_frequency
        if tuning_interval % conformal_retraining_frequency != 0:
            # Round to nearest valid interval
            nearest_multiple = round(tuning_interval / conformal_retraining_frequency)
            self.fixed_interval = (
                max(1, nearest_multiple) * conformal_retraining_frequency
            )
            logger.warning(
                f"Tuning interval {tuning_interval} is not a multiple of conformal_retraining_frequency {conformal_retraining_frequency}. "
                f"Using {self.fixed_interval} instead."
            )
        else:
            self.fixed_interval = tuning_interval

    def select_arm(self) -> Tuple[int, int]:
        return self.fixed_count, self.fixed_interval

    def update(
        self,
        arm: Tuple[int, int],
        reward: float,
        cost: float,
        search_iter: Optional[int] = None,
    ) -> None:
        """Update method that accepts search_iter for API compatibility"""
