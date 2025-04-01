import logging
import numpy as np
from typing import Tuple, Optional, Literal
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)


class ParzenSurrogateTuner:
    def __init__(
        self,
        max_tuning_count: int = 20,
        max_tuning_interval: int = 5,
        conformal_retraining_frequency: int = 1,
        acquisition_function: Literal["ei", "ucb", "pi"] = "ei",
        exploration_weight: float = 0.1,
        bandwidth: float = 0.5,
        random_state: Optional[int] = None,
    ):
        self.max_tuning_count = max_tuning_count
        self.max_tuning_interval = max_tuning_interval
        self.conformal_retraining_frequency = conformal_retraining_frequency
        self.acquisition_function = acquisition_function
        self.exploration_weight = exploration_weight
        self.bandwidth = bandwidth
        self.random_state = random_state

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

        # Initialize observations storage
        self.X_observed = np.empty((0, 2))  # [count, interval]
        self.rewards = np.empty((0,))  # rewards
        self.costs = np.empty((0,))  # costs
        self.ratios = np.empty((0,))  # reward/cost ratios
        self.search_iters = np.empty((0,))  # search iterations (contextual feature)

        # Initialize Parzen estimators
        self.reward_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self.cost_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self.ratio_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)

        # Keep track of best observed value
        self.best_observed_value = -np.inf

        # For noise injection to avoid numerical issues
        self.noise_level = 1e-6

        # Current search iteration
        self.current_iter = 0

    def update(
        self,
        arm: Tuple[int, int],
        reward: float,
        cost: float,
        search_iter: Optional[int] = None,
    ) -> None:
        # Update current iteration if provided
        if search_iter is not None:
            self.current_iter = search_iter

        # Calculate reward-to-cost ratio
        ratio = reward / cost if cost > 0 else 0.0

        # Update best observed value
        if ratio > self.best_observed_value:
            self.best_observed_value = ratio

        # Add observation to our dataset
        x = np.array([[arm[0], arm[1]]])

        self.X_observed = np.vstack([self.X_observed, x]) if self.X_observed.size else x
        self.rewards = np.append(self.rewards, reward)
        self.costs = np.append(self.costs, cost)
        self.ratios = np.append(self.ratios, ratio)
        self.search_iters = np.append(self.search_iters, self.current_iter)

        # Fit the KDE models if we have enough observations (at least 2)
        if len(self.ratios) >= 2:
            # Add small noise to avoid identical values which can cause numerical issues
            if np.allclose(self.rewards, self.rewards[0]):
                self.rewards[-1] += self.noise_level
            if np.allclose(self.costs, self.costs[0]):
                self.costs[-1] += self.noise_level
            if np.allclose(self.ratios, self.ratios[0]):
                self.ratios[-1] += self.noise_level

            # Standardize values for better KDE performance
            X_std = self._standardize_features(self.X_observed)
            search_iters_std = self._standardize_iterations(self.search_iters)
            rewards_std = (self.rewards - np.mean(self.rewards)) / (
                np.std(self.rewards) + self.noise_level
            )
            costs_std = (self.costs - np.mean(self.costs)) / (
                np.std(self.costs) + self.noise_level
            )
            ratios_std = (self.ratios - np.mean(self.ratios)) / (
                np.std(self.ratios) + self.noise_level
            )

            try:
                # Fit KDEs on standardized data, including search iteration as contextual feature
                X_with_iter = np.hstack([X_std, search_iters_std.reshape(-1, 1)])
                X_rewards = np.hstack([X_with_iter, rewards_std.reshape(-1, 1)])
                X_costs = np.hstack([X_with_iter, costs_std.reshape(-1, 1)])
                X_ratios = np.hstack([X_with_iter, ratios_std.reshape(-1, 1)])

                self.reward_kde.fit(X_rewards)
                self.cost_kde.fit(X_costs)
                self.ratio_kde.fit(X_ratios)
            except Exception as e:
                logger.warning(f"KDE fitting failed: {e}")

    def _standardize_features(self, X: np.ndarray) -> np.ndarray:
        """Standardize features to [0, 1] range for better KDE performance"""
        result = X.copy()
        # Normalize count
        result[:, 0] = (result[:, 0] - 1) / (self.max_tuning_count - 1)
        # Normalize interval
        result[:, 1] = (result[:, 1] - 1) / (self.max_tuning_interval - 1)
        return result

    def _standardize_iterations(self, iters: np.ndarray) -> np.ndarray:
        """Standardize search iterations for better KDE performance"""
        if len(iters) == 0:
            return np.array([])

        # Find max iteration for normalization
        max_iter = max(100, np.max(iters))  # Use at least 100 to avoid issues early on
        return iters / max_iter

    def _predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and uncertainty for the specified points using current iteration as context

        Returns:
            Tuple of (mean predictions, uncertainty)
        """
        if len(self.ratios) < 2:
            # Not enough data for prediction
            return np.zeros(len(X)), np.ones(len(X))

        # Standardize input features
        X_std = self._standardize_features(X)

        # Add current iteration as a contextual feature (fixed for all arms)
        iter_std = self._standardize_iterations(np.array([self.current_iter]))
        X_with_iter = np.hstack([X_std, np.tile(iter_std, (len(X_std), 1))])

        # For each point, create query points for each possible ratio value
        # This lets us estimate the probability density for different outcomes
        ratios_mean = np.mean(self.ratios)
        ratios_std = np.std(self.ratios) + self.noise_level

        # Create grid of possible standardized ratio values
        ratio_grid = np.linspace(-3, 3, 50)  # -3 to 3 std deviations

        means = np.zeros(len(X))
        uncertainties = np.zeros(len(X))

        for i, x in enumerate(X_with_iter):
            # Create query points combining this X with all possible ratio values
            query_points = np.tile(x, (len(ratio_grid), 1))
            query_points = np.hstack([query_points, ratio_grid.reshape(-1, 1)])

            # Get log density for all these points
            log_density = self.ratio_kde.score_samples(query_points)
            density = np.exp(log_density)

            # Normalize density to get a proper PDF
            density = density / density.sum()

            # Calculate mean and variance
            mean = np.sum(density * ratio_grid) * ratios_std + ratios_mean
            variance = np.sum(density * (ratio_grid - mean / ratios_std) ** 2) * (
                ratios_std**2
            )

            means[i] = mean
            uncertainties[i] = np.sqrt(variance)

        return means, uncertainties

    def _acquisition(self, X: np.ndarray) -> np.ndarray:
        if len(self.ratios) < 2:
            return np.ones(len(X))  # Uniform when not enough data

        mu, sigma = self._predict(X)

        if self.acquisition_function == "ei":
            # Expected Improvement
            improvement = mu - self.best_observed_value
            mask = sigma > 1e-8
            ei = np.zeros_like(improvement)

            if np.any(mask):
                z = np.zeros_like(improvement)
                z[mask] = improvement[mask] / sigma[mask]
                ei[mask] = improvement[mask] * norm.cdf(z[mask]) + sigma[
                    mask
                ] * norm.pdf(z[mask])

            ei[improvement > 0] = improvement[improvement > 0]
            return ei

        elif self.acquisition_function == "ucb":
            # Upper Confidence Bound
            return mu + self.exploration_weight * sigma

        elif self.acquisition_function == "pi":
            # Probability of Improvement
            improvement = mu - self.best_observed_value - self.exploration_weight
            mask = sigma > 1e-8
            pi = np.zeros_like(mu)

            if np.any(mask):
                z = np.zeros_like(improvement)
                z[mask] = improvement[mask] / sigma[mask]
                pi[mask] = norm.cdf(z[mask])

            return pi

        # Default to UCB
        return mu + self.exploration_weight * sigma

    def select_arm(self) -> Tuple[int, int]:
        if len(self.ratios) < 2:
            # Random exploration if not enough data
            count = np.random.randint(1, self.max_tuning_count + 1)
            interval = np.random.choice(
                self.valid_intervals
            )  # Select from valid intervals
            return (count, interval)

        # Generate grid of all possible valid parameter combinations
        counts = np.arange(1, self.max_tuning_count + 1)
        intervals = np.array(self.valid_intervals)  # Only use valid intervals

        grid = []
        for count in counts:
            for interval in intervals:
                grid.append([count, interval])
        grid = np.array(grid)

        # Compute acquisition function values
        acquisition_values = self._acquisition(grid)

        # Select the arm with highest acquisition value
        best_idx = np.argmax(acquisition_values)

        return tuple(grid[best_idx])


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
