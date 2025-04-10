import logging
import numpy as np
from typing import Tuple, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm

logger = logging.getLogger(__name__)


class BayesianTuner:
    def __init__(
        self,
        max_tuning_count: int = 20,
        max_tuning_interval: int = 5,
        conformal_retraining_frequency: int = 1,
        min_observations: int = 5,  # Changed from 3 to 5
        exploration_weight: float = 0.1,
        random_state: Optional[int] = None,
    ):
        self.max_tuning_count = max_tuning_count
        self.max_tuning_interval = max_tuning_interval
        self.conformal_retraining_frequency = conformal_retraining_frequency
        self.min_observations = min_observations
        self.exploration_weight = exploration_weight
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

        # Observation storage
        self.X_observed = []  # Features: [search_iter, tuning_count, tuning_interval]
        self.y_observed = []  # Target: efficiency (reward/cost)
        self.current_iter = 0

        # Initialize Gaussian Process model with a suitable kernel
        # Matern kernel is good for optimization as it doesn't assume excessive smoothness
        kernel = ConstantKernel() * Matern(nu=2.5, length_scale_bounds=(1e-5, 1e5))
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=random_state,
        )
        self.scaler = StandardScaler()

        # Add efficiency normalization
        self.efficiency_scaler = MinMaxScaler()

        # Flag to indicate if model has been trained
        self.model_trained = False

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

        # Extract the tuning parameters from the arm
        tuning_count, tuning_interval = arm

        # Calculate efficiency directly (reward/cost)
        # Avoid division by zero
        cost = max(cost, 1e-10)
        efficiency = reward / cost

        logger.debug(
            f"Observed efficiency: {efficiency:.4f} (reward={reward:.4f}, cost={cost:.4f})"
        )

        # Store the observation
        self.X_observed.append([self.current_iter, tuning_count, tuning_interval])
        self.y_observed.append(efficiency)

        # Try to fit model if we have enough data
        if len(self.X_observed) >= self.min_observations:
            self._fit_model()

    def _fit_model(self):
        """Fit Gaussian Process model to predict efficiency"""
        if len(self.X_observed) < self.min_observations:
            return

        # Prepare training data
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)

        # Normalize the efficiency values to handle different units
        y_normalized = self.efficiency_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        try:
            # Train Gaussian Process model on normalized data
            self.gp_model.fit(X_scaled, y_normalized)
            self.model_trained = True
            logger.debug(f"GP model trained on {len(self.X_observed)} observations")

        except Exception as e:
            logger.warning(f"Error fitting Gaussian Process model: {e}")
            self.model_trained = False

    def _expected_improvement(self, mean, std, best_f):
        """
        Calculate expected improvement acquisition function

        Args:
            mean: Predicted mean at candidate points
            std: Predicted standard deviation at candidate points
            best_f: Best observed value so far

        Returns:
            Expected improvement values
        """
        # Handle case where std is very small/zero to avoid numerical issues
        std = np.maximum(std, 1e-9)

        # Calculate z-score
        z = (mean - best_f) / std

        # Calculate expected improvement
        phi_z = norm.cdf(z)
        phi_z_pdf = norm.pdf(z)

        ei = (mean - best_f) * phi_z + std * phi_z_pdf

        # Apply exploration weight to balance exploration vs exploitation
        ei = ei * (1 + self.exploration_weight * std)

        return ei

    def select_arm(self) -> Tuple[int, int]:
        """Select the optimal tuning count and interval using Bayesian optimization"""
        if not self.model_trained or len(self.X_observed) < self.min_observations:
            # Not enough data, select random arm
            count = np.random.randint(1, self.max_tuning_count + 1)
            interval = np.random.choice(self.valid_intervals)
            logger.debug(
                f"Insufficient data, selecting random arm: ({count}, {interval})"
            )
            return (count, interval)

        # Generate all possible combinations of tuning count and interval
        # Use current_iter + 1 to predict for the next iteration
        next_iter = self.current_iter + 1
        tuning_counts = np.arange(1, self.max_tuning_count + 1)
        tuning_intervals = np.array(self.valid_intervals)

        all_combinations = []
        for count in tuning_counts:
            for interval in tuning_intervals:
                all_combinations.append([next_iter, count, interval])

        X_candidates = np.array(all_combinations)
        X_candidates_scaled = self.scaler.transform(X_candidates)

        # Predict efficiency mean and standard deviation
        mean_pred, std_pred = self.gp_model.predict(
            X_candidates_scaled, return_std=True
        )

        # Find the best observed normalized value so far
        y_normalized = self.efficiency_scaler.transform(
            np.array(self.y_observed).reshape(-1, 1)
        ).ravel()
        best_observed_value = max(y_normalized) if len(y_normalized) > 0 else 0

        # Calculate expected improvement
        ei = self._expected_improvement(mean_pred, std_pred, best_observed_value)

        # Find the combination with the highest expected improvement
        best_idx = np.argmax(ei)
        _, best_count, best_interval = X_candidates[best_idx]

        logger.debug(
            f"Selected optimal arm for iter {next_iter}: ({int(best_count)}, {int(best_interval)}) with EI={ei[best_idx]:.4f}"
        )
        return (int(best_count), int(best_interval))


class FixedSurrogateTuner:
    def __init__(
        self,
        n_tuning_episodes: int = 10,
        tuning_interval: int = 5,
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
