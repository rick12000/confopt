import logging
from typing import Optional, Union, List
import numpy as np
from confopt.adaptation import DtACI
from confopt.conformalization import (
    LocallyWeightedConformalEstimator,
    QuantileConformalEstimator,
)
from confopt.data_classes import ConformalBounds
from confopt.sampling import (
    LowerBoundSampler,
    ThompsonSampler,
    PessimisticLowerBoundSampler,
)
from confopt.estimation import initialize_estimator

logger = logging.getLogger(__name__)


class BaseConformalSearcher:
    """Base class for conformal searchers with common functionality"""

    def __init__(
        self,
        sampler: Union[
            LowerBoundSampler, ThompsonSampler, PessimisticLowerBoundSampler
        ],
    ):
        self.sampler = sampler
        self.predictions_per_interval = None
        self.primary_estimator_error = None

    def predict(self, X: np.array):
        """Generic prediction method that delegates to sampler-specific methods"""
        if isinstance(self.sampler, LowerBoundSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)
        else:
            raise ValueError(f"Unsupported sampler type: {type(self.sampler)}")

    def _predict_with_ucb(self, X: np.array):
        """Predict using UCB strategy, to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    def _predict_with_thompson(self, X: np.array):
        """Predict using Thompson sampling, to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        """Predict using pessimistic lower bound, to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    def _get_interval_predictions(self, X: np.array) -> List[ConformalBounds]:
        """Helper method to get predictions for all alphas"""
        raise NotImplementedError("Subclasses must implement this method")

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        """Update interval width based on performance feedback"""
        breaches = []
        for interval in self.predictions_per_interval:
            sampled_lower_bound = interval.lower_bounds[sampled_idx]
            sampled_upper_bound = interval.upper_bounds[sampled_idx]

            # Use the contains method from ConformalInterval
            breach = (
                0
                if (sampled_lower_bound <= sampled_performance)
                & (sampled_performance <= sampled_upper_bound)
                else 1
            )
            breaches.append(breach)

        # Update the sampler with the breach information
        self.sampler.update_interval_width(beta=breaches)


class LocallyWeightedConformalSearcher(BaseConformalSearcher):
    def __init__(
        self,
        point_estimator_architecture: str,
        variance_estimator_architecture: str,
        sampler: Union[
            LowerBoundSampler, ThompsonSampler, PessimisticLowerBoundSampler
        ],
    ):
        super().__init__(sampler)
        self.conformal_estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=point_estimator_architecture,
            variance_estimator_architecture=variance_estimator_architecture,
            alphas=self.sampler.fetch_alphas(),
        )

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        self.conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )
        self.primary_estimator_error = self.conformal_estimator.primary_estimator_error

    def _get_interval_predictions(self, X: np.array) -> List[ConformalBounds]:
        """Helper method to get predictions for all alphas"""
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.predictions_per_interval

    def _predict_with_ucb(self, X: np.array):
        interval_predictions = self._get_interval_predictions(X)

        # Get point estimates for beta scaling
        point_estimate = np.array(
            self.conformal_estimator.pe_estimator.predict(X)
        ).reshape(-1, 1)

        # For standard UCB, just use the first interval
        interval_width = (
            interval_predictions[0].upper_bounds - interval_predictions[0].lower_bounds
        )
        # Apply beta scaling
        tracked_lower_bound = point_estimate - self.sampler.beta * interval_width / 2

        self.sampler.update_exploration_step()
        return tracked_lower_bound

    def _predict_with_thompson(self, X: np.array):
        self._get_interval_predictions(X)

        # Vectorized approach for sampling
        n_samples = X.shape[0]
        n_intervals = len(self.predictions_per_interval)

        # Generate random indices for all samples at once
        interval_indices = np.random.choice(n_intervals, size=n_samples)

        # Extract the lower bounds using vectorized operations
        lower_bounds = np.array(
            [
                self.predictions_per_interval[idx].lower_bounds[i]
                for i, idx in enumerate(interval_indices)
            ]
        )

        return lower_bounds

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        interval_predictions = self._get_interval_predictions(X)

        if isinstance(self.sampler.adapter, DtACI):
            best_alpha = self.sampler.fetch_alphas()[
                0
            ]  # Get first element for PessimisticLowerBoundSampler
            for i, alpha in enumerate(self.sampler.fetch_alphas()):
                # When we find the current best alpha, use its lower bound
                if best_alpha == alpha:
                    result_lower_bound = interval_predictions[i].lower_bounds
                    break
        else:
            # For standard pessimistic approach, use the first interval
            result_lower_bound = interval_predictions[0].lower_bounds

        return result_lower_bound


class QuantileConformalSearcher(BaseConformalSearcher):
    def __init__(
        self,
        quantile_estimator_architecture: str,
        sampler: Union[
            LowerBoundSampler, ThompsonSampler, PessimisticLowerBoundSampler
        ],
        n_pre_conformal_trials: int = 20,
        single_fit: bool = False,
    ):
        super().__init__(sampler)
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.single_fit = single_fit
        self.point_estimator = None

        if isinstance(self.sampler, LowerBoundSampler):
            self.sampler.upper_quantile_cap = 0.5
            self.sampler.quantiles = self.sampler._calculate_quantiles()

        # Create the conformal estimator with alphas from the sampler
        self.conformal_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=quantile_estimator_architecture,
            alphas=self.sampler.fetch_alphas(),
            n_pre_conformal_trials=n_pre_conformal_trials,
        )

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """Fit the conformal estimator."""
        # Initialize and fit optimistic estimator if needed for Thompson sampling
        if (
            isinstance(self.sampler, ThompsonSampler)
            and self.sampler.enable_optimistic_sampling
        ):
            self.point_estimator = initialize_estimator(
                estimator_architecture="gbm",
                random_state=random_state,
            )
            self.point_estimator.fit(
                X=np.vstack((X_train, X_val)),
                y=np.concatenate((y_train, y_val)),
            )

        self.conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )

        self.primary_estimator_error = self.conformal_estimator.primary_estimator_error

    def _get_interval_predictions(self, X: np.array) -> List[ConformalBounds]:
        """Helper method to get predictions for all alphas"""
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.predictions_per_interval

    def _predict_with_ucb(self, X: np.array):
        interval_predictions = self._get_interval_predictions(X)

        # For UCB, use the first interval
        interval = interval_predictions[0]
        interval_width = interval.upper_bounds - interval.lower_bounds

        # Apply beta scaling for exploration
        result_lower_bound = interval.upper_bounds - self.sampler.beta * interval_width

        self.sampler.update_exploration_step()
        return result_lower_bound

    def _predict_with_thompson(self, X: np.array):
        self._get_interval_predictions(X)

        # Vectorized approach for sampling
        n_samples = X.shape[0]
        n_intervals = len(self.predictions_per_interval)

        # Generate random indices for all samples at once
        interval_indices = np.random.choice(n_intervals, size=n_samples)

        # Extract the lower bounds using vectorized operations
        lower_bounds = np.array(
            [
                self.predictions_per_interval[idx].lower_bounds[i]
                for i, idx in enumerate(interval_indices)
            ]
        )

        # Apply optimistic sampling if enabled
        if self.sampler.enable_optimistic_sampling and self.point_estimator is not None:
            median_predictions = self.point_estimator.predict(X)
            lower_bounds = np.minimum(lower_bounds, median_predictions)

        return lower_bounds

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        interval_predictions = self._get_interval_predictions(X)

        # For pessimistic approach, use the first interval's lower bound
        return interval_predictions[0].lower_bounds
