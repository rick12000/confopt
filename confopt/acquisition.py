import logging
from typing import Optional, Union, Literal
import numpy as np
from confopt.adaptation import DtACI
from confopt.conformalization import (
    LocallyWeightedConformalEstimator,
    SingleFitQuantileConformalEstimator,
    MultiFitQuantileConformalEstimator,
)
from confopt.sampling import (
    LowerBoundSampler,
    ThompsonSampler,
    PessimisticLowerBoundSampler,
)
from confopt.estimation import initialize_estimator

logger = logging.getLogger(__name__)


class LocallyWeightedConformalSearcher:
    def __init__(
        self,
        point_estimator_architecture: str,
        variance_estimator_architecture: str,
        sampler: Union[
            LowerBoundSampler, ThompsonSampler, PessimisticLowerBoundSampler
        ],
    ):
        self.conformal_estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=point_estimator_architecture,
            variance_estimator_architecture=variance_estimator_architecture,
        )
        self.sampler = sampler
        self.predictions_per_interval = None

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

    def predict(self, X: np.array):
        if isinstance(self.sampler, LowerBoundSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)

    def _predict_with_ucb(self, X: np.array):
        point_estimate = np.array(
            self.conformal_estimator.pe_estimator.predict(X)
        ).reshape(-1, 1)
        if isinstance(self.sampler.adapter, DtACI):
            self.predictions_per_interval = []
            for alpha in self.sampler.fetch_expert_alphas():
                (
                    lower_quantile_value,
                    upper_quantile_value,
                ) = self.conformal_estimator.predict_interval(X=X, alpha=alpha)
                # Apply beta scaling for exploration to the lower bound
                lower_bound = (
                    point_estimate
                    + self.sampler.beta
                    * (upper_quantile_value - lower_quantile_value)
                    / 2
                )

                self.predictions_per_interval.append(
                    np.hstack([lower_quantile_value, upper_quantile_value])
                )
                # Use the current best alpha as the bound
                if self.sampler.fetch_alpha() == alpha:
                    tracked_lower_bound = lower_quantile_value

        else:
            alpha = self.sampler.fetch_alpha()
            (
                lower_quantile_value,
                upper_quantile_value,
            ) = self.conformal_estimator.predict_interval(X=X, alpha=alpha)
            # Apply beta scaling for exploration to the lower bound
            lower_bound = (
                point_estimate
                + self.sampler.beta * (lower_quantile_value - upper_quantile_value) / 2
            )

            self.predictions_per_interval = [
                np.hstack([lower_quantile_value, upper_quantile_value])
            ]
            tracked_lower_bound = lower_bound

        self.sampler.update_exploration_step()
        return tracked_lower_bound

    def _predict_with_thompson(self, X: np.array):
        self.predictions_per_interval = []

        # Get all intervals from the Thompson sampler
        intervals = self.sampler.fetch_intervals()

        # Get predictions for all intervals
        for interval in intervals:
            alpha = 1 - (interval.upper_quantile - interval.lower_quantile)
            lower_bound, upper_bound = self.conformal_estimator.predict_interval(
                X=X, alpha=alpha
            )
            self.predictions_per_interval.append(np.hstack([lower_bound, upper_bound]))

        # Vectorized approach for sampling
        n_samples = X.shape[0]
        n_intervals = len(intervals)

        # Generate random indices for all samples at once
        interval_indices = np.random.choice(n_intervals, size=n_samples)

        # Extract the lower bounds using vectorized operations
        lower_bounds = np.array(
            [
                self.predictions_per_interval[idx][i, 0]
                for i, idx in enumerate(interval_indices)
            ]
        )

        return lower_bounds

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        """
        Predict using Pessimistic Lower Bound sampling strategy.
        """
        if isinstance(self.sampler.adapter, DtACI):
            self.predictions_per_interval = []
            for alpha in self.sampler.fetch_expert_alphas():
                lower_bound, upper_bound = self.conformal_estimator.predict_interval(
                    X=X, alpha=alpha
                )
                self.predictions_per_interval.append(
                    np.hstack([lower_bound, upper_bound])
                )
                # Use the current best alpha as the bound
                if self.sampler.fetch_alpha() == alpha:
                    result_lower_bound = lower_bound
        else:
            alpha = self.sampler.fetch_alpha()
            lower_bound, upper_bound = self.conformal_estimator.predict_interval(
                X=X, alpha=alpha
            )
            self.predictions_per_interval = [np.hstack([lower_bound, upper_bound])]
            result_lower_bound = lower_bound

        return result_lower_bound

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        breaches = []
        for predictions in self.predictions_per_interval:
            sampled_predictions = predictions[sampled_idx, :]
            lower_quantile, upper_quantile = (
                sampled_predictions[0],
                sampled_predictions[1],
            )
            if lower_quantile <= sampled_performance <= upper_quantile:
                breach = 0
            else:
                breach = 1
            breaches.append(breach)
        self.sampler.update_interval_width(breaches=breaches)


class SingleFitQuantileConformalSearcher:
    def __init__(
        self,
        quantile_estimator_architecture: Literal["qknn", "qrf"],
        sampler: Union[
            LowerBoundSampler, ThompsonSampler, PessimisticLowerBoundSampler
        ],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
        if isinstance(self.sampler, LowerBoundSampler):
            self.sampler.upper_quantile_cap = 0.5
            self.sampler.quantiles = self.sampler._calculate_quantiles()
        self.n_pre_conformal_trials = n_pre_conformal_trials

        # Determine intervals to use based on the sampler type
        if isinstance(self.sampler, LowerBoundSampler) or isinstance(
            self.sampler, PessimisticLowerBoundSampler
        ):
            intervals = [self.sampler.fetch_quantile_interval()]
        elif isinstance(self.sampler, ThompsonSampler):
            intervals = self.sampler.fetch_intervals()
        else:
            raise ValueError("Unknown sampler type.")

        # Use a single estimator for all intervals
        self.conformal_estimator = SingleFitQuantileConformalEstimator(
            quantile_estimator_architecture=quantile_estimator_architecture,
            intervals=intervals,
            n_pre_conformal_trials=n_pre_conformal_trials,
        )
        self.point_estimator = None
        self.primary_estimator_error = None
        self.predictions_per_interval = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit the single conformal estimator for all intervals.
        """

        # Initialize and fit optimistic estimator if needed
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

    def predict(self, X: np.array):
        if isinstance(self.sampler, LowerBoundSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)

    def _predict_with_ucb(self, X: np.array):
        # Get the interval from the UCB sampler
        interval = self.sampler.fetch_quantile_interval()

        # Predict interval using the single estimator
        (
            lower_interval,
            upper_interval,
        ) = self.conformal_estimator.predict_interval(X=X, interval=interval)

        # Below upper interval needs to be median and lower bound is lower bound from desired CI
        lower_bound = upper_interval - self.sampler.beta * (
            upper_interval - lower_interval
        )

        # Store predictions for later breach checking
        self.predictions_per_interval = [
            np.column_stack((lower_interval, upper_interval))
        ]

        self.sampler.update_exploration_step()
        return lower_bound

    def _predict_with_thompson(self, X: np.array):
        # Get all intervals from the Thompson sampler
        intervals = self.sampler.fetch_intervals()

        # Get predictions for all intervals using the single estimator
        self.predictions_per_interval = []

        for interval in intervals:
            lower_bound, upper_bound = self.conformal_estimator.predict_interval(
                X=X, interval=interval
            )
            self.predictions_per_interval.append(
                np.column_stack((lower_bound, upper_bound))
            )

        # Vectorized approach for sampling
        n_samples = X.shape[0]
        n_intervals = len(intervals)

        # Generate random indices for all samples at once
        interval_indices = np.random.choice(n_intervals, size=n_samples)

        # Extract the lower bounds using vectorized operations
        lower_bounds = np.array(
            [
                self.predictions_per_interval[idx][i, 0]
                for i, idx in enumerate(interval_indices)
            ]
        )

        # Apply optimistic sampling if enabled - do it once for all samples
        if self.sampler.enable_optimistic_sampling and self.point_estimator is not None:
            # Get all median predictions in one call
            median_predictions = self.point_estimator.predict(X)
            lower_bounds = np.minimum(lower_bounds, median_predictions)

        return lower_bounds

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        # Get the interval from the pessimistic sampler
        interval = self.sampler.fetch_quantile_interval()

        # Predict interval using the single estimator
        (
            lower_interval_bound,
            upper_interval_bound,
        ) = self.conformal_estimator.predict_interval(X=X, interval=interval)

        # Store predictions for later breach checking
        self.predictions_per_interval = [
            np.column_stack((lower_interval_bound, upper_interval_bound))
        ]

        return lower_interval_bound

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        breaches = []
        for predictions in self.predictions_per_interval:
            sampled_predictions = predictions[sampled_idx, :]
            lower_bound, upper_bound = sampled_predictions[0], sampled_predictions[1]

            # Check if the actual performance is within the predicted interval
            breach = 0 if lower_bound <= sampled_performance <= upper_bound else 1
            breaches.append(breach)

        # Update the sampler with the breach information
        self.sampler.update_interval_width(breaches=breaches)


class MultiFitQuantileConformalSearcher:
    def __init__(
        self,
        quantile_estimator_architecture: str,
        sampler: Union[
            LowerBoundSampler, ThompsonSampler, PessimisticLowerBoundSampler
        ],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
        if isinstance(self.sampler, LowerBoundSampler):
            self.sampler.upper_quantile_cap = 0.5
            self.sampler.quantiles = self.sampler._calculate_quantiles()
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.point_estimator = None
        self.primary_estimator_error = None
        self.predictions_per_interval = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit the conformal estimators.
        """
        self.conformal_estimators = []

        # Initialize and fit optimistic estimator if needed
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

        # Get intervals from the sampler
        if isinstance(self.sampler, LowerBoundSampler) or isinstance(
            self.sampler, PessimisticLowerBoundSampler
        ):
            intervals = [self.sampler.fetch_quantile_interval()]
        elif isinstance(self.sampler, ThompsonSampler):
            intervals = self.sampler.fetch_intervals()
        else:
            raise ValueError("Unknown sampler type.")

        # Initialize and fit conformal estimators for each interval
        errors = []
        for interval in intervals:
            estimator = MultiFitQuantileConformalEstimator(
                quantile_estimator_architecture=self.quantile_estimator_architecture,
                interval=interval,
                n_pre_conformal_trials=self.n_pre_conformal_trials,
            )
            estimator.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                tuning_iterations=tuning_iterations,
                random_state=random_state,
            )
            self.conformal_estimators.append(estimator)
            errors.append(estimator.primary_estimator_error)

        self.primary_estimator_error = np.mean(errors)

    def predict(self, X: np.array):
        """
        Predict using the conformal estimators and apply the sampler.
        """
        if isinstance(self.sampler, LowerBoundSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)

    def _predict_with_ucb(self, X: np.array):
        """
        Predict using UCB sampling strategy.
        """
        # With UCB we use only one estimator
        lower_quantile, upper_quantile = self.conformal_estimators[0].predict_interval(
            X=X
        )

        # Apply beta scaling for exploration
        lower_bound = upper_quantile - self.sampler.beta * (
            upper_quantile - lower_quantile
        )

        # Store predictions for later breach checking
        self.predictions_per_interval = [
            np.column_stack((lower_quantile, upper_quantile))
        ]

        self.sampler.update_exploration_step()
        return lower_bound

    def _predict_with_thompson(self, X: np.array):
        """
        Predict using Thompson sampling strategy.
        """
        # Get predictions from all estimators
        self.predictions_per_interval = []

        for estimator in self.conformal_estimators:
            lower_bound, upper_bound = estimator.predict_interval(X=X)
            self.predictions_per_interval.append(
                np.column_stack((lower_bound, upper_bound))
            )

        # Vectorized approach for sampling
        n_samples = X.shape[0]
        n_intervals = len(self.conformal_estimators)

        # Generate random indices for all samples at once
        interval_indices = np.random.choice(n_intervals, size=n_samples)

        # Extract the lower bounds using vectorized operations
        lower_bounds = np.array(
            [
                self.predictions_per_interval[idx][i, 0]
                for i, idx in enumerate(interval_indices)
            ]
        )

        # Apply optimistic sampling if enabled - do it once for all samples
        if self.sampler.enable_optimistic_sampling and self.point_estimator is not None:
            # Get all median predictions in one call
            median_predictions = self.point_estimator.predict(X)
            lower_bounds = np.minimum(lower_bounds, median_predictions)

        return lower_bounds

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        """
        Predict using Pessimistic Lower Bound sampling strategy.
        """
        # With pessimistic lower bound we use only one estimator
        lower_bound, upper_bound = self.conformal_estimators[0].predict_interval(X=X)

        # Store predictions for later breach checking
        self.predictions_per_interval = [np.column_stack((lower_bound, upper_bound))]

        return lower_bound

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        """
        Update interval width based on performance.
        """
        breaches = []
        for predictions in self.predictions_per_interval:
            sampled_predictions = predictions[sampled_idx, :]
            lower_bound, upper_bound = sampled_predictions[0], sampled_predictions[1]

            # Check if the actual performance is within the predicted interval
            breach = 0 if lower_bound <= sampled_performance <= upper_bound else 1
            breaches.append(breach)

        # Update the sampler with the breach information
        self.sampler.update_interval_width(breaches=breaches)
