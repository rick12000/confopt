import logging
from typing import Optional, List, Union, Literal

import numpy as np
from confopt.tracking import RuntimeTracker
from confopt.adaptation import ACI, DtACI
from confopt.conformalization import (
    LocallyWeightedConformalEstimator,
    QuantileInterval,
    SingleFitQuantileConformalEstimator,
    MultiFitQuantileConformalEstimator,
    MedianEstimator,
)

logger = logging.getLogger(__name__)


class UCBSampler:
    def __init__(
        self,
        beta_decay: str = "logarithmic_decay",
        c: float = 1,
        interval_width: float = 0.8,
        adapter_framework: Optional[str] = None,
        upper_quantile_cap: Optional[float] = None,
    ):
        self.beta_decay = beta_decay
        self.c = c
        self.interval_width = interval_width
        self.alpha = 1 - interval_width
        self.upper_quantile_cap = upper_quantile_cap
        self.t = 1
        self.beta = 1

        # Initialize adapter if specified
        self.adapter = self._initialize_adapter(adapter_framework)

        self.quantiles = self._calculate_quantiles()

    def _initialize_adapter(self, framework: Optional[str]):
        if framework == "ACI":
            adapter = ACI(alpha=self.alpha)
        elif framework == "DtACI":
            adapter = DtACI(alpha=self.alpha)
            self.expert_alphas = adapter.alpha_t_values
        else:
            adapter = None
        return adapter

    def _calculate_quantiles(self) -> QuantileInterval:
        if self.upper_quantile_cap:
            interval = QuantileInterval(
                lower_quantile=self.alpha / 2, upper_quantile=self.upper_quantile_cap
            )
        else:
            interval = QuantileInterval(
                lower_quantile=self.alpha / 2, upper_quantile=1 - (self.alpha / 2)
            )
        return interval

    def fetch_alpha(self) -> float:
        return self.alpha

    def fetch_expert_alphas(self) -> List[float]:
        return self.expert_alphas

    def fetch_interval(self) -> QuantileInterval:
        return self.quantiles

    def update_exploration_step(self):
        if self.beta_decay == "logarithmic_decay":
            self.beta = self.c * np.log(self.t) / self.t
        elif self.beta_decay == "logarithmic_growth":
            self.beta = 2 * np.log(self.t + 1)
        self.t += 1

    def update_interval_width(self, breaches: list[int]):
        if isinstance(self.adapter, ACI):
            if len(breaches) != 1:
                raise ValueError("ACI adapter requires a single breach indicator.")
            self.alpha = self.adapter.update(breach_indicator=breaches[0])
            self.quantiles = self._calculate_quantiles()
        elif isinstance(self.adapter, DtACI):
            self.alpha = self.adapter.update(breach_indicators=breaches)
            self.quantiles = self._calculate_quantiles()


class PessimisticLowerBoundSampler:
    def __init__(
        self,
        interval_width: float = 0.8,
        adapter_framework: Optional[str] = None,
    ):
        self.interval_width = interval_width
        self.alpha = 1 - interval_width

        # Initialize adapter if specified
        self.adapter = self._initialize_adapter(adapter_framework)
        self.quantiles = self._calculate_quantiles()

    def _initialize_adapter(self, framework: Optional[str]):
        if framework == "ACI":
            adapter = ACI(alpha=self.alpha)
        elif framework == "DtACI":
            adapter = DtACI(alpha=self.alpha)
            self.expert_alphas = adapter.alpha_t_values
        else:
            adapter = None
        return adapter

    def _calculate_quantiles(self) -> QuantileInterval:
        return QuantileInterval(
            lower_quantile=self.alpha / 2, upper_quantile=1 - (self.alpha / 2)
        )

    def fetch_alpha(self) -> float:
        return self.alpha

    def fetch_expert_alphas(self) -> List[float]:
        if hasattr(self, "expert_alphas"):
            return self.expert_alphas
        return [self.alpha]

    def fetch_interval(self) -> QuantileInterval:
        return self.quantiles

    def update_exploration_step(self):
        # No exploration parameter to update for pessimistic sampler
        pass

    def update_interval_width(self, breaches: list[int]):
        if isinstance(self.adapter, ACI):
            if len(breaches) != 1:
                raise ValueError("ACI adapter requires a single breach indicator.")
            self.alpha = self.adapter.update(breach_indicator=breaches[0])
            self.quantiles = self._calculate_quantiles()
        elif isinstance(self.adapter, DtACI):
            self.alpha = self.adapter.update(breach_indicators=breaches)
            self.quantiles = self._calculate_quantiles()


class ThompsonSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter_framework: Optional[str] = None,
        enable_optimistic_sampling: bool = False,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of Thompson quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.enable_optimistic_sampling = enable_optimistic_sampling

        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, n_quantiles + 1)
        ]
        self.quantiles, self.alphas = self._initialize_quantiles_and_alphas(
            starting_quantiles
        )
        self.adapters = self._initialize_adapters(adapter_framework)

    def _initialize_quantiles_and_alphas(self, starting_quantiles: List[float]):
        quantiles = []
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            quantiles.append(
                QuantileInterval(lower_quantile=lower, upper_quantile=upper)
            )
            alphas.append(1 - (upper - lower))
        return quantiles, alphas

    def _initialize_adapters(self, framework: Optional[str]):
        if not framework:
            return []

        adapter_class = ACI if framework == "ACI" else None
        if not adapter_class:
            raise ValueError(f"Unknown adapter framework: {framework}")

        return [adapter_class(alpha=alpha) for alpha in self.alphas]

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def fetch_intervals(self) -> List[QuantileInterval]:
        return self.quantiles

    def update_interval_width(self, breaches: List[int]):
        for i, (adapter, breach) in enumerate(zip(self.adapters, breaches)):
            updated_alpha = adapter.update(breach_indicator=breach)
            self.alphas[i] = updated_alpha
            self.quantiles[i] = QuantileInterval(
                lower_quantile=updated_alpha / 2, upper_quantile=1 - (updated_alpha / 2)
            )


class LocallyWeightedConformalSearcher:
    """
    Locally weighted conformal regression with sampling.

    Uses a locally weighted conformal estimator and applies sampling strategies
    to form point and variability predictions for y.
    """

    def __init__(
        self,
        point_estimator_architecture: str,
        variance_estimator_architecture: str,
        sampler: Union[UCBSampler, ThompsonSampler, PessimisticLowerBoundSampler],
    ):
        self.conformal_estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=point_estimator_architecture,
            variance_estimator_architecture=variance_estimator_architecture,
        )
        self.sampler = sampler
        self.training_time = None
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
        Fit the conformal estimator.
        """
        self.conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )
        self.training_time = self.conformal_estimator.training_time
        self.primary_estimator_error = self.conformal_estimator.primary_estimator_error

    def predict(self, X: np.array):
        """
        Predict using the conformal estimator and apply the sampler.
        """
        if isinstance(self.sampler, UCBSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)

    def _predict_with_ucb(self, X: np.array):
        """
        Predict using UCB sampling strategy.
        """
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
        """
        Predict using Thompson sampling strategy with locally weighted conformal estimator.
        """
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
    """
    Single-fit quantile conformal regression with sampling.

    Uses a single quantile conformal estimator that can predict any quantile
    after being fitted once, and applies sampling strategies to form predictions.
    """

    def __init__(
        self,
        quantile_estimator_architecture: Literal["qknn", "qrf"],
        sampler: Union[UCBSampler, ThompsonSampler, PessimisticLowerBoundSampler],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
        if isinstance(self.sampler, UCBSampler):
            self.sampler.upper_quantile_cap = 0.5
            self.sampler.quantiles = self.sampler._calculate_quantiles()
        self.n_pre_conformal_trials = n_pre_conformal_trials

        # Use a single estimator for all intervals
        self.conformal_estimator = SingleFitQuantileConformalEstimator(
            quantile_estimator_architecture=quantile_estimator_architecture,
            n_pre_conformal_trials=n_pre_conformal_trials,
        )
        self.median_estimator = None
        self.training_time = None
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
        training_time_tracker = RuntimeTracker()

        # Initialize and fit optimistic estimator if needed
        if (
            isinstance(self.sampler, ThompsonSampler)
            and self.sampler.enable_optimistic_sampling
        ):
            self.median_estimator = MedianEstimator(
                self.quantile_estimator_architecture
            )
            self.median_estimator.fit(
                X=np.vstack((X_train, X_val)),
                y=np.concatenate((y_train, y_val)),
                random_state=random_state,
            )

        # Get all intervals from the sampler
        if isinstance(self.sampler, UCBSampler):
            intervals = [self.sampler.fetch_interval()]
        else:  # ThompsonSampler
            intervals = self.sampler.fetch_intervals()

        # Fit the single conformal estimator with all intervals
        self.conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            intervals=intervals,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )

        self.training_time = training_time_tracker.return_runtime()
        self.primary_estimator_error = self.conformal_estimator.primary_estimator_error

    def predict(self, X: np.array):
        """
        Predict using the conformal estimator and apply the sampler.
        """
        if isinstance(self.sampler, UCBSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)

    def _predict_with_ucb(self, X: np.array):
        """
        Predict using UCB sampling strategy with a single estimator.
        """
        # Get the interval from the UCB sampler
        interval = self.sampler.fetch_interval()

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
        """
        Predict using Thompson sampling strategy with a single estimator.
        """
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
        if (
            self.sampler.enable_optimistic_sampling
            and self.median_estimator is not None
        ):
            # Get all median predictions in one call
            median_predictions = self.median_estimator.predict(X)
            lower_bounds = np.minimum(lower_bounds, median_predictions)

        return lower_bounds

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        """
        Predict using Pessimistic Lower Bound sampling strategy with a single estimator.
        """
        # Get the interval from the pessimistic sampler
        interval = self.sampler.fetch_interval()

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


class MultiFitQuantileConformalSearcher:
    """
    Multi-fit quantile conformal regression with sampling.

    Uses one or more multi-fit quantile conformal estimators and applies
    sampling strategies to form predictions.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        sampler: Union[UCBSampler, ThompsonSampler, PessimisticLowerBoundSampler],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
        if isinstance(self.sampler, UCBSampler):
            self.sampler.upper_quantile_cap = 0.5
            self.sampler.quantiles = self.sampler._calculate_quantiles()
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.conformal_estimators = []
        self.median_estimator = None
        self.training_time = None
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
        training_time_tracker = RuntimeTracker()

        # Initialize and fit optimistic estimator if needed
        if (
            isinstance(self.sampler, ThompsonSampler)
            and self.sampler.enable_optimistic_sampling
        ):
            self.median_estimator = MedianEstimator(
                self.quantile_estimator_architecture
            )
            self.median_estimator.fit(
                X=np.vstack((X_train, X_val)),
                y=np.concatenate((y_train, y_val)),
                random_state=random_state,
            )

        # Get intervals from the sampler
        if isinstance(self.sampler, UCBSampler):
            intervals = [self.sampler.fetch_interval()]
        else:  # ThompsonSampler
            intervals = self.sampler.fetch_intervals()

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

        self.training_time = training_time_tracker.return_runtime()
        self.primary_estimator_error = np.mean(errors)

    def predict(self, X: np.array):
        """
        Predict using the conformal estimators and apply the sampler.
        """
        if isinstance(self.sampler, UCBSampler):
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
        if (
            self.sampler.enable_optimistic_sampling
            and self.median_estimator is not None
        ):
            # Get all median predictions in one call
            median_predictions = self.median_estimator.predict(X)
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
