import logging
from typing import Optional, List, Union, Literal
from pydantic import BaseModel

import numpy as np
from sklearn.metrics import mean_pinball_loss
from confopt.tracking import RuntimeTracker
from confopt.adaptation import ACI, DtACI
from confopt.conformalization import LocallyWeightedConformalEstimator
from confopt.estimation import (
    initialize_point_estimator,
    initialize_quantile_estimator,
    tune,
    SEARCH_MODEL_DEFAULT_CONFIGURATIONS,
)

logger = logging.getLogger(__name__)


class QuantileInterval(BaseModel):
    lower_quantile: float
    upper_quantile: float


class UCBSampler:
    def __init__(
        self,
        beta_decay: str = "logarithmic_decay",
        beta: float = 1,
        c: float = 1,
        interval_width: float = 0.2,
        adapter_framework: Optional[str] = None,
    ):
        self.beta_decay = beta_decay
        self.beta = beta
        self.c = c
        self.interval_width = interval_width
        self.alpha = 1 - interval_width
        self.t = 1

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


class SingleFitQuantileConformalEstimator:
    """
    Single-fit quantile conformal estimator.

    Uses a single model that can predict multiple quantiles with a single fit.
    Can predict any quantile after fitting once.
    """

    def __init__(
        self,
        quantile_estimator_architecture: Literal["qknn", "qrf"],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.quantile_estimator = None
        self.nonconformity_scores = {}  # Store scores by interval
        self.conformalize_predictions = False
        self.training_time = None
        self.primary_estimator_error = None
        self.fitted_quantiles = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        intervals: List[QuantileInterval],
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit the single-fit quantile estimator for multiple intervals with one model.
        """
        training_time_tracker = RuntimeTracker()

        # Extract unique quantiles from all intervals
        all_quantiles = set()
        for interval in intervals:
            all_quantiles.add(interval.lower_quantile)
            all_quantiles.add(interval.upper_quantile)

        # Convert to sorted list
        self.fitted_quantiles = sorted(list(all_quantiles))

        # Tune model parameters if requested
        if tuning_iterations > 1 and len(X_train) > 10:
            initialization_params = tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                quantiles=self.fitted_quantiles,
                random_state=random_state,
            )
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                self.quantile_estimator_architecture
            ].copy()

        # Initialize and fit a single quantile estimator
        self.quantile_estimator = initialize_point_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )

        # Fit the model and calculate nonconformity scores if enough data
        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            self.quantile_estimator.fit(X_train, y_train)

            # Calculate nonconformity scores for each interval on validation data
            for interval in intervals:
                quantiles = [interval.lower_quantile, interval.upper_quantile]
                val_prediction = self.quantile_estimator.predict(
                    X=X_val,
                    quantiles=quantiles,
                )
                lower_conformal_deviations = val_prediction[:, 0] - y_val
                upper_conformal_deviations = y_val - val_prediction[:, 1]
                self.nonconformity_scores[self._interval_key(interval)] = np.maximum(
                    lower_conformal_deviations, upper_conformal_deviations
                )

            self.conformalize_predictions = True
        else:
            self.quantile_estimator.fit(
                X=np.vstack((X_train, X_val)), y=np.concatenate((y_train, y_val))
            )
            self.conformalize_predictions = False

        self.training_time = training_time_tracker.return_runtime()

        # Calculate performance metrics
        scores = []
        for interval in intervals:
            quantiles = [interval.lower_quantile, interval.upper_quantile]
            predictions = self.quantile_estimator.predict(
                X=X_val,
                quantiles=quantiles,
            )
            lo_y_pred = predictions[:, 0]
            hi_y_pred = predictions[:, 1]
            lo_score = mean_pinball_loss(
                y_val, lo_y_pred, alpha=interval.lower_quantile
            )
            hi_score = mean_pinball_loss(
                y_val, hi_y_pred, alpha=interval.upper_quantile
            )
            scores.append((lo_score + hi_score) / 2)

        self.primary_estimator_error = np.mean(scores)

    def _interval_key(self, interval: QuantileInterval) -> str:
        """Create a unique key for an interval to use in the nonconformity scores dictionary."""
        return f"{interval.lower_quantile}_{interval.upper_quantile}"

    def predict_interval(self, X: np.array, interval: QuantileInterval):
        """
        Predict conformal intervals for a specific interval.
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before prediction")

        quantiles = [interval.lower_quantile, interval.upper_quantile]
        prediction = self.quantile_estimator.predict(X=X, quantiles=quantiles)

        if self.conformalize_predictions:
            # Calculate conformity adjustment based on validation scores
            interval_key = self._interval_key(interval)
            if interval_key in self.nonconformity_scores:
                score = np.quantile(
                    self.nonconformity_scores[interval_key],
                    interval.upper_quantile - interval.lower_quantile,
                )
            else:
                # If we don't have exact scores for this interval, use the closest one
                closest_interval = self._find_closest_interval(interval)
                closest_key = self._interval_key(closest_interval)
                score = np.quantile(
                    self.nonconformity_scores[closest_key],
                    interval.upper_quantile - interval.lower_quantile,
                )
        else:
            score = 0

        lower_interval_bound = np.array(prediction[:, 0]) - score
        upper_interval_bound = np.array(prediction[:, 1]) + score

        return lower_interval_bound, upper_interval_bound

    def _find_closest_interval(
        self, target_interval: QuantileInterval
    ) -> QuantileInterval:
        """Find the closest interval in the nonconformity scores dictionary."""
        if not self.nonconformity_scores:
            return target_interval

        best_distance = float("inf")
        closest_interval = target_interval

        for interval_key in self.nonconformity_scores:
            lower, upper = map(float, interval_key.split("_"))
            current_interval = QuantileInterval(
                lower_quantile=lower, upper_quantile=upper
            )

            # Calculate distance between intervals
            distance = abs(
                current_interval.lower_quantile - target_interval.lower_quantile
            ) + abs(current_interval.upper_quantile - target_interval.upper_quantile)

            if distance < best_distance:
                best_distance = distance
                closest_interval = current_interval

        return closest_interval


class MultiFitQuantileConformalEstimator:
    """
    Multi-fit quantile conformal estimator for a single interval.

    Uses a dedicated quantile estimator for a specific interval.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        interval: QuantileInterval,
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.interval = interval
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.quantile_estimator = None
        self.nonconformity_scores = None
        self.conformalize_predictions = False
        self.training_time = None
        self.primary_estimator_error = None

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
        Fit a dedicated quantile estimator for this interval.
        """
        training_time_tracker = RuntimeTracker()

        # Prepare quantiles for this specific interval
        quantiles = [self.interval.lower_quantile, self.interval.upper_quantile]

        # Tune model parameters if requested
        if tuning_iterations > 1 and len(X_train) > 10:
            initialization_params = tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                quantiles=quantiles,
                random_state=random_state,
            )
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                self.quantile_estimator_architecture
            ].copy()

        # Initialize and fit the quantile estimator
        self.quantile_estimator = initialize_quantile_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            pinball_loss_alpha=quantiles,
            random_state=random_state,
        )

        # Fit the model and calculate nonconformity scores if enough data
        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            self.quantile_estimator.fit(X_train, y_train)

            # Calculate nonconformity scores on validation data
            val_prediction = self.quantile_estimator.predict(X_val)
            lower_conformal_deviations = val_prediction[:, 0] - y_val
            upper_conformal_deviations = y_val - val_prediction[:, 1]
            self.nonconformity_scores = np.maximum(
                lower_conformal_deviations, upper_conformal_deviations
            )
            self.conformalize_predictions = True
        else:
            self.quantile_estimator.fit(
                np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))
            )
            self.conformalize_predictions = False

        self.training_time = training_time_tracker.return_runtime()

        # Calculate performance metrics
        predictions = self.quantile_estimator.predict(X_val)
        lo_y_pred = predictions[:, 0]
        hi_y_pred = predictions[:, 1]
        lo_score = mean_pinball_loss(
            y_val, lo_y_pred, alpha=self.interval.lower_quantile
        )
        hi_score = mean_pinball_loss(
            y_val, hi_y_pred, alpha=self.interval.upper_quantile
        )
        self.primary_estimator_error = (lo_score + hi_score) / 2

    def predict_interval(self, X: np.array):
        """
        Predict conformal intervals.
        """
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before prediction")

        prediction = self.quantile_estimator.predict(X)

        if self.conformalize_predictions:
            # Calculate conformity adjustment based on validation scores
            score = np.quantile(
                self.nonconformity_scores,
                self.interval.upper_quantile - self.interval.lower_quantile,
            )
            lower_interval_bound = np.array(prediction[:, 0]) - score
            upper_interval_bound = np.array(prediction[:, 1]) + score
        else:
            lower_interval_bound = np.array(prediction[:, 0])
            upper_interval_bound = np.array(prediction[:, 1])

        return lower_interval_bound, upper_interval_bound


class MedianEstimator:
    """
    Simple wrapper for a median estimator used in optimistic sampling.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.median_estimator = None

    def fit(
        self,
        X: np.array,
        y: np.array,
        random_state: Optional[int] = None,
    ):
        """
        Fit a median (50th percentile) estimator.
        """
        initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
            self.quantile_estimator_architecture
        ].copy()

        self.median_estimator = initialize_quantile_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            pinball_loss_alpha=[0.5],
            random_state=random_state,
        )
        self.median_estimator.fit(X, y)

    def predict(self, X: np.array):
        """
        Predict median values.
        """
        if self.median_estimator is None:
            raise ValueError("Median estimator is not initialized")
        return np.array(self.median_estimator.predict(X)[:, 0])


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
        sampler: Union[UCBSampler, ThompsonSampler],
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

    def _predict_with_ucb(self, X: np.array):
        """
        Predict using UCB sampling strategy.
        """
        if isinstance(self.sampler.adapter, DtACI):
            self.predictions_per_interval = []
            for alpha in self.sampler.fetch_expert_alphas():
                lower_bound, upper_bound = self.conformal_estimator.predict_interval(
                    X=X, alpha=alpha, beta=self.sampler.beta
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
                X=X, alpha=alpha, beta=self.sampler.beta
            )
            self.predictions_per_interval = [np.hstack([lower_bound, upper_bound])]
            result_lower_bound = lower_bound

        self.sampler.update_exploration_step()
        return result_lower_bound

    def _predict_with_thompson(self, y_pred: np.array, var_pred: np.array):
        self.predictions_per_interval = []
        for alpha in self.sampler.fetch_alphas():
            score_quantile = np.quantile(self.nonconformity_scores, 1 - alpha)
            scaled_score = score_quantile * var_pred
            self.predictions_per_interval.append(
                np.hstack([y_pred - scaled_score, y_pred + scaled_score])
            )

        predictions_per_quantile = np.hstack(self.predictions_per_interval)
        lower_bound = []
        for i in range(predictions_per_quantile.shape[0]):
            # Use numpy's choice for reproducibility
            ts_idx = np.random.choice(range(self.sampler.n_quantiles))
            if self.sampler.enable_optimistic_sampling:
                lower_bound.append(
                    min(predictions_per_quantile[i, ts_idx], y_pred[i, 0])
                )
            else:
                lower_bound.append(predictions_per_quantile[i, ts_idx])
        lower_bound = np.array(lower_bound)

        return lower_bound

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
        sampler: Union[UCBSampler, ThompsonSampler],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
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

    def _predict_with_ucb(self, X: np.array):
        """
        Predict using UCB sampling strategy with a single estimator.
        """
        # Get the interval from the UCB sampler
        interval = self.sampler.fetch_interval()

        # Predict interval using the single estimator
        (
            lower_interval_bound,
            upper_interval_bound,
        ) = self.conformal_estimator.predict_interval(X=X, interval=interval)

        # Apply beta scaling for exploration
        lower_bound = lower_interval_bound + self.sampler.beta * (
            upper_interval_bound - lower_interval_bound
        )

        # Store predictions for later breach checking
        self.predictions_per_interval = [
            np.column_stack((lower_interval_bound, upper_interval_bound))
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

        # For each data point, randomly select one interval's lower bound
        n_samples = X.shape[0]
        n_intervals = len(intervals)

        lower_bounds = np.zeros(n_samples)
        for i in range(n_samples):
            # Randomly select an interval
            interval_idx = np.random.choice(n_intervals)

            # Get the lower bound from this interval
            lower_bound_value = self.predictions_per_interval[interval_idx][i, 0]

            # Apply optimistic sampling if enabled
            if (
                self.sampler.enable_optimistic_sampling
                and self.median_estimator is not None
            ):
                median_prediction = self.median_estimator.predict(X[i : i + 1])[0]
                lower_bounds[i] = min(lower_bound_value, median_prediction)
            else:
                lower_bounds[i] = lower_bound_value

        return lower_bounds

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
        sampler: Union[UCBSampler, ThompsonSampler],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
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

    def _predict_with_ucb(self, X: np.array):
        """
        Predict using UCB sampling strategy.
        """
        # With UCB we use only one estimator
        lower_interval_bound, upper_interval_bound = self.conformal_estimators[
            0
        ].predict_interval(X=X)

        # Apply beta scaling for exploration
        lower_bound = lower_interval_bound + self.sampler.beta * (
            upper_interval_bound - lower_interval_bound
        )

        # Store predictions for later breach checking
        self.predictions_per_interval = [
            np.column_stack((lower_interval_bound, upper_interval_bound))
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

        # For each data point, randomly select one interval's lower bound
        n_samples = X.shape[0]
        n_intervals = len(self.conformal_estimators)

        lower_bounds = np.zeros(n_samples)
        for i in range(n_samples):
            # Randomly select an interval
            interval_idx = np.random.choice(n_intervals)

            # Get the lower bound from this interval
            lower_bound_value = self.predictions_per_interval[interval_idx][i, 0]

            # Apply optimistic sampling if enabled
            if (
                self.sampler.enable_optimistic_sampling
                and self.median_estimator is not None
            ):
                median_prediction = self.median_estimator.predict(X[i : i + 1])[0]
                lower_bounds[i] = min(lower_bound_value, median_prediction)
            else:
                lower_bounds[i] = lower_bound_value

        return lower_bounds

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
