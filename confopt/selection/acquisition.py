import logging
from typing import Optional, Union, List
import numpy as np
from abc import ABC, abstractmethod
import random
from copy import deepcopy
from confopt.selection.conformalization import (
    LocallyWeightedConformalEstimator,
    QuantileConformalEstimator,
)
from confopt.wrapping import ConformalBounds
from confopt.selection.sampling import (
    LowerBoundSampler,
    ThompsonSampler,
    PessimisticLowerBoundSampler,
    ExpectedImprovementSampler,
    InformationGainSampler,
)
from confopt.selection.estimation import initialize_estimator

# Import necessary libraries for KDE and entropy calculation
try:
    pass
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Optional dependencies for InformationGain not installed. Install scipy and sklearn."
    )

logger = logging.getLogger(__name__)


def flatten_conformal_bounds(
    predictions_per_interval: List[ConformalBounds],
) -> np.ndarray:
    n_points = len(predictions_per_interval[0].lower_bounds)
    all_bounds = np.zeros((n_points, len(predictions_per_interval) * 2))
    for i, interval in enumerate(predictions_per_interval):
        all_bounds[:, i * 2] = interval.lower_bounds.flatten()
        all_bounds[:, i * 2 + 1] = interval.upper_bounds.flatten()
    return all_bounds


def calculate_ucb_predictions(
    lower_bound: np.ndarray, interval_width: np.ndarray, beta: float
) -> np.ndarray:
    return lower_bound - beta * interval_width


def calculate_thompson_predictions(
    predictions_per_interval: List[ConformalBounds],
    enable_optimistic_sampling: bool = False,
    point_predictions: Optional[np.ndarray] = None,
) -> np.ndarray:
    all_bounds = flatten_conformal_bounds(predictions_per_interval)
    n_points = len(predictions_per_interval[0].lower_bounds)
    n_intervals = all_bounds.shape[1]

    interval_indices = np.random.randint(0, n_intervals, size=n_points)
    sampled_bounds = np.array(
        [all_bounds[i, idx] for i, idx in enumerate(interval_indices)]
    )

    if enable_optimistic_sampling and point_predictions is not None:
        sampled_bounds = np.minimum(sampled_bounds, point_predictions)

    return sampled_bounds


def calculate_expected_improvement(
    predictions_per_interval: List[ConformalBounds],
    current_best_value: float,
    num_samples: int = 20,
) -> np.ndarray:
    all_bounds = flatten_conformal_bounds(predictions_per_interval)
    n_points = len(predictions_per_interval[0].lower_bounds)
    n_intervals = all_bounds.shape[1]

    # Generate all random indices at once
    interval_indices = np.random.randint(0, n_intervals, size=(n_points, num_samples))

    # Vectorized sampling from bounds
    samples = np.zeros((n_points, num_samples))
    for i in range(n_points):
        samples[i] = all_bounds[i, interval_indices[i]]

    # Vectorized improvement calculation
    improvements = np.maximum(0, samples - current_best_value)
    expected_improvements = np.mean(improvements, axis=1)

    # Return negative values for minimization
    return -expected_improvements


def calculate_information_gain(
    X_candidates: np.ndarray,
    conformal_estimator,
    predictions_per_interval: List[ConformalBounds],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_samples: int = 30,
    n_y_samples_per_x: int = 5,  # Default number of Y samples per X candidate
    n_eval_candidates: int = 30,  # Number of candidates to evaluate
    kde_bandwidth: float = 0.5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Calculate information gain for candidate points based on entropy reduction.

    Args:
        X_candidates: Points to evaluate for information gain
        conformal_estimator: The current conformal estimator
        predictions_per_interval: Current conformal prediction intervals
        X_train: Training data points
        y_train: Training target values
        n_samples: Number of samples for the distribution
        n_y_samples_per_x: Number of Y values to sample per X candidate
        n_eval_candidates: Number of candidate points to evaluate (for efficiency)
        kde_bandwidth: Bandwidth for KDE estimation
        random_state: Random seed for reproducibility

    Returns:
        Array of information gain values for each candidate
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Calculate the current distribution p(x*)
    all_bounds = flatten_conformal_bounds(predictions_per_interval)

    # Vectorized sampling from bounds
    # Generate all realizations at once
    n_points = len(X_candidates)
    realizations = np.zeros((n_samples, n_points))

    # Use NumPy's vectorized choice for better performance
    for j in range(n_samples):
        # For each row, randomly select an index
        indices = np.random.randint(0, all_bounds.shape[1], size=n_points)
        # Use advanced indexing to get the values
        realizations[j] = np.array(
            [all_bounds[i, idx] for i, idx in enumerate(indices)]
        )

    # Find xstar indices (argmin) for each realization
    xstar_indices = np.argmin(realizations, axis=1)

    # Count frequencies directly instead of using KDE
    unique_indices, counts = np.unique(xstar_indices, return_counts=True)
    prior_probs = np.zeros(n_points)
    prior_probs[unique_indices] = counts / n_samples

    # Calculate entropy directly from probabilities
    # Only consider non-zero probabilities to avoid log(0)
    mask = prior_probs > 0
    prior_entropy = -np.sum(prior_probs[mask] * np.log(prior_probs[mask]))

    # Initialize results array for all candidates
    information_gains = np.zeros(len(X_candidates))

    # Randomly sample a subset of candidates to evaluate (for efficiency)
    n_eval = min(n_eval_candidates, len(X_candidates))
    eval_indices = np.random.choice(len(X_candidates), size=n_eval, replace=False)

    # Pre-compute the dataset split once outside the loop
    train_ratio = 0.8

    # Cache X_train shape for efficient stacking
    X_train.shape

    for i in eval_indices:
        x = X_candidates[i].reshape(1, -1)

        # Get the predictions for this point from the already computed intervals
        all_bounds_for_point = all_bounds[i]

        # Sample Y values all at once
        y_samples = np.random.choice(all_bounds_for_point, size=n_y_samples_per_x)

        # For each X candidate, calculate posterior entropies for multiple Y samples
        posterior_entropies = []

        for y_idx in range(n_y_samples_per_x):
            y_sampled = y_samples[y_idx]

            # Create new dataset efficiently
            X_new = np.vstack([X_train, x])
            y_new = np.append(y_train, y_sampled)

            # Retrain conformal estimator (this is the irreducible bottleneck)
            new_estimator = deepcopy(conformal_estimator)

            try:
                # Split the dataset
                if len(X_new) >= 10:
                    train_size = int(train_ratio * len(X_new))
                    X_train_new, y_train_new = X_new[:train_size], y_new[:train_size]
                    X_val_new, y_val_new = X_new[train_size:], y_new[train_size:]

                    # Fit with minimal tuning
                    new_estimator.fit(
                        X_train=X_train_new,
                        y_train=y_train_new,
                        X_val=X_val_new,
                        y_val=y_val_new,
                        tuning_iterations=0,
                    )

                    # Generate new predictions
                    new_predictions = new_estimator.predict_intervals(X_candidates)
                    new_bounds = flatten_conformal_bounds(new_predictions)

                    # Vectorized sampling from new bounds
                    posterior_realizations = np.zeros((n_samples, n_points))
                    for j in range(n_samples):
                        indices = np.random.randint(
                            0, new_bounds.shape[1], size=n_points
                        )
                        posterior_realizations[j] = np.array(
                            [new_bounds[k, idx] for k, idx in enumerate(indices)]
                        )

                    # Find argmin indices
                    posterior_xstar_indices = np.argmin(posterior_realizations, axis=1)

                    # Count frequencies directly
                    unique_posterior_indices, posterior_counts = np.unique(
                        posterior_xstar_indices, return_counts=True
                    )
                    posterior_probs = np.zeros(n_points)
                    posterior_probs[unique_posterior_indices] = (
                        posterior_counts / n_samples
                    )

                    # Calculate entropy directly
                    mask = posterior_probs > 0
                    if np.any(mask):
                        posterior_entropy = -np.sum(
                            posterior_probs[mask] * np.log(posterior_probs[mask])
                        )
                        posterior_entropies.append(posterior_entropy)
            except Exception as e:
                logger.warning(f"Error during posterior entropy calculation: {e}")
                continue

        # Calculate expected posterior entropy
        if posterior_entropies:
            expected_posterior_entropy = np.mean(posterior_entropies)
            information_gains[i] = prior_entropy - expected_posterior_entropy

    # Return negative values for minimization
    return -information_gains


class BaseConformalSearcher(ABC):
    def __init__(
        self,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
            InformationGainSampler,
        ],
    ):
        self.sampler = sampler
        self.conformal_estimator = None
        self.X_train = None
        self.y_train = None

    def predict(self, X: np.array):
        if isinstance(self.sampler, LowerBoundSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)
        elif isinstance(self.sampler, ExpectedImprovementSampler):
            return self._predict_with_expected_improvement(X)
        elif isinstance(self.sampler, InformationGainSampler):
            return self._predict_with_information_gain(X)
        else:
            raise ValueError(f"Unsupported sampler type: {type(self.sampler)}")

    @abstractmethod
    def _predict_with_ucb(self, X: np.array):
        pass

    @abstractmethod
    def _predict_with_thompson(self, X: np.array):
        pass

    @abstractmethod
    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        pass

    @abstractmethod
    def _predict_with_expected_improvement(self, X: np.array):
        pass

    @abstractmethod
    def _predict_with_information_gain(self, X: np.array):
        pass

    @abstractmethod
    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        pass

    def update(self, X: np.array, y_true: float) -> None:
        # Store training data for information gain calculation
        if self.X_train is not None:
            self.X_train = np.vstack([self.X_train, X])
            self.y_train = np.append(self.y_train, y_true)
        else:
            self.X_train = X.reshape(1, -1)
            self.y_train = np.array([y_true])

        if isinstance(self.sampler, ExpectedImprovementSampler):
            self.sampler.update_best_value(y_true)

        if isinstance(self.sampler, LowerBoundSampler):
            self.sampler.update_exploration_step()

        if self.conformal_estimator.nonconformity_scores is not None:
            if hasattr(self.sampler, "adapter") or hasattr(self.sampler, "adapters"):
                betas = self._calculate_betas(X, y_true)
                if isinstance(
                    self.sampler,
                    (
                        ThompsonSampler,
                        ExpectedImprovementSampler,
                        InformationGainSampler,
                    ),
                ):
                    self.sampler.update_interval_width(betas=betas)
                elif isinstance(
                    self.sampler, (PessimisticLowerBoundSampler, LowerBoundSampler)
                ):
                    if len(betas) == 1:
                        self.sampler.update_interval_width(beta=betas[0])
                    else:
                        raise ValueError(
                            "Multiple betas returned for single beta sampler."
                        )


class LocallyWeightedConformalSearcher(BaseConformalSearcher):
    def __init__(
        self,
        point_estimator_architecture: str,
        variance_estimator_architecture: str,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
            InformationGainSampler,
        ],
    ):
        super().__init__(sampler)
        self.point_estimator_architecture = point_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture
        self.conformal_estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=self.point_estimator_architecture,
            variance_estimator_architecture=self.variance_estimator_architecture,
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
        self.X_train = X_train
        self.y_train = y_train

        self.conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )
        self.primary_estimator_error = self.conformal_estimator.primary_estimator_error

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.predictions_per_interval[0].lower_bounds

    def _predict_with_ucb(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_estimates = np.array(
            self.conformal_estimator.pe_estimator.predict(X)
        ).reshape(-1, 1)
        interval = self.predictions_per_interval[0]
        interval_width = (interval.upper_bounds - interval.lower_bounds).reshape(
            -1, 1
        ) / 2
        tracked_lower_bounds = calculate_ucb_predictions(
            lower_bound=point_estimates,
            interval_width=interval_width,
            beta=self.sampler.beta,
        )
        self.sampler.update_exploration_step()
        return tracked_lower_bounds

    def _predict_with_thompson(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_predictions = None
        if self.sampler.enable_optimistic_sampling:
            point_predictions = self.conformal_estimator.pe_estimator.predict(X)
        return calculate_thompson_predictions(
            predictions_per_interval=self.predictions_per_interval,
            enable_optimistic_sampling=self.sampler.enable_optimistic_sampling,
            point_predictions=point_predictions,
        )

    def _predict_with_expected_improvement(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return calculate_expected_improvement(
            predictions_per_interval=self.predictions_per_interval,
            current_best_value=self.sampler.current_best_value,
            num_samples=self.sampler.num_ei_samples,
        )

    def _predict_with_information_gain(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)

        # Calculate information gain for each point in X
        information_gains = calculate_information_gain(
            X_candidates=X,
            conformal_estimator=self.conformal_estimator,
            predictions_per_interval=self.predictions_per_interval,
            X_train=self.X_train,
            y_train=self.y_train,
            n_samples=self.sampler.n_samples,
            n_y_samples_per_x=self.sampler.n_y_samples_per_x,
            n_eval_candidates=self.sampler.n_candidates,
            random_state=None,  # Allow randomness for diversity
        )

        return information_gains

    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        return self.conformal_estimator.calculate_betas(X, y_true)


class QuantileConformalSearcher(BaseConformalSearcher):
    def __init__(
        self,
        quantile_estimator_architecture: str,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
            InformationGainSampler,
        ],
        n_pre_conformal_trials: int = 20,
    ):
        super().__init__(sampler)
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.conformal_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=self.quantile_estimator_architecture,
            alphas=self.sampler.fetch_alphas(),
            n_pre_conformal_trials=self.n_pre_conformal_trials,
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
        self.X_train = X_train
        self.y_train = y_train

        if isinstance(self.sampler, (PessimisticLowerBoundSampler, LowerBoundSampler)):
            upper_quantile_cap = 0.5
        elif isinstance(self.sampler, (ThompsonSampler, InformationGainSampler)):
            upper_quantile_cap = None
            if (
                hasattr(self.sampler, "enable_optimistic_sampling")
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
        elif isinstance(self.sampler, ExpectedImprovementSampler):
            upper_quantile_cap = None
        else:
            raise ValueError(f"Unsupported sampler type: {type(self.sampler)}")

        self.conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
            upper_quantile_cap=upper_quantile_cap,
        )
        self.primary_estimator_error = self.conformal_estimator.primary_estimator_error

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.predictions_per_interval[0].lower_bounds

    def _predict_with_ucb(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        interval = self.predictions_per_interval[0]
        interval_width = interval.upper_bounds - interval.lower_bounds
        tracked_lower_bounds = calculate_ucb_predictions(
            lower_bound=interval.upper_bounds,
            interval_width=interval_width,
            beta=self.sampler.beta,
        )
        self.sampler.update_exploration_step()
        return tracked_lower_bounds

    def _predict_with_thompson(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_predictions = None
        if self.sampler.enable_optimistic_sampling:
            point_predictions = getattr(self, "point_estimator", None)
            if point_predictions:
                point_predictions = point_predictions.predict(X)
        lower_bounds = calculate_thompson_predictions(
            predictions_per_interval=self.predictions_per_interval,
            enable_optimistic_sampling=self.sampler.enable_optimistic_sampling,
            point_predictions=point_predictions,
        )
        return lower_bounds

    def _predict_with_expected_improvement(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return calculate_expected_improvement(
            predictions_per_interval=self.predictions_per_interval,
            current_best_value=self.sampler.current_best_value,
            num_samples=self.sampler.num_ei_samples,
        )

    def _predict_with_information_gain(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)

        # Calculate information gain for each point in X
        information_gains = calculate_information_gain(
            X_candidates=X,
            conformal_estimator=self.conformal_estimator,
            predictions_per_interval=self.predictions_per_interval,
            X_train=self.X_train,
            y_train=self.y_train,
            n_samples=self.sampler.n_samples,
            n_y_samples_per_x=self.sampler.n_y_samples_per_x,
            n_eval_candidates=self.sampler.n_candidates,
            random_state=None,  # Allow randomness for diversity
        )

        return information_gains

    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        return self.conformal_estimator.calculate_betas(X, y_true)
