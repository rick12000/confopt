import logging
from typing import Optional, Union, List
import numpy as np
from abc import ABC, abstractmethod
from confopt.selection.adaptation import DtACI
from confopt.selection.conformalization import (
    LocallyWeightedConformalEstimator,
    QuantileConformalEstimator,
)
from confopt.wrapping import ConformalBounds
from confopt.selection.sampling import (
    LowerBoundSampler,
    ThompsonSampler,
    PessimisticLowerBoundSampler,
)
from confopt.selection.estimation import initialize_estimator

logger = logging.getLogger(__name__)


def calculate_ucb_predictions(
    lower_bound: np.ndarray, interval_width: np.ndarray, beta: float
) -> np.ndarray:
    return lower_bound - beta * interval_width


def calculate_thompson_predictions(
    predictions_per_interval: List[ConformalBounds],
    enable_optimistic_sampling: bool = False,
    point_predictions: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Get the number of samples from the first interval's bounds
    n_samples = len(predictions_per_interval[0].lower_bounds)
    n_intervals = len(predictions_per_interval)

    interval_indices = np.random.choice(n_intervals, size=n_samples)
    lower_bounds = np.array(
        [
            predictions_per_interval[idx].lower_bounds[i]
            for i, idx in enumerate(interval_indices)
        ]
    )

    if enable_optimistic_sampling and point_predictions is not None:
        lower_bounds = np.minimum(lower_bounds, point_predictions)

    return lower_bounds


class BaseConformalSearcher(ABC):
    def __init__(
        self,
        sampler: Union[
            LowerBoundSampler, ThompsonSampler, PessimisticLowerBoundSampler
        ],
    ):
        self.sampler = sampler

    def predict(self, X: np.array):
        if isinstance(self.sampler, LowerBoundSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)
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
    def _get_interval_predictions(self, X: np.array) -> List[ConformalBounds]:
        pass

    @abstractmethod
    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        pass

    def update_interval_width(self, X: np.array, y_true: float) -> list[float]:
        if isinstance(self.sampler.adapter, DtACI):
            betas = self._calculate_betas(X, y_true)
        if isinstance(self.sampler, ThompsonSampler):
            self.sampler.update_interval_width(betas=betas)
        elif isinstance(
            self.sampler, (PessimisticLowerBoundSampler, LowerBoundSampler)
        ):
            if len(betas) == 1:
                self.sampler.update_interval_width(beta=betas[0])
            else:
                raise ValueError("Multiple betas returned for single beta sampler.")
        else:
            raise ValueError(f"Unsupported sampler type: {type(self.sampler)}")


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
        self.point_estimator_architecture = point_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        self.conformal_estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=self.point_estimator_architecture,
            variance_estimator_architecture=self.variance_estimator_architecture,
            alphas=self.sampler.fetch_alphas(),
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

    def _calculate_betas(self, X: np.array, y_true: float) -> float:
        return self.conformal_estimator.calculate_betas(X, y_true)


class QuantileConformalSearcher(BaseConformalSearcher):
    def __init__(
        self,
        quantile_estimator_architecture: str,
        sampler: Union[
            LowerBoundSampler, ThompsonSampler, PessimisticLowerBoundSampler
        ],
        n_pre_conformal_trials: int = 20,
    ):
        super().__init__(sampler)
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.n_pre_conformal_trials = n_pre_conformal_trials

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        self.conformal_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=self.quantile_estimator_architecture,
            alphas=self.sampler.fetch_alphas(),
            n_pre_conformal_trials=self.n_pre_conformal_trials,
        )

        if isinstance(self.sampler, (PessimisticLowerBoundSampler, LowerBoundSampler)):
            upper_quantile_cap = 0.5
        elif isinstance(self.sampler, ThompsonSampler):
            upper_quantile_cap = None
            if self.sampler.enable_optimistic_sampling:
                self.point_estimator = initialize_estimator(
                    estimator_architecture="gbm",
                    random_state=random_state,
                )
                self.point_estimator.fit(
                    X=np.vstack((X_train, X_val)),
                    y=np.concatenate((y_train, y_val)),
                )
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

    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        return self.conformal_estimator.calculate_betas(X, y_true)
