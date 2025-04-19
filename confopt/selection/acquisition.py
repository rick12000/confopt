import logging
from typing import Optional, Union, List
import numpy as np
from abc import ABC, abstractmethod
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

    idx = np.random.randint(0, n_intervals, size=n_points)
    sampled_bounds = np.array([all_bounds[i, idx[i]] for i in range(n_points)])

    if enable_optimistic_sampling and point_predictions is not None:
        sampled_bounds = np.minimum(sampled_bounds, point_predictions)

    return sampled_bounds


def calculate_expected_improvement(
    predictions_per_interval: List[ConformalBounds],
    best_historical_y: float,
    num_samples: int = 100,
) -> np.ndarray:
    all_bounds = flatten_conformal_bounds(predictions_per_interval)

    n_observations = len(predictions_per_interval[0].lower_bounds)
    idxs = np.random.randint(0, all_bounds.shape[1], size=(n_observations, num_samples))

    y_samples_per_observation = np.zeros((n_observations, num_samples))
    for i in range(n_observations):
        y_samples_per_observation[i] = all_bounds[i, idxs[i]]

    improvements = np.maximum(0, y_samples_per_observation - best_historical_y)
    expected_improvements = np.mean(improvements, axis=1)

    return -expected_improvements


def calculate_information_gain(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_space: np.ndarray,
    conformal_estimator,
    predictions_per_interval: List[ConformalBounds],
    n_paths: int = 100,
    n_X_candidates: int = 10,
    n_y_candidates_per_x: int = 3,
    sampling_strategy: str = "uniform",
) -> np.ndarray:
    all_bounds = flatten_conformal_bounds(predictions_per_interval)
    n_observations = len(X_space)

    y_paths = np.zeros((n_paths, n_observations))
    for i in range(n_paths):
        idxs = np.random.randint(0, all_bounds.shape[1], size=n_observations)
        y_paths[i] = np.array([all_bounds[j, idxs[j]] for j in range(n_observations)])
    minimization_idxs_per_path, counts = np.unique(
        np.argmin(y_paths, axis=1), return_counts=True
    )

    best_x_distribution = np.zeros(n_observations)
    best_x_distribution[minimization_idxs_per_path] = counts / n_paths
    non_zero_idxs = best_x_distribution > 0
    best_x_entropy = -np.sum(
        best_x_distribution[non_zero_idxs] * np.log(best_x_distribution[non_zero_idxs])
    )

    capped_n_X_candidates = min(n_X_candidates, n_observations)
    if sampling_strategy == "uniform":
        X_candidate_idxs = np.random.choice(
            n_observations, size=capped_n_X_candidates, replace=False
        )
    elif sampling_strategy == "thompson":
        thompson_samples = np.array(
            [
                all_bounds[i, np.random.randint(0, all_bounds.shape[1])]
                for i in range(n_observations)
            ]
        )
        X_candidate_idxs = np.argsort(thompson_samples)[:capped_n_X_candidates]
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    information_gain_per_X = np.zeros(n_observations)
    for i in X_candidate_idxs:
        X_cand = X_space[i].reshape(1, -1)
        y_range = np.random.choice(all_bounds[i], size=n_y_candidates_per_x)

        entropy_per_y_candidate = []
        for y_cand in y_range:
            X_expanded = np.vstack([X_train, X_cand])
            y_expanded = np.append(y_train, y_cand)

            cand_estimator = deepcopy(conformal_estimator)
            cand_estimator.fit(
                X_train=X_expanded,
                y_train=y_expanded,
                X_val=X_val,
                y_val=y_val,
                tuning_iterations=0,
            )

            cand_predictions = cand_estimator.predict_intervals(X_space)
            cand_bounds = flatten_conformal_bounds(cand_predictions)

            conditional_y_paths = np.zeros((n_paths, n_observations))
            for j in range(n_paths):
                idxs = np.random.randint(0, cand_bounds.shape[1], size=n_observations)
                conditional_y_paths[j] = np.array(
                    [cand_bounds[k, idxs[k]] for k in range(n_observations)]
                )
            conditional_minimization_idxs_per_path, posterior_counts = np.unique(
                np.argmin(conditional_y_paths, axis=1), return_counts=True
            )

            conditional_best_X_distribution = np.zeros(n_observations)
            conditional_best_X_distribution[conditional_minimization_idxs_per_path] = (
                posterior_counts / n_paths
            )
            non_zero_idxs = conditional_best_X_distribution > 0
            if np.any(non_zero_idxs):
                candidate_conditional_entropy = -np.sum(
                    conditional_best_X_distribution[non_zero_idxs]
                    * np.log(conditional_best_X_distribution[non_zero_idxs])
                )
                entropy_per_y_candidate.append(candidate_conditional_entropy)

        information_gain_per_X[i] = best_x_entropy - np.mean(entropy_per_y_candidate)

    return -information_gain_per_X


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
        self.X_val = X_val  # Store validation data
        self.y_val = y_val  # Store validation data

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
        width = (interval.upper_bounds - interval.lower_bounds).reshape(-1, 1) / 2
        bounds = calculate_ucb_predictions(
            lower_bound=point_estimates,
            interval_width=width,
            beta=self.sampler.beta,
        )
        self.sampler.update_exploration_step()
        return bounds

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
            best_historical_y=self.sampler.current_best_value,
            num_samples=self.sampler.num_ei_samples,
        )

    def _predict_with_information_gain(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return calculate_information_gain(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_space=X,
            conformal_estimator=self.conformal_estimator,
            predictions_per_interval=self.predictions_per_interval,
            n_paths=self.sampler.n_paths,
            n_y_candidates_per_x=self.sampler.n_y_candidates_per_x,
            n_X_candidates=self.sampler.n_X_candidates,
            sampling_strategy=self.sampler.sampling_strategy,
        )

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
        self.X_val = X_val  # Store validation data
        self.y_val = y_val  # Store validation data

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
        width = interval.upper_bounds - interval.lower_bounds
        bounds = calculate_ucb_predictions(
            lower_bound=interval.upper_bounds,
            interval_width=width,
            beta=self.sampler.beta,
        )
        self.sampler.update_exploration_step()
        return bounds

    def _predict_with_thompson(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_predictions = None
        if self.sampler.enable_optimistic_sampling:
            point_predictor = getattr(self, "point_estimator", None)
            if point_predictor:
                point_predictions = point_predictor.predict(X)
        return calculate_thompson_predictions(
            predictions_per_interval=self.predictions_per_interval,
            enable_optimistic_sampling=self.sampler.enable_optimistic_sampling,
            point_predictions=point_predictions,
        )

    def _predict_with_expected_improvement(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return calculate_expected_improvement(
            predictions_per_interval=self.predictions_per_interval,
            best_historical_y=self.sampler.current_best_value,
            num_samples=self.sampler.num_ei_samples,
        )

    def _predict_with_information_gain(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return calculate_information_gain(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_space=X,
            conformal_estimator=self.conformal_estimator,
            predictions_per_interval=self.predictions_per_interval,
            n_paths=self.sampler.n_paths,
            n_y_candidates_per_x=self.sampler.n_y_candidates_per_x,
            n_X_candidates=self.sampler.n_X_candidates,
            sampling_strategy=self.sampler.sampling_strategy,
        )

    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        return self.conformal_estimator.calculate_betas(X, y_true)
