import logging
from typing import Optional, Union, List, Tuple
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

from scipy.stats import qmc

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
    MaxValueEntropySearchSampler,
)
from confopt.selection.estimation import initialize_estimator

logger = logging.getLogger(__name__)

DEFAULT_IG_SAMPLER_RANDOM_STATE = 1234


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

    improvements = np.maximum(0, best_historical_y - y_samples_per_observation)
    expected_improvements = np.mean(improvements, axis=1)

    return -expected_improvements


def _calculate_best_x_entropy(
    all_bounds: np.ndarray,
    n_observations: int,
    n_paths: int,
    entropy_method: str = "distance",
    alpha: float = 0.1,
) -> Tuple[float, np.ndarray]:
    indices_for_paths = np.vstack([np.arange(n_observations)] * n_paths)
    idxs = np.random.randint(0, all_bounds.shape[1], size=(n_paths, n_observations))
    y_paths = all_bounds[indices_for_paths, idxs]

    minimization_idxs = np.argmin(y_paths, axis=1)
    min_values = np.array([y_paths[i, minimization_idxs[i]] for i in range(n_paths)])
    best_x_entropy = _differential_entropy_estimator(
        min_values, alpha, method=entropy_method
    )

    return best_x_entropy, indices_for_paths


def calculate_variance(
    predictions_per_interval: List[ConformalBounds],
    num_samples: int = 100,
) -> np.ndarray:
    all_bounds = flatten_conformal_bounds(predictions_per_interval)
    n_observations = len(predictions_per_interval[0].lower_bounds)
    idxs = np.random.randint(0, all_bounds.shape[1], size=(n_observations, num_samples))
    y_samples_per_observation = np.zeros((n_observations, num_samples))
    for i in range(n_observations):
        y_samples_per_observation[i] = all_bounds[i, idxs[i]]
    conditional_variances = np.var(y_samples_per_observation, axis=1)
    return conditional_variances


def _select_candidates(
    predictions_per_interval: List[ConformalBounds],
    n_candidates: int,
    sampling_strategy: str,
    X_space: Optional[np.ndarray] = None,
    best_historical_y: Optional[float] = None,
    best_historical_x: Optional[np.ndarray] = None,
    perturbation_scale: float = 0.1,
) -> np.ndarray:
    all_bounds = flatten_conformal_bounds(predictions_per_interval)
    n_observations = len(predictions_per_interval[0].lower_bounds)
    capped_n_candidates = min(n_candidates, n_observations)

    if sampling_strategy == "uniform":
        return np.random.choice(n_observations, size=capped_n_candidates, replace=False)

    elif sampling_strategy == "thompson":
        thompson_samples = calculate_thompson_predictions(
            predictions_per_interval=predictions_per_interval,
            enable_optimistic_sampling=False,
        )
        return np.argsort(thompson_samples)[:capped_n_candidates]

    elif sampling_strategy == "expected_improvement":
        if best_historical_y is None:
            best_historical_y = np.min(np.mean(all_bounds, axis=1))
            logger.warning(
                "No best_historical_y provided for expected improvement selection, using calculated minimum."
            )

        ei_values = calculate_expected_improvement(
            predictions_per_interval=predictions_per_interval,
            best_historical_y=best_historical_y,
            num_samples=100,
        )
        return np.argsort(ei_values)[:capped_n_candidates]

    elif sampling_strategy == "variance":
        variances = calculate_variance(
            predictions_per_interval=predictions_per_interval, num_samples=100
        )
        return np.argsort(-variances)[:capped_n_candidates]

    elif sampling_strategy == "sobol":
        if X_space is None:
            raise ValueError("X_space must be provided for space-filling designs")
        n_dim = X_space.shape[1]
        sampler = qmc.Sobol(d=n_dim, scramble=True)
        points = sampler.random(n=capped_n_candidates)
        X_min = np.min(X_space, axis=0)
        X_range = np.max(X_space, axis=0) - X_min
        X_normalized = (X_space - X_min) / (X_range + 1e-10)
        selected_indices = []
        for point in points:
            distances = np.sqrt(np.sum((X_normalized - point) ** 2, axis=1))
            selected_idx = np.argmin(distances)
            selected_indices.append(selected_idx)
        return np.array(selected_indices)

    elif sampling_strategy == "latin_hypercube":
        if X_space is None:
            raise ValueError("X_space must be provided for space-filling designs")
        n_dim = X_space.shape[1]
        sampler = qmc.LatinHypercube(d=n_dim)
        points = sampler.random(n=capped_n_candidates)
        X_min = np.min(X_space, axis=0)
        X_range = np.max(X_space, axis=0) - X_min
        X_normalized = (X_space - X_min) / (X_range + 1e-10)
        selected_indices = []
        for point in points:
            distances = np.sqrt(np.sum((X_normalized - point) ** 2, axis=1))
            selected_idx = np.argmin(distances)
            selected_indices.append(selected_idx)
        return np.array(selected_indices)

    elif sampling_strategy == "perturbation":
        if X_space is None:
            raise ValueError("X_space must be provided for perturbation sampling")
        if best_historical_x is None or best_historical_y is None:
            logger.warning(
                "No best historical point provided for perturbation sampling, using uniform sampling."
            )
            return np.random.choice(
                n_observations, size=capped_n_candidates, replace=False
            )
        n_dim = X_space.shape[1]
        X_min = np.min(X_space, axis=0)
        X_max = np.max(X_space, axis=0)
        X_range = X_max - X_min
        lower_bounds = np.maximum(
            best_historical_x - perturbation_scale * X_range, X_min
        )
        upper_bounds = np.minimum(
            best_historical_x + perturbation_scale * X_range, X_max
        )
        random_points = np.random.uniform(
            lower_bounds, upper_bounds, size=(capped_n_candidates, n_dim)
        )
        selected_indices = []
        for point in random_points:
            distances = np.sqrt(np.sum((X_space - point) ** 2, axis=1))
            selected_idx = np.argmin(distances)
            selected_indices.append(selected_idx)
        return np.array(selected_indices)

    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")


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
    entropy_method: str = "distance",
    alpha: float = 0.1,
    n_jobs: int = -1,
) -> np.ndarray:
    all_bounds = flatten_conformal_bounds(predictions_per_interval)
    n_observations = len(predictions_per_interval[0].lower_bounds)
    prior_entropy, indices_for_paths = _calculate_best_x_entropy(
        all_bounds, n_observations, n_paths, entropy_method, alpha
    )
    best_historical_y = None
    best_historical_x = None
    if y_train is not None and len(y_train) > 0:
        if y_val is not None and len(y_val) > 0:
            combined_y = np.concatenate((y_train, y_val))
            combined_X = np.vstack((X_train, X_val))
            if sampling_strategy in ["expected_improvement", "perturbation"]:
                best_idx = np.argmin(combined_y)
                best_historical_y = combined_y[best_idx]
                best_historical_x = combined_X[best_idx].reshape(1, -1)
        else:
            if sampling_strategy in ["expected_improvement", "perturbation"]:
                best_idx = np.argmin(y_train)
                best_historical_y = y_train[best_idx]
                best_historical_x = X_train[best_idx].reshape(1, -1)
    candidate_idxs = _select_candidates(
        predictions_per_interval=predictions_per_interval,
        n_candidates=n_X_candidates,
        sampling_strategy=sampling_strategy,
        X_space=X_space,
        best_historical_y=best_historical_y,
        best_historical_x=best_historical_x,
    )

    def process_candidate(idx):
        X_cand = X_space[idx].reshape(1, -1)
        y_cand_idxs = np.random.randint(
            0, all_bounds.shape[1], size=n_y_candidates_per_x
        )
        y_range = all_bounds[idx, y_cand_idxs]
        information_gains = []
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
                random_state=DEFAULT_IG_SAMPLER_RANDOM_STATE,
            )
            cand_predictions = cand_estimator.predict_intervals(X_space)
            cand_bounds = flatten_conformal_bounds(cand_predictions)
            cond_idxs = np.random.randint(
                0, cand_bounds.shape[1], size=(n_paths, n_observations)
            )
            conditional_y_paths = cand_bounds[
                np.vstack([np.arange(n_observations)] * n_paths), cond_idxs
            ]
            cond_minimizers = np.argmin(conditional_y_paths, axis=1)
            conditional_samples = np.array(
                [conditional_y_paths[i, cond_minimizers[i]] for i in range(n_paths)]
            )
            posterior_entropy = _differential_entropy_estimator(
                conditional_samples, alpha, method=entropy_method
            )
            information_gains.append(prior_entropy - posterior_entropy)
        return idx, np.mean(information_gains) if information_gains else 0.0

    information_gain = np.zeros(n_observations)
    results = _run_parallel_or_sequential(
        lambda idx_list: process_candidate(idx_list[0]),
        [[idx] for idx in candidate_idxs],
        n_jobs=n_jobs,
        desc="Calculating information gain",
    )
    for idx, ig_value in results:
        information_gain[idx] = ig_value
    return -information_gain


def calculate_max_value_entropy_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_space: np.ndarray,
    conformal_estimator,
    predictions_per_interval: List[ConformalBounds],
    n_min_samples: int = 100,
    n_y_samples: int = 20,
    alpha: float = 0.1,
    entropy_method: str = "distance",
    n_jobs: int = -1,
) -> np.ndarray:
    import joblib

    all_bounds = flatten_conformal_bounds(predictions_per_interval)
    n_observations = len(predictions_per_interval[0].lower_bounds)
    idxs = np.random.randint(
        0, all_bounds.shape[1], size=(n_min_samples, n_observations)
    )
    sampled_funcs = np.zeros((n_min_samples, n_observations))
    for i in range(n_min_samples):
        sampled_funcs[i] = all_bounds[np.arange(n_observations), idxs[i]]
    min_values = np.min(sampled_funcs, axis=1)
    h_prior = _differential_entropy_estimator(min_values, alpha, method=entropy_method)

    def process_batch(batch_indices):
        batch_mes = np.zeros(len(batch_indices))
        for i, idx in enumerate(batch_indices):
            y_sample_idxs = np.random.randint(0, all_bounds.shape[1], size=n_y_samples)
            candidate_y_samples = all_bounds[idx, y_sample_idxs]
            updated_min_values = np.minimum(
                min_values[np.newaxis, :], candidate_y_samples[:, np.newaxis]
            )
            h_posteriors = np.array(
                [
                    _differential_entropy_estimator(
                        updated_min_values[j], alpha, method=entropy_method
                    )
                    for j in range(n_y_samples)
                ]
            )
            sample_mes = h_prior - h_posteriors
            batch_mes[i] = np.mean(sample_mes)
        return batch_indices, batch_mes

    batch_size = min(
        100, max(1, n_observations // (joblib.cpu_count() if n_jobs <= 0 else n_jobs))
    )
    batches = [
        list(range(i, min(i + batch_size, n_observations)))
        for i in range(0, n_observations, batch_size)
    ]
    mes_values = np.zeros(n_observations)
    results = _run_parallel_or_sequential(
        process_batch,
        batches,
        n_jobs=n_jobs,
        desc="Calculating max value entropy search",
    )
    for indices, values in results:
        mes_values[indices] = values
    return -mes_values


def _differential_entropy_estimator(
    samples: np.ndarray, alpha: float = 0.1, method: str = "distance"
) -> float:
    n_samples = len(samples)
    if n_samples <= 1:
        return 0.0
    if method == "distance":
        sorted_samples = np.sort(samples)
        distances = np.diff(sorted_samples)
        distances = np.append(distances, np.median(distances))
        distances = np.maximum(distances, alpha)
        log_distances = np.log(distances)
        entropy = np.mean(log_distances) + np.log(n_samples)
        return entropy
    elif method == "histogram":
        n_bins = int(np.sqrt(n_samples))
        hist, bin_edges = np.histogram(samples, bins=n_bins, density=True)
        bin_widths = np.diff(bin_edges)
        entropy = -np.sum(hist * np.log(hist + 1e-12) * bin_widths)
        return entropy
    else:
        raise ValueError(f"Unknown entropy estimation method: {method}")


def _run_parallel_or_sequential(func, items, n_jobs=-1, desc=None):
    import joblib
    from tqdm.auto import tqdm

    if n_jobs == 1:
        results = []
        for item in tqdm(items, desc=desc, disable=desc is None):
            results.append(func(item))
        return results
    else:
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
            if desc:
                with tqdm(total=len(items), desc=desc) as progress_bar:

                    def update_progress(*args, **kwargs):
                        progress_bar.update()

                    results = joblib.Parallel()(
                        joblib.delayed(func)(item) for item in items
                    )
                    return results
            else:
                return joblib.Parallel()(joblib.delayed(func)(item) for item in items)


class BaseConformalSearcher(ABC):
    def __init__(
        self,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
            InformationGainSampler,
            MaxValueEntropySearchSampler,
        ],
    ):
        self.sampler = sampler
        self.conformal_estimator = None
        self.X_train = None
        self.y_train = None
        self.last_beta = None

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
        elif isinstance(self.sampler, MaxValueEntropySearchSampler):
            return self._predict_with_max_value_entropy_search(X)
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
    def _predict_with_max_value_entropy_search(self, X: np.array):
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
            uses_adaptation = hasattr(self.sampler, "adapter") or hasattr(
                self.sampler, "adapters"
            )
            if uses_adaptation:
                betas = self._calculate_betas(X, y_true)
                if isinstance(
                    self.sampler,
                    (
                        ThompsonSampler,
                        ExpectedImprovementSampler,
                        InformationGainSampler,
                        MaxValueEntropySearchSampler,
                    ),
                ):
                    self.sampler.update_interval_width(betas=betas)
                elif isinstance(
                    self.sampler, (PessimisticLowerBoundSampler, LowerBoundSampler)
                ):
                    if len(betas) == 1:
                        self.last_beta = betas[0]
                        self.sampler.update_interval_width(beta=betas[0])
                    else:
                        raise ValueError(
                            "Multiple betas returned for single beta sampler."
                        )
                self.conformal_estimator.update_alphas(self.sampler.fetch_alphas())


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
            MaxValueEntropySearchSampler,
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
        self.X_val = X_val
        self.y_val = y_val
        if isinstance(self.sampler, InformationGainSampler) and random_state is None:
            random_state = DEFAULT_IG_SAMPLER_RANDOM_STATE
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

    def _predict_with_max_value_entropy_search(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return calculate_max_value_entropy_search(
            X_train=self.X_train,
            y_train=self.y_train,
            X_space=X,
            conformal_estimator=self.conformal_estimator,
            predictions_per_interval=self.predictions_per_interval,
            n_min_samples=self.sampler.n_min_samples,
            n_y_samples=self.sampler.n_y_samples,
            alpha=self.sampler.alpha,
            entropy_method="distance"
            if not hasattr(self.sampler, "entropy_method")
            else self.sampler.entropy_method,
            n_jobs=1,
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
            MaxValueEntropySearchSampler,
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
        self.X_val = X_val
        self.y_val = y_val
        random_state = random_state
        if isinstance(self.sampler, InformationGainSampler) and random_state is None:
            random_state = DEFAULT_IG_SAMPLER_RANDOM_STATE
        if isinstance(self.sampler, (PessimisticLowerBoundSampler, LowerBoundSampler)):
            upper_quantile_cap = 0.5
        elif isinstance(
            self.sampler,
            (ThompsonSampler, InformationGainSampler, MaxValueEntropySearchSampler),
        ):
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

    def _predict_with_max_value_entropy_search(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return calculate_max_value_entropy_search(
            X_train=self.X_train,
            y_train=self.y_train,
            X_space=X,
            conformal_estimator=self.conformal_estimator,
            predictions_per_interval=self.predictions_per_interval,
            n_min_samples=self.sampler.n_min_samples,
            n_y_samples=self.sampler.n_y_samples,
            alpha=self.sampler.alpha,
            entropy_method="distance"
            if not hasattr(self.sampler, "entropy_method")
            else self.sampler.entropy_method,
            n_jobs=1,
        )

    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        return self.conformal_estimator.calculate_betas(X, y_true)
