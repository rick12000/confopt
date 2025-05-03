from typing import Optional, List, Literal, Tuple
import numpy as np
from confopt.selection.adaptation import DtACI
import warnings
from confopt.wrapping import ConformalBounds
import joblib
from tqdm.auto import tqdm
from copy import deepcopy


def flatten_conformal_bounds(
    predictions_per_interval: List[ConformalBounds],
) -> np.ndarray:
    n_points = len(predictions_per_interval[0].lower_bounds)
    all_bounds = np.zeros((n_points, len(predictions_per_interval) * 2))
    for i, interval in enumerate(predictions_per_interval):
        all_bounds[:, i * 2] = interval.lower_bounds.flatten()
        all_bounds[:, i * 2 + 1] = interval.upper_bounds.flatten()
    return all_bounds


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


class PessimisticLowerBoundSampler:
    def __init__(
        self,
        interval_width: float = 0.8,
        adapter: Optional[Literal["DtACI"]] = None,
    ):
        self.interval_width = interval_width

        self.alpha = 1 - interval_width
        self.adapter = self._initialize_adapter(adapter)

    def _initialize_adapter(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[DtACI]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return DtACI(alpha=self.alpha, gamma_values=[0.05, 0.01, 0.1])
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return [self.alpha]

    def update_interval_width(self, beta: float) -> None:
        if self.adapter is not None:
            self.alpha = self.adapter.update(beta=beta)
        else:
            warnings.warn(
                "'update_interval_width()' method was called, but no adapter was initialized."
            )


class LowerBoundSampler(PessimisticLowerBoundSampler):
    def __init__(
        self,
        interval_width: float = 0.8,
        adapter: Optional[Literal["DtACI"]] = None,
        beta_decay: Optional[
            Literal[
                "inverse_square_root_decay",
                "logarithmic_decay",
            ]
        ] = "logarithmic_decay",
        c: float = 1,
        beta_max: float = 10,
    ):
        super().__init__(interval_width, adapter)
        self.beta_decay = beta_decay
        self.c = c
        self.t = 1
        self.beta = 1
        self.beta_max = beta_max
        self.mu_max = float("-inf")

    def update_exploration_step(self):
        self.t += 1
        if self.beta_decay == "inverse_square_root_decay":
            self.beta = np.sqrt(self.c / self.t)
        elif self.beta_decay == "logarithmic_decay":
            self.beta = np.sqrt((self.c * np.log(self.t)) / self.t)
        elif self.beta_decay is None:
            self.beta = 1
        else:
            raise ValueError(
                "beta_decay must be 'inverse_square_root_decay', 'logarithmic_decay', or None."
            )

    def calculate_ucb_predictions(
        self,
        predictions_per_interval: List[ConformalBounds],
        point_estimates: np.ndarray = None,
        interval_width: np.ndarray = None,
    ) -> np.ndarray:
        if point_estimates is None or interval_width is None:
            interval = predictions_per_interval[0]
            point_estimates = (interval.upper_bounds + interval.lower_bounds) / 2
            interval_width = (interval.upper_bounds - interval.lower_bounds) / 2

        return point_estimates - self.beta * interval_width


class ThompsonSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI"]] = None,
        enable_optimistic_sampling: bool = False,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of Thompson quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.enable_optimistic_sampling = enable_optimistic_sampling

        self.alphas = self._initialize_alphas()
        self.adapters = self._initialize_adapters(adapter)

    def _initialize_alphas(self) -> list[float]:
        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, self.n_quantiles + 1)
        ]
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            alphas.append(1 - (upper - lower))
        return alphas

    def _initialize_adapters(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[List[DtACI]]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return [
                DtACI(alpha=alpha, gamma_values=[0.05, 0.01, 0.1])
                for alpha in self.alphas
            ]
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        if self.adapters:
            for i, (adapter, beta) in enumerate(zip(self.adapters, betas)):
                updated_alpha = adapter.update(beta=beta)
                self.alphas[i] = updated_alpha

    def calculate_thompson_predictions(
        self,
        predictions_per_interval: List[ConformalBounds],
        point_predictions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        n_points = len(predictions_per_interval[0].lower_bounds)
        n_intervals = all_bounds.shape[1]

        idx = np.random.randint(0, n_intervals, size=n_points)
        sampled_bounds = np.array([all_bounds[i, idx[i]] for i in range(n_points)])

        if self.enable_optimistic_sampling and point_predictions is not None:
            sampled_bounds = np.minimum(sampled_bounds, point_predictions)

        return sampled_bounds


class ExpectedImprovementSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI"]] = None,
        current_best_value: float = float("inf"),
        num_ei_samples: int = 20,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.current_best_value = current_best_value
        self.num_ei_samples = num_ei_samples

        self.alphas = self._initialize_alphas()
        self.adapters = self._initialize_adapters(adapter)

    def update_best_value(self, value: float):
        """Update the current best value found in optimization."""
        self.current_best_value = min(self.current_best_value, value)

    def _initialize_alphas(self) -> list[float]:
        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, self.n_quantiles + 1)
        ]
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            alphas.append(1 - (upper - lower))
        return alphas

    def _initialize_adapters(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[List[DtACI]]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return [
                DtACI(alpha=alpha, gamma_values=[0.05, 0.01, 0.1])
                for alpha in self.alphas
            ]
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        if self.adapters:
            for i, (adapter, beta) in enumerate(zip(self.adapters, betas)):
                updated_alpha = adapter.update(beta=beta)
                self.alphas[i] = updated_alpha

    def calculate_expected_improvement(
        self,
        predictions_per_interval: List[ConformalBounds],
    ) -> np.ndarray:
        all_bounds = flatten_conformal_bounds(predictions_per_interval)

        n_observations = len(predictions_per_interval[0].lower_bounds)
        idxs = np.random.randint(
            0, all_bounds.shape[1], size=(n_observations, self.num_ei_samples)
        )

        y_samples_per_observation = np.zeros((n_observations, self.num_ei_samples))
        for i in range(n_observations):
            y_samples_per_observation[i] = all_bounds[i, idxs[i]]

        improvements = np.maximum(
            0, self.current_best_value - y_samples_per_observation
        )
        expected_improvements = np.mean(improvements, axis=1)

        return -expected_improvements


class InformationGainSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI"]] = None,
        n_paths: int = 100,
        n_X_candidates: int = 10,
        n_y_candidates_per_x: int = 3,
        sampling_strategy: str = "uniform",
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.n_paths = n_paths
        self.n_X_candidates = n_X_candidates
        self.n_y_candidates_per_x = n_y_candidates_per_x
        self.sampling_strategy = sampling_strategy

        self.alphas = self._initialize_alphas()
        self.adapters = self._initialize_adapters(adapter)

    def _initialize_alphas(self) -> list[float]:
        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, self.n_quantiles + 1)
        ]
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            alphas.append(1 - (upper - lower))
        return alphas

    def _initialize_adapters(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[List[DtACI]]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return [
                DtACI(alpha=alpha, gamma_values=[0.05, 0.01, 0.1])
                for alpha in self.alphas
            ]
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        if self.adapters:
            for i, (adapter, beta) in enumerate(zip(self.adapters, betas)):
                updated_alpha = adapter.update(beta=beta)
                self.alphas[i] = updated_alpha

    def _calculate_best_x_entropy(
        self,
        all_bounds: np.ndarray,
        n_observations: int,
        entropy_method: str = "distance",
        alpha: float = 0.1,
    ) -> Tuple[float, np.ndarray]:
        indices_for_paths = np.vstack([np.arange(n_observations)] * self.n_paths)
        idxs = np.random.randint(
            0, all_bounds.shape[1], size=(self.n_paths, n_observations)
        )
        y_paths = all_bounds[indices_for_paths, idxs]

        minimization_idxs = np.argmin(y_paths, axis=1)
        min_values = np.array(
            [y_paths[i, minimization_idxs[i]] for i in range(self.n_paths)]
        )
        best_x_entropy = _differential_entropy_estimator(
            min_values, alpha, method=entropy_method
        )

        return best_x_entropy, indices_for_paths

    def _select_candidates(
        self,
        predictions_per_interval: List[ConformalBounds],
        X_space: np.ndarray,
        best_historical_y: Optional[float] = None,
        best_historical_x: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        n_observations = len(predictions_per_interval[0].lower_bounds)
        capped_n_candidates = min(self.n_X_candidates, n_observations)

        if self.sampling_strategy == "thompson":
            thompson_sampler = ThompsonSampler()
            thompson_samples = thompson_sampler.calculate_thompson_predictions(
                predictions_per_interval=predictions_per_interval
            )
            return np.argsort(thompson_samples)[:capped_n_candidates]

        elif self.sampling_strategy == "expected_improvement":
            if best_historical_y is None:
                best_historical_y = np.min(np.mean(all_bounds, axis=1))

            ei_sampler = ExpectedImprovementSampler(
                current_best_value=best_historical_y
            )
            ei_values = ei_sampler.calculate_expected_improvement(
                predictions_per_interval=predictions_per_interval
            )
            return np.argsort(ei_values)[:capped_n_candidates]

        elif self.sampling_strategy == "sobol":
            if X_space is None:
                raise ValueError("X_space must be provided for space-filling designs")
            n_dim = X_space.shape[1]
            from scipy.stats import qmc

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

        elif self.sampling_strategy == "perturbation":
            if X_space is None:
                raise ValueError("X_space must be provided for perturbation sampling")
            if best_historical_x is None or best_historical_y is None:
                return np.random.choice(
                    n_observations, size=capped_n_candidates, replace=False
                )
            n_dim = X_space.shape[1]
            X_min = np.min(X_space, axis=0)
            X_max = np.max(X_space, axis=0)
            X_range = X_max - X_min
            perturbation_scale = 0.1
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
            return np.random.choice(
                n_observations, size=capped_n_candidates, replace=False
            )

    def calculate_information_gain(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_space: np.ndarray,
        conformal_estimator,
        predictions_per_interval: List[ConformalBounds],
        entropy_method: str = "distance",
        alpha: float = 0.1,
        n_jobs: int = -1,
    ) -> np.ndarray:
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        n_observations = len(predictions_per_interval[0].lower_bounds)
        prior_entropy, indices_for_paths = self._calculate_best_x_entropy(
            all_bounds, n_observations, entropy_method, alpha
        )

        best_historical_y = None
        best_historical_x = None
        if y_train is not None and len(y_train) > 0:
            if y_val is not None and len(y_val) > 0:
                combined_y = np.concatenate((y_train, y_val))
                combined_X = np.vstack((X_train, X_val))
                if self.sampling_strategy in ["expected_improvement", "perturbation"]:
                    best_idx = np.argmin(combined_y)
                    best_historical_y = combined_y[best_idx]
                    best_historical_x = combined_X[best_idx].reshape(1, -1)
            else:
                if self.sampling_strategy in ["expected_improvement", "perturbation"]:
                    best_idx = np.argmin(y_train)
                    best_historical_y = y_train[best_idx]
                    best_historical_x = X_train[best_idx].reshape(1, -1)

        candidate_idxs = self._select_candidates(
            predictions_per_interval=predictions_per_interval,
            X_space=X_space,
            best_historical_y=best_historical_y,
            best_historical_x=best_historical_x,
        )

        def process_candidate(idx):
            X_cand = X_space[idx].reshape(1, -1)
            y_cand_idxs = np.random.randint(
                0, all_bounds.shape[1], size=self.n_y_candidates_per_x
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
                    random_state=1234,
                )
                cand_predictions = cand_estimator.predict_intervals(X_space)
                cand_bounds = flatten_conformal_bounds(cand_predictions)
                cond_idxs = np.random.randint(
                    0, cand_bounds.shape[1], size=(self.n_paths, n_observations)
                )
                conditional_y_paths = cand_bounds[
                    np.vstack([np.arange(n_observations)] * self.n_paths), cond_idxs
                ]
                cond_minimizers = np.argmin(conditional_y_paths, axis=1)
                conditional_samples = np.array(
                    [
                        conditional_y_paths[i, cond_minimizers[i]]
                        for i in range(self.n_paths)
                    ]
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


class MaxValueEntropySearchSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI"]] = None,
        n_min_samples: int = 100,  # Number of samples to estimate minimum value distribution
        n_y_samples: int = 20,  # Number of y samples to evaluate per candidate point
        alpha: float = 0.1,  # Parameter for entropy estimation
        sampling_strategy: str = "uniform",  # Strategy for selecting initial candidate points if needed
        entropy_method: str = "distance",
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.n_min_samples = n_min_samples
        self.n_y_samples = n_y_samples
        self.alpha = alpha
        self.sampling_strategy = sampling_strategy
        self.entropy_method = entropy_method

        self.alphas = self._initialize_alphas()
        self.adapters = self._initialize_adapters(adapter)

    def _initialize_alphas(self) -> list[float]:
        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, self.n_quantiles + 1)
        ]
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            alphas.append(1 - (upper - lower))
        return alphas

    def _initialize_adapters(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[List[DtACI]]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return [
                DtACI(alpha=alpha, gamma_values=[0.05, 0.01, 0.1])
                for alpha in self.alphas
            ]
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        if self.adapters:
            for i, (adapter, beta) in enumerate(zip(self.adapters, betas)):
                updated_alpha = adapter.update(beta=beta)
                self.alphas[i] = updated_alpha

    def calculate_max_value_entropy_search(
        self,
        predictions_per_interval: List[ConformalBounds],
        n_jobs: int = -1,
    ) -> np.ndarray:
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        n_observations = len(predictions_per_interval[0].lower_bounds)
        idxs = np.random.randint(
            0, all_bounds.shape[1], size=(self.n_min_samples, n_observations)
        )
        sampled_funcs = np.zeros((self.n_min_samples, n_observations))
        for i in range(self.n_min_samples):
            sampled_funcs[i] = all_bounds[np.arange(n_observations), idxs[i]]
        min_values = np.min(sampled_funcs, axis=1)
        h_prior = _differential_entropy_estimator(
            min_values, self.alpha, method=self.entropy_method
        )

        def process_batch(batch_indices):
            batch_mes = np.zeros(len(batch_indices))
            for i, idx in enumerate(batch_indices):
                y_sample_idxs = np.random.randint(
                    0, all_bounds.shape[1], size=self.n_y_samples
                )
                candidate_y_samples = all_bounds[idx, y_sample_idxs]
                updated_min_values = np.minimum(
                    min_values[np.newaxis, :], candidate_y_samples[:, np.newaxis]
                )
                h_posteriors = np.array(
                    [
                        _differential_entropy_estimator(
                            updated_min_values[j],
                            self.alpha,
                            method=self.entropy_method,
                        )
                        for j in range(self.n_y_samples)
                    ]
                )
                sample_mes = h_prior - h_posteriors
                batch_mes[i] = np.mean(sample_mes)
            return batch_indices, batch_mes

        batch_size = min(
            100,
            max(1, n_observations // (joblib.cpu_count() if n_jobs <= 0 else n_jobs)),
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
