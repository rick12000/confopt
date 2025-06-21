from typing import Optional, List, Literal, Tuple
import numpy as np
from confopt.selection.adaptation import DtACI
import warnings
from confopt.wrapping import ConformalBounds
import joblib
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
    samples: np.ndarray, method: Literal["distance", "histogram"] = "distance"
) -> float:
    """
    Estimate the differential entropy of samples using various methods.

    Parameters:
    -----------
    samples : np.ndarray
        The samples used to estimate differential entropy
    method : str
        The method to use for entropy estimation:
        - 'distance': Based on nearest-neighbor distances (Vasicek estimator)
        - 'histogram': Based on binned probability density

    Returns:
    --------
    float: The estimated differential entropy
    """
    n_samples = len(samples)
    if n_samples <= 1:
        return 0.0

    # Check if all samples are identical (constant)
    if np.all(samples == samples[0]):
        return 0.0

    # Try to use the optimized Cython implementation if available
    try:
        from confopt.selection.cy_entropy import cy_differential_entropy

        return cy_differential_entropy(samples, method)
    except ImportError:
        # Fall back to pure Python implementation
        if method == "distance":
            # Vasicek estimator based on spacings
            m = int(np.sqrt(n_samples))  # Window size
            if m >= n_samples:
                m = max(1, n_samples // 2)

            sorted_samples = np.sort(samples)
            # Handle boundary cases by wrapping around
            wrapped_samples = np.concatenate([sorted_samples, sorted_samples[:m]])

            spacings = wrapped_samples[m : n_samples + m] - wrapped_samples[:n_samples]
            # Avoid log of zero by setting very small spacings to a minimum value
            spacings = np.maximum(spacings, np.finfo(float).eps)

            # Vasicek estimator formula
            entropy = np.sum(np.log(n_samples * spacings / m)) / n_samples
            return entropy

        elif method == "histogram":
            # Use Scott's rule for bin width selection
            std = np.std(samples)
            if std == 0:  # Handle constant samples
                return 0.0

            # Scott's rule: bin_width = 3.49 * std * n^(-1/3)
            bin_width = 3.49 * std * (n_samples ** (-1 / 3))
            data_range = np.max(samples) - np.min(samples)
            n_bins = max(1, int(np.ceil(data_range / bin_width)))

            # First get frequencies (counts) in each bin
            hist, bin_edges = np.histogram(samples, bins=n_bins)

            # Convert counts to probabilities (relative frequencies)
            probs = hist / n_samples

            # Remove zero probabilities (bins with no samples)
            positive_idx = probs > 0
            positive_probs = probs[positive_idx]

            # Bin width is needed for conversion from discrete to differential entropy
            bin_widths = np.diff(bin_edges)

            # Differential entropy = discrete entropy + log(bin width)
            # H(X) ≈ -Σ p(i)log(p(i)) + log(Δ)
            # where Δ is the bin width

            # Calculate discrete entropy component
            discrete_entropy = -np.sum(positive_probs * np.log(positive_probs))

            # Add log of average bin width to convert to differential entropy
            # This is a standard correction factor when estimating differential entropy with histograms
            avg_bin_width = np.mean(bin_widths)
            differential_entropy = discrete_entropy + np.log(avg_bin_width)

            return differential_entropy
        else:
            raise ValueError(
                f"Unknown entropy estimation method: {method}. Choose from 'distance' or 'histogram'."
            )


def _run_parallel_or_sequential(func, items, n_jobs=-1):
    if n_jobs == 1:
        results = []
        for item in items:
            results.append(func(item))
        return results
    else:
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
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
        entropy_method: Literal["distance", "histogram"] = "distance",
        use_caching: bool = True,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.n_paths = n_paths
        self.n_X_candidates = n_X_candidates
        self.n_y_candidates_per_x = n_y_candidates_per_x
        self.sampling_strategy = sampling_strategy
        self.entropy_method = entropy_method
        self.use_caching = use_caching
        self._entropy_cache = {}  # Cache for entropy calculations

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

    def _get_cached_entropy(self, samples):
        """Get cached entropy value for the given samples if available"""
        if not self.use_caching:
            return None

        # Use a hash of the sample data as the cache key
        key = hash(samples.tobytes())
        return self._entropy_cache.get(key)

    def _set_cached_entropy(self, samples, entropy_value):
        """Cache the entropy value for the given samples"""
        if not self.use_caching:
            return

        key = hash(samples.tobytes())
        self._entropy_cache[key] = entropy_value

        # Limit cache size to prevent memory issues
        if len(self._entropy_cache) > 1000:
            # Remove a random key if cache gets too large
            self._entropy_cache.pop(next(iter(self._entropy_cache)))

    def _calculate_best_x_entropy(
        self,
        all_bounds: np.ndarray,
        n_observations: int,
    ) -> Tuple[float, np.ndarray]:
        """Calculate the entropy of the best function value across the candidate space"""
        # Process in batches to manage memory for large observation sets
        batch_size = min(1000, self.n_paths)
        indices_for_paths = np.vstack([np.arange(n_observations)] * self.n_paths)
        min_values = np.zeros(self.n_paths)

        for batch_start in range(0, self.n_paths, batch_size):
            batch_end = min(batch_start + batch_size, self.n_paths)
            batch_size_actual = batch_end - batch_start

            # Generate random indices for this batch
            idxs = np.random.randint(
                0, all_bounds.shape[1], size=(batch_size_actual, n_observations)
            )

            # Process each path in the batch
            for i in range(batch_size_actual):
                path_idx = batch_start + i
                # Get samples for this path
                path_samples = all_bounds[np.arange(n_observations), idxs[i]]
                # Find minimum value
                min_values[path_idx] = np.min(path_samples)

        # Calculate entropy using the cached version if available
        cached_entropy = self._get_cached_entropy(min_values)
        if cached_entropy is not None:
            best_x_entropy = cached_entropy
        else:
            best_x_entropy = _differential_entropy_estimator(
                min_values, method=self.entropy_method
            )
            self._set_cached_entropy(min_values, best_x_entropy)

        return best_x_entropy, indices_for_paths

    def _select_candidates(
        self,
        predictions_per_interval: List[ConformalBounds],
        X_space: np.ndarray,
        best_historical_y: Optional[float] = None,
        best_historical_x: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Select candidate points for information gain calculation"""
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
            try:
                from scipy.stats import qmc

                # If X_space is not provided or is too small, fall back to random sampling
                if X_space is None or len(X_space) < capped_n_candidates:
                    return np.random.choice(
                        n_observations, size=capped_n_candidates, replace=False
                    )

                n_dim = X_space.shape[1]
                sampler = qmc.Sobol(d=n_dim, scramble=True)
                points = sampler.random(n=capped_n_candidates)

                # Normalize the input space
                X_min = np.min(X_space, axis=0)
                X_range = np.max(X_space, axis=0) - X_min
                X_range[X_range == 0] = 1.0  # Avoid division by zero
                X_normalized = (X_space - X_min) / X_range

                # Find closest points in the X_space to the Sobol points
                selected_indices = []
                for point in points:
                    distances = np.sqrt(np.sum((X_normalized - point) ** 2, axis=1))
                    selected_idx = np.argmin(distances)
                    selected_indices.append(selected_idx)

                return np.array(selected_indices)
            except ImportError:
                # Fall back to random sampling if scipy.stats.qmc is not available
                return np.random.choice(
                    n_observations, size=capped_n_candidates, replace=False
                )

        elif self.sampling_strategy == "perturbation":
            # If no historical best point is available or X_space is invalid, use random sampling
            if (
                X_space is None
                or len(X_space) < 1
                or best_historical_x is None
                or best_historical_y is None
            ):
                return np.random.choice(
                    n_observations, size=capped_n_candidates, replace=False
                )

            try:
                n_dim = X_space.shape[1]

                # Compute valid bounds for perturbation
                X_min = np.min(X_space, axis=0)
                X_max = np.max(X_space, axis=0)
                X_range = X_max - X_min

                # Scale perturbation based on data range
                perturbation_scale = 0.1
                # Ensure best_historical_x is 2D for proper broadcasting
                if best_historical_x.ndim == 1:
                    best_historical_x = best_historical_x.reshape(1, -1)

                # Compute perturbation bounds
                lower_bounds = np.maximum(
                    best_historical_x - perturbation_scale * X_range, X_min
                )
                upper_bounds = np.minimum(
                    best_historical_x + perturbation_scale * X_range, X_max
                )

                # Generate random perturbed points
                perturbed_points = np.random.uniform(
                    lower_bounds, upper_bounds, size=(capped_n_candidates, n_dim)
                )

                # Find closest X_space points to the perturbed points
                selected_indices = []
                for point in perturbed_points:
                    distances = np.sqrt(np.sum((X_space - point) ** 2, axis=1))
                    selected_idx = np.argmin(distances)
                    if selected_idx not in selected_indices:
                        selected_indices.append(selected_idx)

                # If we didn't get enough unique points, fill with random ones
                while len(selected_indices) < capped_n_candidates:
                    idx = np.random.randint(0, n_observations)
                    if idx not in selected_indices:
                        selected_indices.append(idx)

                return np.array(selected_indices)
            except Exception:
                # Fall back to random sampling if there are any issues
                return np.random.choice(
                    n_observations, size=capped_n_candidates, replace=False
                )
        else:
            # Default to uniform random sampling
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
        n_jobs: int = 1,
    ) -> np.ndarray:
        """
        Calculate the information gain for each candidate point.

        Optimized version with:
        1. Entropy calculation caching
        2. Memory management for large candidate spaces
        3. Efficient parallelization
        """
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        n_observations = len(predictions_per_interval[0].lower_bounds)

        # Calculate prior entropy with caching
        prior_entropy, indices_for_paths = self._calculate_best_x_entropy(
            all_bounds, n_observations
        )

        # Get historical best values for candidate selection
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

        # Select candidates more efficiently
        candidate_idxs = self._select_candidates(
            predictions_per_interval=predictions_per_interval,
            X_space=X_space,
            best_historical_y=best_historical_y,
            best_historical_x=best_historical_x,
        )

        def process_candidate(idx):
            """Process a single candidate with optimizations"""
            X_cand = X_space[idx].reshape(1, -1)
            # Generate all y candidate indices at once
            y_cand_idxs = np.random.randint(
                0, all_bounds.shape[1], size=self.n_y_candidates_per_x
            )
            # Get all y candidates at once
            y_range = all_bounds[idx, y_cand_idxs]

            information_gains = []

            # Process y candidates in smaller batches to manage memory
            batch_size = min(5, self.n_y_candidates_per_x)
            for batch_start in range(0, self.n_y_candidates_per_x, batch_size):
                batch_end = min(batch_start + batch_size, self.n_y_candidates_per_x)
                batch_y_candidates = y_range[batch_start:batch_end]

                for y_cand in batch_y_candidates:
                    # Create expanded dataset with the candidate point
                    X_expanded = np.vstack([X_train, X_cand])
                    y_expanded = np.append(y_train, y_cand)

                    # Create a copy of the estimator for this candidate
                    cand_estimator = deepcopy(conformal_estimator)

                    # Fit the estimator with the expanded dataset
                    cand_estimator.fit(
                        X_train=X_expanded,
                        y_train=y_expanded,
                        X_val=X_val,
                        y_val=y_val,
                        tuning_iterations=0,
                        random_state=1234,
                    )

                    # Get predictions using the updated model
                    cand_predictions = cand_estimator.predict_intervals(X_space)
                    cand_bounds = flatten_conformal_bounds(cand_predictions)

                    # Process paths in batches to reduce memory usage
                    path_batch_size = min(50, self.n_paths)
                    conditional_samples = np.zeros(self.n_paths)

                    for path_batch_start in range(0, self.n_paths, path_batch_size):
                        path_batch_end = min(
                            path_batch_start + path_batch_size, self.n_paths
                        )
                        batch_size_actual = path_batch_end - path_batch_start

                        # Generate random indices for this batch
                        cond_idxs_batch = np.random.randint(
                            0,
                            cand_bounds.shape[1],
                            size=(batch_size_actual, n_observations),
                        )

                        # Get samples and find minimizers for each path
                        for i in range(batch_size_actual):
                            path_idx = path_batch_start + i
                            # Extract samples for this path
                            path_idx_in_batch = i
                            path_samples = cand_bounds[
                                np.arange(n_observations),
                                cond_idxs_batch[path_idx_in_batch],
                            ]
                            # Find minimizer and its value
                            cond_minimizer = np.argmin(path_samples)
                            conditional_samples[path_idx] = path_samples[cond_minimizer]

                    # Calculate posterior entropy with caching
                    cached_posterior = self._get_cached_entropy(conditional_samples)
                    if cached_posterior is not None:
                        posterior_entropy = cached_posterior
                    else:
                        posterior_entropy = _differential_entropy_estimator(
                            conditional_samples, method=self.entropy_method
                        )
                        self._set_cached_entropy(conditional_samples, posterior_entropy)

                    # Calculate information gain
                    information_gains.append(prior_entropy - posterior_entropy)

            # Return the mean information gain for this candidate
            return idx, np.mean(information_gains) if information_gains else 0.0

        # Initialize information gain array
        information_gain = np.zeros(n_observations)

        # Process candidates in parallel or sequentially
        results = _run_parallel_or_sequential(
            process_candidate,
            candidate_idxs,
            n_jobs=n_jobs,
        )

        # Collect results
        for idx, ig_value in results:
            information_gain[idx] = ig_value

        return -information_gain


class MaxValueEntropySearchSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI"]] = None,
        n_min_samples: int = 100,
        n_y_samples: int = 20,
        entropy_method: Literal["distance", "histogram"] = "distance",
        use_caching: bool = True,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.n_min_samples = n_min_samples
        self.n_y_samples = n_y_samples
        self.entropy_method = entropy_method
        self.use_caching = use_caching
        self._entropy_cache = {}  # Cache for entropy calculations

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

    def _get_cached_entropy(self, samples):
        """Get cached entropy value for the given samples if available"""
        if not self.use_caching:
            return None

        key = hash(samples.tobytes())
        return self._entropy_cache.get(key)

    def _set_cached_entropy(self, samples, entropy_value):
        """Cache the entropy value for the given samples"""
        if not self.use_caching:
            return

        key = hash(samples.tobytes())
        self._entropy_cache[key] = entropy_value

        # Limit cache size to prevent memory issues
        if len(self._entropy_cache) > 1000:
            self._entropy_cache.pop(next(iter(self._entropy_cache)))

    def calculate_max_value_entropy_search(
        self,
        predictions_per_interval: List[ConformalBounds],
        n_jobs: int = 2,
    ) -> np.ndarray:
        """
        Calculate the max value entropy search acquisition function for each candidate point.

        Parameters:
        -----------
        predictions_per_interval: List of ConformalBounds
            Predicted confidence intervals for each point
        n_jobs: int
            Number of parallel jobs to run

        Returns:
        --------
        np.ndarray: Acquisition function values (negated for minimization)
        """
        n_observations = len(predictions_per_interval[0].lower_bounds)

        # Flatten conformal bounds for easier processing
        all_bounds = flatten_conformal_bounds(predictions_per_interval)

        # Generate indices for sampling the prior
        idxs = np.random.randint(
            0, all_bounds.shape[1], size=(self.n_min_samples, n_observations)
        )

        # Calculate min values
        min_values = np.zeros(self.n_min_samples)
        for i in range(self.n_min_samples):
            min_values[i] = np.min(all_bounds[np.arange(n_observations), idxs[i]])

        # Try to use Cython implementation if available
        try:
            from confopt.selection.cy_entropy import cy_differential_entropy

            h_prior = cy_differential_entropy(min_values, self.entropy_method)
        except ImportError:
            # Check cache first
            cached_entropy = self._get_cached_entropy(min_values)
            if cached_entropy is not None:
                h_prior = cached_entropy
            else:
                h_prior = _differential_entropy_estimator(
                    min_values, method=self.entropy_method
                )
                self._set_cached_entropy(min_values, h_prior)

        # Pre-calculate min/max values for fast-path checks
        min_of_mins = np.min(min_values)
        max_of_mins = np.max(min_values)

        def process_batch(batch_indices):
            """Process a batch of points"""
            batch_mes = np.zeros(len(batch_indices))

            for i, idx in enumerate(batch_indices):
                # Generate y samples
                y_idxs = np.random.randint(
                    0, all_bounds.shape[1], size=self.n_y_samples
                )
                y_samples = all_bounds[idx, y_idxs]

                h_posteriors = np.zeros(self.n_y_samples)

                # Process each y sample
                for j in range(self.n_y_samples):
                    y = y_samples[j]

                    # Fast path 1: y greater than all min values
                    if y > max_of_mins:
                        h_posteriors[j] = h_prior
                        continue

                    # Fast path 2: y smaller than all min values
                    if y < min_of_mins:
                        h_posteriors[j] = 0.0
                        continue

                    # Calculate updated min values
                    updated_mins = np.minimum(min_values, y)

                    # Check entropy cache
                    cached = self._get_cached_entropy(updated_mins)
                    if cached is not None:
                        h_posteriors[j] = cached
                    else:
                        # Try to use the Cython implementation
                        try:
                            from confopt.selection.cy_entropy import (
                                cy_differential_entropy,
                            )

                            h_posteriors[j] = cy_differential_entropy(
                                updated_mins, self.entropy_method
                            )
                        except ImportError:
                            h_posteriors[j] = _differential_entropy_estimator(
                                updated_mins, method=self.entropy_method
                            )
                        # Cache the result
                        self._set_cached_entropy(updated_mins, h_posteriors[j])

                # Calculate information gain
                h_diff = h_prior - h_posteriors
                sample_mes = np.maximum(0, h_diff)
                batch_mes[i] = np.mean(sample_mes)

            return batch_indices, batch_mes

        # Create batches for parallel processing
        batch_size = max(5, n_observations // (n_jobs * 2))
        all_indices = np.arange(n_observations)
        batches = [
            all_indices[i : min(i + batch_size, n_observations)]
            for i in range(0, n_observations, batch_size)
        ]

        # Process batches
        mes_values = np.zeros(n_observations)
        results = _run_parallel_or_sequential(
            process_batch,
            batches,
            n_jobs=n_jobs,
        )

        # Collect results
        for indices, values in results:
            mes_values[indices] = values

        return -mes_values
