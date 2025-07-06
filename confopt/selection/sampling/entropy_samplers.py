"""
Information-theoretic acquisition strategies for conformal prediction optimization.

This module implements entropy-based acquisition functions that use information gain
to guide optimization decisions. The strategies quantify the expected reduction in
uncertainty about the global optimum location through information-theoretic measures,
providing principled exploration that balances between high-information regions and
promising optimization areas.

Key methodological approaches:
- Differential entropy estimation using distance-based and histogram methods
- Monte Carlo simulation for optimum location uncertainty quantification
- Information gain computation through conditional entropy reduction
- Efficient candidate selection using various sampling strategies

The module provides two main acquisition strategies:
1. Entropy Search: Full information gain computation with model updates
2. Max Value Entropy Search: Simplified entropy reduction for computational efficiency

Integration with conformal prediction enables robust uncertainty quantification
without requiring explicit probabilistic models, making the approaches suitable
for diverse optimization scenarios with complex objective functions.
"""

from typing import Optional, List, Literal
import numpy as np
import joblib
from copy import deepcopy
from confopt.wrapping import ConformalBounds
from confopt.selection.sampling.thompson_samplers import (
    flatten_conformal_bounds,
    ThompsonSampler,
)
from confopt.selection.sampling.expected_improvement_samplers import (
    ExpectedImprovementSampler,
)
from confopt.selection.sampling.utils import (
    initialize_quantile_alphas,
    initialize_multi_adapters,
    update_multi_interval_widths,
    validate_even_quantiles,
)
from scipy.stats import qmc
import logging

logger = logging.getLogger(__name__)


def calculate_entropy(
    samples: np.ndarray, method: Literal["distance", "histogram"] = "distance"
) -> float:
    """
    Compute differential entropy using non-parametric estimation methods.

    This function estimates the differential entropy of continuous distributions
    from sample data using either distance-based (Vasicek) or histogram-based
    (Scott's rule) approaches. The estimation is crucial for information gain
    computation in entropy-based acquisition strategies.

    The implementation prioritizes accuracy and robustness, handling edge cases
    like identical samples and small sample sizes while providing fallback
    implementations when optimized Cython versions are unavailable.

    Args:
        samples: 1D array of sample values for entropy estimation. Should contain
            sufficient samples for reliable entropy estimation (typically >10).
        method: Estimation method. "distance" uses Vasicek k-nearest neighbor
            spacing estimator, "histogram" uses Scott's rule with discrete
            entropy correction.

    Returns:
        Estimated differential entropy value. Returns 0.0 for degenerate cases
        (<=1 samples or all identical values).
    """
    n_samples = len(samples)
    if n_samples <= 1:
        return 0.0
    if np.all(samples == samples[0]):
        return 0.0
    try:
        from confopt.selection.sampling import cy_differential_entropy

        return cy_differential_entropy(samples, method)
    except ImportError:
        logger.warning(
            "Cython differential entropy implementation not found. Falling back to pure Python. This may hurt performance significantly."
        )
        if method == "distance":
            # Vasicek estimator using k-nearest neighbor spacing
            k = int(np.sqrt(n_samples))
            if k >= n_samples:
                k = max(1, n_samples // 2)

            sorted_samples = np.sort(samples)
            total_log_spacing = 0.0

            for i in range(n_samples):
                # Calculate k-nearest neighbor distance
                left_idx = max(0, i - k // 2)
                right_idx = min(n_samples - 1, i + k // 2)

                # Ensure we have k neighbors
                if right_idx - left_idx + 1 < k:
                    if left_idx == 0:
                        right_idx = min(n_samples - 1, left_idx + k - 1)
                    else:
                        left_idx = max(0, right_idx - k + 1)

                spacing = max(
                    sorted_samples[right_idx] - sorted_samples[left_idx],
                    np.finfo(float).eps,
                )
                total_log_spacing += np.log(spacing * n_samples / k)

            entropy = total_log_spacing / n_samples

        elif method == "histogram":
            std = np.std(samples)
            if std == 0:
                return 0.0
            bin_width = 3.49 * std * (n_samples ** (-1 / 3))
            data_range = np.max(samples) - np.min(samples)
            n_bins = max(1, int(np.ceil(data_range / bin_width)))
            hist, bin_edges = np.histogram(samples, bins=n_bins)
            probs = hist / n_samples

            # Calculate discrete entropy only for positive probabilities
            discrete_entropy = 0.0
            for prob in probs:
                if prob > 0:
                    discrete_entropy -= prob * np.log(prob)

            bin_widths = np.diff(bin_edges)
            avg_bin_width = np.mean(bin_widths)
            entropy = discrete_entropy + np.log(avg_bin_width)
        else:
            raise ValueError(
                f"Unknown entropy estimation method: {method}. Choose from 'distance' or 'histogram'."
            )

    return entropy


def _run_parallel_or_sequential(func, items, n_jobs=-1):
    """
    Execute function over items with optional parallelization.

    Provides unified interface for parallel or sequential execution based on
    n_jobs parameter, enabling flexible computation strategies for different
    hardware configurations and problem sizes.

    Args:
        func: Function to apply to each item. Should accept single item argument.
        items: Iterable of items to process.
        n_jobs: Number of parallel jobs. Use 1 for sequential execution,
            -1 for all available cores.

    Returns:
        List of function results in same order as input items.
    """
    if n_jobs == 1:
        results = []
        for item in items:
            results.append(func(item))
        return results
    else:
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
            return joblib.Parallel()(joblib.delayed(func)(item) for item in items)


class EntropySearchSampler:
    """
    Entropy Search acquisition strategy using information gain maximization.

    This class implements full Entropy Search for optimization under uncertainty,
    computing information gain about the global optimum location through Monte Carlo
    simulation and conditional entropy reduction. The approach provides theoretically
    principled exploration by selecting candidates that maximally reduce uncertainty
    about the optimum location.

    The implementation uses conformal prediction intervals for uncertainty quantification
    and supports multiple candidate selection strategies for computational efficiency.
    Information gain is computed by comparing prior and posterior entropy of the
    optimum location distribution after hypothetical observations.

    Methodological approach:
    - Monte Carlo simulation of possible objective function realizations
    - Prior entropy computation for current optimum location uncertainty
    - Conditional entropy estimation after hypothetical observations
    - Information gain calculation as entropy reduction

    Performance characteristics:
    - High computational cost due to model refitting for each candidate
    - Excellent exploration properties with strong theoretical foundation
    - Suitable for expensive optimization problems where acquisition cost is justified
    """

    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        n_paths: int = 100,
        n_x_candidates: int = 10,
        n_y_candidates_per_x: int = 3,
        sampling_strategy: str = "uniform",
        entropy_measure: Literal["distance", "histogram"] = "distance",
    ):
        """
        Initialize Entropy Search sampler with configuration parameters.

        Args:
            n_quantiles: Number of quantiles for interval construction. Must be even
                for symmetric pairing. Higher values provide finer uncertainty
                resolution but increase computational cost.
            adapter: Interval width adaptation strategy for coverage maintenance.
                "DtACI" provides aggressive adaptation, "ACI" conservative adaptation.
            n_paths: Number of Monte Carlo paths for entropy estimation. Higher
                values provide more accurate entropy estimates but increase cost.
                Typical values: 50-200.
            n_x_candidates: Number of candidates to evaluate for information gain.
                Computational cost scales linearly with this parameter.
            n_y_candidates_per_x: Number of hypothetical y-values per candidate.
                Higher values improve information gain estimates but increase cost.
            sampling_strategy: Candidate selection strategy. Options include
                "uniform", "thompson", "expected_improvement", "sobol", "perturbation".
            entropy_measure: Entropy estimation method. "distance" uses Vasicek
                estimator, "histogram" uses Scott's rule with bin correction.
        """
        validate_even_quantiles(n_quantiles, "Information Gain")
        self.n_quantiles = n_quantiles
        self.n_paths = n_paths
        self.n_x_candidates = n_x_candidates
        self.n_y_candidates_per_x = n_y_candidates_per_x
        self.sampling_strategy = sampling_strategy
        self.entropy_measure = entropy_measure
        self.alphas = initialize_quantile_alphas(n_quantiles)
        self.adapters = initialize_multi_adapters(self.alphas, adapter)

    def fetch_alphas(self) -> List[float]:
        """
        Retrieve current alpha values for interval construction.

        Returns:
            List of alpha values (miscoverage rates) for each confidence level.
        """
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        """
        Update interval widths using observed coverage rates.

        Args:
            betas: Observed coverage rates for each interval, used to adjust
                alpha parameters for better coverage maintenance.
        """
        self.alphas = update_multi_interval_widths(self.adapters, self.alphas, betas)

    def get_entropy_of_optimum_location(
        self,
        all_bounds: np.ndarray,
        n_observations: int,
    ) -> float:
        """
        Compute entropy of global optimum location using Monte Carlo simulation.

        This method estimates the current uncertainty about the global optimum
        location by simulating multiple realizations of the objective function
        and computing the entropy of the resulting minimum locations.

        Args:
            all_bounds: Flattened conformal bounds matrix of shape
                (n_observations, n_intervals * 2).
            n_observations: Number of candidate points.

        Returns:
            Estimated entropy of optimum location distribution.
        """
        optimum_locations = np.zeros(self.n_paths)
        idxs = np.random.randint(
            0, all_bounds.shape[1], size=(self.n_paths, n_observations)
        )
        for i in range(self.n_paths):
            path_samples = all_bounds[np.arange(n_observations), idxs[i]]
            optimum_locations[i] = np.min(path_samples)
        optimum_location_entropy = calculate_entropy(
            optimum_locations, method=self.entropy_measure
        )
        return optimum_location_entropy

    def select_candidates(
        self,
        predictions_per_interval: List[ConformalBounds],
        candidate_space: np.ndarray,
        best_historical_y: Optional[float] = None,
        best_historical_x: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select candidate points for information gain evaluation using specified strategy.

        This method implements multiple candidate selection strategies to balance
        computational efficiency with exploration effectiveness. Different strategies
        are appropriate for different phases of optimization and problem characteristics.

        Args:
            predictions_per_interval: List of ConformalBounds objects for uncertainty
                quantification of candidate points.
            candidate_space: Array of candidate points with shape (n_candidates, n_dims).
            best_historical_y: Current best observed objective value for improvement-based
                strategies.
            best_historical_x: Current best observed point for perturbation-based
                strategies.

        Returns:
            Array of selected candidate indices for information gain evaluation.
        """
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        n_observations = len(predictions_per_interval[0].lower_bounds)
        capped_n_candidates = min(self.n_x_candidates, n_observations)
        if self.sampling_strategy == "thompson":
            thompson_sampler = ThompsonSampler()
            thompson_samples = thompson_sampler.calculate_thompson_predictions(
                predictions_per_interval=predictions_per_interval
            )
            candidates = np.argsort(thompson_samples)[:capped_n_candidates]
        elif self.sampling_strategy == "expected_improvement":
            if best_historical_y is None:
                best_historical_y = np.min(np.mean(all_bounds, axis=1))
            ei_sampler = ExpectedImprovementSampler(
                current_best_value=best_historical_y
            )
            ei_values = ei_sampler.calculate_expected_improvement(
                predictions_per_interval=predictions_per_interval
            )
            candidates = np.argsort(ei_values)[:capped_n_candidates]
        elif self.sampling_strategy == "sobol":
            if candidate_space is None or len(candidate_space) < capped_n_candidates:
                candidates = np.random.choice(
                    n_observations, size=capped_n_candidates, replace=False
                )
            n_dim = candidate_space.shape[1]
            sampler = qmc.Sobol(d=n_dim, scramble=True)
            points = sampler.random(n=capped_n_candidates)
            X_min = np.min(candidate_space, axis=0)
            X_range = np.max(candidate_space, axis=0) - X_min
            X_range[X_range == 0] = 1.0
            X_normalized = (candidate_space - X_min) / X_range
            selected_indices = []
            for point in points:
                distances = np.sqrt(np.sum((X_normalized - point) ** 2, axis=1))
                selected_idx = np.argmin(distances)
                selected_indices.append(selected_idx)
            candidates = np.array(selected_indices)
        elif self.sampling_strategy == "perturbation":
            if (
                candidate_space is None
                or len(candidate_space) < 1
                or best_historical_x is None
                or best_historical_y is None
            ):
                candidates = np.random.choice(
                    n_observations, size=capped_n_candidates, replace=False
                )
            n_dim = candidate_space.shape[1]
            X_min = np.min(candidate_space, axis=0)
            X_max = np.max(candidate_space, axis=0)
            X_range = X_max - X_min
            perturbation_scale = 0.1
            if best_historical_x.ndim == 1:
                best_historical_x = best_historical_x.reshape(1, -1)
            lower_bounds = np.maximum(
                best_historical_x - perturbation_scale * X_range, X_min
            )
            upper_bounds = np.minimum(
                best_historical_x + perturbation_scale * X_range, X_max
            )
            perturbed_points = np.random.uniform(
                lower_bounds, upper_bounds, size=(capped_n_candidates, n_dim)
            )
            selected_indices = []
            for point in perturbed_points:
                distances = np.sqrt(np.sum((candidate_space - point) ** 2, axis=1))
                selected_idx = np.argmin(distances)
                if selected_idx not in selected_indices:
                    selected_indices.append(selected_idx)
            while len(selected_indices) < capped_n_candidates:
                idx = np.random.randint(0, n_observations)
                if idx not in selected_indices:
                    selected_indices.append(idx)
            candidates = np.array(selected_indices)
        else:
            logger.warning(
                f"Unknown sampling strategy '{self.sampling_strategy}'. Defaulting to uniform random sampling."
            )
            candidates = np.random.choice(
                n_observations, size=capped_n_candidates, replace=False
            )
        return candidates

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
        Calculate information gain for candidate points through model updates.

        This method computes the expected information gain about the global optimum
        location by evaluating how much each candidate point would reduce uncertainty
        if observed. The computation involves fitting updated models with hypothetical
        observations and comparing resulting entropy estimates.

        Args:
            X_train: Training input data for model fitting.
            y_train: Training target values for model fitting.
            X_val: Validation input data for conformal calibration.
            y_val: Validation target values for conformal calibration.
            X_space: Full candidate space for entropy computation.
            conformal_estimator: Conformal predictor instance for model updates.
            predictions_per_interval: Current predictions for all candidates.
            n_jobs: Number of parallel jobs for computation.

        Returns:
            Array of information gain values (negated for minimization compatibility).
            Higher information gain (more negative values) indicates more informative
            candidates.
        """
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        n_observations = len(predictions_per_interval[0].lower_bounds)
        optimum_location_entropy = self.get_entropy_of_optimum_location(
            all_bounds, n_observations
        )
        combined_y = np.concatenate((y_train, y_val))
        combined_X = np.vstack((X_train, X_val))
        if self.sampling_strategy in ["expected_improvement", "perturbation"]:
            best_idx = np.argmin(combined_y)
            best_historical_y = combined_y[best_idx]
            best_historical_x = combined_X[best_idx].reshape(1, -1)
        else:
            best_historical_y = None
            best_historical_x = None

        candidate_idxs = self.select_candidates(
            predictions_per_interval=predictions_per_interval,
            candidate_space=X_space,
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

                conditional_samples = np.zeros(self.n_paths)
                cond_idxs = np.random.randint(
                    0,
                    cand_bounds.shape[1],
                    size=(self.n_paths, n_observations),
                )

                for i in range(self.n_paths):
                    path_samples = cand_bounds[
                        np.arange(n_observations),
                        cond_idxs[i],
                    ]
                    cond_minimizer = np.argmin(path_samples)
                    conditional_samples[i] = path_samples[cond_minimizer]

                conditional_optimum_location_entropy = calculate_entropy(
                    conditional_samples, method=self.entropy_measure
                )

                information_gains.append(
                    optimum_location_entropy - conditional_optimum_location_entropy
                )

            return idx, np.mean(information_gains) if information_gains else 0.0

        information_gains = np.zeros(n_observations)

        results = _run_parallel_or_sequential(
            process_candidate,
            candidate_idxs,
            n_jobs=n_jobs,
        )

        for idx, ig_value in results:
            information_gains[idx] = ig_value

        return -information_gains


class MaxValueEntropySearchSampler:
    """
    Max Value Entropy Search acquisition strategy for computational efficiency.

    This class implements a simplified version of Entropy Search that focuses on
    entropy reduction of the maximum (minimum for minimization) value rather than
    the full optimum location. This approach provides significant computational
    savings while maintaining strong exploration properties through information-
    theoretic principles.

    The method computes information gain by comparing the entropy of current
    optimum value estimates with conditional entropy after hypothetical observations,
    avoiding expensive model refitting while preserving exploration effectiveness.

    Methodological approach:
    - Direct entropy computation of optimum value distribution
    - Conditional entropy estimation through value capping
    - Information gain as entropy reduction without model updates
    - Efficient vectorized computation for large candidate sets

    Performance characteristics:
    - Significantly lower computational cost than full Entropy Search
    - Good exploration properties through information-theoretic guidance
    - Suitable for moderate to large-scale optimization problems
    """

    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        n_paths: int = 100,
        n_y_candidates_per_x: int = 20,
        entropy_method: Literal["distance", "histogram"] = "distance",
    ):
        """
        Initialize Max Value Entropy Search sampler.

        Args:
            n_quantiles: Number of quantiles for interval construction. Must be even
                for symmetric pairing. Higher values provide finer uncertainty
                resolution.
            adapter: Interval width adaptation strategy for coverage maintenance.
            n_paths: Number of Monte Carlo paths for entropy estimation. Higher
                values improve accuracy but increase computational cost.
            n_y_candidates_per_x: Number of hypothetical y-values per candidate
                for conditional entropy estimation.
            entropy_method: Entropy estimation method. "distance" uses Vasicek
                estimator, "histogram" uses Scott's rule.
        """
        validate_even_quantiles(n_quantiles, "Max Value Entropy Search")

        self.n_quantiles = n_quantiles
        self.n_paths = n_paths
        self.n_y_candidates_per_x = n_y_candidates_per_x
        self.entropy_method = entropy_method

        self.alphas = initialize_quantile_alphas(n_quantiles)
        self.adapters = initialize_multi_adapters(self.alphas, adapter)

    def fetch_alphas(self) -> List[float]:
        """
        Retrieve current alpha values for interval construction.

        Returns:
            List of alpha values (miscoverage rates) for each confidence level.
        """
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        """
        Update interval widths using observed coverage rates.

        Args:
            betas: Observed coverage rates for each interval, used to adjust
                alpha parameters for better coverage maintenance.
        """
        self.alphas = update_multi_interval_widths(self.adapters, self.alphas, betas)

    def calculate_information_gain(
        self,
        predictions_per_interval: List[ConformalBounds],
        n_jobs: int = 2,
    ) -> np.ndarray:
        """
        Calculate information gain using max value entropy reduction.

        This method computes information gain by estimating how much each candidate
        point would reduce uncertainty about the global optimum value. The approach
        uses direct entropy computation without requiring model refitting, providing
        computational efficiency while maintaining exploration effectiveness.

        Args:
            predictions_per_interval: List of ConformalBounds objects containing
                prediction intervals for all candidate points.
            n_jobs: Number of parallel jobs for batch processing.

        Returns:
            Array of information gain values (negated for minimization compatibility).
            Higher information gain (more negative values) indicates candidates that
            would provide more information about the optimum value.
        """
        n_observations = len(predictions_per_interval[0].lower_bounds)
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        idxs = np.random.randint(
            0, all_bounds.shape[1], size=(self.n_paths, n_observations)
        )

        optimums = np.zeros(self.n_paths)
        for i in range(self.n_paths):
            optimums[i] = np.min(all_bounds[np.arange(n_observations), idxs[i]])

        try:
            from confopt.selection.sampling import cy_differential_entropy

            entropy_of_optimum = cy_differential_entropy(optimums, self.entropy_method)
        except ImportError:
            logger.warning(
                "Cython differential entropy implementation not found. Falling back to pure Python. This may hurt performance significantly."
            )
            entropy_of_optimum = calculate_entropy(optimums, method=self.entropy_method)

        optimum_min = np.min(optimums)
        optimum_max = np.max(optimums)

        def process_batch(batch_indices):
            batch_information_gain = np.zeros(len(batch_indices))

            for i, idx in enumerate(batch_indices):
                y_idxs = np.random.randint(
                    0, all_bounds.shape[1], size=self.n_y_candidates_per_x
                )
                y_samples = all_bounds[idx, y_idxs]

                conditional_optimum_entropies = np.zeros(self.n_y_candidates_per_x)
                for j in range(self.n_y_candidates_per_x):
                    y = y_samples[j]

                    if y > optimum_max:
                        conditional_optimum_entropies[j] = entropy_of_optimum
                        continue

                    if y < optimum_min:
                        conditional_optimum_entropies[j] = 0.0
                        continue

                    adjusted_optimums = np.minimum(optimums, y)

                    try:
                        from confopt.selection.sampling import (
                            cy_differential_entropy,
                        )

                        conditional_optimum_entropies[j] = cy_differential_entropy(
                            adjusted_optimums, self.entropy_method
                        )
                    except ImportError:
                        logger.warning(
                            "Cython differential entropy implementation not found. Falling back to pure Python. This may hurt performance significantly."
                        )
                        conditional_optimum_entropies[j] = calculate_entropy(
                            adjusted_optimums, method=self.entropy_method
                        )

                information_gains = entropy_of_optimum - conditional_optimum_entropies
                positive_information_gains = np.maximum(0, information_gains)
                batch_information_gain[i] = np.mean(positive_information_gains)

            return batch_indices, batch_information_gain

        batch_size = max(5, n_observations // (n_jobs * 2))
        all_indices = np.arange(n_observations)
        batches = [
            all_indices[i : min(i + batch_size, n_observations)]
            for i in range(0, n_observations, batch_size)
        ]

        information_gains = np.zeros(n_observations)
        results = _run_parallel_or_sequential(
            process_batch,
            batches,
            n_jobs=n_jobs,
        )

        # Collect results
        for indices, values in results:
            information_gains[indices] = values

        return -information_gains
