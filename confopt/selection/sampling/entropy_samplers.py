"""
Max Value Entropy Search acquisition strategy for conformal prediction optimization.

This module implements entropy-based acquisition functions for optimization under
uncertainty. The strategy quantifies the expected reduction in uncertainty about
the global optimum value through information-theoretic measures, providing
principled exploration that balances between high-information regions and
promising optimization areas.

Key methodological approaches:
- Differential entropy estimation using distance-based and histogram methods
- Monte Carlo simulation for optimum value uncertainty quantification
- Efficient entropy computation without requiring model refitting
- Direct value-based entropy reduction for computational efficiency

The module provides the Max Value Entropy Search acquisition strategy:
- Max Value Entropy Search: Simplified entropy reduction for computational efficiency

Integration with conformal prediction enables robust uncertainty quantification
without requiring explicit probabilistic models, making the approaches suitable
for diverse optimization scenarios with complex objective functions.
"""

from typing import Optional, List, Literal
import numpy as np
import joblib
from confopt.wrapping import ConformalBounds
from confopt.selection.sampling.thompson_samplers import (
    flatten_conformal_bounds,
)
from confopt.selection.sampling.utils import (
    initialize_quantile_alphas,
    initialize_multi_adapters,
    update_multi_interval_widths,
    validate_even_quantiles,
)
import logging

logger = logging.getLogger(__name__)

# Try to import Cython implementation once at module level
try:
    from confopt.selection.sampling.cy_entropy import (
        cy_differential_entropy,
        cy_batch_differential_entropy,
    )

    CYTHON_AVAILABLE = True
except ImportError:
    logger.info(
        "Cython differential entropy implementation not available. Using pure Python fallback."
    )
    cy_differential_entropy = None
    cy_batch_differential_entropy = None
    CYTHON_AVAILABLE = False


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

    if CYTHON_AVAILABLE:
        return cy_differential_entropy(samples, method)

    # Pure Python fallback
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

        # Optimized Monte Carlo sampling using vectorized operations
        # Sample column indices for all paths and observations at once
        col_indices = np.random.randint(
            0, all_bounds.shape[1], size=(self.n_paths, n_observations)
        )

        # Use meshgrid-like approach for fully vectorized indexing
        # Create row indices that match the shape of col_indices
        row_indices = np.arange(n_observations)[np.newaxis, :].repeat(
            self.n_paths, axis=0
        )

        # Vectorized sampling: use advanced indexing to sample all at once
        sampled_matrix = all_bounds[row_indices.ravel(), col_indices.ravel()].reshape(
            self.n_paths, n_observations
        )

        # Find minimum across observations for each path (vectorized)
        optimums = np.min(sampled_matrix, axis=1)

        if CYTHON_AVAILABLE:
            entropy_of_optimum = cy_differential_entropy(optimums, self.entropy_method)
        else:
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

                # Conservative optimization: keep original logic with minimal vectorization
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

                    if CYTHON_AVAILABLE:
                        conditional_optimum_entropies[j] = cy_differential_entropy(
                            adjusted_optimums, self.entropy_method
                        )
                    else:
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
