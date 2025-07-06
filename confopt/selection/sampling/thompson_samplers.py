"""
Thompson sampling strategy for conformal prediction acquisition.

This module implements Thompson sampling for conformal prediction, providing
a probabilistic approach to exploration-exploitation trade-offs in optimization
under uncertainty. The implementation uses random sampling from prediction
intervals to approximate posterior sampling, enabling efficient acquisition
function optimization with proper uncertainty quantification.

Thompson sampling methodology:
The sampler randomly draws values from available prediction intervals to simulate
sampling from posterior distributions over the objective function. This approach
naturally balances exploration of uncertain regions with exploitation of
promising areas, providing theoretical guarantees for regret minimization in
bandit-style optimization problems.

Key features:
- Quantile-based interval construction with symmetric pairing
- Adaptive interval width adjustment using coverage feedback
- Optional optimistic sampling with point estimate integration
- Efficient vectorized sampling across multiple intervals
- Integration with conformal prediction uncertainty quantification

The module integrates with the broader conformal optimization framework by
accepting ConformalBounds objects and providing standardized interfaces for
alpha value management and interval width adaptation.
"""

from typing import Optional, List, Literal
import numpy as np
from confopt.wrapping import ConformalBounds
from confopt.selection.sampling.utils import (
    initialize_quantile_alphas,
    initialize_multi_adapters,
    update_multi_interval_widths,
    validate_even_quantiles,
    flatten_conformal_bounds,
)


class ThompsonSampler:
    """
    Thompson sampling acquisition strategy for conformal prediction optimization.

    This class implements Thompson sampling using conformal prediction intervals
    as approximations to posterior distributions. The sampler randomly draws
    values from prediction intervals to balance exploration and exploitation,
    providing a principled approach to acquisition function optimization under
    uncertainty.

    The implementation supports multiple confidence levels through quantile-based
    interval construction, adaptive interval width adjustment based on coverage
    feedback, and optional optimistic sampling for enhanced exploration.

    Methodological approach:
    - Constructs nested prediction intervals using symmetric quantile pairing
    - Samples randomly from flattened interval representations
    - Optionally incorporates point estimates for optimistic exploration
    - Adapts interval widths using empirical coverage rates

    Performance characteristics:
    - O(n_intervals * n_observations) sampling complexity
    - Efficient vectorized operations for large candidate sets
    - Minimal memory overhead through flattened representations
    """

    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        enable_optimistic_sampling: bool = False,
    ):
        """
        Initialize Thompson sampler with quantile-based interval construction.

        Args:
            n_quantiles: Number of quantiles for interval construction. Must be even
                to enable symmetric pairing. Higher values provide finer uncertainty
                granularity but increase computational cost. Typical values: 4-8.
            adapter: Interval width adaptation strategy. "DtACI" provides aggressive
                multi-scale adaptation, "ACI" offers conservative single-scale
                adaptation, None disables adaptation.
            enable_optimistic_sampling: Whether to incorporate point estimates for
                optimistic exploration. When enabled, sampled values are capped
                by point predictions to encourage exploitation of promising regions.
        """
        validate_even_quantiles(n_quantiles, "Thompson")

        self.n_quantiles = n_quantiles
        self.enable_optimistic_sampling = enable_optimistic_sampling

        # Initialize symmetric quantile-based alpha values
        self.alphas = initialize_quantile_alphas(n_quantiles)
        # Configure adapters for interval width adjustment
        self.adapters = initialize_multi_adapters(self.alphas, adapter)

    def fetch_alphas(self) -> List[float]:
        """
        Retrieve current alpha values for interval construction.

        Returns:
            List of alpha values (miscoverage rates) for each confidence level,
            ordered from lowest to highest confidence (decreasing alpha values).
        """
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        """
        Update interval widths using observed coverage rates.

        This method applies adaptive interval width adjustment based on empirical
        coverage feedback. Each interval's alpha parameter is updated independently
        using its corresponding observed coverage rate, allowing for fine-grained
        control over uncertainty quantification accuracy.

        Args:
            betas: Observed coverage rates for each interval, in the same order
                as the alpha values. Values should be in [0, 1] representing
                the fraction of true values falling within each interval.
        """
        self.alphas = update_multi_interval_widths(self.adapters, self.alphas, betas)

    def calculate_thompson_predictions(
        self,
        predictions_per_interval: List[ConformalBounds],
        point_predictions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate Thompson sampling predictions through random interval sampling.

        This method implements the core Thompson sampling logic by randomly
        selecting values from the available prediction intervals. The sampling
        process approximates drawing from posterior distributions over the
        objective function, enabling principled exploration-exploitation
        trade-offs.

        Methodology:
        1. Flatten prediction intervals into efficient matrix representation
        2. Randomly sample column indices for each observation
        3. Extract corresponding interval bounds
        4. Optionally apply optimistic capping using point estimates

        Args:
            predictions_per_interval: List of ConformalBounds objects containing
                lower and upper bounds for each confidence level. All bounds
                must have the same number of observations.
            point_predictions: Optional point estimates for optimistic sampling.
                When provided and optimistic sampling is enabled, sampled values
                are capped at point estimates to encourage exploitation.

        Returns:
            Array of sampled predictions with shape (n_observations,). Each value
            represents a random draw from the corresponding observation's
            prediction intervals, potentially capped by point estimates.
        """
        # Flatten intervals into efficient matrix representation
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        n_observations = len(predictions_per_interval[0].lower_bounds)
        n_intervals = all_bounds.shape[1]

        # Randomly sample interval bounds for each observation
        idx = np.random.randint(0, n_intervals, size=n_observations)
        sampled_bounds = np.array(
            [all_bounds[i, idx[i]] for i in range(n_observations)]
        )

        # Apply optimistic capping if enabled and point predictions available
        if self.enable_optimistic_sampling and point_predictions is not None:
            sampled_bounds = np.minimum(sampled_bounds, point_predictions)

        return sampled_bounds
