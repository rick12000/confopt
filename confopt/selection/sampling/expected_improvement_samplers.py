"""
Expected Improvement acquisition strategy for conformal prediction optimization.

This module implements Expected Improvement (EI) acquisition functions using
conformal prediction intervals to quantify uncertainty. The approach extends
classical Bayesian optimization's Expected Improvement to conformal prediction
settings, enabling efficient acquisition function optimization without requiring
explicit posterior distributions.

Expected Improvement methodology:
The acquisition function computes the expected value of improvement over the
current best observation by sampling from prediction intervals. This provides
a natural exploration-exploitation balance, with high values indicating either
high predicted improvement (exploitation) or high uncertainty (exploration).

Mathematical foundation:
EI(x) = E[max(f_min - f(x), 0)] where f_min is the current best value and
the expectation is computed by Monte Carlo sampling from prediction intervals.

Key features:
- Monte Carlo estimation of expected improvement using interval sampling
- Adaptive current best value tracking for dynamic optimization
- Quantile-based interval construction with symmetric pairing
- Adaptive interval width adjustment using coverage feedback
- Efficient vectorized computation for large candidate sets

The module integrates with conformal prediction frameworks by accepting
ConformalBounds objects and providing standardized interfaces for uncertainty
quantification and acquisition function optimization.
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


class ExpectedImprovementSampler:
    """
    Expected Improvement acquisition strategy using conformal prediction intervals.

    This class implements Expected Improvement for optimization under uncertainty
    using conformal prediction intervals as uncertainty quantification. The
    sampler estimates expected improvement through Monte Carlo sampling from
    prediction intervals, providing a principled approach to balancing
    exploration and exploitation without requiring explicit posterior models.

    Methodological approach:
    - Constructs nested prediction intervals using symmetric quantile pairing
    - Estimates expected improvement via Monte Carlo sampling from intervals
    - Tracks current best value for improvement computation
    - Adapts interval widths using empirical coverage feedback

    The acquisition function naturally balances exploration (high uncertainty
    regions) with exploitation (promising low-value regions) by computing
    expected improvements over the current best observation.

    Performance characteristics:
    - O(n_samples * n_intervals * n_observations) for EI computation
    - Efficient vectorized operations for batch evaluation
    - Adaptive complexity through configurable sample count
    """

    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        current_best_value: float = float("inf"),
        num_ei_samples: int = 20,
    ):
        """
        Initialize Expected Improvement sampler with interval construction.

        Args:
            n_quantiles: Number of quantiles for interval construction. Must be even
                for symmetric pairing. Higher values provide finer uncertainty
                granularity but increase computational cost. Typical values: 4-8.
            adapter: Interval width adaptation strategy. "DtACI" provides aggressive
                multi-scale adaptation, "ACI" offers conservative adaptation,
                None disables adaptation.
            current_best_value: Initial best observed value for improvement
                calculation. Should be set to the minimum observed objective
                value. Updated automatically through update_best_value().
            num_ei_samples: Number of Monte Carlo samples for EI estimation.
                Higher values provide more accurate estimates but increase
                computational cost. Typical values: 10-50.
        """
        validate_even_quantiles(n_quantiles, "Expected Improvement")

        self.n_quantiles = n_quantiles
        self.current_best_value = current_best_value
        self.num_ei_samples = num_ei_samples

        # Initialize symmetric quantile-based alpha values
        self.alphas = initialize_quantile_alphas(n_quantiles)
        # Configure adapters for interval width adjustment
        self.adapters = initialize_multi_adapters(self.alphas, adapter)

    def update_best_value(self, value: float):
        """
        Update current best observed value for improvement computation.

        This method should be called after each new observation to maintain
        accurate improvement calculations. The best value serves as the baseline
        for computing expected improvements in subsequent acquisition decisions.

        Args:
            value: Newly observed objective value to compare with current best.
                For minimization problems, this updates the minimum observed value.
        """
        self.current_best_value = min(self.current_best_value, value)

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
        to maintain target coverage while optimizing interval efficiency for
        accurate expected improvement estimation.

        Args:
            betas: Observed coverage rates for each interval, in the same order
                as alpha values. Values should be in [0, 1] representing the
                fraction of true values falling within each interval.
        """
        self.alphas = update_multi_interval_widths(self.adapters, self.alphas, betas)

    def calculate_expected_improvement(
        self,
        predictions_per_interval: List[ConformalBounds],
    ) -> np.ndarray:
        """
        Calculate Expected Improvement for each candidate point using Monte Carlo sampling.

        This method estimates the expected improvement acquisition function by
        Monte Carlo sampling from prediction intervals. For each candidate point,
        multiple samples are drawn from its prediction intervals, improvements
        over the current best are computed, and the expectation is estimated
        as the sample mean.

        Methodology:
        1. Flatten prediction intervals into efficient matrix representation
        2. Generate random samples from intervals for each observation
        3. Compute improvements: max(0, current_best - sampled_value)
        4. Estimate expected improvement as sample mean
        5. Return negated values for minimization compatibility

        Args:
            predictions_per_interval: List of ConformalBounds objects containing
                lower and upper bounds for each confidence level. All bounds
                must have the same number of observations.

        Returns:
            Array of expected improvement values with shape (n_observations,).
            Values are negated for minimization (higher EI = more negative value).
            Points with higher expected improvement are more attractive for
            next evaluation.
        """
        # Flatten intervals into efficient matrix representation
        all_bounds = flatten_conformal_bounds(predictions_per_interval)

        n_observations = len(predictions_per_interval[0].lower_bounds)

        # Generate random sample indices for Monte Carlo estimation
        idxs = np.random.randint(
            0, all_bounds.shape[1], size=(n_observations, self.num_ei_samples)
        )

        # Extract interval samples for each observation
        realizations_per_observation = np.zeros((n_observations, self.num_ei_samples))
        for i in range(n_observations):
            realizations_per_observation[i] = all_bounds[i, idxs[i]]

        # Compute improvements over current best value
        improvements_per_observation = np.maximum(
            0, self.current_best_value - realizations_per_observation
        )

        # Estimate expected improvement as sample mean
        expected_improvements = np.mean(improvements_per_observation, axis=1)

        # Return negated for minimization compatibility
        return -expected_improvements
