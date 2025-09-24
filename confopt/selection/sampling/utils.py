"""
Utility functions for sampling strategies in conformal prediction.

This module provides shared functionality used across different sampler implementations,
including alpha initialization strategies, adapter configuration for interval width
adjustment, and common preprocessing utilities. The module implements quantile-based
alpha initialization following symmetric quantile pairing methodology and provides
standardized interfaces for interval width adaptation using coverage rate feedback.

Key architectural components:
- Quantile-based alpha value initialization using symmetric pairing
- Multi-adapter configuration for complex sampling strategies
- Interval width update mechanisms with coverage rate feedback
- Validation utilities for sampling parameter constraints
- Conformal bounds preprocessing for efficient computation

Integration context:
The utilities in this module are designed to be used by all sampling strategy
implementations, providing consistent interfaces for common operations while
allowing each sampler to implement its specific acquisition logic.
"""

from typing import Optional, List, Literal
import warnings
from confopt.selection.adaptation import DtACI
from confopt.wrapping import ConformalBounds
import numpy as np


def initialize_quantile_alphas(n_quantiles: int) -> List[float]:
    """
    Initialize alpha values using symmetric quantile pairing methodology.

    This function implements a symmetric quantile initialization strategy where
    quantiles are paired symmetrically around the median, and alpha values are
    computed as the complement of the quantile interval width. This approach
    ensures balanced coverage across different uncertainty levels while maintaining
    proper nesting of prediction intervals.

    The methodology creates quantiles using equal spacing in the cumulative
    distribution, then pairs them symmetrically to form nested intervals with
    decreasing alpha values (increasing confidence levels).

    Args:
        n_quantiles: Number of quantiles to generate. Must be even to ensure
            symmetric pairing. Typical values are 4, 6, or 8 depending on the
            desired granularity of uncertainty quantification.

    Returns:
        List of alpha values in decreasing order, corresponding to increasing
        confidence levels. Length is n_quantiles // 2.

    Raises:
        ValueError: If n_quantiles is not even, preventing symmetric pairing.

    Example:
        >>> alphas = initialize_quantile_alphas(4)
        >>> print(alphas)  # [0.4, 0.2] for 60% and 80% confidence intervals
    """
    if n_quantiles % 2 != 0:
        raise ValueError("Number of quantiles must be even.")

    starting_quantiles = [
        round(i / (n_quantiles + 1), 2) for i in range(1, n_quantiles + 1)
    ]
    alphas = []
    half_length = len(starting_quantiles) // 2

    for i in range(half_length):
        lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
        alphas.append(1 - (upper - lower))
    return alphas


def initialize_multi_adapters(
    alphas: List[float], adapter: Optional[Literal["DtACI", "ACI"]] = None
) -> Optional[List[DtACI]]:
    """
    Initialize multiple adapters for dynamic interval width adjustment.

    This function creates individual adapters for each alpha value in multi-interval
    sampling strategies. Each adapter maintains its own coverage tracking and
    adjustment mechanism, allowing for independent width optimization across
    different confidence levels.

    The DtACI adapter uses multiple gamma values for robust adaptation, while
    ACI uses a single gamma value for simpler, more conservative adjustment.

    Args:
        alphas: List of alpha values, each requiring its own adapter instance.
            Each alpha corresponds to a different confidence level in the
            multi-interval sampling strategy.
        adapter: Adaptation strategy type. "DtACI" provides aggressive adaptation
            with multiple gamma parameters, while "ACI" provides conservative
            adaptation with a single gamma parameter.

    Returns:
        List of initialized adapters corresponding to each alpha value, or None
        if no adaptation is requested. Each adapter maintains independent state
        for coverage tracking and interval adjustment.

    Raises:
        ValueError: If adapter type is not recognized or supported.
    """
    if adapter is None:
        return None
    elif adapter == "DtACI":
        return [
            DtACI(
                alpha=alpha,
                gamma_values=[0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128],
            )
            for alpha in alphas
        ]
    elif adapter == "ACI":
        return [DtACI(alpha=alpha, gamma_values=[0.005]) for alpha in alphas]
    else:
        raise ValueError("adapter must be None, 'DtACI', or 'ACI'")


def initialize_single_adapter(
    alpha: float, adapter: Optional[Literal["DtACI", "ACI"]] = None
) -> Optional[DtACI]:
    """
    Initialize a single adapter for interval width adjustment in single-alpha samplers.

    This function creates a single adapter instance for samplers that operate with
    a single confidence level. The adapter tracks coverage rates and adjusts the
    alpha parameter to maintain target coverage while optimizing interval width.

    Args:
        alpha: The alpha value (miscoverage rate) for the prediction interval.
            Typical values range from 0.05 to 0.2, corresponding to 95% to 80%
            confidence levels.
        adapter: Adaptation strategy type. "DtACI" uses multiple gamma values
            for robust adaptation across different time scales, while "ACI"
            uses conservative single-gamma adaptation.

    Returns:
        Initialized adapter instance for the specified alpha value, or None
        if no adaptation is requested.

    Raises:
        ValueError: If adapter type is not recognized.
    """
    if adapter is None:
        return None
    elif adapter == "DtACI":
        return DtACI(
            alpha=alpha,
            gamma_values=[0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128],
        )
    elif adapter == "ACI":
        return DtACI(alpha=alpha, gamma_values=[0.005])
    else:
        raise ValueError("adapter must be None, 'DtACI', or 'ACI'")


def update_multi_interval_widths(
    adapters: Optional[List[DtACI]], alphas: List[float], betas: List[float]
) -> List[float]:
    """
    Update multiple interval widths using coverage rate feedback.

    This function applies adaptive interval width adjustment across multiple
    confidence levels simultaneously. Each adapter receives its corresponding
    observed coverage rate and updates its alpha parameter independently,
    allowing for fine-grained control over interval widths at different
    confidence levels.

    The update mechanism uses empirical coverage rates to adjust miscoverage
    parameters, tightening intervals when coverage exceeds targets and
    widening them when coverage falls short.

    Args:
        adapters: List of adapter instances, one per interval. If None,
            no adaptation is performed and original alphas are returned.
        alphas: Current alpha values for each interval. These serve as
            fallback values if no adapters are provided.
        betas: Observed coverage rates for each interval, used to drive
            the adaptation process. Should have same length as alphas.

    Returns:
        Updated alpha values after applying coverage-based adaptation.
        If no adapters are provided, returns the original alpha values.
    """
    if adapters:
        updated_alphas = []
        for i, (adapter, beta) in enumerate(zip(adapters, betas)):
            updated_alpha = adapter.update(beta=beta)
            updated_alphas.append(updated_alpha)
        return updated_alphas
    else:
        return alphas


def update_single_interval_width(
    adapter: Optional[DtACI], alpha: float, beta: float
) -> float:
    """
    Update a single interval width using observed coverage rate feedback.

    This function applies adaptive interval width adjustment for single-interval
    samplers. The adapter uses the observed coverage rate to adjust the alpha
    parameter, balancing between maintaining target coverage and optimizing
    interval efficiency.

    Args:
        adapter: The adapter instance for interval width adjustment. If None,
            a warning is issued and the original alpha is returned unchanged.
        alpha: Current alpha value (miscoverage rate) for the interval.
        beta: Observed coverage rate used to drive the adaptation process.

    Returns:
        Updated alpha value after applying coverage-based adaptation, or
        the original alpha if no adapter is provided.

    Warns:
        UserWarning: If update is requested but no adapter was initialized.
    """
    if adapter is not None:
        return adapter.update(beta=beta)
    else:
        warnings.warn(
            "'update_interval_width()' method was called, but no adapter was initialized."
        )
        return alpha


def validate_even_quantiles(n_quantiles: int, sampler_name: str = "sampler") -> None:
    """
    Validate quantile count constraints for symmetric sampling strategies.

    This validation function ensures that sampling strategies requiring symmetric
    quantile pairing receive appropriate input parameters. Many sampling methods
    rely on symmetric interval construction, which requires even numbers of
    quantiles for proper mathematical formulation.

    Args:
        n_quantiles: Number of quantiles to validate.
        sampler_name: Name of the sampler for descriptive error messages.

    Raises:
        ValueError: If n_quantiles is not even, preventing symmetric pairing.
    """
    if n_quantiles % 2 != 0:
        raise ValueError(f"Number of {sampler_name} quantiles must be even.")


def flatten_conformal_bounds(
    predictions_per_interval: List[ConformalBounds],
) -> np.ndarray:
    """
    Flatten conformal prediction bounds into efficient matrix representation.

    This preprocessing function transforms a list of ConformalBounds objects
    into a 2D numpy array for efficient vectorized operations. The flattening
    interleaves lower and upper bounds to maintain interval relationships
    while enabling fast numerical computations across all intervals and
    observations simultaneously.

    The resulting matrix structure supports efficient sampling operations,
    statistical computations, and vectorized interval manipulations required
    by acquisition functions.

    Args:
        predictions_per_interval: List of ConformalBounds objects, each containing
            lower_bounds and upper_bounds arrays. All bounds objects must have
            the same number of observations.

    Returns:
        Flattened bounds array of shape (n_observations, n_intervals * 2) where
        columns alternate between lower and upper bounds for each interval.

    Example:
        For 2 intervals and 3 observations:
        Column order: [interval1_lower, interval1_upper, interval2_lower, interval2_upper]
    """
    n_points = len(predictions_per_interval[0].lower_bounds)
    all_bounds = np.zeros((n_points, len(predictions_per_interval) * 2))
    for i, interval in enumerate(predictions_per_interval):
        all_bounds[:, i * 2] = interval.lower_bounds.flatten()
        all_bounds[:, i * 2 + 1] = interval.upper_bounds.flatten()
    return all_bounds
