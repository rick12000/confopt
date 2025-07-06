"""
Sampling-based acquisition strategies for conformal prediction optimization.

This package provides a comprehensive suite of acquisition strategies that use
conformal prediction intervals for uncertainty quantification in optimization
under uncertainty. The strategies implement different methodological approaches
to balance exploration and exploitation, each with distinct theoretical foundations
and computational characteristics.

Available acquisition strategies:
- Thompson Sampling: Probabilistic exploration through random interval sampling
- Expected Improvement: Classical Bayesian optimization extended to conformal settings
- Entropy Search: Information-theoretic acquisition with full model updates
- Max Value Entropy Search: Efficient entropy-based acquisition without refitting
- Bound-based Samplers: Conservative and UCB-style confidence bound strategies

The package provides standardized interfaces for alpha value management, adaptive
interval width adjustment, and efficient conformal bounds processing, enabling
consistent integration across different optimization pipelines and modeling
approaches.
"""

from .thompson_samplers import ThompsonSampler
from .expected_improvement_samplers import ExpectedImprovementSampler
from .entropy_samplers import EntropySearchSampler, MaxValueEntropySearchSampler
from .bound_samplers import PessimisticLowerBoundSampler, LowerBoundSampler
from .utils import (
    initialize_quantile_alphas,
    initialize_multi_adapters,
    initialize_single_adapter,
    update_multi_interval_widths,
    update_single_interval_width,
    fetch_alphas,
    validate_even_quantiles,
    flatten_conformal_bounds,
)

__all__ = [
    "ThompsonSampler",
    "ExpectedImprovementSampler",
    "EntropySearchSampler",
    "MaxValueEntropySearchSampler",
    "PessimisticLowerBoundSampler",
    "LowerBoundSampler",
    "initialize_quantile_alphas",
    "initialize_multi_adapters",
    "initialize_single_adapter",
    "update_multi_interval_widths",
    "update_single_interval_width",
    "fetch_alphas",
    "validate_even_quantiles",
    "flatten_conformal_bounds",
]
