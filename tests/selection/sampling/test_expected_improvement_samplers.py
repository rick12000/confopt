"""
Tests for Expected Improvement acquisition strategies in conformal prediction optimization.

This module tests the Expected Improvement sampler that estimates expected improvement
through Monte Carlo sampling from prediction intervals. Tests focus on mathematical
correctness of EI estimation, exploration-exploitation balance, adaptive interval
width adjustment, and acquisition function properties.

Test coverage includes:
- ExpectedImprovementSampler: Monte Carlo EI estimation with conformal intervals
- Best value tracking and improvement computation accuracy
- Adaptive interval width mechanisms and coverage feedback
- Mathematical properties of EI acquisition function
- Edge cases and boundary conditions
"""

import pytest
import numpy as np
from confopt.selection.sampling.expected_improvement_samplers import (
    ExpectedImprovementSampler,
)


class TestExpectedImprovementSampler:
    """Test Expected Improvement acquisition strategy using conformal prediction intervals."""

    @pytest.mark.parametrize("n_quantiles", [4, 6, 8])
    def test_initialization_even_quantiles(self, n_quantiles):
        """Test initialization with valid even quantile numbers."""
        sampler = ExpectedImprovementSampler(n_quantiles=n_quantiles)

        assert sampler.n_quantiles == n_quantiles
        assert len(sampler.alphas) == n_quantiles // 2

    @pytest.mark.parametrize("n_quantiles", [3, 5, 7])
    def test_initialization_odd_quantiles_raises_error(self, n_quantiles):
        """Test that odd quantile numbers raise validation errors."""
        with pytest.raises(ValueError):
            ExpectedImprovementSampler(n_quantiles=n_quantiles)

    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_initialization_adapter_types(self, adapter):
        """Test initialization with different adapter configurations."""
        sampler = ExpectedImprovementSampler(n_quantiles=4, adapter=adapter)

        if adapter is None:
            assert sampler.adapters is None
        else:
            assert sampler.adapters is not None
            assert len(sampler.adapters) == len(sampler.alphas)

    def test_update_best_value(self):
        """Test best value updates with improving values."""
        sampler = ExpectedImprovementSampler(current_best_value=10.0)

        # Better value should update
        sampler.update_best_value(5.0)
        assert sampler.current_best_value == 5.0

        # Should not update if new value is worse
        sampler.update_best_value(10.0)
        assert sampler.current_best_value == 5.0

    def test_fetch_alphas_returns_correct_format(self):
        """Test alpha retrieval returns proper list format."""
        sampler = ExpectedImprovementSampler(n_quantiles=6)
        alphas = sampler.fetch_alphas()

        assert isinstance(alphas, list)
        assert len(alphas) == 3  # n_quantiles // 2
        assert all(0 < alpha < 1 for alpha in alphas)

    def test_calculate_expected_improvement_negative_values(
        self, simple_conformal_bounds
    ):
        """Test EI values are negative for minimization compatibility."""
        sampler = ExpectedImprovementSampler(
            n_quantiles=4, num_ei_samples=20, current_best_value=0.1
        )

        ei_values = sampler.calculate_expected_improvement(simple_conformal_bounds)

        # All EI values should be non-positive (negated for minimization)
        assert np.all(ei_values <= 0)
        n_observations = len(simple_conformal_bounds[0].lower_bounds)
        assert ei_values.shape == (n_observations,)

    def test_calculate_expected_improvement_deterministic_sampling(
        self, simple_conformal_bounds
    ):
        """Test EI calculation consistency with fixed random seed."""
        sampler = ExpectedImprovementSampler(n_quantiles=4, num_ei_samples=50)

        # Calculate EI with fixed seed
        np.random.seed(42)
        ei_values1 = sampler.calculate_expected_improvement(simple_conformal_bounds)

        np.random.seed(42)
        ei_values2 = sampler.calculate_expected_improvement(simple_conformal_bounds)

        # Results should be identical with same seed
        np.testing.assert_array_almost_equal(ei_values1, ei_values2)
