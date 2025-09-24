"""
Tests for bound-based acquisition strategies in conformal prediction optimization.

This module tests the bound-based acquisition samplers that use prediction interval
bounds for optimization decisions. Tests focus on methodological correctness of
bound extraction, exploration-exploitation balance, adaptive interval width
adjustment, and mathematical properties of the acquisition functions.

Test coverage includes:
- PessimisticLowerBoundSampler: Conservative bound-based acquisition
- LowerBoundSampler: LCB-style exploration with decay schedules
- Adaptive interval width mechanisms and coverage feedback
- Mathematical properties and edge cases
"""

import pytest
import numpy as np
from unittest.mock import patch
from confopt.selection.sampling.bound_samplers import (
    PessimisticLowerBoundSampler,
    LowerBoundSampler,
)


class TestPessimisticLowerBoundSampler:
    """Test conservative acquisition strategy using pessimistic lower bounds."""

    @pytest.mark.parametrize("interval_width", [0.7, 0.8, 0.9, 0.95])
    def test_initialization_interval_width(self, interval_width):
        """Test initialization with different interval widths."""
        sampler = PessimisticLowerBoundSampler(interval_width=interval_width)

        assert sampler.interval_width == interval_width
        assert sampler.alpha == 1 - interval_width
        assert 0 < sampler.alpha < 1

    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_initialization_adapter_types(self, adapter):
        """Test initialization with different adapter configurations."""
        sampler = PessimisticLowerBoundSampler(interval_width=0.8, adapter=adapter)

        if adapter is None:
            assert sampler.adapter is None
        else:
            assert sampler.adapter is not None

    def test_fetch_alphas_single_value(self):
        """Test alpha retrieval returns single value list."""
        sampler = PessimisticLowerBoundSampler(interval_width=0.85)
        alphas = sampler.fetch_alphas()

        assert isinstance(alphas, list)
        assert len(alphas) == 1
        assert abs(alphas[0] - 0.15) < 1e-10

    def test_fetch_alphas_consistency(self):
        """Test alpha values remain consistent with interval width."""
        interval_widths = [0.7, 0.8, 0.9]
        for width in interval_widths:
            sampler = PessimisticLowerBoundSampler(interval_width=width)
            alphas = sampler.fetch_alphas()
            assert alphas[0] == 1 - width

    @patch("confopt.selection.sampling.bound_samplers.update_single_interval_width")
    def test_update_interval_width_with_adapter(self, mock_update):
        """Test interval width update with adapter present."""
        mock_update.return_value = 0.12
        sampler = PessimisticLowerBoundSampler(interval_width=0.8, adapter="ACI")
        original_alpha = sampler.alpha

        sampler.update_interval_width(beta=0.85)

        mock_update.assert_called_once_with(sampler.adapter, original_alpha, 0.85)
        assert sampler.alpha == 0.12

    @patch("confopt.selection.sampling.bound_samplers.update_single_interval_width")
    def test_update_interval_width_without_adapter(self, mock_update):
        """Test interval width update without adapter."""
        mock_update.return_value = 0.2
        sampler = PessimisticLowerBoundSampler(interval_width=0.8, adapter=None)
        original_alpha = sampler.alpha

        sampler.update_interval_width(beta=0.85)

        mock_update.assert_called_once_with(None, original_alpha, 0.85)
        assert sampler.alpha == 0.2

    @pytest.mark.parametrize("beta", [0.5, 0.75, 0.85, 0.95])
    def test_update_interval_width_coverage_range(self, beta):
        """Test update with different coverage rates."""
        sampler = PessimisticLowerBoundSampler(interval_width=0.8, adapter="ACI")
        sampler.alpha

        sampler.update_interval_width(beta=beta)

        # Alpha should be adjusted based on coverage
        assert isinstance(sampler.alpha, float)
        assert 0 < sampler.alpha < 1

    def test_interval_width_bounds(self):
        """Test interval width parameter bounds."""
        # Valid ranges
        for width in [0.5, 0.8, 0.99]:
            sampler = PessimisticLowerBoundSampler(interval_width=width)
            assert 0 < sampler.alpha < 1

        # Edge case: very high confidence
        sampler = PessimisticLowerBoundSampler(interval_width=0.999)
        assert abs(sampler.alpha - 0.001) < 1e-10

    def test_alpha_interval_width_relationship(self):
        """Test mathematical relationship between alpha and interval width."""
        widths = np.linspace(0.5, 0.95, 10)
        for width in widths:
            sampler = PessimisticLowerBoundSampler(interval_width=width)
            assert abs(sampler.alpha + sampler.interval_width - 1.0) < 1e-10


class TestLowerBoundSampler:
    """Test LCB acquisition strategy with adaptive exploration."""

    @pytest.mark.parametrize("interval_width", [0.7, 0.8, 0.9])
    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_initialization_inheritance(self, interval_width, adapter):
        """Test proper inheritance from PessimisticLowerBoundSampler."""
        sampler = LowerBoundSampler(interval_width=interval_width, adapter=adapter)

        assert sampler.interval_width == interval_width
        assert sampler.alpha == 1 - interval_width
        if adapter is None:
            assert sampler.adapter is None
        else:
            assert sampler.adapter is not None

    @pytest.mark.parametrize(
        "beta_decay", [None, "inverse_square_root_decay", "logarithmic_decay"]
    )
    def test_initialization_decay_strategies(self, beta_decay):
        """Test initialization with different decay strategies."""
        sampler = LowerBoundSampler(beta_decay=beta_decay)

        assert sampler.beta_decay == beta_decay
        assert sampler.t == 1
        assert sampler.beta == 1

    @pytest.mark.parametrize("c", [0.1, 1.0, 5.0, 10.0])
    def test_initialization_exploration_constant(self, c):
        """Test initialization with different exploration constants."""
        sampler = LowerBoundSampler(c=c)

        assert sampler.c == c

    @pytest.mark.parametrize("beta_max", [1.0, 5.0, 10.0, 20.0])
    def test_initialization_beta_max(self, beta_max):
        """Test initialization with different maximum beta values."""
        sampler = LowerBoundSampler(beta_max=beta_max)

        assert sampler.beta_max == beta_max

    def test_time_step_initialization(self):
        """Test initial time step and exploration parameter."""
        sampler = LowerBoundSampler()

        assert sampler.t == 1
        assert sampler.beta == 1
        assert sampler.mu_max == float("-inf")

    def test_update_exploration_step_time_increment(self):
        """Test time step increment in exploration update."""
        sampler = LowerBoundSampler()
        initial_t = sampler.t

        sampler.update_exploration_step()

        assert sampler.t == initial_t + 1

    @pytest.mark.parametrize(
        "decay_type", ["inverse_square_root_decay", "logarithmic_decay"]
    )
    def test_update_exploration_decay_formulas(self, decay_type):
        """Test exploration decay formula implementations."""
        c = 2.0
        sampler = LowerBoundSampler(beta_decay=decay_type, c=c)

        # Run multiple steps to test decay
        betas = []
        for _ in range(10):
            sampler.update_exploration_step()
            betas.append(sampler.beta)

        # Beta should generally decrease (with possible fluctuations due to log term)
        assert betas[-1] < betas[0]
        assert all(beta >= 0 for beta in betas)

    def test_update_exploration_inverse_square_root_decay(self):
        """Test inverse square root decay implementation."""
        c = 4.0
        sampler = LowerBoundSampler(beta_decay="inverse_square_root_decay", c=c)

        sampler.update_exploration_step()  # t=2
        expected_beta = np.sqrt(c / 2)
        assert abs(sampler.beta - expected_beta) < 1e-10

        sampler.update_exploration_step()  # t=3
        expected_beta = np.sqrt(c / 3)
        assert abs(sampler.beta - expected_beta) < 1e-10

    def test_update_exploration_logarithmic_decay(self):
        """Test logarithmic decay implementation."""
        c = 2.0
        sampler = LowerBoundSampler(beta_decay="logarithmic_decay", c=c)

        sampler.update_exploration_step()  # t=2
        expected_beta = np.sqrt((c * np.log(2)) / 2)
        assert abs(sampler.beta - expected_beta) < 1e-10

        sampler.update_exploration_step()  # t=3
        expected_beta = np.sqrt((c * np.log(3)) / 3)
        assert abs(sampler.beta - expected_beta) < 1e-10

    def test_update_exploration_no_decay(self):
        """Test behavior when no decay is specified."""
        sampler = LowerBoundSampler(beta_decay=None)
        initial_beta = sampler.beta

        for _ in range(5):
            sampler.update_exploration_step()
            assert sampler.beta == initial_beta

    def test_update_exploration_invalid_decay(self):
        """Test error handling for invalid decay strategies."""
        sampler = LowerBoundSampler()
        sampler.beta_decay = "invalid_decay"

        with pytest.raises(ValueError, match="beta_decay must be"):
            sampler.update_exploration_step()

    def test_calculate_ucb_predictions_basic(self, test_predictions_and_widths):
        """Test basic LCB calculation functionality."""
        point_estimates, interval_widths = test_predictions_and_widths
        sampler = LowerBoundSampler()

        lcb_values = sampler.calculate_ucb_predictions(point_estimates, interval_widths)

        assert lcb_values.shape == point_estimates.shape
        assert isinstance(lcb_values, np.ndarray)

    def test_calculate_ucb_predictions_formula(self, test_predictions_and_widths):
        """Test LCB formula implementation."""
        point_estimates, interval_widths = test_predictions_and_widths
        beta = 2.0
        sampler = LowerBoundSampler()
        sampler.beta = beta

        lcb_values = sampler.calculate_ucb_predictions(point_estimates, interval_widths)
        expected_values = point_estimates - beta * interval_widths

        np.testing.assert_array_almost_equal(lcb_values, expected_values)

    def test_calculate_ucb_predictions_beta_effect(self, test_predictions_and_widths):
        """Test effect of different beta values on LCB calculations."""
        point_estimates, interval_widths = test_predictions_and_widths

        beta_low = LowerBoundSampler()
        beta_low.beta = 0.5

        beta_high = LowerBoundSampler()
        beta_high.beta = 3.0

        lcb_low = beta_low.calculate_ucb_predictions(point_estimates, interval_widths)
        lcb_high = beta_high.calculate_ucb_predictions(point_estimates, interval_widths)

        # Higher beta should lead to lower (more conservative) LCB values
        assert np.all(lcb_high < lcb_low)

    def test_calculate_ucb_predictions_edge_cases(self):
        """Test LCB calculation with edge case inputs."""
        sampler = LowerBoundSampler()

        # Zero interval widths
        point_estimates = np.array([1, 2, 3])
        interval_widths = np.zeros(3)
        lcb_values = sampler.calculate_ucb_predictions(point_estimates, interval_widths)
        np.testing.assert_array_equal(lcb_values, point_estimates)

        # Single point
        single_point = np.array([5.0])
        single_width = np.array([1.0])
        lcb_single = sampler.calculate_ucb_predictions(single_point, single_width)
        assert lcb_single.shape == (1,)

    def test_calculate_ucb_predictions_negative_inputs(self):
        """Test LCB calculation with negative inputs."""
        sampler = LowerBoundSampler()
        sampler.beta = 1.5

        point_estimates = np.array([-2, -1, 0, 1, 2])
        interval_widths = np.array([0.5, 1.0, 1.5, 1.0, 0.5])

        lcb_values = sampler.calculate_ucb_predictions(point_estimates, interval_widths)
        expected = point_estimates - 1.5 * interval_widths

        np.testing.assert_array_almost_equal(lcb_values, expected)

    @pytest.mark.parametrize("t_steps", [1, 5, 10, 50])
    def test_exploration_decay_convergence(self, t_steps):
        """Test exploration parameter convergence over multiple steps."""
        sampler = LowerBoundSampler(beta_decay="logarithmic_decay", c=1.0)

        for _ in range(t_steps):
            sampler.update_exploration_step()

        # Beta should decrease as t increases
        assert sampler.beta < 1.0
        assert sampler.beta > 0
        assert sampler.t == t_steps + 1

    def test_exploration_decay_asymptotic_behavior(self):
        """Test asymptotic behavior of exploration decay."""
        sampler = LowerBoundSampler(beta_decay="inverse_square_root_decay", c=1.0)

        # Run many steps
        for _ in range(1000):
            sampler.update_exploration_step()

        # Beta should be very small but positive
        assert 0 < sampler.beta < 0.1

    def test_inheritance_method_access(self):
        """Test access to inherited methods from parent class."""
        sampler = LowerBoundSampler(interval_width=0.85, adapter="ACI")

        # Should have access to parent methods
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 1
        assert abs(alphas[0] - 0.15) < 1e-10

        # Should be able to update interval width
        sampler.update_interval_width(beta=0.8)
        assert isinstance(sampler.alpha, float)

    def test_mathematical_properties_lcb_ordering(self, test_predictions_and_widths):
        """Test mathematical ordering properties of LCB values."""
        point_estimates, interval_widths = test_predictions_and_widths
        sampler = LowerBoundSampler()
        sampler.beta = 1.0

        lcb_values = sampler.calculate_ucb_predictions(point_estimates, interval_widths)

        # LCB should be lower than point estimates when interval_widths > 0
        mask = interval_widths > 0
        assert np.all(lcb_values[mask] <= point_estimates[mask])

    def test_exploration_constant_impact(self, test_predictions_and_widths):
        """Test impact of exploration constant on acquisition behavior."""
        point_estimates, interval_widths = test_predictions_and_widths

        sampler_conservative = LowerBoundSampler(c=0.1)
        sampler_conservative.update_exploration_step()

        sampler_aggressive = LowerBoundSampler(c=10.0)
        sampler_aggressive.update_exploration_step()

        lcb_conservative = sampler_conservative.calculate_ucb_predictions(
            point_estimates, interval_widths
        )
        lcb_aggressive = sampler_aggressive.calculate_ucb_predictions(
            point_estimates, interval_widths
        )

        # Aggressive exploration should lead to lower LCB values
        assert np.mean(lcb_aggressive) < np.mean(lcb_conservative)

    def test_beta_max_constraint(self):
        """Test that beta values respect maximum constraint."""
        beta_max = 5.0
        sampler = LowerBoundSampler(
            beta_max=beta_max, c=100.0
        )  # Large c to potentially exceed beta_max

        # Even with large c, beta should not exceed beta_max in early iterations
        assert sampler.beta <= beta_max

    @pytest.mark.parametrize("array_size", [1, 10, 100, 1000])
    def test_calculate_ucb_predictions_scalability(self, array_size):
        """Test LCB calculation scalability with different array sizes."""
        sampler = LowerBoundSampler()

        point_estimates = np.random.uniform(-5, 5, array_size)
        interval_widths = np.random.uniform(0.1, 2.0, array_size)

        lcb_values = sampler.calculate_ucb_predictions(point_estimates, interval_widths)

        assert lcb_values.shape == (array_size,)
        assert len(lcb_values) == array_size

    def test_state_consistency_after_updates(self):
        """Test state consistency after multiple operations."""
        sampler = LowerBoundSampler(interval_width=0.8, adapter="ACI", c=2.0)
        original_interval_width = sampler.interval_width

        # Perform multiple operations
        sampler.update_exploration_step()
        sampler.update_interval_width(beta=0.85)
        sampler.update_exploration_step()

        # State should remain consistent
        assert isinstance(sampler.alpha, float)
        assert 0 < sampler.alpha < 1
        assert sampler.t >= 1
        assert sampler.beta >= 0
        # interval_width remains unchanged even when alpha is updated
        assert sampler.interval_width == original_interval_width
