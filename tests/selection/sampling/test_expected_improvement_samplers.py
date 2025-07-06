import pytest
import numpy as np
from unittest.mock import patch
from confopt.selection.sampling.expected_improvement_samplers import (
    ExpectedImprovementSampler,
)
from confopt.selection.sampling.utils import initialize_quantile_alphas


class TestExpectedImprovementSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            ExpectedImprovementSampler(n_quantiles=5)

    def test_initialize_alphas_via_utils(self):
        # Test the utility function directly since the method is now abstracted
        alphas = initialize_quantile_alphas(4)
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_fetch_alphas(self):
        sampler = ExpectedImprovementSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_update_best_value(self):
        sampler = ExpectedImprovementSampler(current_best_value=0.5)
        assert sampler.current_best_value == 0.5

        sampler.update_best_value(0.7)
        assert sampler.current_best_value == 0.5

        sampler.update_best_value(0.3)
        assert sampler.current_best_value == 0.3

    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_update_interval_width(self, adapter):
        sampler = ExpectedImprovementSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter in ["DtACI", "ACI"]:
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas

    def test_calculate_expected_improvement_detailed(self, simple_conformal_bounds):
        sampler = ExpectedImprovementSampler(current_best_value=0.4, num_ei_samples=1)

        with patch.object(
            np.random,
            "randint",
            side_effect=[np.array([[0], [1], [2]]), np.array([[0], [1], [2]])],
        ):
            result = sampler.calculate_expected_improvement(
                predictions_per_interval=simple_conformal_bounds
            )

        expected = np.array([-0.3, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

        sampler.current_best_value = 0.6
        with patch.object(
            np.random,
            "randint",
            side_effect=[np.array([[0], [1], [2]]), np.array([[0], [1], [2]])],
        ):
            result = sampler.calculate_expected_improvement(
                predictions_per_interval=simple_conformal_bounds
            )

        expected = np.array([-0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_expected_improvement_randomized(self, conformal_bounds):
        np.random.seed(42)

        sampler = ExpectedImprovementSampler(current_best_value=0.5, num_ei_samples=10)
        ei = sampler.calculate_expected_improvement(
            predictions_per_interval=conformal_bounds
        )

        assert len(ei) == 5
        assert np.all(ei <= 0)
