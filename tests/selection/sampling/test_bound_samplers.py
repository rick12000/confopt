import pytest
import numpy as np
from confopt.selection.sampling.bound_samplers import (
    PessimisticLowerBoundSampler,
    LowerBoundSampler,
)


class TestPessimisticLowerBoundSampler:
    @pytest.mark.parametrize(
        "interval_width,expected_alpha", [(0.8, 0.2), (0.9, 0.1), (0.95, 0.05)]
    )
    def test_fetch_alphas(self, interval_width, expected_alpha):
        sampler = PessimisticLowerBoundSampler(interval_width=interval_width)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 1
        assert alphas[0] == pytest.approx(expected_alpha)

    @pytest.mark.parametrize("interval_width", [0.8, 0.9])
    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_update_interval_width(self, interval_width, adapter):
        sampler = PessimisticLowerBoundSampler(
            interval_width=interval_width, adapter=adapter
        )

        beta = 0.5
        sampler.update_interval_width(beta)

        if adapter in ["DtACI", "ACI"]:
            assert sampler.alpha != pytest.approx(1 - interval_width)
        else:
            assert sampler.alpha == pytest.approx(1 - interval_width)

    def test_adapter_initialization(self):
        sampler_aci = PessimisticLowerBoundSampler(interval_width=0.8, adapter="ACI")
        assert sampler_aci.adapter is not None
        assert sampler_aci.adapter.gamma_values.tolist() == [0.005]

        sampler_dtaci = PessimisticLowerBoundSampler(
            interval_width=0.8, adapter="DtACI"
        )
        assert sampler_dtaci.adapter is not None
        assert sampler_dtaci.adapter.gamma_values.tolist() == [0.05, 0.01, 0.1]


class TestLowerBoundSampler:
    @pytest.mark.parametrize(
        "interval_width,expected_alpha",
        [(0.8, 0.2)],
    )
    def test_fetch_alphas(self, interval_width, expected_alpha):
        sampler = LowerBoundSampler(interval_width=interval_width)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 1
        assert alphas[0] == pytest.approx(expected_alpha)

    @pytest.mark.parametrize(
        "beta_decay,c,expected_beta",
        [
            ("inverse_square_root_decay", 2.0, lambda t: np.sqrt(2.0 / t)),
            ("logarithmic_decay", 2.0, lambda t: np.sqrt((2.0 * np.log(t)) / t)),
        ],
    )
    def test_update_exploration_step(self, beta_decay, c, expected_beta):
        sampler = LowerBoundSampler(beta_decay=beta_decay, c=c, beta_max=10.0)
        sampler.update_exploration_step()
        assert sampler.t == 2
        assert sampler.beta == pytest.approx(expected_beta(2))

    def test_calculate_ucb_predictions_with_point_estimates(self, conformal_bounds):
        sampler = LowerBoundSampler(interval_width=0.8, beta_decay=None)
        sampler.beta = 0.5

        point_estimates = np.array([0.5, 0.7, 0.3, 0.9, 0.6])
        interval_width = np.array([0.2, 0.1, 0.3, 0.05, 0.15])

        result = sampler.calculate_ucb_predictions(
            predictions_per_interval=conformal_bounds,
            point_estimates=point_estimates,
            interval_width=interval_width,
        )

        expected = point_estimates - 0.5 * interval_width
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_ucb_predictions_from_intervals(self, conformal_bounds):
        sampler = LowerBoundSampler(interval_width=0.8, beta_decay=None)
        sampler.beta = 0.75

        result = sampler.calculate_ucb_predictions(
            predictions_per_interval=conformal_bounds
        )

        interval = conformal_bounds[0]
        point_estimates = (interval.upper_bounds + interval.lower_bounds) / 2
        width = (interval.upper_bounds - interval.lower_bounds) / 2
        expected = point_estimates - 0.75 * width

        np.testing.assert_array_almost_equal(result, expected)
