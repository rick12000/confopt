import pytest
import numpy as np
from confopt.selection.sampling import (
    PessimisticLowerBoundSampler,
    LowerBoundSampler,
    ThompsonSampler,
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
    @pytest.mark.parametrize("adapter", [None, "DtACI"])
    def test_update_interval_width(self, interval_width, adapter):
        sampler = PessimisticLowerBoundSampler(
            interval_width=interval_width, adapter=adapter
        )

        beta = 0.5
        sampler.update_interval_width(beta)

        if adapter == "DtACI":
            assert sampler.alpha != pytest.approx(1 - interval_width)
        else:
            assert sampler.alpha == pytest.approx(1 - interval_width)


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
            (
                "adaptive_sequential_decay",
                2.0,
                lambda t, alpha, stag, max_beta: min(
                    np.sqrt(1 / t) * (1 + alpha) ** stag, max_beta
                ),
            ),
        ],
    )
    def test_update_exploration_step(self, beta_decay, c, expected_beta):
        sampler = LowerBoundSampler(beta_decay=beta_decay, c=c, beta_max=10.0)
        sampler.update_exploration_step()
        assert sampler.t == 2

        if beta_decay in ["inverse_square_root_decay", "logarithmic_decay"]:
            assert sampler.beta == pytest.approx(expected_beta(2))
        elif beta_decay == "adaptive_sequential_decay":
            assert sampler.beta == pytest.approx(
                expected_beta(2, sampler.alpha, sampler.stagnation, sampler.beta_max)
            )

    def test_update_stagnation(self):
        sampler = LowerBoundSampler()

        # Initial state
        assert sampler.stagnation == 0
        assert sampler.mu_max == float("-inf")

        # First value sets mu_max and keeps stagnation at 0
        sampler.update_stagnation(10.0)
        assert sampler.mu_max == 10.0
        assert sampler.stagnation == 0

        # Lower value increases stagnation
        sampler.update_stagnation(9.0)
        assert sampler.mu_max == 10.0
        assert sampler.stagnation == 1

        # Equal value increases stagnation
        sampler.update_stagnation(10.0)
        assert sampler.mu_max == 10.0
        assert sampler.stagnation == 2

        # Higher value resets stagnation and updates mu_max
        sampler.update_stagnation(12.0)
        assert sampler.mu_max == 12.0
        assert sampler.stagnation == 0

    def test_adaptive_sequential_decay(self):
        sampler = LowerBoundSampler(
            beta_decay="adaptive_sequential_decay", beta_max=10.0
        )

        # Check initial state
        assert sampler.beta == 1
        assert sampler.stagnation == 0

        # Simulate stagnation and check beta increases
        sampler.update_stagnation(5.0)  # First reward
        sampler.update_exploration_step()
        initial_beta = sampler.beta

        # No improvement - stagnation increases
        sampler.update_stagnation(4.0)
        sampler.update_exploration_step()
        stagnation_beta = sampler.beta

        # Beta should increase with stagnation
        assert stagnation_beta > initial_beta

        # Improvement - stagnation resets
        sampler.update_stagnation(10.0)
        sampler.update_exploration_step()
        reset_beta = sampler.beta

        # Beta should decrease after improvement
        assert reset_beta < stagnation_beta


class TestThompsonSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            ThompsonSampler(n_quantiles=5)

    def test_initialize_alphas(self):
        sampler = ThompsonSampler(n_quantiles=4)
        alphas = sampler._initialize_alphas()

        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)  # 1 - (0.8 - 0.2)
        assert alphas[1] == pytest.approx(0.8)  # 1 - (0.6 - 0.4)

    def test_fetch_alphas(self):
        sampler = ThompsonSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    @pytest.mark.parametrize("adapter", [None, "DtACI"])
    def test_update_interval_width(self, adapter):
        sampler = ThompsonSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter == "DtACI":
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas
