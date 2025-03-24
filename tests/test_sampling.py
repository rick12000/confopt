import pytest
import numpy as np
from confopt.sampling import (
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

    @pytest.mark.parametrize("interval_width", [(0.8), (0.9)])
    def test_calculate_quantiles(self, interval_width):
        sampler = PessimisticLowerBoundSampler(interval_width=interval_width)
        interval = sampler._calculate_quantiles()
        expected_alpha = 1 - interval_width
        assert interval.lower_quantile == pytest.approx(expected_alpha / 2)
        assert interval.upper_quantile == pytest.approx(1 - (expected_alpha / 2))


class TestLowerBoundSampler:
    @pytest.mark.parametrize(
        "interval_width,cap,expected_lower,expected_upper",
        [(0.8, 0.5, 0.1, 0.5), (0.8, None, 0.1, 0.9)],
    )
    def test_calculate_quantiles(
        self, interval_width, cap, expected_lower, expected_upper
    ):
        sampler = LowerBoundSampler(
            interval_width=interval_width, upper_quantile_cap=cap
        )
        interval = sampler._calculate_quantiles()
        assert interval.lower_quantile == pytest.approx(expected_lower)
        assert interval.upper_quantile == pytest.approx(expected_upper)

    @pytest.mark.parametrize(
        "beta_decay,c,expected_beta",
        [
            ("inverse_square_root_decay", 2.0, lambda t: np.sqrt(2.0 / t)),
            ("logarithmic_decay", 2.0, lambda t: np.sqrt((2.0 * np.log(t)) / t)),
        ],
    )
    def test_update_exploration_step(self, beta_decay, c, expected_beta):
        sampler = LowerBoundSampler(beta_decay=beta_decay, c=c)
        sampler.update_exploration_step()
        assert sampler.t == 2
        assert sampler.beta == pytest.approx(expected_beta(2))


class TestThompsonSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(
            ValueError, match="Number of Thompson quantiles must be even"
        ):
            ThompsonSampler(n_quantiles=5)

    def test_initialize_quantiles_and_alphas(self):
        sampler = ThompsonSampler(n_quantiles=4)
        quantiles, alphas = sampler._initialize_quantiles_and_alphas()

        assert len(quantiles) == 2
        assert len(alphas) == 2

        assert quantiles[0].lower_quantile == pytest.approx(0.2)
        assert quantiles[0].upper_quantile == pytest.approx(0.8)
        assert alphas[0] == pytest.approx(0.4)  # 1 - (0.8 - 0.2)

        assert quantiles[1].lower_quantile == pytest.approx(0.4)
        assert quantiles[1].upper_quantile == pytest.approx(0.6)
        assert alphas[1] == pytest.approx(0.8)  # 1 - (0.6 - 0.4)

    def test_fetch_methods(self):
        sampler = ThompsonSampler(n_quantiles=4)

        # Test fetch_alphas
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)
