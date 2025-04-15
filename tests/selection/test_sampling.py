import pytest
import numpy as np
from confopt.selection.sampling import (
    PessimisticLowerBoundSampler,
    LowerBoundSampler,
    ThompsonSampler,
    ExpectedImprovementSampler,
    InformationGainSampler,
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
        ],
    )
    def test_update_exploration_step(self, beta_decay, c, expected_beta):
        sampler = LowerBoundSampler(beta_decay=beta_decay, c=c, beta_max=10.0)
        sampler.update_exploration_step()
        assert sampler.t == 2
        assert sampler.beta == pytest.approx(expected_beta(2))


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


class TestExpectedImprovementSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            ExpectedImprovementSampler(n_quantiles=5)

    def test_initialize_alphas(self):
        sampler = ExpectedImprovementSampler(n_quantiles=4)
        alphas = sampler._initialize_alphas()

        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)  # 1 - (0.8 - 0.2)
        assert alphas[1] == pytest.approx(0.8)  # 1 - (0.6 - 0.4)

    def test_fetch_alphas(self):
        sampler = ExpectedImprovementSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_update_best_value(self):
        sampler = ExpectedImprovementSampler(current_best_value=0.5)
        assert sampler.current_best_value == 0.5

        # Test that it only updates if new value is better
        sampler.update_best_value(0.3)
        assert sampler.current_best_value == 0.5

        sampler.update_best_value(0.7)
        assert sampler.current_best_value == 0.7

    @pytest.mark.parametrize("adapter", [None, "DtACI"])
    def test_update_interval_width(self, adapter):
        sampler = ExpectedImprovementSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter == "DtACI":
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas


class TestInformationGainSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            InformationGainSampler(n_quantiles=5)

    def test_initialize_alphas(self):
        sampler = InformationGainSampler(n_quantiles=4)
        alphas = sampler._initialize_alphas()

        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)  # 1 - (0.8 - 0.2)
        assert alphas[1] == pytest.approx(0.8)  # 1 - (0.6 - 0.4)

    def test_fetch_alphas(self):
        sampler = InformationGainSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_parameter_initialization(self):
        sampler = InformationGainSampler(
            n_quantiles=6, n_samples=50, n_candidates=100, n_y_samples_per_x=10
        )
        assert sampler.n_samples == 50
        assert sampler.n_candidates == 100
        assert sampler.n_y_samples_per_x == 10
        assert len(sampler.alphas) == 3  # 6 quantiles = 3 alphas

    @pytest.mark.parametrize("adapter", [None, "DtACI"])
    def test_update_interval_width(self, adapter):
        sampler = InformationGainSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter == "DtACI":
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas
