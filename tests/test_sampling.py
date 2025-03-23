import pytest
import numpy as np
from confopt.sampling import (
    PessimisticLowerBoundSampler,
    LowerBoundSampler,
    ThompsonSampler,
)
from confopt.adaptation import ACI, DtACI
from confopt.data_classes import QuantileInterval


class TestPessimisticLowerBoundSampler:
    @pytest.mark.parametrize("adapter_framework", ["ACI", "DtACI", None])
    def test_init_custom_parameters(self, adapter_framework):
        sampler = PessimisticLowerBoundSampler(
            interval_width=0.9, adapter=adapter_framework
        )
        assert sampler.interval_width == pytest.approx(0.9)
        assert sampler.alpha == pytest.approx(0.1)
        if adapter_framework == "ACI":
            assert isinstance(sampler.adapter, ACI)
        elif adapter_framework == "DtACI":
            assert isinstance(sampler.adapter, DtACI)
            assert hasattr(sampler, "expert_alphas")
        elif adapter_framework is None:
            assert sampler.adapter is None

    @pytest.mark.parametrize(
        "framework,expected_type,check_attr",
        [
            ("ACI", ACI, None),
            ("DtACI", DtACI, "expert_alphas"),
            (None, type(None), None),
        ],
    )
    def test_initialize_adapter(self, framework, expected_type, check_attr):
        sampler = PessimisticLowerBoundSampler()
        adapter = sampler._initialize_adapter(framework)
        assert isinstance(adapter, expected_type)
        if check_attr:
            assert hasattr(sampler, check_attr)
        if framework == "ACI":
            assert adapter.alpha == pytest.approx(0.2)
        elif framework == "DtACI":
            assert (adapter.alpha_t_values == [pytest.approx(0.2)]).all()

    def test_initialize_adapter_invalid(self):
        sampler = PessimisticLowerBoundSampler()
        with pytest.raises(ValueError, match="Unknown adapter framework"):
            sampler._initialize_adapter("InvalidAdapter")

    @pytest.mark.parametrize(
        "interval_width,expected_alpha", [(0.8, 0.2), (0.9, 0.1), (0.95, 0.05)]
    )
    def test_fetch_alpha(self, interval_width, expected_alpha):
        sampler = PessimisticLowerBoundSampler(interval_width=interval_width)
        assert sampler.fetch_alpha() == pytest.approx(expected_alpha)

    def test_fetch_quantile_interval(self):
        sampler = PessimisticLowerBoundSampler(interval_width=0.9)
        interval = sampler.fetch_quantile_interval()
        assert isinstance(interval, QuantileInterval)
        assert interval.lower_quantile == pytest.approx(0.05)
        assert interval.upper_quantile == pytest.approx(0.95)

    @pytest.mark.parametrize(
        "adapter_framework,breaches,should_raise",
        [
            ("ACI", [1], False),
            ("ACI", [1, 0], True),
            ("DtACI", [1, 0, 1, 0, 1, 0, 0, 1], False),
        ],
    )
    def test_update_interval_width(self, adapter_framework, breaches, should_raise):
        sampler = PessimisticLowerBoundSampler(adapter=adapter_framework)
        initial_alpha = sampler.alpha

        if should_raise:
            with pytest.raises(
                ValueError, match="ACI adapter requires a single breach indicator"
            ):
                sampler.update_interval_width(breaches)
        else:
            sampler.update_interval_width(breaches)
            assert sampler.alpha != initial_alpha

    @pytest.mark.parametrize(
        "interval_width,adapter_framework", [(0.8, None), (0.9, "ACI")]
    )
    def test_calculate_quantiles(self, interval_width, adapter_framework):
        sampler = PessimisticLowerBoundSampler(
            interval_width=interval_width, adapter=adapter_framework
        )
        interval = sampler._calculate_quantiles()
        expected_alpha = 1 - interval_width
        assert interval.lower_quantile == pytest.approx(expected_alpha / 2)
        assert interval.upper_quantile == pytest.approx(1 - (expected_alpha / 2))


class TestLowerBoundSampler:
    def test_init_custom_parameters(self):
        sampler = LowerBoundSampler(
            beta_decay="inverse_square_root_decay",
            c=2.0,
            interval_width=0.9,
            adapter_framework="ACI",
            upper_quantile_cap=0.5,
        )
        assert sampler.beta_decay == "inverse_square_root_decay"
        assert sampler.c == pytest.approx(2.0)
        assert sampler.interval_width == pytest.approx(0.9)
        assert sampler.alpha == pytest.approx(0.1)
        assert isinstance(sampler.adapter, ACI)
        assert sampler.upper_quantile_cap == pytest.approx(0.5)

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
    @pytest.mark.parametrize(
        "n_quantiles,adapter_framework,optimistic,expected_len",
        [(4, None, False, 2), (6, "ACI", True, 3)],
    )
    def test_init_parameters(
        self, n_quantiles, adapter_framework, optimistic, expected_len
    ):
        sampler = ThompsonSampler(
            n_quantiles=n_quantiles,
            adapter_framework=adapter_framework,
            enable_optimistic_sampling=optimistic,
        )
        assert sampler.n_quantiles == n_quantiles
        assert sampler.enable_optimistic_sampling is optimistic
        assert len(sampler.quantiles) == expected_len
        assert len(sampler.alphas) == expected_len

        if adapter_framework:
            assert len(sampler.adapters) == expected_len
        else:
            assert sampler.adapters is None

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

    def test_initialize_adapters_invalid(self):
        sampler = ThompsonSampler(n_quantiles=4)
        with pytest.raises(ValueError, match="Unknown adapter framework"):
            sampler._initialize_adapters("InvalidAdapter")

    def test_fetch_methods(self):
        sampler = ThompsonSampler(n_quantiles=4)

        # Test fetch_alphas
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

        # Test fetch_intervals
        intervals = sampler.fetch_intervals()
        assert len(intervals) == 2
        assert intervals[0].lower_quantile == pytest.approx(0.2)
        assert intervals[0].upper_quantile == pytest.approx(0.8)
        assert intervals[1].lower_quantile == pytest.approx(0.4)
        assert intervals[1].upper_quantile == pytest.approx(0.6)

    def test_update_interval_width(self):
        sampler = ThompsonSampler(n_quantiles=4, adapter_framework="ACI")
        initial_alphas = sampler.alphas.copy()
        breaches = [1, 0]

        sampler.update_interval_width(breaches)

        assert sampler.alphas[0] != initial_alphas[0]
        assert sampler.quantiles[0].lower_quantile == pytest.approx(
            sampler.alphas[0] / 2
        )
        assert sampler.quantiles[0].upper_quantile == pytest.approx(
            1 - (sampler.alphas[0] / 2)
        )
