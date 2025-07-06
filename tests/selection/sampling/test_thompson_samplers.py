import pytest
import numpy as np
from unittest.mock import patch
from confopt.selection.sampling.thompson_samplers import (
    ThompsonSampler,
    flatten_conformal_bounds,
)
from confopt.selection.sampling.utils import initialize_quantile_alphas


class TestThompsonSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            ThompsonSampler(n_quantiles=5)

    def test_initialize_alphas_via_utils(self):
        # Test the utility function directly since the method is now abstracted
        alphas = initialize_quantile_alphas(4)
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_fetch_alphas(self):
        sampler = ThompsonSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_update_interval_width(self, adapter):
        sampler = ThompsonSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter in ["DtACI", "ACI"]:
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas

    @pytest.mark.parametrize(
        "enable_optimistic, point_predictions",
        [(False, None), (True, np.array([0.05, 0.35, 0.75, 0.25, 0.95]))],
    )
    def test_calculate_thompson_predictions(
        self, conformal_bounds, enable_optimistic, point_predictions
    ):
        sampler = ThompsonSampler(
            n_quantiles=4, enable_optimistic_sampling=enable_optimistic
        )

        fixed_indices = np.array([0, 3, 5, 1, 4])

        with patch.object(np.random, "randint", return_value=fixed_indices):
            result = sampler.calculate_thompson_predictions(
                predictions_per_interval=conformal_bounds,
                point_predictions=point_predictions,
            )

        flattened_bounds = flatten_conformal_bounds(conformal_bounds)
        expected_sampled_bounds = np.array(
            [flattened_bounds[i, idx] for i, idx in enumerate(fixed_indices)]
        )

        if enable_optimistic and point_predictions is not None:
            expected = np.minimum(expected_sampled_bounds, point_predictions)
        else:
            expected = expected_sampled_bounds

        np.testing.assert_array_almost_equal(result, expected)

    def test_thompson_predictions_randomized(self, conformal_bounds):
        np.random.seed(42)

        sampler = ThompsonSampler(n_quantiles=4)
        predictions = sampler.calculate_thompson_predictions(conformal_bounds)
        assert len(predictions) == 5

        sampler = ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=True)
        point_predictions = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        predictions = sampler.calculate_thompson_predictions(
            conformal_bounds,
            point_predictions=point_predictions,
        )
        assert len(predictions) == 5
        assert np.all(predictions <= point_predictions) or np.all(predictions < np.inf)


def test_flatten_conformal_bounds_detailed(simple_conformal_bounds):
    flattened = flatten_conformal_bounds(simple_conformal_bounds)

    assert flattened.shape == (3, 4)

    expected = np.array(
        [
            [0.1, 0.4, 0.2, 0.5],
            [0.3, 0.6, 0.4, 0.7],
            [0.5, 0.8, 0.6, 0.9],
        ]
    )

    np.testing.assert_array_equal(flattened, expected)


def test_flatten_conformal_bounds(conformal_bounds):
    flattened = flatten_conformal_bounds(conformal_bounds)

    assert flattened.shape == (5, len(conformal_bounds) * 2)

    for i, interval in enumerate(conformal_bounds):
        assert np.array_equal(flattened[:, i * 2], interval.lower_bounds.flatten())
        assert np.array_equal(flattened[:, i * 2 + 1], interval.upper_bounds.flatten())
