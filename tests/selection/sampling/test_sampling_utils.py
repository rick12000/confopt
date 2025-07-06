import pytest
import numpy as np
from confopt.selection.sampling.utils import (
    initialize_quantile_alphas,
    initialize_multi_adapters,
    initialize_single_adapter,
    update_multi_interval_widths,
    update_single_interval_width,
    validate_even_quantiles,
    flatten_conformal_bounds,
)


@pytest.mark.parametrize("n_quantiles", [2, 4, 6, 8, 10])
def test_initialize_quantile_alphas_even_counts(n_quantiles):
    """Test quantile alpha initialization with valid even counts."""
    alphas = initialize_quantile_alphas(n_quantiles)

    # Should return half the input quantiles
    assert len(alphas) == n_quantiles // 2

    # All alphas should be in valid range
    assert all(0 < alpha < 1 for alpha in alphas)

    # Spot check:
    if n_quantiles == 4:
        expected_alphas = [0.4, 0.8]
        np.testing.assert_allclose(alphas, expected_alphas, rtol=1e-10)


@pytest.mark.parametrize("n_quantiles", [1, 3, 5, 7])
def test_initialize_quantile_alphas_odd_counts_raises(n_quantiles):
    """Test that odd quantile counts raise appropriate errors."""
    with pytest.raises(ValueError):
        initialize_quantile_alphas(n_quantiles)


def test_update_multi_interval_widths_with_adapters(coverage_feedback):
    """Test multi-interval width updates with adaptation."""
    alphas = [0.2, 0.1, 0.05]
    adapters = initialize_multi_adapters(alphas, "DtACI")

    # Store initial alphas
    initial_alphas = alphas.copy()

    # Update with coverage feedback
    updated_alphas = update_multi_interval_widths(adapters, alphas, coverage_feedback)

    # Should return list of same length
    assert len(updated_alphas) == len(initial_alphas)

    # Alphas should be updated (likely different from initial)
    assert isinstance(updated_alphas, list)
    assert all(isinstance(alpha, float) for alpha in updated_alphas)

    # All alphas should remain in valid range
    assert all(0 < alpha < 1 for alpha in updated_alphas)


def test_update_multi_interval_widths_without_adapters():
    """Test multi-interval width updates without adaptation."""
    alphas = [0.2, 0.1, 0.05]
    betas = [0.8, 0.9, 0.95]

    updated_alphas = update_multi_interval_widths(None, alphas, betas)

    # Should return original alphas unchanged
    assert updated_alphas == alphas


def test_update_single_interval_width():
    """Test single interval width update with adaptation."""
    alpha = 0.1
    adapter = initialize_single_adapter(alpha, "DtACI")
    beta = 0.85

    updated_alpha = update_single_interval_width(adapter, alpha, beta)

    # Should return a float in valid range
    assert isinstance(updated_alpha, float)
    assert 0 < updated_alpha < 1
    assert updated_alpha != alpha  # Should be updated


def test_validate_even_quantiles_valid():
    """Test validation passes for even quantiles."""
    # Should not raise any exception
    validate_even_quantiles(4, "test_sampler")
    validate_even_quantiles(6, "another_sampler")


@pytest.mark.parametrize("n_quantiles", [1, 3, 5, 7])
def test_validate_even_quantiles_invalid(n_quantiles):
    """Test validation raises for odd quantiles."""
    with pytest.raises(
        ValueError, match="Number of test_sampler quantiles must be even"
    ):
        validate_even_quantiles(n_quantiles, "test_sampler")


def test_flatten_conformal_bounds_structure(multi_interval_bounds):
    """Test conformal bounds flattening produces correct structure."""
    flattened = flatten_conformal_bounds(multi_interval_bounds)

    n_obs = len(multi_interval_bounds[0].lower_bounds)
    n_intervals = len(multi_interval_bounds)
    expected_shape = (n_obs, n_intervals * 2)

    # Should have correct shape
    assert flattened.shape == expected_shape

    # Should be numpy array
    assert isinstance(flattened, np.ndarray)


def test_flatten_conformal_bounds_interleaving(small_dataset):
    """Test that bounds are correctly interleaved in flattened representation."""
    flattened = flatten_conformal_bounds(small_dataset)

    # Check that columns alternate between lower and upper bounds
    for i, bounds in enumerate(small_dataset):
        lower_col = i * 2
        upper_col = i * 2 + 1

        np.testing.assert_array_equal(flattened[:, lower_col], bounds.lower_bounds)
        np.testing.assert_array_equal(flattened[:, upper_col], bounds.upper_bounds)


def test_flatten_conformal_bounds_preserves_intervals(nested_intervals):
    """Test that flattening preserves interval relationships."""
    flattened = flatten_conformal_bounds(nested_intervals)

    # Check that nested relationships are preserved
    for obs_idx in range(flattened.shape[0]):
        # Extract bounds for this observation
        wide_lower, wide_upper = flattened[obs_idx, 0], flattened[obs_idx, 1]
        med_lower, med_upper = flattened[obs_idx, 2], flattened[obs_idx, 3]
        narrow_lower, narrow_upper = flattened[obs_idx, 4], flattened[obs_idx, 5]

        # Verify nesting: narrow ⊆ medium ⊆ wide
        assert wide_lower <= med_lower <= narrow_lower
        assert narrow_upper <= med_upper <= wide_upper
