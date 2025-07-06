import pytest
import numpy as np
from confopt.selection.sampling.utils import (
    initialize_quantile_alphas,
    initialize_multi_adapters,
    initialize_single_adapter,
    update_multi_interval_widths,
    update_single_interval_width,
    fetch_alphas,
    validate_even_quantiles,
    flatten_conformal_bounds,
)


@pytest.mark.parametrize("n_quantiles", [2, 4, 6, 8, 10])
def test_initialize_quantile_alphas_even_counts(n_quantiles):
    """Test quantile alpha initialization with valid even counts."""
    alphas = initialize_quantile_alphas(n_quantiles)

    # Should return half the input quantiles
    assert len(alphas) == n_quantiles // 2

    # Alphas should be decreasing (increasing confidence)
    assert alphas == sorted(alphas, reverse=True)

    # All alphas should be in valid range
    assert all(0 < alpha < 1 for alpha in alphas)

    # For symmetric quantiles, specific mathematical relationships should hold
    if n_quantiles == 4:
        expected_alphas = [0.4, 0.2]  # 60%, 80% confidence
        np.testing.assert_allclose(alphas, expected_alphas, rtol=1e-10)


@pytest.mark.parametrize("n_quantiles", [1, 3, 5, 7])
def test_initialize_quantile_alphas_odd_counts_raises(n_quantiles):
    """Test that odd quantile counts raise appropriate errors."""
    with pytest.raises(ValueError, match="Number of quantiles must be even"):
        initialize_quantile_alphas(n_quantiles)


def test_initialize_quantile_alphas_mathematical_properties():
    """Test mathematical properties of symmetric quantile initialization."""
    alphas = initialize_quantile_alphas(6)

    # Should produce three alpha values
    assert len(alphas) == 3

    # Check symmetric pairing property: alphas should correspond to
    # intervals with equal tail probabilities
    expected = [0.6, 0.4, 0.2]  # From quantile pairs (0.2,0.8), (0.3,0.7), (0.4,0.6)
    np.testing.assert_allclose(alphas, expected, rtol=1e-10)


@pytest.mark.parametrize("adapter", ["DtACI", "ACI", None])
def test_initialize_multi_adapters(adapter):
    """Test multi-adapter initialization with different strategies."""
    alphas = [0.1, 0.05, 0.01]
    adapters = initialize_multi_adapters(alphas, adapter)

    if adapter is None:
        assert adapters is None
    else:
        assert len(adapters) == len(alphas)
        assert all(hasattr(a, "update") for a in adapters)
        # Each adapter should have the correct alpha
        for adapter_obj, alpha in zip(adapters, alphas):
            assert adapter_obj.alpha_0 == alpha


def test_initialize_multi_adapters_invalid_type():
    """Test that invalid adapter types raise errors."""
    alphas = [0.1, 0.05]
    with pytest.raises(ValueError, match="adapter must be None, 'DtACI', or 'ACI'"):
        initialize_multi_adapters(alphas, "InvalidAdapter")


@pytest.mark.parametrize("adapter", ["DtACI", "ACI", None])
def test_initialize_single_adapter(adapter):
    """Test single adapter initialization."""
    alpha = 0.1
    adapter_obj = initialize_single_adapter(alpha, adapter)

    if adapter is None:
        assert adapter_obj is None
    else:
        assert hasattr(adapter_obj, "update")
        assert adapter_obj.alpha_0 == alpha


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


def test_update_single_interval_width_with_adapter():
    """Test single interval width update with adaptation."""
    alpha = 0.1
    adapter = initialize_single_adapter(alpha, "DtACI")
    beta = 0.85

    updated_alpha = update_single_interval_width(adapter, alpha, beta)

    # Should return a float in valid range
    assert isinstance(updated_alpha, float)
    assert 0 < updated_alpha < 1


def test_update_single_interval_width_without_adapter():
    """Test single interval width update without adapter issues warning."""
    alpha = 0.1
    beta = 0.85

    with pytest.warns(UserWarning, match="'update_interval_width()' method was called"):
        updated_alpha = update_single_interval_width(None, alpha, beta)

    # Should return original alpha unchanged
    assert updated_alpha == alpha


@pytest.mark.parametrize("alpha_type", ["uniform", "quantile"])
@pytest.mark.parametrize("n_quantiles", [2, 4, 6])
def test_fetch_alphas(alpha_type, n_quantiles):
    """Test alpha fetching with different strategies."""
    alphas = fetch_alphas(n_quantiles, alpha_type)

    if alpha_type == "uniform":
        # Should return uniform weights
        expected_length = n_quantiles
        expected_values = [1.0 / n_quantiles] * n_quantiles
        assert len(alphas) == expected_length
        np.testing.assert_allclose(alphas, expected_values)
    else:  # quantile
        # Should return quantile-based alphas
        expected_length = n_quantiles // 2
        assert len(alphas) == expected_length
        assert alphas == sorted(alphas, reverse=True)


def test_fetch_alphas_invalid_type():
    """Test that invalid alpha types raise errors."""
    with pytest.raises(ValueError, match="alpha_type must be 'uniform' or 'quantile'"):
        fetch_alphas(4, "invalid_type")


@pytest.mark.parametrize("n_quantiles", [1, 3, 5])
def test_fetch_alphas_odd_quantiles_raises(n_quantiles):
    """Test that odd quantile counts raise errors in fetch_alphas."""
    with pytest.raises(ValueError, match="Number of quantiles must be even"):
        fetch_alphas(n_quantiles, "quantile")


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
