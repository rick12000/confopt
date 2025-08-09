"""
Tests for entropy-based acquisition strategies in conformal prediction optimization.

This module tests the core functionality of entropy samplers including entropy
calculation correctness, sampler initialization, and information gain computation.
"""

import pytest
import numpy as np
from unittest.mock import patch
from confopt.selection.sampling.entropy_samplers import (
    calculate_entropy,
    _run_parallel_or_sequential,
    MaxValueEntropySearchSampler,
)

POS_TOL: float = 0.3  # Allow up to 30% positive information gains due to noise


def test_entropy_edge_cases_and_basic_properties(
    entropy_samples_identical, entropy_samples_linear
):
    # Test edge cases: empty, single, identical samples
    assert calculate_entropy(np.array([]), method="distance") == 0.0
    assert calculate_entropy(np.array([5.0]), method="distance") == 0.0
    assert calculate_entropy(entropy_samples_identical, method="distance") == 0.0

    # Test invalid method raises error
    with pytest.raises(ValueError, match="Unknown entropy estimation method"):
        calculate_entropy(entropy_samples_linear, method="invalid_method")

    # Test basic mathematical properties
    entropy_distance = calculate_entropy(entropy_samples_linear, method="distance")
    entropy_histogram = calculate_entropy(entropy_samples_linear, method="histogram")
    assert np.isfinite(entropy_distance) and entropy_distance > 0.0
    assert np.isfinite(entropy_histogram) and entropy_histogram != 0.0


@pytest.mark.parametrize("method", ["distance", "histogram"])
def test_entropy_distribution_comparison(
    method, entropy_samples_gaussian, entropy_samples_uniform
):
    # Wider distributions should have higher entropy
    np.random.seed(42)
    narrow_samples = np.random.normal(0, 0.1, 100)
    wide_samples = np.random.normal(0, 2.0, 100)

    narrow_entropy = calculate_entropy(narrow_samples, method=method)
    wide_entropy = calculate_entropy(wide_samples, method=method)
    gaussian_entropy = calculate_entropy(entropy_samples_gaussian, method=method)
    uniform_entropy = calculate_entropy(entropy_samples_uniform, method=method)

    assert wide_entropy > narrow_entropy
    assert gaussian_entropy > 0.0 and np.isfinite(gaussian_entropy)
    assert uniform_entropy > 0.0 and np.isfinite(uniform_entropy)


def test_entropy_cython_python_consistency(
    entropy_samples_gaussian, entropy_samples_uniform
):
    # First get Cython results (if available)
    cython_entropy_gaussian = calculate_entropy(
        entropy_samples_gaussian, method="distance"
    )
    cython_entropy_uniform = calculate_entropy(
        entropy_samples_uniform, method="distance"
    )

    # Force Python fallback by mocking import error
    with patch("builtins.__import__") as mock_import:

        def side_effect(name, *args, **kwargs):
            if "cy_differential_entropy" in str(args):
                raise ImportError("Cython not available")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        python_entropy_gaussian = calculate_entropy(
            entropy_samples_gaussian, method="distance"
        )
        python_entropy_uniform = calculate_entropy(
            entropy_samples_uniform, method="distance"
        )

    # Both implementations should produce finite, positive results
    assert np.isfinite(python_entropy_gaussian) and python_entropy_gaussian > 0.0
    assert np.isfinite(python_entropy_uniform) and python_entropy_uniform > 0.0

    # If Cython was available, results should be similar (within numerical tolerance)
    if not np.isnan(cython_entropy_gaussian):
        np.testing.assert_allclose(
            python_entropy_gaussian, cython_entropy_gaussian, rtol=0.1
        )
        np.testing.assert_allclose(
            python_entropy_uniform, cython_entropy_uniform, rtol=0.1
        )


def test_parallel_execution_utility():
    def square(x):
        return x**2

    items = [1, 2, 3, 4]

    # Test sequential execution
    sequential_results = _run_parallel_or_sequential(square, items, n_jobs=1)
    assert sequential_results == [1, 4, 9, 16]

    # Test parallel execution (should produce same results)
    parallel_results = _run_parallel_or_sequential(square, items, n_jobs=2)
    assert parallel_results == [1, 4, 9, 16]

    # Test edge cases
    assert _run_parallel_or_sequential(square, [], n_jobs=1) == []
    assert _run_parallel_or_sequential(lambda x: x, [42], n_jobs=1) == [42]


@pytest.mark.parametrize("n_quantiles", [2, 4, 6, 8])
def test_max_value_entropy_sampler_initialization_and_properties(n_quantiles):
    # Test valid initialization
    sampler = MaxValueEntropySearchSampler(n_quantiles=n_quantiles)
    assert sampler.n_quantiles == n_quantiles
    assert len(sampler.alphas) == n_quantiles // 2
    assert all(0 < alpha < 1 for alpha in sampler.alphas)

    # Test alpha fetching
    alphas = sampler.fetch_alphas()
    assert isinstance(alphas, list)
    assert len(alphas) == n_quantiles // 2

    # Test with different parameters
    sampler_custom = MaxValueEntropySearchSampler(
        n_quantiles=n_quantiles, n_paths=50, entropy_method="histogram", adapter="DtACI"
    )
    assert sampler_custom.n_paths == 50
    assert sampler_custom.entropy_method == "histogram"
    assert sampler_custom.adapters is not None


@pytest.mark.parametrize("n_quantiles", [1, 3, 5, 7])
def test_max_value_entropy_sampler_invalid_quantiles(n_quantiles):
    with pytest.raises(ValueError, match="quantiles must be even"):
        MaxValueEntropySearchSampler(n_quantiles=n_quantiles)


def test_max_value_entropy_sampler_functionality(monte_carlo_bounds_simple):
    sampler = MaxValueEntropySearchSampler(
        n_quantiles=4, n_y_candidates_per_x=5, n_paths=15, entropy_method="distance"
    )

    # Test alpha update
    original_alphas = sampler.alphas.copy()
    betas = [0.80, 0.95]
    sampler.update_interval_width(betas)
    assert len(sampler.alphas) == len(original_alphas)

    # Test information gain computation
    info_gains = sampler.calculate_information_gain(
        predictions_per_interval=monte_carlo_bounds_simple, n_jobs=1
    )

    assert isinstance(info_gains, np.ndarray)
    assert info_gains.shape == (len(monte_carlo_bounds_simple[0].lower_bounds),)
    assert all(np.isfinite(gain) for gain in info_gains)
    assert all(
        gain <= 0 for gain in info_gains
    )  # Should be consistently negative for this simpler case


def test_max_value_entropy_deterministic_behavior(monte_carlo_bounds_simple):
    sampler = MaxValueEntropySearchSampler(
        n_quantiles=4, n_paths=10, n_y_candidates_per_x=3, entropy_method="distance"
    )

    # Test deterministic behavior with same seed
    np.random.seed(42)
    info_gains1 = sampler.calculate_information_gain(
        predictions_per_interval=monte_carlo_bounds_simple, n_jobs=1
    )

    np.random.seed(42)
    info_gains2 = sampler.calculate_information_gain(
        predictions_per_interval=monte_carlo_bounds_simple, n_jobs=1
    )

    np.testing.assert_array_equal(info_gains1, info_gains2)
    assert all(np.isfinite(gain) for gain in info_gains1)
