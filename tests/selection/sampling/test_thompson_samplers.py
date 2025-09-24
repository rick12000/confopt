import pytest
import numpy as np
from confopt.selection.sampling.thompson_samplers import ThompsonSampler
from confopt.wrapping import ConformalBounds


@pytest.mark.parametrize("n_quantiles", [2, 4, 6, 8])
def test_thompson_sampler_initialization_valid_quantiles(n_quantiles):
    """Test Thompson sampler initialization with valid even quantile counts."""
    sampler = ThompsonSampler(n_quantiles=n_quantiles)

    assert len(sampler.alphas) == n_quantiles // 2
    assert sampler.n_quantiles == n_quantiles
    assert not sampler.enable_optimistic_sampling
    assert sampler.adapters is None  # Default no adapter


@pytest.mark.parametrize("adapter", ["DtACI", "ACI", None])
def test_thompson_sampler_initialization_with_adapters(adapter):
    """Test Thompson sampler initialization with different adapter strategies."""
    sampler = ThompsonSampler(n_quantiles=4, adapter=adapter)

    if adapter is None:
        assert sampler.adapters is None
    else:
        assert len(sampler.adapters) == 2  # n_quantiles // 2
        assert all(hasattr(a, "update") for a in sampler.adapters)


def test_update_interval_width_with_adapters(coverage_feedback):
    """Test interval width updating with adaptation enabled."""
    sampler = ThompsonSampler(n_quantiles=6, adapter="DtACI")
    initial_alphas = sampler.alphas.copy()

    sampler.update_interval_width(coverage_feedback)

    assert len(sampler.alphas) == len(initial_alphas)
    # Alphas should have changed based on coverage feedback
    assert not np.array_equal(sampler.alphas, initial_alphas)


def test_update_interval_width_without_adapters():
    """Test interval width updating when no adapters are configured."""
    sampler = ThompsonSampler(n_quantiles=4, adapter=None)
    initial_alphas = sampler.alphas.copy()
    betas = [0.85, 0.92]

    # Should return original alphas unchanged when no adapters
    sampler.update_interval_width(betas)
    assert np.array_equal(sampler.alphas, initial_alphas)


def test_calculate_thompson_predictions_shape(simple_conformal_bounds):
    """Test Thompson predictions return correct shape."""
    sampler = ThompsonSampler(n_quantiles=4)
    predictions = sampler.calculate_thompson_predictions(simple_conformal_bounds)

    n_observations = len(simple_conformal_bounds[0].lower_bounds)
    assert predictions.shape == (n_observations,)


def test_calculate_thompson_predictions_values_within_bounds(simple_conformal_bounds):
    """Test that Thompson predictions fall within conformal bounds."""
    sampler = ThompsonSampler(n_quantiles=4)
    predictions = sampler.calculate_thompson_predictions(simple_conformal_bounds)

    # Get overall bounds across all intervals
    all_lower = np.minimum(
        simple_conformal_bounds[0].lower_bounds, simple_conformal_bounds[1].lower_bounds
    )
    all_upper = np.maximum(
        simple_conformal_bounds[0].upper_bounds, simple_conformal_bounds[1].upper_bounds
    )

    # All predictions should be within the overall bounds
    assert np.all(predictions >= all_lower)
    assert np.all(predictions <= all_upper)


@pytest.mark.parametrize("n_quantiles", [2, 4, 6])
def test_calculate_thompson_predictions_stochasticity(
    simple_conformal_bounds, n_quantiles
):
    """Test that Thompson predictions show appropriate stochastic behavior."""
    sampler = ThompsonSampler(n_quantiles=n_quantiles)

    # Generate multiple samples
    samples = []
    for _ in range(50):
        predictions = sampler.calculate_thompson_predictions(simple_conformal_bounds)
        samples.append(predictions)

    samples_array = np.array(samples)

    # Check that predictions vary across runs (stochastic behavior)
    variance_per_observation = np.var(samples_array, axis=0)
    assert np.all(variance_per_observation > 0)  # Should have non-zero variance


def test_calculate_thompson_predictions_optimistic_sampling_enabled(
    simple_conformal_bounds,
):
    """Test Thompson predictions with optimistic sampling enabled."""
    sampler = ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=True)
    point_estimates = np.array([0.2, 0.4, 0.6])  # Conservative point estimates

    predictions = sampler.calculate_thompson_predictions(
        simple_conformal_bounds, point_predictions=point_estimates
    )

    # Predictions should be capped at point estimates
    assert np.all(predictions <= point_estimates)


def test_calculate_thompson_predictions_mathematical_properties(
    simple_conformal_bounds,
):
    """Test mathematical properties of Thompson sampling distribution.

    Thompson sampling uniformly samples from the flattened bounds matrix,
    which contains all lower and upper bounds from all intervals.
    For simple_conformal_bounds, each observation should sample uniformly
    from the set of bounds: [lower1, upper1, lower2, upper2].
    """
    sampler = ThompsonSampler(n_quantiles=4)  # Creates 2 intervals

    # Extract expected values for each observation from the bounds
    expected_values_per_obs = []
    for obs_idx in range(len(simple_conformal_bounds[0].lower_bounds)):
        values = [
            simple_conformal_bounds[0].lower_bounds[obs_idx],  # interval 1 lower
            simple_conformal_bounds[0].upper_bounds[obs_idx],  # interval 1 upper
            simple_conformal_bounds[1].lower_bounds[obs_idx],  # interval 2 lower
            simple_conformal_bounds[1].upper_bounds[obs_idx],  # interval 2 upper
        ]
        expected_values_per_obs.append(values)

    # Generate many samples for statistical analysis
    n_samples = 10000
    samples = []
    for _ in range(n_samples):
        predictions = sampler.calculate_thompson_predictions(simple_conformal_bounds)
        samples.append(predictions)

    samples_array = np.array(samples)

    # For each observation, rigorously test uniform sampling from expected values
    for obs_idx in range(len(simple_conformal_bounds[0].lower_bounds)):
        obs_samples = samples_array[:, obs_idx]
        expected_values = expected_values_per_obs[obs_idx]

        # Test 1: All samples should be from the expected discrete set
        unique_samples = np.unique(obs_samples)
        np.testing.assert_array_almost_equal(
            np.sort(unique_samples),
            np.sort(expected_values),
            decimal=10,
            err_msg=f"Observation {obs_idx} samples not from expected bounds set",
        )

        # Test 2: Each value should appear with approximately equal frequency (uniform)
        expected_freq = n_samples / len(expected_values)
        tolerance = 0.05 * n_samples  # 5% tolerance for randomness

        for value in expected_values:
            actual_freq = np.sum(np.isclose(obs_samples, value))
            assert abs(actual_freq - expected_freq) < tolerance, (
                f"Observation {obs_idx}, value {value}: expected ~{expected_freq:.0f} "
                f"occurrences, got {actual_freq}"
            )

        # Test 3: Sample mean should equal theoretical mean of uniform distribution
        theoretical_mean = np.mean(expected_values)
        sample_mean = np.mean(obs_samples)

        # With large sample size, sample mean should be very close to theoretical
        mean_tolerance = 0.01 * abs(theoretical_mean)  # 1% tolerance
        assert abs(sample_mean - theoretical_mean) < mean_tolerance, (
            f"Observation {obs_idx}: theoretical mean {theoretical_mean:.6f}, "
            f"sample mean {sample_mean:.6f}"
        )


def test_thompson_sampler_deterministic_with_seed():
    """Test that Thompson sampler produces deterministic results with fixed seed."""
    sampler = ThompsonSampler(n_quantiles=4)

    # Create fixed bounds
    bounds = [
        ConformalBounds(
            lower_bounds=np.array([0.1, 0.2]), upper_bounds=np.array([0.5, 0.6])
        )
    ]

    # Set seed and get predictions
    np.random.seed(42)
    predictions1 = sampler.calculate_thompson_predictions(bounds)

    # Reset seed and get predictions again
    np.random.seed(42)
    predictions2 = sampler.calculate_thompson_predictions(bounds)

    # Should be identical with same seed
    np.testing.assert_array_equal(predictions1, predictions2)
