import pytest
import numpy as np
import time

# Import both implementations
try:
    # Import the Cython implementation if available
    from confopt.utils.cy_entropy import cy_differential_entropy

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


# Python implementation (copied from the original code)
def py_differential_entropy_estimator(
    samples: np.ndarray, method: str = "distance"
) -> float:
    """
    Pure Python implementation of the differential entropy estimator
    """
    n_samples = len(samples)
    if n_samples <= 1:
        return 0.0

    # Check if all samples are identical (constant)
    if np.all(samples == samples[0]):
        return 0.0

    if method == "distance":
        # Vasicek estimator based on spacings
        m = int(np.sqrt(n_samples))  # Window size
        if m >= n_samples:
            m = max(1, n_samples // 2)

        sorted_samples = np.sort(samples)
        # Handle boundary cases by wrapping around
        wrapped_samples = np.concatenate([sorted_samples, sorted_samples[:m]])

        spacings = wrapped_samples[m : n_samples + m] - wrapped_samples[:n_samples]
        # Avoid log of zero by setting very small spacings to a minimum value
        spacings = np.maximum(spacings, np.finfo(float).eps)

        # Vasicek estimator formula
        entropy = np.sum(np.log(n_samples * spacings / m)) / n_samples
        return entropy

    elif method == "histogram":
        # Use Scott's rule for bin width selection
        std = np.std(samples)
        if std == 0:  # Handle constant samples
            return 0.0

        # Scott's rule: bin_width = 3.49 * std * n^(-1/3)
        bin_width = 3.49 * std * (n_samples ** (-1 / 3))
        data_range = np.max(samples) - np.min(samples)
        n_bins = max(1, int(np.ceil(data_range / bin_width)))

        # First get frequencies (counts) in each bin
        hist, bin_edges = np.histogram(samples, bins=n_bins)

        # Convert counts to probabilities (relative frequencies)
        probs = hist / n_samples

        # Remove zero probabilities (bins with no samples)
        positive_idx = probs > 0
        positive_probs = probs[positive_idx]

        # Bin width is needed for conversion from discrete to differential entropy
        bin_widths = np.diff(bin_edges)

        # Calculate discrete entropy = -Σ p(i)log(p(i))
        discrete_entropy = -np.sum(positive_probs * np.log(positive_probs))

        # Add log of average bin width to convert to differential entropy
        avg_bin_width = np.mean(bin_widths)
        differential_entropy = discrete_entropy + np.log(avg_bin_width)

        return differential_entropy
    else:
        raise ValueError(
            f"Unknown entropy estimation method: {method}. Choose from 'distance' or 'histogram'."
        )


def benchmark_function(func, *args, **kwargs):
    """Benchmark the runtime of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython implementation not available")
def test_cy_entropy_correctness():
    """Test that Cython and Python implementations give the same results"""
    # Generate random samples for testing
    np.random.seed(42)
    samples = np.random.normal(0, 1, size=1000)

    # Test the distance method
    py_result = py_differential_entropy_estimator(samples, method="distance")
    cy_result = cy_differential_entropy(samples, method="distance")

    # Results should be very close (allowing for small floating-point differences)
    assert (
        abs(py_result - cy_result) < 1e-10
    ), f"Results differ: Python={py_result}, Cython={cy_result}"

    # Test the histogram method
    py_result = py_differential_entropy_estimator(samples, method="histogram")
    cy_result = cy_differential_entropy(samples, method="histogram")

    # Results should be very close
    assert (
        abs(py_result - cy_result) < 1e-10
    ), f"Results differ: Python={py_result}, Cython={cy_result}"


@pytest.mark.parametrize("sample_size", [100, 1000, 5000, 10000])
@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython implementation not available")
def test_cy_entropy_performance(sample_size):
    """Benchmark the performance difference between Cython and Python implementations"""
    # Generate random samples for testing
    np.random.seed(42)
    samples = np.random.normal(0, 1, size=sample_size)

    # Benchmark the distance method
    print(f"\nTesting with sample size {sample_size}:")

    _, py_time_distance = benchmark_function(
        py_differential_entropy_estimator, samples, "distance"
    )
    _, cy_time_distance = benchmark_function(
        cy_differential_entropy, samples, "distance"
    )

    print(
        f"  Distance method - Python: {py_time_distance:.6f}s, Cython: {cy_time_distance:.6f}s"
    )
    print(f"  Speed improvement: {py_time_distance / cy_time_distance:.2f}x faster")

    _, py_time_hist = benchmark_function(
        py_differential_entropy_estimator, samples, "histogram"
    )
    _, cy_time_hist = benchmark_function(cy_differential_entropy, samples, "histogram")

    print(
        f"  Histogram method - Python: {py_time_hist:.6f}s, Cython: {cy_time_hist:.6f}s"
    )
    print(f"  Speed improvement: {py_time_hist / cy_time_hist:.2f}x faster")

    # We expect the Cython implementation to be significantly faster
    assert (
        cy_time_distance < py_time_distance
    ), "Cython should be faster than Python for distance method"
    assert (
        cy_time_hist < py_time_hist
    ), "Cython should be faster than Python for histogram method"
