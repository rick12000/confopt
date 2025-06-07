import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt, ceil

def cy_differential_entropy(np.ndarray[double, ndim=1] samples, str method='distance'):
    """
    Optimized Cython implementation of differential entropy estimator

    Parameters:
    -----------
    samples : np.ndarray
        1D array of samples for entropy calculation
    method : str
        Method to use ('distance' or 'histogram')

    Returns:
    --------
    float: The estimated differential entropy
    """
    cdef int n_samples = len(samples)
    cdef double eps = np.finfo(float).eps

    # Quick returns for trivial cases
    if n_samples <= 1:
        return 0.0

    # Check if all samples are identical
    cdef double first_sample = samples[0]
    cdef bint all_same = True
    cdef int i

    for i in range(1, n_samples):
        if samples[i] != first_sample:
            all_same = False
            break

    if all_same:
        return 0.0

    if method == 'distance':
        # Vasicek estimator (spacing method)
        cdef int m = int(sqrt(n_samples))
        if m >= n_samples:
            m = max(1, n_samples // 2)

        # Sort the samples
        cdef np.ndarray[double, ndim=1] sorted_samples = np.sort(samples)

        # Create wrapped samples
        cdef np.ndarray[double, ndim=1] wrapped_samples = np.concatenate([sorted_samples, sorted_samples[:m]])

        cdef np.ndarray[double, ndim=1] spacings = np.zeros(n_samples, dtype=np.float64)
        cdef double total_log_spacing = 0.0

        for i in range(n_samples):
            spacings[i] = max(wrapped_samples[i+m] - wrapped_samples[i], eps)
            total_log_spacing += log(n_samples * spacings[i] / m)

        return total_log_spacing / n_samples

    elif method == 'histogram':
        # Scott's rule for bin width
        cdef double std = np.std(samples)
        if std == 0:
            return 0.0

        cdef double bin_width = 3.49 * std * (n_samples ** (-1.0/3.0))
        cdef double data_range = np.max(samples) - np.min(samples)
        cdef int n_bins = max(1, int(ceil(data_range / bin_width)))

        # Calculate histogram
        hist, bin_edges = np.histogram(samples, bins=n_bins)

        # Convert to probabilities
        cdef np.ndarray[double, ndim=1] probs = hist.astype(np.float64) / n_samples

        # Remove zeros
        cdef np.ndarray[double, ndim=1] positive_probs = probs[probs > 0]

        # Calculate discrete entropy
        cdef double discrete_entropy = -np.sum(positive_probs * np.log(positive_probs))

        # Add log of bin width for differential entropy
        cdef np.ndarray[double, ndim=1] bin_widths = np.diff(bin_edges)
        cdef double avg_bin_width = np.mean(bin_widths)

        return discrete_entropy + log(avg_bin_width)

    else:
        raise ValueError(f"Unknown entropy estimation method: {method}")
