import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt, ceil, fabs, pow
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy
cimport cython

# C comparison function for qsort
cdef int compare_doubles(const void *a, const void *b) noexcept nogil:
    cdef double diff = (<double*>a)[0] - (<double*>b)[0]
    return 1 if diff > 0 else (-1 if diff < 0 else 0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_differential_entropy(double[::1] samples, str method='distance'):
    """
    Highly optimized Cython implementation of differential entropy estimator

    Parameters:
    -----------
    samples : memoryview of double
        1D array of samples for entropy calculation
    method : str
        Method to use ('distance' or 'histogram')

    Returns:
    --------
    float: The estimated differential entropy
    """
    cdef int n_samples = samples.shape[0]
    cdef double eps = 2.220446049250313e-16  # np.finfo(float).eps hardcoded for speed
    cdef double first_sample, total_log_spacing, spacing, sum_val, sum_sq, mean_val, std_val
    cdef double bin_width, data_range, discrete_entropy, min_val, max_val, bin_start
    cdef int i, j, k, left_idx, right_idx, n_bins, bin_idx
    cdef bint all_same = True
    cdef double *sorted_data = NULL
    cdef int *hist_counts = NULL

    # Quick returns for trivial cases
    if n_samples <= 1:
        return 0.0

    # Check if all samples are identical (optimized)
    first_sample = samples[0]
    for i in range(1, n_samples):
        if fabs(samples[i] - first_sample) > eps:
            all_same = False
            break

    if all_same:
        return 0.0

    if method == 'distance':
        # Vasicek estimator using k-nearest neighbor spacing
        k = <int>sqrt(n_samples)
        if k >= n_samples:
            k = max(1, n_samples // 2)

        # Allocate memory for sorted samples
        sorted_data = <double*>malloc(n_samples * sizeof(double))
        if sorted_data == NULL:
            raise MemoryError("Failed to allocate memory for sorted samples")

        try:
            # Copy data to C array
            for i in range(n_samples):
                sorted_data[i] = samples[i]

            # Use C qsort for maximum speed
            qsort(sorted_data, n_samples, sizeof(double), compare_doubles)

            total_log_spacing = 0.0

            # Optimized spacing calculation
            for i in range(n_samples):
                # Calculate k-nearest neighbor distance
                left_idx = max(0, i - k // 2)
                right_idx = min(n_samples - 1, i + k // 2)

                # Ensure we have k neighbors
                if right_idx - left_idx + 1 < k:
                    if left_idx == 0:
                        right_idx = min(n_samples - 1, left_idx + k - 1)
                    else:
                        left_idx = max(0, right_idx - k + 1)

                spacing = sorted_data[right_idx] - sorted_data[left_idx]
                if spacing <= eps:
                    spacing = eps
                total_log_spacing += log(spacing * n_samples / k)

            return total_log_spacing / n_samples

        finally:
            free(sorted_data)

    elif method == 'histogram':
        # Optimized histogram method with manual statistics computation

        # Compute mean and std manually for speed
        sum_val = 0.0
        for i in range(n_samples):
            sum_val += samples[i]
        mean_val = sum_val / n_samples

        sum_sq = 0.0
        min_val = samples[0]
        max_val = samples[0]
        for i in range(n_samples):
            sum_sq += (samples[i] - mean_val) * (samples[i] - mean_val)
            if samples[i] < min_val:
                min_val = samples[i]
            if samples[i] > max_val:
                max_val = samples[i]

        std_val = sqrt(sum_sq / (n_samples - 1)) if n_samples > 1 else 0.0
        if std_val <= eps:
            return 0.0

        # Scott's rule for bin width
        bin_width = 3.49 * std_val * pow(n_samples, -1.0/3.0)
        data_range = max_val - min_val
        n_bins = max(1, <int>ceil(data_range / bin_width))

        # Allocate histogram array
        hist_counts = <int*>malloc(n_bins * sizeof(int))
        if hist_counts == NULL:
            raise MemoryError("Failed to allocate memory for histogram")

        try:
            # Initialize histogram
            for i in range(n_bins):
                hist_counts[i] = 0

            # Fill histogram manually
            bin_start = min_val
            for i in range(n_samples):
                bin_idx = <int>((samples[i] - bin_start) / bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                elif bin_idx < 0:
                    bin_idx = 0
                hist_counts[bin_idx] += 1

            # Calculate discrete entropy
            discrete_entropy = 0.0
            for i in range(n_bins):
                if hist_counts[i] > 0:
                    prob = <double>hist_counts[i] / n_samples
                    discrete_entropy -= prob * log(prob)

            # Add log of bin width for differential entropy
            return discrete_entropy + log(bin_width)

        finally:
            free(hist_counts)

    else:
        raise ValueError(f"Unknown entropy estimation method: {method}")
