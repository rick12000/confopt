Sampling Utilities Module
=========================

Overview
--------

The ``sampling.utils`` module provides essential utility functions for implementing sampling strategies in conformal prediction optimization. This module serves as the foundation for all sampling-based acquisition strategies, offering standardized interfaces for common operations including alpha value initialization, adapter configuration, interval width updates, and conformal bounds preprocessing.

The module implements key methodological components that ensure consistency across different sampling strategies while maintaining computational efficiency and proper uncertainty quantification.

Key Features
------------

* **Symmetric Quantile Initialization**: Methodologically-driven alpha value computation using symmetric quantile pairing
* **Multi-Scale Adaptation**: Support for multiple adapters with independent coverage tracking
* **Flexible Configuration**: Uniform and quantile-based alpha initialization strategies
* **Efficient Preprocessing**: Vectorized conformal bounds flattening for computational performance
* **Validation Utilities**: Parameter constraint checking for sampling strategy requirements

Architecture
------------

The module follows a functional design pattern with utility functions organized into logical groups:

**Initialization Functions**
  - ``initialize_quantile_alphas()``: Symmetric quantile-based alpha computation
  - ``initialize_multi_adapters()``: Multi-interval adapter configuration
  - ``initialize_single_adapter()``: Single-interval adapter setup

**Update Functions**
  - ``update_multi_interval_widths()``: Batch interval width adjustment
  - ``update_single_interval_width()``: Single interval adaptation

**Utility Functions**
  - ``fetch_alphas()``: Convenient alpha value retrieval
  - ``validate_even_quantiles()``: Parameter validation
  - ``flatten_conformal_bounds()``: Efficient matrix representation

Mathematical Foundation and Derivation
-------------------------------------

The sampling utilities provide the mathematical foundation for interval construction and adaptation across all sampling strategies.

**Symmetric Quantile Initialization**

The symmetric quantile approach creates nested intervals with theoretically grounded confidence levels:

1. **Quantile Generation**: For :math:`n$ quantiles (even), generate equally spaced points:

   .. math::
      q_i = \frac{i}{n+1}, \quad i = 1, 2, \ldots, n

2. **Symmetric Pairing**: Form pairs :math:`(q_i, q_{n+1-i})$ to ensure symmetry around the median.

3. **Alpha Computation**: Calculate miscoverage rates:

   .. math::
      \alpha_j = 1 - (q_{n+1-j} - q_j), \quad j = 1, 2, \ldots, n/2

4. **Interval Nesting**: This produces nested intervals:

   .. math::
      I_{\alpha_1}(x) \supseteq I_{\alpha_2}(x) \supseteq \cdots \supseteq I_{\alpha_{n/2}}(x)

**Example for n=4**:
- Quantiles: :math:`q_1 = 0.2, q_2 = 0.4, q_3 = 0.6, q_4 = 0.8$
- Pairs: :math:`(0.2, 0.8)$ and :math:`(0.4, 0.6)$
- Alphas: :math:`\alpha_1 = 1 - (0.8 - 0.2) = 0.4`, :math:`\alpha_2 = 1 - (0.6 - 0.4) = 0.8$

**Adaptive Interval Width Management**

The adaptation mechanism maintains target coverage while optimizing interval efficiency:

**Coverage Tracking**: For interval with target miscoverage :math:`\alpha$, track empirical coverage:

.. math::
   \hat{\beta}_t = \frac{1}{t} \sum_{i=1}^t \mathbf{1}[y_i \in [L_{\alpha}(x_i), U_{\alpha}(x_i)]]

**Adaptation Rule**: Update :math:`\alpha$ based on coverage deviation:

.. math::
   \alpha_{t+1} = \alpha_t + \gamma (\alpha_t - (1 - \hat{\beta}_t))

where :math:`\gamma > 0$ is the adaptation rate.

**Multi-Adapter Independence**: For multiple intervals, each adapter operates independently:

.. math::
   \alpha_{j,t+1} = \text{adapter}_j(\alpha_{j,t}, \hat{\beta}_{j,t})

**Conformal Bounds Flattening**

The flattening operation creates efficient matrix representations:

**Input Structure**: List of :math:`k$ ConformalBounds objects, each with :math:`n$ observations.

**Output Matrix**: :math:`\mathbf{B} \in \mathbb{R}^{n \times 2k}$ where:

.. math::
   \mathbf{B}[i, 2j-1] = L_j(x_i), \quad \mathbf{B}[i, 2j] = U_j(x_i)

for observation :math:`i$ and interval :math:`j$.

**Sampling Efficiency**: This representation enables vectorized sampling:

.. math::
   \tilde{y}_i \sim \text{Uniform}(\{\mathbf{B}[i, j] : j = 1, \ldots, 2k\})

**Validation and Constraints**

**Even Quantile Requirement**: Symmetric pairing requires even :math:`n$:

.. math::
   n \bmod 2 = 0

This ensures each quantile has a symmetric partner around the median.

**Coverage Rate Ordering**: For proper nesting, coverage rates must satisfy:

.. math::
   \hat{\beta}_1 \leq \hat{\beta}_2 \leq \cdots \leq \hat{\beta}_{n/2}

corresponding to decreasing confidence levels.

**Alpha Value Properties**:
- Monotonicity: :math:`\alpha_1 > \alpha_2 > \cdots > \alpha_{n/2}`
- Bounds: :math:`0 < \alpha_j < 1$ for all :math:`j`
- Symmetry: Equal tail probabilities for each interval

Symmetric Quantile Initialization
---------------------------------

The symmetric quantile initialization methodology creates nested prediction intervals with theoretically-grounded confidence levels. The approach uses equal spacing in the cumulative distribution and pairs quantiles symmetrically around the median.

**Mathematical Foundation**

Given :math:`n` quantiles (where :math:`n` is even), the algorithm generates quantiles:

.. math::
   q_i = \frac{i}{n+1}, \quad i = 1, 2, \ldots, n

Symmetric pairs are formed as :math:`(q_i, q_{n+1-i})`, and alpha values are computed as:

.. math::
   \alpha_i = 1 - (q_{n+1-i} - q_i)

This ensures proper nesting of intervals with decreasing alpha values (increasing confidence levels).

**Example**

For ``n_quantiles = 4``:

.. code-block:: python

   from confopt.selection.sampling.utils import initialize_quantile_alphas

   alphas = initialize_quantile_alphas(4)
   print(alphas)  # [0.4, 0.2] for 60% and 80% confidence

Adaptive Interval Width Management
----------------------------------

The module supports dynamic interval width adjustment through adapter configuration. Two adaptation strategies are provided:

**DtACI (Dynamic Threshold ACI)**
  Aggressive adaptation with multiple gamma values for robust adjustment across different time scales.

**ACI (Adaptive Conformal Inference)**
  Conservative adaptation with single gamma value for stable coverage maintenance.

**Multi-Interval Adaptation**

.. code-block:: python

   from confopt.selection.sampling.utils import (
       initialize_quantile_alphas,
       initialize_multi_adapters,
       update_multi_interval_widths
   )

   # Initialize for 4 quantiles with DtACI adaptation
   alphas = initialize_quantile_alphas(4)
   adapters = initialize_multi_adapters(alphas, "DtACI")

   # Update based on observed coverage rates
   observed_betas = [0.85, 0.78]  # Coverage for 60% and 80% intervals
   updated_alphas = update_multi_interval_widths(adapters, alphas, observed_betas)

Efficient Conformal Bounds Processing
-------------------------------------

The ``flatten_conformal_bounds()`` function transforms lists of ConformalBounds objects into efficient matrix representations for vectorized operations.

**Matrix Structure**

For :math:`n` observations and :math:`k` intervals, the output matrix has shape :math:`(n, 2k)` with columns arranged as:

.. math::
   \begin{bmatrix}
   l_1^{(1)} & u_1^{(1)} & l_1^{(2)} & u_1^{(2)} & \cdots & l_1^{(k)} & u_1^{(k)} \\
   l_2^{(1)} & u_2^{(1)} & l_2^{(2)} & u_2^{(2)} & \cdots & l_2^{(k)} & u_2^{(k)} \\
   \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
   l_n^{(1)} & u_n^{(1)} & l_n^{(2)} & u_n^{(2)} & \cdots & l_n^{(k)} & u_n^{(k)}
   \end{bmatrix}

where :math:`l_i^{(j)}` and :math:`u_i^{(j)}` are the lower and upper bounds for observation :math:`i` and interval :math:`j`.

Usage Examples
--------------

**Basic Alpha Initialization**

.. code-block:: python

   from confopt.selection.sampling.utils import initialize_quantile_alphas

   # Symmetric quantile initialization
   alphas = initialize_quantile_alphas(6)  # [0.6, 0.4, 0.2]

   # Uniform initialization
   from confopt.selection.sampling.utils import fetch_alphas
   uniform_alphas = fetch_alphas(6, alpha_type="uniform")  # [0.167, 0.167, ...]

**Adapter Configuration and Updates**

.. code-block:: python

   from confopt.selection.sampling.utils import (
       initialize_single_adapter,
       update_single_interval_width
   )

   # Single interval with adaptation
   alpha = 0.2  # 80% confidence interval
   adapter = initialize_single_adapter(alpha, "DtACI")

   # Update based on observed coverage
   observed_coverage = 0.85
   updated_alpha = update_single_interval_width(adapter, alpha, observed_coverage)

**Conformal Bounds Processing**

.. code-block:: python

   from confopt.selection.sampling.utils import flatten_conformal_bounds
   import numpy as np

   # Assuming predictions_per_interval is a list of ConformalBounds
   flattened_bounds = flatten_conformal_bounds(predictions_per_interval)

   # Efficient sampling from all intervals
   n_obs, n_bounds = flattened_bounds.shape
   random_indices = np.random.randint(0, n_bounds, size=n_obs)
   sampled_values = flattened_bounds[np.arange(n_obs), random_indices]

Performance Considerations
-------------------------

**Computational Complexity**
- Alpha initialization: O(n_quantiles)
- Adapter updates: O(n_adapters) per update
- Bounds flattening: O(n_observations × n_intervals)
- Memory usage: O(n_observations × n_intervals) for flattened representation

**Optimization Guidelines**
- Use even numbers of quantiles for symmetric pairing
- Batch adapter updates when possible for efficiency
- Cache flattened bounds for repeated sampling operations
- Consider memory usage for large candidate sets

**Scaling Considerations**
- Adapter overhead scales linearly with number of intervals
- Flattened representation enables efficient vectorized operations
- Validation functions add minimal computational overhead

Integration Points
-----------------

The utilities module integrates with several framework components:

**Sampling Strategies**
  All sampling classes depend on these utilities for consistent alpha management and bounds processing.

**Adaptation Framework**
  Direct integration with ``DtACI`` adapters for interval width adjustment.

**Conformal Prediction**
  Processes ``ConformalBounds`` objects from conformal predictors.

**Optimization Pipeline**
  Provides standardized interfaces for acquisition function computation.

Common Pitfalls
---------------

**Quantile Count Validation**
  Always ensure even numbers of quantiles for symmetric initialization:

.. code-block:: python

   # Correct
   alphas = initialize_quantile_alphas(4)  # Works

   # Incorrect
   alphas = initialize_quantile_alphas(3)  # Raises ValueError

**Adapter Lifecycle Management**
  Initialize adapters once and reuse for consistent coverage tracking:

.. code-block:: python

   # Correct: Initialize once, update multiple times
   adapters = initialize_multi_adapters(alphas, "DtACI")
   for coverage_batch in coverage_data:
       alphas = update_multi_interval_widths(adapters, alphas, coverage_batch)

   # Incorrect: Reinitializing loses adaptation history
   for coverage_batch in coverage_data:
       adapters = initialize_multi_adapters(alphas, "DtACI")  # Wrong!

**Coverage Rate Ordering**
  Ensure coverage rates match alpha value ordering:

.. code-block:: python

   # Alphas: [0.4, 0.2] for 60%, 80% confidence
   # Betas must correspond: [coverage_60%, coverage_80%]
   betas = [0.65, 0.82]  # Correct ordering

See Also
--------

* :doc:`thompson_samplers` - Thompson sampling implementation using these utilities
* :doc:`expected_improvement_samplers` - Expected Improvement with utility integration
* :doc:`entropy_samplers` - Entropy-based sampling strategies
* :doc:`bound_samplers` - Bound-based acquisition strategies
* :doc:`../adaptation/adaptation` - Interval width adaptation mechanisms
