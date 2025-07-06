Entropy-Based Sampling Module
=============================

Overview
--------

The ``entropy_samplers`` module implements information-theoretic acquisition strategies for conformal prediction optimization. These strategies use entropy and information gain principles to guide optimization decisions, providing theoretically principled exploration that balances between high-information regions and promising optimization areas.

The module focuses on quantifying and reducing uncertainty about the global optimum through information-theoretic measures, offering two complementary approaches: full Entropy Search with model updates and efficient Max Value Entropy Search without refitting.

Key Features
------------

* **Information-Theoretic Foundation**: Principled exploration using entropy and information gain
* **Differential Entropy Estimation**: Robust non-parametric entropy computation using distance and histogram methods
* **Multiple Acquisition Strategies**: Full Entropy Search and computationally efficient Max Value Entropy Search
* **Flexible Candidate Selection**: Multiple strategies including Thompson sampling, Expected Improvement, and Sobol sequences
* **Parallel Processing Support**: Efficient computation through configurable parallelization
* **Adaptive Interval Widths**: Coverage-based adjustment for accurate uncertainty quantification

Architecture
------------

The module provides two main classes implementing different information-theoretic approaches:

**EntropySearchSampler**
  Full information gain computation with model updates and candidate evaluation

**MaxValueEntropySearchSampler**
  Efficient entropy reduction focusing on optimum value without model refitting

**Supporting Functions**
  - ``calculate_entropy()``: Non-parametric differential entropy estimation
  - ``_run_parallel_or_sequential()``: Unified parallel/sequential execution interface

Mathematical Foundation and Derivation
-------------------------------------

Information-theoretic acquisition strategies use entropy and information gain to guide optimization by quantifying uncertainty reduction about the global optimum.

**Information Gain Framework**

The fundamental principle is to maximize information gain about the optimum location :math:`x^*`:

.. math::
   IG(x) = H[p(x^*)] - \mathbb{E}_{y|x}[H[p(x^*|y)]

where :math:`H[\cdot]` denotes differential entropy.

**Entropy Search Derivation**

1. **Prior Optimum Distribution**: Define :math:`p(x^*)` as the current belief about optimum location.

2. **Posterior Update**: After observing :math:`y` at candidate :math:`x`, update beliefs:

   .. math::
      p(x^*|y) \propto p(y|x^*, x) p(x^*)

3. **Information Gain**: Compute expected entropy reduction:

   .. math::
      IG(x) = H[p(x^*)] - \int p(y|x) H[p(x^*|y)] dy

**Monte Carlo Implementation**

Since analytical computation is intractable, we use Monte Carlo estimation:

1. **Function Sampling**: Generate :math:`M$ function realizations from prediction intervals:

   .. math::
      f^{(i)} = \{\tilde{y}^{(i)}(x_j)\}_{j=1}^n, \quad i = 1, \ldots, M

2. **Optimum Location Sampling**: For each realization, find the optimum:

   .. math::
      x^{*(i)} = \arg\min_{x_j} \tilde{y}^{(i)}(x_j)

3. **Prior Entropy**: Estimate entropy of optimum locations:

   .. math::
      H[p(x^*)] \approx H[\{x^{*(i)}\}_{i=1}^M]

4. **Conditional Entropy**: For each candidate :math:`x$ and hypothetical observation :math:`y$:

   .. math::
      H[p(x^*|y)] \approx H[\{x^{*(i)} : \tilde{y}^{(i)}(x) = y\}]

**Max Value Entropy Search Simplification**

Instead of tracking optimum location, focus on optimum value :math:`f^* = \min_x f(x)$:

.. math::
   IG_{MV}(x) = H[p(f^*)] - \mathbb{E}_{y|x}[H[p(f^*|y)]

This avoids expensive model refitting by using value capping:

.. math::
   f^{*|y} = \min(f^*, y)

when candidate :math:`x$ achieves value :math:`y$.

**Differential Entropy Estimation**

Two robust estimators are implemented:

**Vasicek Estimator (Distance-based)**:

.. math::
   \hat{H} = \frac{1}{n} \sum_{i=1}^{n} \log\left(\frac{n}{k}(X_{(i+k)} - X_{(i-k)})\right)

where :math:`X_{(i)}$ are order statistics and :math:`k = \lfloor\sqrt{n}\rfloor`.

**Histogram Estimator (Scott's Rule)**:

.. math::
   \hat{H} = -\sum_{i=1}^{B} p_i \log p_i + \log(\Delta)

where :math:`p_i = n_i/n$ are bin probabilities, :math:`\Delta$ is average bin width, and bin width follows:

.. math::
   \Delta = 3.49 \sigma n^{-1/3}

**Acquisition Decision**

Select the candidate maximizing information gain:

.. math::
   x^* = \arg\max_{x \in \mathcal{X}} IG(x)

This naturally balances:
- **High uncertainty regions**: Large :math:`H[p(x^*)]$ contributes to high :math:`IG$
- **Informative observations**: Large entropy reduction :math:`H[p(x^*)] - H[p(x^*|y)]`

Information-Theoretic Methodology
---------------------------------

The acquisition strategies are based on maximizing information gain about the global optimum location or value. This approach provides principled exploration by selecting candidates that maximally reduce uncertainty.

**Information Gain Framework**

Information gain quantifies the expected reduction in uncertainty about the optimum:

.. math::
   IG(x) = H[p(x^*)] - \mathbb{E}_{y|x}[H[p(x^*|y)]]

where :math:`H[\cdot]` denotes entropy, :math:`x^*` is the optimum location, and :math:`y` is the observed value at candidate :math:`x`.

**Entropy Search Approach**

Full Entropy Search computes information gain by:

1. Estimating prior entropy of optimum location distribution
2. Simulating posterior distributions after hypothetical observations
3. Computing conditional entropy for each scenario
4. Averaging information gain across scenarios

**Max Value Entropy Search**

The simplified approach focuses on optimum value rather than location:

.. math::
   IG_{MV}(x) = H[f^*] - \mathbb{E}_{y|x}[H[f^*|y]]

where :math:`f^*` is the optimum value, avoiding expensive model refitting.

Differential Entropy Estimation
------------------------------

Accurate entropy estimation is crucial for information gain computation. The module implements two robust non-parametric methods:

**Distance-Based Estimation (Vasicek)**

Uses k-nearest neighbor spacing for entropy estimation:

.. math::
   \hat{H} = \frac{1}{n} \sum_{i=1}^{n} \log\left(\frac{n}{k}(X_{(i+k)} - X_{(i-k)})\right)

where :math:`X_{(i)}` are order statistics and :math:`k = \sqrt{n}`.

**Histogram-Based Estimation (Scott's Rule)**

Combines discrete entropy with bin width correction:

.. math::
   \hat{H} = -\sum_{i} p_i \log p_i + \log(\Delta)

where :math:`p_i` are bin probabilities and :math:`\Delta` is the average bin width.

**Implementation Optimization**

.. code-block:: python

   # Cython optimization with pure Python fallback
   try:
       from confopt.selection.sampling import cy_differential_entropy
       entropy = cy_differential_entropy(samples, method)
   except ImportError:
       # Fallback to pure Python implementation
       entropy = calculate_entropy(samples, method)

Usage Examples
--------------

**Basic Entropy Search**

.. code-block:: python

   from confopt.selection.sampling.entropy_samplers import EntropySearchSampler

   # Initialize with standard configuration
   entropy_sampler = EntropySearchSampler(
       n_quantiles=4,
       n_paths=100,
       n_x_candidates=10,
       sampling_strategy="thompson"
   )

   # Calculate information gain for all candidates
   information_gains = entropy_sampler.calculate_information_gain(
       X_train=X_train,
       y_train=y_train,
       X_val=X_val,
       y_val=y_val,
       X_space=candidate_space,
       conformal_estimator=predictor,
       predictions_per_interval=predictions
   )

   # Select candidate with highest information gain
   selected_idx = np.argmin(information_gains)  # Most negative = highest gain

**Max Value Entropy Search**

.. code-block:: python

   from confopt.selection.sampling.entropy_samplers import MaxValueEntropySearchSampler

   # Initialize efficient variant
   mv_sampler = MaxValueEntropySearchSampler(
       n_quantiles=4,
       n_paths=100,
       n_y_candidates_per_x=20
   )

   # Calculate information gain (no model refitting required)
   information_gains = mv_sampler.calculate_information_gain(
       predictions_per_interval=predictions,
       n_jobs=4  # Parallel processing
   )

**Candidate Selection Strategies**

.. code-block:: python

   # Thompson sampling for exploration-exploitation balance
   thompson_sampler = EntropySearchSampler(
       sampling_strategy="thompson",
       n_x_candidates=15
   )

   # Expected Improvement for exploitation focus
   ei_sampler = EntropySearchSampler(
       sampling_strategy="expected_improvement",
       n_x_candidates=10
   )

   # Sobol sequences for space-filling exploration
   sobol_sampler = EntropySearchSampler(
       sampling_strategy="sobol",
       n_x_candidates=20
   )

**Adaptive Configuration**

.. code-block:: python

   # Adaptive interval widths with DtACI
   adaptive_sampler = EntropySearchSampler(
       n_quantiles=6,
       adapter="DtACI",
       entropy_measure="distance"
   )

   # Update interval widths based on coverage
   coverage_rates = [0.62, 0.81, 0.91]  # For 60%, 80%, 90% intervals
   adaptive_sampler.update_interval_width(coverage_rates)

Performance Considerations
-------------------------

**Computational Complexity**

*Entropy Search*
- Initialization: O(n_quantiles)
- Information gain: O(n_candidates × n_y_candidates × n_paths × model_fit_cost)
- Memory: O(n_observations × n_quantiles + n_paths)

*Max Value Entropy Search*
- Initialization: O(n_quantiles)
- Information gain: O(n_observations × n_y_candidates × n_paths)
- Memory: O(n_observations × n_quantiles + n_paths)

**Scaling Guidelines**

.. code-block:: python

   # For expensive optimization (few evaluations, high accuracy)
   expensive_config = {
       'n_paths': 200,
       'n_x_candidates': 20,
       'n_y_candidates_per_x': 5,
       'sampling_strategy': 'expected_improvement'
   }

   # For moderate cost optimization
   balanced_config = {
       'n_paths': 100,
       'n_x_candidates': 10,
       'n_y_candidates_per_x': 3,
       'sampling_strategy': 'thompson'
   }

   # For fast exploration (many evaluations, moderate accuracy)
   fast_config = {
       'n_paths': 50,
       'n_x_candidates': 5,
       'n_y_candidates_per_x': 2,
       'sampling_strategy': 'uniform'
   }

**Optimization Strategies**

.. code-block:: python

   # Efficient parallel processing
   def parallel_entropy_search(sampler, prediction_batches, n_jobs=4):
       results = []
       for batch in prediction_batches:
           ig_values = sampler.calculate_information_gain(
               predictions_per_interval=batch,
               n_jobs=n_jobs
           )
           results.append(ig_values)
       return np.concatenate(results)

   # Memory-efficient batch processing
   def batch_entropy_computation(sampler, large_candidate_set, batch_size=1000):
       n_candidates = len(large_candidate_set)
       all_gains = []

       for start_idx in range(0, n_candidates, batch_size):
           end_idx = min(start_idx + batch_size, n_candidates)
           batch_predictions = large_candidate_set[start_idx:end_idx]

           batch_gains = sampler.calculate_information_gain(batch_predictions)
           all_gains.extend(batch_gains)

       return np.array(all_gains)

Integration Points
-----------------

**Conformal Prediction Framework**
  Directly processes ConformalBounds objects from any conformal predictor, enabling seamless uncertainty quantification across different modeling approaches.

**Optimization Pipelines**
  Provides acquisition values compatible with sequential optimization, multi-armed bandit frameworks, and batch evaluation scenarios.

**Parallel Computing**
  Supports joblib-based parallelization for efficient computation on multi-core systems and distributed environments.

**Model Adaptation**
  Integrates with DtACI and ACI adapters for dynamic interval width adjustment based on empirical coverage feedback.

Common Pitfalls
---------------

**Sample Size for Entropy Estimation**
  Ensure sufficient samples for reliable entropy computation:

.. code-block:: python

   # Good: Sufficient paths for stable entropy estimates
   reliable_sampler = EntropySearchSampler(n_paths=100)

   # Risky: Too few paths may cause noisy entropy estimates
   unreliable_sampler = EntropySearchSampler(n_paths=10)  # May be unstable

**Candidate Selection Strategy**
  Choose appropriate strategy for optimization phase:

.. code-block:: python

   # Early exploration: Use space-filling strategies
   early_phase = EntropySearchSampler(sampling_strategy="sobol")

   # Later exploitation: Use improvement-based strategies
   later_phase = EntropySearchSampler(sampling_strategy="expected_improvement")

**Memory Management for Large Problems**
  Monitor memory usage with large candidate sets:

.. code-block:: python

   # Memory-efficient: Process in batches
   def memory_efficient_entropy_search(sampler, large_predictions):
       batch_size = 500  # Adjust based on available memory
       results = []

       for i in range(0, len(large_predictions), batch_size):
           batch = large_predictions[i:i+batch_size]
           batch_results = sampler.calculate_information_gain(batch)
           results.extend(batch_results)

       return np.array(results)

**Parallel Processing Configuration**
  Balance parallelization with memory constraints:

.. code-block:: python

   # Conservative: Avoid memory issues
   safe_sampler = MaxValueEntropySearchSampler(n_jobs=2)

   # Aggressive: Maximum parallelization (ensure sufficient memory)
   fast_sampler = MaxValueEntropySearchSampler(n_jobs=-1)

**Entropy Method Selection**
  Choose entropy estimation method based on data characteristics:

.. code-block:: python

   # For smooth, continuous distributions
   distance_sampler = EntropySearchSampler(entropy_measure="distance")

   # For discrete or multimodal distributions
   histogram_sampler = EntropySearchSampler(entropy_measure="histogram")

See Also
--------

* :doc:`sampling_utils` - Utility functions for interval management and preprocessing
* :doc:`thompson_samplers` - Probabilistic acquisition strategy implementation
* :doc:`expected_improvement_samplers` - Expected improvement acquisition functions
* :doc:`bound_samplers` - Confidence bound-based acquisition strategies
* :doc:`../adaptation/adaptation` - Interval width adaptation mechanisms
