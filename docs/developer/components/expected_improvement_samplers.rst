Expected Improvement Acquisition Functions
==========================================

Overview
--------

The ``expected_improvement_samplers`` module implements Expected Improvement (EI) acquisition functions adapted for conformal prediction optimization. This approach extends the classical Bayesian optimization framework to conformal prediction settings, providing a principled method for balancing exploration and exploitation without requiring explicit posterior distributions over the objective function.

The implementation leverages Monte Carlo sampling from conformal prediction intervals to estimate expected improvements, offering robust uncertainty quantification while maintaining computational efficiency for large-scale optimization problems.

Key Features
------------

- **Adaptive Interval Widths**: Automatically adjusts to the local density of data and uncertainty, using more intervals where the function is complex and fewer where it is simple.
- **Multi-Quantile Support**: Simultaneously optimize for multiple quantiles of the predictive distribution, enabling a more comprehensive exploration of the objective function.
- **Batch Sampling**: Efficiently generate and evaluate multiple candidate solutions in parallel, significantly speeding up the optimization process.
- **Integration with Conformal Prediction**: Seamlessly works with any conformal predictor, providing flexibility in uncertainty quantification.

Architecture
------------

The module is structured around the `ExpectedImprovementSampler` class, which encapsulates the logic for sampling and selecting candidate solutions based on expected improvement.

- **Initialization**: Configure the sampler with the desired number of quantiles, initial best value, and other parameters.
- **Interval Sampling**: For each candidate, sample from the conformal prediction intervals to estimate the potential improvement.
- **EI Calculation**: Compute the expected improvement for each candidate based on the sampled intervals.
- **Selection**: Choose the candidate with the highest expected improvement for evaluation.

Mathematical Foundation and Derivation
-------------------------------------

The Expected Improvement acquisition function provides a principled approach to optimization under uncertainty by quantifying the expected benefit of evaluating a candidate point.

**Classical Expected Improvement**

In the Gaussian process setting, Expected Improvement is defined as:

.. math::
   \text{EI}(x) = \mathbb{E}[\max(f_{\min} - f(x), 0)]

where :math:`f_{\min}` is the current best observed value and :math:`f(x)` follows a Gaussian posterior distribution.

For a Gaussian posterior :math:`f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))`, this has the closed form:

.. math::
   \text{EI}(x) = (\mu(x) - f_{\min})\Phi(Z) + \sigma(x)\phi(Z)

where :math:`Z = \frac{\mu(x) - f_{\min}}{\sigma(x)}`, :math:`\Phi` is the standard normal CDF, and :math:`\phi` is the standard normal PDF.

**Conformal Prediction Adaptation**

In conformal prediction settings, we lack explicit posterior distributions but have prediction intervals. The adaptation uses Monte Carlo estimation:

.. math::
   \text{EI}(x) = \mathbb{E}[\max(f_{\min} - \tilde{y}(x), 0)]

where :math:`\tilde{y}(x)` is sampled from the prediction intervals.

**Monte Carlo Estimation Process**

1. **Interval Sampling**: For candidate :math:`x`, draw :math:`M` samples from its prediction intervals:

   .. math::
      \tilde{y}_i(x) \sim \text{Uniform}(\mathcal{I}(x))

   where :math:`\mathcal{I}(x) = \{[L_j(x), U_j(x)]\}_{j=1}^k` represents the set of conformal intervals.

2. **Improvement Computation**: Calculate individual improvements:

   .. math::
      I_i(x) = \max(0, f_{\min} - \tilde{y}_i(x))

3. **Expectation Approximation**: Estimate expected improvement:

   .. math::
      \widehat{\text{EI}}(x) = \frac{1}{M} \sum_{i=1}^{M} I_i(x)

**Theoretical Properties**

The Monte Carlo estimator is unbiased:

.. math::
   \mathbb{E}[\widehat{\text{EI}}(x)] = \text{EI}(x)

with variance decreasing as :math:`O(1/M)`, ensuring convergence to the true expected improvement as sample size increases.

**Acquisition Decision Rule**

The optimal next evaluation point is:

.. math::
   x^* = \arg\max_{x \in \mathcal{X}} \widehat{\text{EI}}(x)

This naturally balances:
- **Exploitation**: High improvement potential (low predicted values)
- **Exploration**: High uncertainty (wide prediction intervals)

Expected Improvement Methodology
-------------------------------

**Initialization**

The sampler is initialized with a set of quantiles and an initial best value. The quantiles determine the points in the distribution of the objective function that are of interest (e.g., 60th, 80th percentiles), and the best value is used to calculate the improvement.

.. code-block:: python

   # Initialize sampler
   sampler = ExpectedImprovementSampler(
       n_quantiles=4,
       current_best_value=1.5,  # Known best value
       num_ei_samples=30
   )

**Adaptive Configuration**

.. code-block:: python

   # Initialize with adaptive interval widths
   adaptive_sampler = ExpectedImprovementSampler(
       n_quantiles=6,
       adapter="DtACI",
       num_ei_samples=50
   )

   # Update interval widths based on coverage
   coverage_rates = [0.62, 0.81, 0.91]  # For 60%, 80%, 90% intervals
   adaptive_sampler.update_interval_width(coverage_rates)

**Sample Count Trade-offs**

.. code-block:: python

   # High accuracy, higher computational cost
   precise_sampler = ExpectedImprovementSampler(num_ei_samples=100)

   # Fast computation, lower accuracy
   fast_sampler = ExpectedImprovementSampler(num_ei_samples=10)

   # Balanced approach
   balanced_sampler = ExpectedImprovementSampler(num_ei_samples=20)

Performance Considerations
-------------------------

**Computational Complexity**
- Initialization: O(n_quantiles)
- EI computation: O(n_observations × n_quantiles × n_samples)
- Memory usage: O(n_observations × n_quantiles) for interval storage
- Best value update: O(1)

**Scaling Guidelines**
- Sample count affects accuracy vs. computational cost trade-off
- More quantiles improve uncertainty resolution but increase cost
- Vectorized operations enable efficient batch processing
- Consider memory usage for large candidate sets

**Parameter Selection Guidelines**

.. code-block:: python

   # For quick exploration (early optimization phases)
   quick_config = {
       'n_quantiles': 4,
       'num_ei_samples': 10,
       'adapter': None
   }

   # For precise optimization (later phases)
   precise_config = {
       'n_quantiles': 6,
       'num_ei_samples': 50,
       'adapter': "DtACI"
   }

   # For balanced performance
   balanced_config = {
       'n_quantiles': 4,
       'num_ei_samples': 20,
       'adapter': "ACI"
   }

Integration Points
-----------------

**Conformal Prediction Framework**
  Directly processes ConformalBounds objects from any conformal predictor, enabling seamless integration with different uncertainty quantification approaches.

**Optimization Algorithms**
  Provides acquisition values compatible with gradient-free optimization routines, multi-armed bandit frameworks, and sequential decision making pipelines.

**Ensemble Strategies**
  Can be combined with other acquisition functions for portfolio optimization or used in multi-objective settings with appropriate scalarization.

**Parallel Evaluation**
  Supports batch candidate evaluation for parallel objective function evaluation scenarios.

Common Pitfalls
---------------

**Best Value Initialization**
  Always initialize with a reasonable best value to avoid poor early performance:

.. code-block:: python

   # Good: Initialize with known minimum
   if historical_data_available:
       best_val = np.min(historical_y_values)
       sampler = ExpectedImprovementSampler(current_best_value=best_val)

   # Acceptable: Conservative initialization
   else:
       sampler = ExpectedImprovementSampler(current_best_value=float("inf"))

**Sample Count Selection**
  Balance accuracy with computational requirements:

.. code-block:: python

   # Too few samples: Noisy EI estimates
   unreliable_sampler = ExpectedImprovementSampler(num_ei_samples=3)  # Risky

   # Too many samples: Unnecessary computation
   wasteful_sampler = ExpectedImprovementSampler(num_ei_samples=1000)  # Overkill

   # Balanced: Sufficient for reliable estimates
   good_sampler = ExpectedImprovementSampler(num_ei_samples=20)  # Good

**Best Value Updates**
  Don't forget to update the best value after each evaluation:

.. code-block:: python

   for iteration in optimization_loop:
       ei_values = sampler.calculate_expected_improvement(predictions)
       selected_idx = np.argmin(ei_values)

       new_y = objective_function(candidates[selected_idx])
       sampler.update_best_value(new_y)  # Critical step!

**Interval Ordering Consistency**
  Ensure coverage rates match alpha value ordering:

.. code-block:: python

   # For n_quantiles=4: alphas=[0.4, 0.2] (60%, 80% confidence)
   # Coverage rates must match: [coverage_60%, coverage_80%]
   correct_coverage = [0.63, 0.82]  # Correct ordering
   sampler.update_interval_width(correct_coverage)

See Also
--------

* :doc:`sampling_utils` - Utility functions for interval management and preprocessing
* :doc:`thompson_samplers` - Alternative probabilistic acquisition strategy
* :doc:`entropy_samplers` - Information-theoretic acquisition approaches
* :doc:`bound_samplers` - Confidence bound-based strategies
* :doc:`../adaptation/adaptation` - Interval width adaptation mechanisms
