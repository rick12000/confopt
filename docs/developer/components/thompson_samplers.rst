Thompson Sampling Module
========================

Overview
--------

The ``thompson_samplers`` module implements Thompson sampling for conformal prediction optimization, providing a probabilistic approach to exploration-exploitation trade-offs in Bayesian optimization. The implementation adapts classical Thompson sampling to conformal prediction settings by using random sampling from prediction intervals to approximate posterior sampling over the objective function.

Thompson sampling naturally balances exploration of uncertain regions with exploitation of promising areas through randomization, offering theoretical guarantees for regret minimization in bandit-style optimization problems.

Key Features
------------

* **Interval-Based Posterior Approximation**: Uses conformal prediction intervals as surrogates for posterior distributions
* **Symmetric Quantile Construction**: Methodologically-grounded confidence level selection
* **Adaptive Interval Widths**: Dynamic adjustment based on empirical coverage feedback
* **Optimistic Sampling Option**: Enhanced exploration through point estimate integration
* **Vectorized Implementation**: Efficient computation for large candidate sets
* **Multi-Scale Uncertainty**: Support for multiple confidence levels simultaneously

Architecture
------------

The module implements a single ``ThompsonSampler`` class that encapsulates the complete Thompson sampling methodology:

**Core Components**
  - Quantile-based alpha initialization for nested interval construction
  - Multi-adapter configuration for independent interval width adjustment
  - Random sampling mechanism for posterior approximation
  - Optional optimistic exploration enhancement

**Integration Points**
  - Accepts ``ConformalBounds`` objects from conformal predictors
  - Uses adaptation framework for coverage-based interval adjustment
  - Provides standardized interfaces for acquisition function optimization

Mathematical Foundation and Derivation
-------------------------------------

Thompson sampling provides a principled probabilistic approach to the exploration-exploitation trade-off by sampling from posterior distributions over the objective function.

**Classical Thompson Sampling**

In the multi-armed bandit setting, Thompson sampling selects actions by:

1. **Posterior Sampling**: Sample a function realization from the posterior:

   .. math::
      \tilde{f} \sim p(f | \mathcal{D})

   where :math:`\mathcal{D} = \{(x_i, y_i)\}_{i=1}^t` is the observed data.

2. **Optimistic Action**: Select the action that optimizes the sampled function:

   .. math::
      a_t = \arg\max_a \tilde{f}(a)

**Conformal Prediction Adaptation**

In conformal prediction settings, we adapt this by treating prediction intervals as implicit posterior representations:

1. **Interval-Based Sampling**: For each candidate :math:`x`, sample from its prediction intervals:

   .. math::
      \tilde{y}(x) \sim \text{Uniform}(\mathcal{I}(x))

   where :math:`\mathcal{I}(x) = \bigcup_{j=1}^k [L_j(x), U_j(x)]` represents the union of conformal intervals.

2. **Acquisition Decision**: Select the candidate with the most optimistic sample:

   .. math::
      x_t = \arg\min_{x \in \mathcal{X}} \tilde{y}(x)

**Multi-Interval Construction**

The nested interval structure follows symmetric quantile pairing:

.. math::
   \alpha_i = 1 - (q_{n+1-i} - q_i)

where :math:`q_i = \frac{i}{n+1}` for :math:`i = 1, \ldots, n`.

This produces nested intervals:

.. math::
   I_{\alpha_1}(x) \supseteq I_{\alpha_2}(x) \supseteq \cdots \supseteq I_{\alpha_k}(x)

with decreasing miscoverage rates :math:`\alpha_1 > \alpha_2 > \cdots > \alpha_k`.

**Sampling Mechanism**

The uniform sampling across all interval bounds creates an implicit probability distribution:

.. math::
   p(\tilde{y}(x)) = \frac{1}{2k} \sum_{j=1}^k [\delta(L_j(x)) + \delta(U_j(x))]

where :math:`\delta(\cdot)` is the Dirac delta function.

**Optimistic Enhancement**

When point predictions :math:`\hat{y}(x)` are available, optimistic sampling applies:

.. math::
   \tilde{y}_{\text{opt}}(x) = \min(\tilde{y}(x), \hat{y}(x))

This modification encourages exploitation of regions where point estimates are optimistic relative to interval samples.

**Regret Guarantees**

Under appropriate conditions, Thompson sampling achieves sublinear regret:

.. math::
   R_T = O(\sqrt{T \log T})

where :math:`T` is the number of evaluations, making it competitive with UCB-based strategies while maintaining computational simplicity.

Thompson Sampling Methodology
-----------------------------

Thompson sampling addresses the exploration-exploitation dilemma in optimization under uncertainty by randomly sampling from posterior distributions over the objective function. In conformal prediction settings, prediction intervals serve as approximations to these posterior distributions.

**Theoretical Foundation**

Classical Thompson sampling selects actions by sampling from posterior distributions:

.. math::
   a_t = \arg\max_{a} \tilde{f}(a)

where :math:`\tilde{f}` is sampled from the posterior over the objective function.

**Conformal Adaptation**

The conformal version approximates this by random sampling from prediction intervals:

.. math::
   x_t = \arg\min_{x} \tilde{y}(x)

where :math:`\tilde{y}(x)` is randomly sampled from the prediction interval :math:`[L(x), U(x)]`.

**Regret Guarantees**

Under appropriate conditions, Thompson sampling achieves :math:`O(\sqrt{T \log T})` regret bounds, making it competitive with other acquisition strategies while maintaining computational simplicity.

Multi-Interval Construction
---------------------------

The sampler constructs nested prediction intervals using symmetric quantile pairing, enabling multi-scale uncertainty quantification:

**Quantile Selection**

For :math:`n` quantiles (even), symmetric pairs :math:`(q_i, q_{n+1-i})` generate alpha values:

.. math::
   \alpha_i = 1 - (q_{n+1-i} - q_i)

**Nested Intervals**

This produces nested intervals with decreasing alpha values:

.. math::
   I_1(x) \supseteq I_2(x) \supseteq \cdots \supseteq I_k(x)

where :math:`I_j(x)` represents the :math:`j`-th confidence interval.

**Sampling Strategy**

Random sampling uniformly selects from all available interval bounds, naturally weighting by interval width and confidence level.

Optimistic Sampling Enhancement
-------------------------------

The optional optimistic sampling feature combines Thompson sampling with point estimate exploitation:

.. math::
   \tilde{y}_{\text{opt}}(x) = \min(\tilde{y}(x), \hat{y}(x))

where :math:`\hat{y}(x)` is the point prediction and :math:`\tilde{y}(x)` is the interval sample.

This modification encourages exploitation of regions where point estimates are optimistic relative to sampled values, potentially accelerating convergence in well-modeled regions.

Usage Examples
--------------

**Basic Thompson Sampling**

.. code-block:: python

   from confopt.selection.sampling.thompson_samplers import ThompsonSampler

   # Initialize sampler with 4 quantiles
   sampler = ThompsonSampler(n_quantiles=4)

   # Get current alpha values
   alphas = sampler.fetch_alphas()  # [0.4, 0.2] for 60%, 80% confidence

   # Calculate Thompson sampling predictions
   thompson_values = sampler.calculate_thompson_predictions(
       predictions_per_interval=conformal_bounds
   )

   # Select candidate with minimum sampled value
   selected_idx = np.argmin(thompson_values)

**Adaptive Interval Width Management**

.. code-block:: python

   # Initialize with DtACI adaptation
   adaptive_sampler = ThompsonSampler(
       n_quantiles=6,
       adapter="DtACI"
   )

   # Update interval widths based on observed coverage
   observed_coverage = [0.65, 0.82, 0.91]  # For 60%, 80%, 90% intervals
   adaptive_sampler.update_interval_width(observed_coverage)

   # Updated alphas reflect coverage feedback
   updated_alphas = adaptive_sampler.fetch_alphas()

**Optimistic Exploration**

.. code-block:: python

   # Enable optimistic sampling for enhanced exploitation
   optimistic_sampler = ThompsonSampler(
       n_quantiles=4,
       enable_optimistic_sampling=True
   )

   # Provide point predictions for optimistic capping
   thompson_values = optimistic_sampler.calculate_thompson_predictions(
       predictions_per_interval=conformal_bounds,
       point_predictions=point_estimates
   )

**Integration with Optimization Loop**

.. code-block:: python

   import numpy as np
   from confopt.selection.sampling.thompson_samplers import ThompsonSampler

   def optimization_loop(conformal_predictor, candidate_space, n_iterations=50):
       sampler = ThompsonSampler(n_quantiles=4, adapter="DtACI")

       for iteration in range(n_iterations):
           # Get conformal predictions for all candidates
           predictions = conformal_predictor.predict_intervals(candidate_space)

           # Calculate Thompson sampling values
           acquisition_values = sampler.calculate_thompson_predictions(predictions)

           # Select candidate with minimum sampled value
           selected_idx = np.argmin(acquisition_values)
           selected_x = candidate_space[selected_idx]

           # Evaluate objective function
           observed_y = objective_function(selected_x)

           # Update model and adaptation (coverage tracking would go here)
           conformal_predictor.update(selected_x, observed_y)

Advanced Configuration
---------------------

**Multi-Scale Quantile Selection**

Different quantile counts provide different exploration characteristics:

.. code-block:: python

   # Conservative: Fewer intervals, more focused sampling
   conservative_sampler = ThompsonSampler(n_quantiles=4)

   # Aggressive: More intervals, finer uncertainty resolution
   aggressive_sampler = ThompsonSampler(n_quantiles=8)

   # Balanced: Moderate complexity with good performance
   balanced_sampler = ThompsonSampler(n_quantiles=6)

**Adaptation Strategy Selection**

.. code-block:: python

   # No adaptation: Fixed interval widths
   static_sampler = ThompsonSampler(adapter=None)

   # Conservative adaptation: Stable coverage maintenance
   conservative_sampler = ThompsonSampler(adapter="ACI")

   # Aggressive adaptation: Rapid width adjustment
   aggressive_sampler = ThompsonSampler(adapter="DtACI")

Performance Considerations
-------------------------

**Computational Complexity**
- Initialization: O(n_quantiles)
- Prediction: O(n_observations × n_quantiles)
- Adaptation: O(n_quantiles) per update
- Memory: O(n_observations × n_quantiles) for flattened bounds

**Scaling Guidelines**
- Quantile count affects both accuracy and computational cost
- Vectorized implementation enables efficient batch processing
- Flattened bounds representation optimizes memory access patterns

**Parameter Selection**
- 4-6 quantiles typically provide good exploration-exploitation balance
- More quantiles increase computational cost with diminishing returns
- Adaptation frequency should balance responsiveness with stability

**Performance Optimization**

.. code-block:: python

   # Efficient batch processing
   def batch_thompson_sampling(sampler, prediction_batches):
       results = []
       for batch in prediction_batches:
           thompson_values = sampler.calculate_thompson_predictions(batch)
           results.append(thompson_values)
       return np.concatenate(results)

Integration Points
-----------------

**Conformal Prediction Framework**
  Directly processes ``ConformalBounds`` objects from any conformal predictor implementing the standard interface.

**Adaptation Mechanisms**
  Integrates with ``DtACI`` and ``ACI`` adapters for dynamic interval width adjustment based on coverage feedback.

**Optimization Pipelines**
  Provides acquisition values compatible with standard optimization routines and multi-armed bandit frameworks.

**Ensemble Methods**
  Can be combined with other acquisition strategies for hybrid approaches or used in portfolio optimization settings.

Common Pitfalls
---------------

**Quantile Count Constraints**
  Always use even numbers of quantiles for symmetric pairing:

.. code-block:: python

   # Correct
   sampler = ThompsonSampler(n_quantiles=4)  # Works

   # Incorrect
   sampler = ThompsonSampler(n_quantiles=5)  # Raises ValueError

**Coverage Rate Ordering**
  Ensure coverage rates match alpha value ordering when updating:

.. code-block:: python

   # For alphas [0.4, 0.2] (60%, 80% confidence)
   coverage_rates = [0.62, 0.81]  # Must correspond to [60%, 80%]
   sampler.update_interval_width(coverage_rates)

**Point Prediction Compatibility**
  When using optimistic sampling, ensure point predictions have compatible shapes:

.. code-block:: python

   # Correct: Matching shapes
   n_candidates = len(predictions_per_interval[0].lower_bounds)
   point_preds = np.array([...])  # Shape: (n_candidates,)

   # Calculate with proper shapes
   values = sampler.calculate_thompson_predictions(
       predictions_per_interval=predictions,
       point_predictions=point_preds
   )

**Adaptation State Management**
  Don't reinitialize samplers during optimization to preserve adaptation state:

.. code-block:: python

   # Correct: Reuse sampler instance
   sampler = ThompsonSampler(adapter="DtACI")
   for iteration in optimization_loop:
       # Use same sampler instance
       values = sampler.calculate_thompson_predictions(predictions)
       sampler.update_interval_width(coverage_rates)

   # Incorrect: Loses adaptation history
   for iteration in optimization_loop:
       sampler = ThompsonSampler(adapter="DtACI")  # Wrong!

See Also
--------

* :doc:`sampling_utils` - Utility functions used by Thompson sampling
* :doc:`expected_improvement_samplers` - Alternative acquisition strategy
* :doc:`entropy_samplers` - Information-theoretic acquisition strategies
* :doc:`bound_samplers` - Confidence bound acquisition strategies
* :doc:`../adaptation/adaptation` - Interval width adaptation mechanisms
