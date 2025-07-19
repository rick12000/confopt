Sampling Strategies
===================

The sampling module (``confopt.selection.sampling``) implements diverse acquisition strategies that define how the optimization algorithm selects the next configuration to evaluate. These strategies operate within the conformal prediction framework to balance exploration and exploitation while maintaining statistical coverage guarantees.

Overview
--------

Sampling strategies serve as the core decision-making components in conformal optimization, determining which candidate configurations are most promising for evaluation. Each strategy implements a different approach to the exploration-exploitation trade-off:

- **Bound-based Samplers**: Use confidence bounds for conservative or aggressive exploration
- **Thompson Sampling**: Probabilistic posterior sampling for balanced exploration
- **Expected Improvement**: Improvement-based acquisition for efficient optimization
- **Entropy-based Methods**: Information-theoretic approaches for complex landscapes

All samplers integrate with the adaptive conformal inference framework, allowing dynamic adjustment of exploration behavior based on empirical coverage performance.

Architecture
------------

.. mermaid::

   graph TD
       subgraph "Sampling Strategies"
           LBS["LowerBoundSampler<br/>UCB with Exploration Decay"]
           PLBS["PessimisticLowerBoundSampler<br/>Conservative Lower Bounds"]
           TS["ThompsonSampler<br/>Posterior Sampling"]
           EIS["ExpectedImprovementSampler<br/>Monte Carlo EI"]
           ESS["EntropySearchSampler<br/>Information Gain"]
           MVES["MaxValueEntropySearchSampler<br/>Simplified Entropy"]
       end

       subgraph "Adaptive Components"
           DTACI["DtACI Adaptation<br/>Multi-Expert Learning"]
           UTILS["Sampling Utils<br/>Alpha Initialization<br/>Adapter Management"]
       end

       subgraph "Conformal Integration"
           CB["ConformalBounds<br/>Interval Representations"]
           SEARCHER["BaseConformalSearcher<br/>Strategy Orchestration"]
       end

       LBS --> DTACI
       PLBS --> DTACI
       TS --> DTACI
       EIS --> DTACI
       ESS --> DTACI
       MVES --> DTACI

       DTACI --> UTILS
       UTILS --> CB
       CB --> SEARCHER

Bound-based Samplers
--------------------

Bound-based samplers utilize specific bounds from prediction intervals to make acquisition decisions, providing direct interpretable acquisition values while maintaining uncertainty quantification.

PessimisticLowerBoundSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements conservative acquisition using pessimistic lower bounds from prediction intervals. This strategy prioritizes risk-averse decision making by focusing on worst-case scenarios.

**Mathematical Framework:**

For prediction interval :math:`[L(x), U(x)]` with confidence level :math:`1-\alpha`:

.. math::

   \text{Acquisition}(x) = L(x)

**Key Features:**

- **Conservative Bias**: Assumes pessimistic scenarios for robust optimization
- **Single Interval**: Uses one confidence level for computational efficiency
- **Interpretable Values**: Direct lower bound extraction for acquisition decisions
- **Adaptive Width**: Optional DtACI integration for interval adjustment

**Implementation Details:**

``__init__(interval_width=0.8, adapter=None)``
   Initializes with specified confidence level and optional adaptation mechanism.

``calculate_pessimistic_lower_bound_predictions(predictions_per_interval)``
   Extracts lower bounds from conformal prediction intervals for acquisition ranking.

**Usage Scenarios:**

- **Risk-averse Optimization**: When conservative estimates are preferred
- **Safety-critical Applications**: Where worst-case performance matters
- **Stable Objectives**: Functions with predictable uncertainty patterns

LowerBoundSampler
~~~~~~~~~~~~~~~~~

Extends pessimistic lower bound sampling with sophisticated exploration control through time-dependent exploration parameters. Implements Lower Confidence Bound (LCB) strategy adapted for minimization.

**Mathematical Framework:**

.. math::

   \text{LCB}(x) = \mu(x) - \beta(t) \cdot \sigma(x)

Where:
- :math:`\mu(x)`: Point estimate
- :math:`\sigma(x)`: Interval width (uncertainty estimate)
- :math:`\beta(t)`: Time-dependent exploration parameter

**Exploration Decay Strategies:**

**Logarithmic Decay** (default):
   :math:`\beta(t) = \min\left(\beta_{\max}, c\sqrt{\frac{\log t}{t}}\right)`

**Inverse Square Root Decay**:
   :math:`\beta(t) = \min\left(\beta_{\max}, c\sqrt{\frac{1}{t}}\right)`

**Key Features:**

- **Theoretical Guarantees**: Regret bounds under appropriate decay schedules
- **Adaptive Exploration**: Balances exploitation and uncertainty quantification
- **Exploration Control**: Configurable decay parameters and maximum values
- **UCB Adaptation**: Lower confidence bound variant for minimization problems

**Implementation Details:**

``__init__(interval_width=0.8, adapter=None, beta_decay="logarithmic_decay", c=1, beta_max=10)``
   Configures LCB with exploration decay schedule and bounds.

``update_exploration_step()``
   Updates time step and recalculates exploration parameter according to decay schedule.

``calculate_ucb_predictions(point_estimates, interval_width)``
   Computes LCB acquisition values combining point predictions with exploration bonuses.

**Performance Characteristics:**

- **Regret Bounds**: :math:`O(\sqrt{T \log T})` for logarithmic decay
- **Convergence**: Guaranteed convergence to global optimum under regularity conditions
- **Computational Cost**: O(1) per evaluation with efficient vectorized operations

Thompson Sampling
------------------

Implements probabilistic posterior sampling for conformal prediction, providing a principled approach to exploration-exploitation balance through random sampling from prediction intervals.

**Mathematical Framework:**

Thompson sampling approximates posterior sampling by randomly drawing values from prediction intervals:

1. **Interval Construction**: Create nested intervals using symmetric quantile pairing
2. **Random Sampling**: Draw random values from flattened interval representation
3. **Optimistic Capping**: Optional point estimate integration for exploitation

**Key Features:**

- **Theoretical Foundation**: Regret guarantees for bandit-style optimization
- **Multi-Interval Support**: Uses multiple confidence levels for fine-grained uncertainty
- **Optimistic Mode**: Optional point estimate capping for enhanced exploitation
- **Adaptive Intervals**: DtACI integration for dynamic interval adjustment

**Implementation Details:**

``__init__(n_quantiles=4, adapter=None, enable_optimistic_sampling=False)``
   Initializes with quantile-based intervals and optional optimistic sampling.

``calculate_thompson_predictions(predictions_per_interval, point_predictions=None)``
   Generates Thompson sampling predictions through random interval sampling.

**Quantile-based Alpha Initialization:**

Uses symmetric quantile pairing for nested interval construction:

.. math::

   \alpha_i = \frac{2i}{n_{\text{quantiles}}} \quad \text{for } i = 1, 2, \ldots, \frac{n_{\text{quantiles}}}{2}

**Algorithm Steps:**

1. **Flatten Intervals**: Convert nested intervals to efficient matrix representation
2. **Random Sampling**: Draw column indices for each observation
3. **Value Extraction**: Extract corresponding interval bounds
4. **Optimistic Capping**: Apply point estimate bounds if enabled

**Performance Characteristics:**

- **Sampling Complexity**: O(n_intervals × n_observations)
- **Memory Usage**: O(n_intervals × n_observations) for flattened representation
- **Regret Properties**: Matches theoretical Thompson sampling guarantees

Expected Improvement Sampling
-----------------------------

Implements Expected Improvement (EI) acquisition using Monte Carlo estimation from conformal prediction intervals, extending classical Bayesian optimization to conformal settings.

**Mathematical Framework:**

Expected Improvement computes the expected value of improvement over the current best:

.. math::

   \text{EI}(x) = \mathbb{E}[\max(f_{\min} - f(x), 0)]

Where the expectation is estimated through Monte Carlo sampling from prediction intervals.

**Monte Carlo Estimation:**

1. **Sample Generation**: Draw random samples from prediction intervals
2. **Improvement Calculation**: Compute improvements over current best
3. **Expectation Estimation**: Average improvements across samples

**Key Features:**

- **Improvement Focus**: Directly optimizes expected improvement over current best
- **Monte Carlo Flexibility**: Adapts to arbitrary interval shapes through sampling
- **Dynamic Best Tracking**: Automatically updates current best value
- **Efficient Computation**: Vectorized operations for batch evaluation

**Implementation Details:**

``__init__(n_quantiles=4, adapter=None, current_best_value=float("inf"), num_ei_samples=20)``
   Configures EI with interval construction and sampling parameters.

``calculate_expected_improvement(predictions_per_interval)``
   Estimates expected improvement through Monte Carlo sampling from intervals.

``update_best_value(y_observed)``
   Updates current best value for improvement computation.

**Algorithm Steps:**

1. **Interval Flattening**: Convert prediction intervals to sampling matrix
2. **Random Sampling**: Generate Monte Carlo samples from intervals
3. **Improvement Computation**: Calculate improvements over current best
4. **Expectation Estimation**: Compute sample mean of improvements

**Performance Characteristics:**

- **Sampling Complexity**: O(n_samples × n_intervals × n_observations)
- **Accuracy**: Improves with number of Monte Carlo samples
- **Convergence**: Approaches true EI as sample count increases

Information-Theoretic Samplers
-------------------------------

Information-theoretic samplers use entropy-based measures to quantify and maximize information gain about the global optimum location, providing principled exploration for complex optimization landscapes.

EntropySearchSampler
~~~~~~~~~~~~~~~~~~~~

Implements full Entropy Search using information gain maximization through Monte Carlo simulation and conditional entropy reduction.

**Mathematical Framework:**

Information gain is computed as the reduction in entropy about the optimum location:

.. math::

   \text{IG}(x) = H[p_{\min}] - \mathbb{E}_{y|x}[H[p_{\min}|y]]

Where:
- :math:`H[p_{\min}]`: Current entropy of optimum location distribution
- :math:`H[p_{\min}|y]`: Conditional entropy after observing y at x

**Key Features:**

- **Full Information Gain**: Computes exact information gain through model updates
- **Candidate Selection**: Multiple strategies for efficient candidate screening
- **Entropy Estimation**: Distance-based and histogram methods for entropy calculation
- **Model Refitting**: Updates conformal estimators for each candidate evaluation

**Implementation Details:**

``__init__(n_quantiles=4, adapter=None, n_paths=100, n_x_candidates=10, n_y_candidates_per_x=3, sampling_strategy="uniform", entropy_measure="distance")``
   Configures entropy search with simulation and candidate selection parameters.

``calculate_information_gain(X_train, y_train, X_val, y_val, X_space, conformal_estimator, predictions_per_interval, n_jobs=1)``
   Computes information gain through model refitting and entropy estimation.

**Candidate Selection Strategies:**

- **Thompson Sampling**: Uses Thompson samples for candidate screening
- **Expected Improvement**: EI-based candidate selection
- **Sobol Sampling**: Low-discrepancy sequences for space-filling selection
- **Uniform Random**: Simple random candidate selection
- **Perturbation**: Local search around current best

**Entropy Estimation Methods:**

**Distance-based (Vasicek Estimator)**:
   :math:`\hat{H} = \frac{1}{n} \sum_{i=1}^n \log\left(\frac{n+1}{m}(X_{(i+m)} - X_{(i-m)})\right)`

**Histogram-based (Scott's Rule)**:
   :math:`\hat{H} = -\sum_{i=1}^{n_{\text{bins}}} p_i \log p_i`

**Performance Characteristics:**

- **Computational Cost**: High due to model refitting for each candidate
- **Information Quality**: Excellent exploration properties with strong theoretical foundation
- **Scalability**: Suitable for expensive function evaluations where acquisition cost is justified

MaxValueEntropySearchSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements simplified entropy search focusing on maximum value entropy reduction, providing computational efficiency while maintaining information-theoretic principles.

**Mathematical Framework:**

Focuses on entropy reduction of the maximum value rather than full optimum location:

.. math::

   \text{MES}(x) = H[f_{\max}] - \mathbb{E}_{y|x}[H[f_{\max}|y]]

**Key Features:**

- **Computational Efficiency**: Avoids expensive model refitting
- **Value-focused**: Directly targets maximum value uncertainty
- **Vectorized Operations**: Efficient batch evaluation
- **Simplified Entropy**: Direct entropy computation without model updates

**Implementation Details:**

``__init__(n_quantiles=4, adapter=None, n_paths=100, n_y_candidates_per_x=20, entropy_method="distance")``
   Configures MES with entropy estimation parameters.

``calculate_max_value_entropy_search(predictions_per_interval)``
   Computes simplified entropy search acquisition values.

**Algorithm Steps:**

1. **Prior Entropy**: Estimate entropy of current maximum value distribution
2. **Conditional Sampling**: Generate hypothetical observations for each candidate
3. **Conditional Entropy**: Estimate entropy after hypothetical observations
4. **Information Gain**: Compute entropy reduction for each candidate

**Performance Characteristics:**

- **Computational Cost**: Significantly lower than full entropy search
- **Exploration Quality**: Good information-theoretic guidance
- **Scalability**: Suitable for moderate to large-scale optimization

Sampling Utilities
-------------------

The utilities module (``confopt.selection.sampling.utils``) provides shared functionality for sampling strategy implementations, including alpha initialization, adapter management, and preprocessing utilities.

**Key Functions:**

``initialize_quantile_alphas(n_quantiles)``
   Creates symmetric quantile-based alpha values for nested interval construction.

``initialize_multi_adapters(alphas, adapter)``
   Sets up independent DtACI instances for multi-interval samplers.

``initialize_single_adapter(alpha, adapter)``
   Creates single DtACI instance for single-interval samplers.

``update_multi_interval_widths(predictions_per_interval, adapters, betas)``
   Updates interval widths using coverage feedback from multiple adapters.

``validate_even_quantiles(n_quantiles, sampler_name)``
   Ensures even number of quantiles for symmetric pairing.

``flatten_conformal_bounds(predictions_per_interval)``
   Converts nested intervals to efficient matrix representation for sampling.

Integration Patterns
---------------------

Samplers integrate with the broader optimization framework through standardized interfaces:

**Initialization Phase:**

1. **Sampler Creation**: Instantiate with configuration parameters
2. **Alpha Setup**: Initialize alpha values for interval construction
3. **Adapter Configuration**: Set up adaptive components if requested

**Optimization Loop:**

1. **Prediction Request**: Acquisition function calls sampler methods
2. **Interval Processing**: Convert conformal bounds to acquisition values
3. **Value Return**: Provide acquisition scores for configuration ranking
4. **Adaptation Update**: Adjust parameters based on coverage feedback

**Common Interface Methods:**

``fetch_alphas()``
   Returns current alpha values for conformal estimator configuration.

``calculate_*_predictions()``
   Strategy-specific acquisition value computation.

``update_*()`` (when applicable)
   Updates sampler state based on new observations.

Performance Comparison
----------------------

**Computational Complexity:**

- **Bound Samplers**: O(1) per evaluation - most efficient
- **Thompson Sampling**: O(n_intervals) per evaluation - moderate cost
- **Expected Improvement**: O(n_samples × n_intervals) - higher cost
- **Entropy Search**: O(n_candidates × model_refit_cost) - highest cost
- **Max-Value Entropy**: O(n_paths × n_candidates) - moderate-high cost

**Exploration Quality:**

- **Information Gain**: Excellent for complex, multi-modal functions
- **Thompson Sampling**: Good general-purpose exploration with guarantees
- **Expected Improvement**: Effective for unimodal functions
- **Lower Bound**: Simple and reliable for well-behaved objectives
- **Pessimistic Bound**: Conservative exploration for risk-averse scenarios

**Theoretical Guarantees:**

- **Thompson Sampling**: Regret bounds matching optimal Bayesian strategies
- **Lower Bound**: UCB-style regret guarantees under regularity conditions
- **Expected Improvement**: Convergence guarantees for GP-based optimization
- **Entropy Methods**: Information-theoretic optimality under uncertainty

Best Practices
---------------

**Strategy Selection:**

- **Thompson Sampling**: Default choice for balanced exploration-exploitation
- **Expected Improvement**: Use for expensive evaluations with clear improvement focus
- **Information Gain**: Best for complex landscapes with multiple modes
- **Lower Bound**: Simple and effective for smooth, unimodal functions
- **Pessimistic Bound**: Conservative choice for safety-critical applications

**Parameter Tuning:**

- **n_quantiles**: 4-8 for most applications, higher for fine-grained uncertainty
- **n_samples**: 20-50 for Monte Carlo methods, balance accuracy vs. cost
- **adaptation**: Use "DtACI" for robust adaptation, "ACI" for conservative adjustment
- **exploration parameters**: Tune based on optimization horizon and noise level

**Common Pitfalls:**

- **Insufficient quantiles**: Too few levels may miss important uncertainty structure
- **Over-sampling**: Excessive Monte Carlo samples provide diminishing returns
- **Aggressive adaptation**: Too fast alpha adjustment can destabilize coverage
- **Strategy mismatch**: Wrong sampler choice for objective function characteristics

**Integration Guidelines:**

- **Warm-up period**: Allow sufficient random search before conformal prediction
- **Coverage monitoring**: Track empirical coverage vs. target levels
- **Computational budgets**: Balance acquisition cost vs. evaluation cost
- **Multi-objective**: Consider different samplers for different optimization phases
