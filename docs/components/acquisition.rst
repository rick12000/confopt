Acquisition Functions
====================

The acquisition module (``confopt.selection.acquisition``) provides the core interface between conformal prediction estimators and optimization algorithms. It implements uncertainty-aware point selection for hyperparameter optimization through conformal prediction-based acquisition functions that maintain finite-sample coverage guarantees while optimizing exploration-exploitation trade-offs.

Overview
--------

The acquisition module bridges conformal prediction estimators with various acquisition strategies, enabling adaptive optimization that adjusts exploration based on prediction uncertainty and coverage feedback. All acquisition functions provide theoretical coverage guarantees while supporting different optimization objectives.

The module follows a strategy pattern architecture where:

- **BaseConformalSearcher**: Defines the common interface and orchestrates acquisition strategies
- **LocallyWeightedConformalSearcher**: Implements variance-adaptive conformal acquisition
- **QuantileConformalSearcher**: Implements direct quantile-based conformal acquisition
- **Sampling Strategies**: Pluggable acquisition behaviors (Thompson sampling, Expected Improvement, etc.)

Architecture
------------

.. mermaid::

   graph TD
       subgraph "Acquisition Layer"
           BCS["BaseConformalSearcher<br/>predict()<br/>update()<br/>calculate_breach()"]
           LWCS["LocallyWeightedConformalSearcher<br/>fit()<br/>_predict_with_*()"]
           QCS["QuantileConformalSearcher<br/>fit()<br/>_predict_with_*()"]
       end

       subgraph "Conformal Estimators"
           LWCE["LocallyWeightedConformalEstimator<br/>Point + Variance Modeling"]
           QCE["QuantileConformalEstimator<br/>Direct Quantile Modeling"]
       end

       subgraph "Sampling Strategies"
           LBS["LowerBoundSampler<br/>UCB with Exploration Decay"]
           TS["ThompsonSampler<br/>Posterior Sampling"]
           PLBS["PessimisticLowerBoundSampler<br/>Conservative Lower Bounds"]
           EIS["ExpectedImprovementSampler<br/>EI via Monte Carlo"]
           ESS["EntropySearchSampler<br/>Information Gain"]
           MVES["MaxValueEntropySearchSampler<br/>Simplified Entropy Search"]
       end

       subgraph "Tuning Integration"
           CT["ConformalTuner<br/>search()<br/>_run_trials()"]
       end

       BCS --> LWCS
       BCS --> QCS
       LWCS --> LWCE
       QCS --> QCE

       BCS --> LBS
       BCS --> TS
       BCS --> PLBS
       BCS --> EIS
       BCS --> ESS
       BCS --> MVES

       CT --> LWCS
       CT --> QCS

BaseConformalSearcher
---------------------

The abstract base class that defines the common interface for all conformal acquisition functions. It implements the Template Method pattern with strategy injection, where acquisition behavior is delegated to samplers while coverage tracking and adaptive behavior are handled by the searcher framework.

**Key Responsibilities:**

- **Strategy Orchestration**: Routes prediction requests to appropriate sampler methods
- **Coverage Tracking**: Manages alpha adaptation through coverage feedback
- **Interface Standardization**: Provides unified API for different acquisition approaches
- **Interval Caching**: Stores prediction intervals for efficient reuse

**Core Methods:**

``predict(X)``
   Routes acquisition function evaluation to the appropriate sampler-specific method based on the configured strategy. Caches interval predictions for potential reuse by update() method.

``update(X, y_true)``
   Updates adaptive alpha values using coverage feedback from observed performance. Calculates beta values (coverage probabilities) and applies adaptive adjustment mechanisms.

``calculate_breach(X, y_true)``
   Determines if observed values fall outside prediction intervals for single-alpha samplers. Returns 1 for breaches (miscoverage) and 0 for coverage.

**Sampler Integration:**

The base class supports six acquisition strategies through polymorphic method dispatch:

- **Upper Confidence Bound**: ``_predict_with_ucb()`` for exploration-exploitation balance
- **Thompson Sampling**: ``_predict_with_thompson()`` for posterior sampling
- **Pessimistic Lower Bound**: ``_predict_with_pessimistic_lower_bound()`` for conservative selection
- **Expected Improvement**: ``_predict_with_expected_improvement()`` for improvement-based acquisition
- **Information Gain**: ``_predict_with_information_gain()`` for entropy-based exploration
- **Max-Value Entropy Search**: ``_predict_with_max_value_entropy_search()`` for simplified entropy search

LocallyWeightedConformalSearcher
---------------------------------

Implements acquisition functions using locally weighted conformal prediction, where prediction intervals adapt to local variance patterns in the objective function. This approach excels when the objective function exhibits heteroscedastic noise, as it can narrow intervals in low-uncertainty regions while expanding them in high-noise areas.

**Mathematical Framework:**

The searcher uses two-stage estimation:

1. **Point Estimation**: :math:`\hat{\mu}(x) = E[Y|X=x]` using point estimator
2. **Variance Estimation**: :math:`\hat{\sigma}^2(x) = E[r^2|X=x]` using residuals from point estimator
3. **Interval Construction**: :math:`[\hat{\mu}(x) \pm q_{1-\alpha}(R) \times \hat{\sigma}(x)]`

Where nonconformity scores are: :math:`R_i = \frac{|y_{val,i} - \hat{\mu}(X_{val,i})|}{\max(\hat{\sigma}(X_{val,i}), \epsilon)}`

**Key Features:**

- **Heteroscedastic Adaptation**: Intervals adapt to local prediction uncertainty
- **Dual Estimator Architecture**: Separate optimization of point and variance estimators
- **Coverage Guarantees**: Maintains finite-sample coverage through conformal calibration
- **Flexible Architectures**: Supports any estimator registered in ESTIMATOR_REGISTRY

**Implementation Details:**

``fit(X_train, y_train, X_val, y_val, tuning_iterations, random_state)``
   Trains both point and variance estimators using split conformal methodology. The training data is further split internally to ensure proper separation between point estimation, variance estimation, and conformal calibration.

``_predict_with_*()`` methods
   Each acquisition strategy method combines point predictions with uncertainty estimates from the variance model. The specific combination depends on the sampler:

   - **UCB**: :math:`\hat{\mu}(x) - \beta(t) \times \hat{\sigma}(x)` with time-dependent exploration
   - **Thompson**: Random sampling from intervals with optional optimistic capping
   - **Expected Improvement**: Monte Carlo estimation using interval sampling

**Usage in Optimization:**

The locally weighted approach is particularly effective for:

- **Engineering Optimization**: Where measurement noise varies across the design space
- **Neural Architecture Search**: Where validation performance uncertainty depends on architecture complexity
- **Hyperparameter Optimization**: Where objective function noise varies with parameter settings

QuantileConformalSearcher
-------------------------

Implements acquisition functions using quantile-based conformal prediction, directly estimating prediction quantiles and applying conformal adjustments when sufficient calibration data is available. This approach automatically switches between conformalized and non-conformalized modes based on data availability.

**Mathematical Framework:**

The searcher operates in two modes:

**Conformalized Mode** (when n_samples ≥ n_pre_conformal_trials):
   :math:`[q_{\alpha/2}(x) - C_\alpha, q_{1-\alpha/2}(x) + C_\alpha]`

**Non-conformalized Mode** (when n_samples < n_pre_conformal_trials):
   :math:`[q_{\alpha/2}(x), q_{1-\alpha/2}(x)]`

Where :math:`C_\alpha` is the conformal adjustment computed from nonconformity scores on the validation set.

**Key Features:**

- **Asymmetric Intervals**: Naturally handles asymmetric prediction uncertainty
- **Automatic Mode Selection**: Switches between conformalized/non-conformalized based on data availability
- **Direct Quantile Modeling**: No separate variance estimation required
- **Flexible Quantile Architectures**: Supports both multi-fit and single-fit quantile estimators

**Implementation Details:**

``fit(X_train, y_train, X_val, y_val, tuning_iterations, random_state)``
   Trains the quantile estimator and sets up conformal calibration. Handles sampler-specific configurations and optional point estimator setup for optimistic Thompson sampling.

**Mode Selection Logic:**
   - Uses total sample count (n_train + n_val) to determine mode
   - Conformalized mode provides stronger coverage guarantees
   - Non-conformalized mode offers computational efficiency for small datasets

**Quantile Estimator Integration:**

The searcher supports various quantile architectures:

- **Gradient Boosting**: LightGBM and scikit-learn implementations
- **Random Forest**: Quantile random forest variants
- **Neural Networks**: Deep quantile regression models
- **Gaussian Processes**: GP-based quantile estimation
- **Ensemble Methods**: Stacked quantile estimators

Integration with Tuning Process
--------------------------------

The acquisition functions integrate with the main optimization loop through ``ConformalTuner``:

**Initialization Phase:**
   1. Tuner creates searcher instance with specified architecture and sampler
   2. Random search phase collects initial data for model training
   3. Searcher.fit() trains conformal estimators on collected data

**Optimization Phase:**
   1. Tuner calls searcher.predict() on candidate configurations
   2. Searcher returns acquisition values for configuration selection
   3. Tuner evaluates selected configuration and observes performance
   4. Searcher.update() adjusts alpha values using coverage feedback

**Adaptive Behavior:**
   - Alpha values adapt based on empirical coverage rates
   - Model retraining occurs periodically to incorporate new data
   - Exploration-exploitation balance evolves through sampler-specific mechanisms

**Data Flow:**

.. mermaid::

   sequenceDiagram
       participant Tuner
       participant Searcher
       participant Estimator
       participant Sampler

       Tuner->>Searcher: fit(X_train, y_train, X_val, y_val)
       Searcher->>Estimator: fit() with conformal calibration

       loop Optimization
           Tuner->>Searcher: predict(X_candidates)
           Searcher->>Estimator: predict_intervals(X_candidates)
           Searcher->>Sampler: calculate_*_predictions()
           Sampler-->>Searcher: acquisition_values
           Searcher-->>Tuner: acquisition_values

           Tuner->>Searcher: update(X_selected, y_observed)
           Searcher->>Sampler: update alpha adaptation
       end

Performance Characteristics
---------------------------

**Computational Complexity:**

- **LocallyWeighted**: O(n_train) for dual estimator training + O(n_val) for calibration
- **Quantile**: O(n_train × n_quantiles) for multi-fit or O(n_train) for single-fit
- **Prediction**: O(1) per candidate point for both approaches
- **Update**: O(n_alphas) for alpha adaptation

**Memory Requirements:**

- **Training Data**: Stored for potential model retraining
- **Nonconformity Scores**: O(n_val) for conformal calibration
- **Interval Predictions**: Cached for efficient sampler access
- **Alpha Adaptation**: O(n_alphas × n_experts) for DtACI adaptation

**Scalability Considerations:**

- Both approaches scale linearly with training data size
- Quantile approach scales with number of quantile levels
- Information gain samplers have higher computational cost due to model refitting
- Parallel evaluation possible for batch acquisition scenarios

Best Practices
---------------

**Architecture Selection:**

- **LocallyWeighted**: Use when objective function has heteroscedastic noise
- **Quantile**: Use when asymmetric uncertainty or limited data availability
- **Point Estimator**: Choose based on problem characteristics (smoothness, dimensionality)
- **Variance Estimator**: Should complement point estimator choice

**Sampler Selection:**

- **Thompson Sampling**: Good general-purpose choice with theoretical guarantees
- **Expected Improvement**: Effective for expensive function evaluations
- **Information Gain**: Best for complex, multi-modal objective functions
- **Lower Bound**: Simple and efficient for well-behaved functions

**Hyperparameter Tuning:**

- **n_candidate_configurations**: Balance between exploration and computational cost
- **tuning_iterations**: More iterations for complex estimator architectures
- **n_pre_conformal_trials**: Adjust based on desired coverage vs. efficiency trade-off
- **alpha values**: Start with standard levels (0.1, 0.05) and allow adaptation

**Common Pitfalls:**

- Insufficient validation data for reliable conformal calibration
- Mismatched estimator architectures for point and variance estimation
- Overly aggressive alpha adaptation leading to coverage violations
- Inadequate warm-up phase before conformal prediction activation
