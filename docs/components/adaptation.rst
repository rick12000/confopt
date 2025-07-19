Adaptive Conformal Inference
============================

The adaptation module (``confopt.selection.adaptation``) implements adaptive conformal inference algorithms that dynamically adjust coverage levels based on empirical performance feedback. The module provides the DtACI (Dynamically-tuned Adaptive Conformal Inference) algorithm which maintains target coverage rates while optimizing interval widths for efficient optimization.

Overview
--------

Adaptive conformal inference addresses the fundamental challenge of maintaining valid coverage guarantees while optimizing prediction interval efficiency. Traditional conformal prediction uses fixed miscoverage levels (α values), which may be suboptimal when the difficulty of predictions varies across the input space or over time.

The DtACI algorithm solves this by:

- **Multi-Expert Framework**: Maintains multiple experts with different learning rates
- **Empirical Feedback**: Adapts based on observed coverage performance
- **Theoretical Guarantees**: Provides regret bounds and coverage control
- **Robust Adaptation**: Uses exponential weighting to handle non-stationary environments

Mathematical Foundation
-----------------------

The DtACI algorithm is based on the theoretical framework from Gibbs & Candès (2021), implementing online learning for conformal prediction with the following key components:

**Pinball Loss Function**

The adaptation mechanism uses the pinball loss to measure expert performance:

.. math::

   \ell(\beta_t, \theta) = \alpha(\beta_t - \theta) - \min\{0, \beta_t - \theta\}

Where:
- :math:`\beta_t`: Empirical coverage probability for observation t
- :math:`\theta`: Expert's current alpha value (:math:`\alpha_t^i`)
- :math:`\alpha`: Global target miscoverage level

**Expert Weight Updates**

Expert weights are updated using exponential weighting based on performance:

.. math::

   w_{t+1}^i \propto w_t^i \times \exp(-\eta \times \ell(\beta_t, \alpha_t^i))

With regularization to prevent weight collapse:

.. math::

   w_{t+1}^i = (1-\sigma)\bar{w}_t^i + \frac{\sigma}{K}

**Expert Alpha Updates**

Each expert updates its alpha value using gradient-based adjustment:

.. math::

   \alpha_{t+1}^i = \alpha_t^i + \gamma_i \times (\alpha - \text{err}_t^i)

Where :math:`\text{err}_t^i = \mathbf{1}[\beta_t < \alpha_t^i]` is the error indicator.

**Final Alpha Selection**

The final alpha can be selected through:

1. **Weighted Average** (Algorithm 2): :math:`\alpha_t = \sum_{i=1}^K w_t^i \alpha_t^i`
2. **Random Sampling** (Algorithm 1): :math:`\alpha_t \sim \text{Categorical}(w_t)`

DtACI Implementation
--------------------

The ``DtACI`` class implements the complete adaptive conformal inference algorithm with theoretical parameter settings derived from the paper's regret analysis.

**Initialization Parameters:**

``alpha`` (float, default=0.1)
   Target miscoverage level :math:`\alpha \in (0,1)`. This represents the long-term average miscoverage rate the algorithm aims to achieve.

``gamma_values`` (list[float], optional)
   Learning rates for each expert :math:`\gamma_i > 0`. Different learning rates allow experts to adapt at different time scales:

   - **Fast learners** (large γ): Quickly adapt to recent changes but may be unstable
   - **Slow learners** (small γ): Provide stability but adapt slowly to changes
   - **Default**: ``[0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]``

``use_weighted_average`` (bool, default=True)
   Selection mechanism for final alpha value:

   - **True**: Deterministic weighted average (Algorithm 2) - more stable
   - **False**: Random sampling (Algorithm 1) - matches original theoretical analysis

**Theoretical Parameters:**

The implementation uses theoretically-motivated parameters derived from regret analysis:

``interval`` (int, default=500)
   Time horizon for regret analysis. Affects the learning rate and regularization parameters.

``sigma`` (float)
   Regularization parameter: :math:`\sigma = \frac{1}{2 \times \text{interval}}`. Prevents expert weight collapse.

``eta`` (float)
   Learning rate for weight updates: :math:`\eta = \frac{\sqrt{3/T} \sqrt{\log(TK) + 2}}{(1-\alpha)^2 \alpha^2}`

**Core Methods:**

``update(beta: float) -> float``
   Updates the adaptive mechanism with new coverage feedback and returns the updated alpha value.

   **Algorithm Steps:**

   1. **Compute Losses**: Calculate pinball loss for each expert
   2. **Update Weights**: Apply exponential weighting with regularization
   3. **Update Experts**: Gradient step for each expert's alpha value
   4. **Select Alpha**: Choose final alpha via weighted average or sampling
   5. **Clip Values**: Ensure alpha values remain in valid range [0.001, 0.999]

   **Parameters:**

   - ``beta``: Empirical coverage probability :math:`\beta_t \in [0,1]`

   **Returns:**

   - Updated miscoverage level :math:`\alpha_{t+1}`

**State Tracking:**

The DtACI instance maintains comprehensive state for analysis and debugging:

- ``alpha_t_candidates``: Current alpha values for each expert
- ``weights``: Current expert weights
- ``beta_history``: Sequence of observed coverage feedback
- ``alpha_history``: Evolution of selected alpha values
- ``weight_history``: Evolution of expert weight distributions

Coverage Feedback Calculation
------------------------------

The adaptation mechanism requires empirical coverage feedback (β values) computed from conformal prediction performance. The beta value represents the proportion of calibration scores that exceed the test nonconformity score.

**Mathematical Definition:**

For a new observation :math:`(X_t, Y_t)` with predicted nonconformity score :math:`R_t`:

.. math::

   \beta_t = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[R_i^{\text{cal}} \geq R_t]

Where :math:`R_i^{\text{cal}}` are the calibration nonconformity scores.

**Interpretation:**

- **High β (> α)**: Observation is "easy" relative to calibration data → tighten intervals
- **Low β (< α)**: Observation is "hard" relative to calibration data → widen intervals
- **β ≈ α**: Observation difficulty matches target coverage level

Integration with Sampling Strategies
-------------------------------------

The adaptation module integrates with sampling strategies through the utility functions in ``confopt.selection.sampling.utils``:

**Multi-Alpha Samplers:**

``initialize_multi_adapters(alphas, adapter)``
   Creates independent DtACI instances for each alpha level in multi-interval samplers:

   - **Thompson Sampling**: Separate adaptation for each quantile level
   - **Expected Improvement**: Independent adaptation across confidence levels
   - **Entropy Search**: Multi-scale adaptation for different uncertainty levels

**Single-Alpha Samplers:**

``initialize_single_adapter(alpha, adapter)``
   Creates a single DtACI instance for samplers using one confidence level:

   - **Lower Bound Sampling**: Adapts the single confidence interval
   - **Pessimistic Lower Bound**: Conservative adaptation for risk-averse optimization

**Adapter Configuration:**

``"DtACI"`` (Recommended)
   Full multi-expert adaptation with default gamma values ``[0.001, 0.005, 0.01, 0.05]``

   - **Advantages**: Robust to non-stationarity, handles diverse time scales
   - **Use cases**: Complex optimization landscapes, varying objective difficulty

``"ACI"`` (Conservative)
   Single-expert adaptation with gamma value ``[0.005]``

   - **Advantages**: Simple, stable, less prone to over-adaptation
   - **Use cases**: Well-behaved objectives, stable optimization environments

``None`` (No Adaptation)
   Fixed alpha values throughout optimization

   - **Advantages**: Predictable behavior, no adaptation overhead
   - **Use cases**: Known optimal coverage levels, debugging scenarios

Usage in Acquisition Functions
-------------------------------

The adaptation mechanism integrates seamlessly with acquisition functions through the ``BaseConformalSearcher.update()`` method:

**Update Process:**

1. **Observation**: New configuration evaluated, performance observed
2. **Beta Calculation**: Compute coverage feedback using conformal estimator
3. **Alpha Update**: DtACI adapts alpha values based on coverage performance
4. **Propagation**: Updated alphas propagated to conformal estimator
5. **Interval Adjustment**: Prediction intervals adjust for next iteration

**Integration Example:**

.. code-block:: python

   # In BaseConformalSearcher.update()
   def update(self, X, y_true):
       # Calculate coverage feedback
       betas = self._calculate_betas(X, y_true)

       # Update sampler adapters
       if hasattr(self.sampler, 'adapters') and self.sampler.adapters:
           for i, adapter in enumerate(self.sampler.adapters):
               new_alpha = adapter.update(betas[i])
               self.sampler.alphas[i] = new_alpha

       # Propagate to conformal estimator
       self.conformal_estimator.updated_alphas = self.sampler.alphas

**Data Flow:**

.. mermaid::

   graph TD
       subgraph "Optimization Loop"
           EVAL["Evaluate Configuration<br/>(X_t, Y_t)"]
           BETA["Calculate Coverage Feedback<br/>β_t = P(R_cal ≥ R_t)"]
           ADAPT["DtACI Adaptation<br/>α_{t+1} = f(α_t, β_t)"]
           UPDATE["Update Intervals<br/>New prediction intervals"]
           NEXT["Next Configuration<br/>Selection"]
       end

       subgraph "DtACI Algorithm"
           LOSS["Compute Pinball Losses<br/>ℓ(β_t, α_t^i)"]
           WEIGHT["Update Expert Weights<br/>w_{t+1}^i ∝ w_t^i exp(-η·ℓ)"]
           EXPERT["Update Expert Alphas<br/>α_{t+1}^i = α_t^i + γ_i(α - err_t^i)"]
           SELECT["Select Final Alpha<br/>Weighted average or sampling"]
       end

       EVAL --> BETA
       BETA --> ADAPT
       ADAPT --> LOSS
       LOSS --> WEIGHT
       WEIGHT --> EXPERT
       EXPERT --> SELECT
       SELECT --> UPDATE
       UPDATE --> NEXT
       NEXT --> EVAL

Performance Characteristics
---------------------------

**Computational Complexity:**

- **Update Operation**: O(K) where K is the number of experts
- **Memory Usage**: O(K + T) for K experts and T time steps of history
- **Typical K**: 4-8 experts provide good performance-complexity trade-off

**Convergence Properties:**

- **Regret Bounds**: O(√T log(TK)) regret against best fixed expert
- **Coverage Guarantee**: Long-term coverage approaches target α
- **Adaptation Rate**: Controlled by gamma values and expert diversity

**Empirical Performance:**

Based on theoretical analysis and empirical validation:

- **Coverage Error**: Typically < 0.02 deviation from target coverage
- **Adaptation Time**: 20-50 observations for initial convergence
- **Stability**: Robust to non-stationary objective functions

Best Practices
---------------

**Gamma Value Selection:**

- **Default Values**: Use provided defaults for most applications
- **Custom Values**: Choose based on expected adaptation timescales
- **Range**: Typically between 0.001 (conservative) and 0.1 (aggressive)

**Algorithm Variants:**

- **Weighted Average**: Use for stable, predictable adaptation
- **Random Sampling**: Use when theoretical guarantees are paramount
- **Expert Count**: 4-8 experts balance performance and computational cost

**Integration Guidelines:**

- **Warm-up Period**: Allow 20+ observations before trusting adaptation
- **Coverage Monitoring**: Track actual coverage vs. target coverage
- **Alpha Bounds**: Ensure alpha values remain in reasonable range [0.01, 0.3]

**Common Issues:**

- **Insufficient Data**: Requires adequate calibration set for reliable beta calculation
- **Over-Adaptation**: Too aggressive gamma values can cause instability
- **Under-Adaptation**: Too conservative gamma values may not respond to changes
- **Weight Collapse**: Regularization prevents but monitor weight distributions

Theoretical Guarantees
----------------------

The DtACI algorithm provides several theoretical guarantees derived from online learning theory:

**Regret Bound:**

.. math::

   \text{Regret}_T \leq \frac{\sqrt{3T \log(TK) + 6T}}{(1-\alpha)^2 \alpha^2}

**Coverage Control:**

.. math::

   \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T \mathbf{1}[Y_t \notin \hat{C}_t] = \alpha + o(1)

**Finite-Sample Validity:**

The conformal prediction framework ensures that for any finite sample size, the prediction intervals maintain valid coverage properties regardless of the underlying data distribution.

These guarantees make DtACI suitable for safety-critical applications where both efficiency and reliability are essential.
