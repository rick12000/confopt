Adaptation Module
=================

Overview
--------

The adaptation module implements adaptive conformal inference algorithms for maintaining coverage guarantees under distribution shift. It provides the Dt-ACI (Distribution-free Adaptive Conformal Inference) algorithm from Gibbs & Candès (2021), which dynamically adjusts miscoverage levels based on empirical coverage feedback to ensure robust prediction intervals despite changing data distributions.

The module serves as a core component for online conformal prediction, enabling automatic adaptation to distribution shifts without requiring prior knowledge of the shift magnitude or timing. This makes it particularly valuable for real-world applications where data distributions evolve over time.

Key Features
------------

* **Distribution-free adaptation**: No assumptions about the nature or magnitude of distribution shifts
* **Theoretical coverage guarantees**: Provable regret bounds ensuring asymptotic coverage convergence
* **Multi-expert framework**: Maintains multiple candidate alpha values with different learning rates
* **Exponential weighting**: Uses principled weight updates based on pinball loss performance
* **Numerical stability**: Robust implementation with appropriate bounds and regularization
* **Real-time operation**: Efficient online updates suitable for streaming applications

Architecture
------------

The module implements a single-class architecture centered around the ``DtACI`` class:

**Core Components**:

* **Pinball Loss Function**: Asymmetric loss function measuring miscoverage costs
* **Expert System**: Multiple alpha candidates with different learning rates (gamma values)
* **Exponential Weighting**: Principled weight updates based on expert performance
* **Gradient Updates**: Alpha value adjustments using stochastic gradient ascent

**Design Pattern**:
The implementation follows an online learning paradigm where:

1. Multiple experts (alpha candidates) compete based on performance
2. Expert weights are updated using exponential weighting with regularization
3. Alpha values are adjusted using gradient steps with different learning rates
4. Current alpha is sampled from the expert distribution

Dt-ACI Algorithm
----------------

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

The Dt-ACI algorithm addresses the fundamental challenge of maintaining coverage under distribution shift by adaptively adjusting the miscoverage level α based on empirical feedback.

**Core Algorithm Steps**:

1. **Initialization**: Start with k experts, each with alpha value α and learning rate γⁱ
2. **Feedback Reception**: Receive empirical coverage βₜ from conformal predictor
3. **Loss Computation**: Calculate pinball losses for each expert
4. **Weight Update**: Update expert weights using exponential weighting
5. **Alpha Update**: Adjust alpha values using gradient ascent
6. **Selection**: Sample current alpha from expert distribution

**Pinball Loss Function**:

The asymmetric pinball loss measures the cost of miscoverage:

.. math::

   L(β, θ, α) = α × \max(θ - β, 0) + (1-α) × \max(β - θ, 0)

Where:
- β: Empirical coverage (fraction of calibration scores ≥ test score)
- θ: Target coverage level (1 - αᵢ for expert i)
- α: Original miscoverage level for asymmetric penalty weighting

**Expert Weight Updates**:

Weights are updated using exponential weighting with regularization:

.. math::

   \tilde{w}_t^i &= w_{t-1}^i × \exp(-η × L_t^i)

   w_t^i &= (1-σ) × \frac{\tilde{w}_t^i}{||\tilde{w}_t||_1} + \frac{σ}{k}

Where:
- η: Learning rate for exponential weights
- σ: Regularization parameter (mixing with uniform distribution)
- k: Number of experts

**Alpha Value Updates**:

Each expert's alpha is updated using gradient ascent:

.. math::

   α_t^i = \text{clip}(α_{t-1}^i + γ^i × (α - \mathbf{1}_{β_t < α_{t-1}^i}), ε, 1-ε)

Where:
- γⁱ: Learning rate for expert i
- 𝟙_{βₜ < αₜ₋₁ⁱ}: Indicator function for under-coverage
- ε: Numerical stability bounds (0.01, 0.99)

Theoretical Guarantees
~~~~~~~~~~~~~~~~~~~~~~

**Regret Bound**:
Under mild assumptions, the algorithm achieves:

.. math::

   R_T ≤ O(\sqrt{T \log(T·k)})

This ensures that the cumulative regret grows sublinearly, guaranteeing asymptotic convergence to optimal coverage.

**Coverage Properties**:
- **Finite-sample validity**: Maintains coverage guarantees at each time step
- **Adaptive convergence**: Converges to target coverage under stationary conditions
- **Robustness**: Handles arbitrary distribution shifts without prior knowledge

Parameter Selection
~~~~~~~~~~~~~~~~~~~

**Learning Rates (gamma_values)**:
Default exponentially spaced values: [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

- Smaller values: More conservative, stable under noise
- Larger values: More aggressive, faster adaptation to shifts
- Multiple values: Hedge against uncertainty in optimal learning rate

**Algorithm Parameters**:
- **interval (T)**: Window size for regret analysis (default: 500)
- **sigma (σ)**: Regularization parameter = 1/(2T)
- **eta (η)**: Exponential weights learning rate (theoretical formula)

Usage Examples
--------------

Basic Dt-ACI Setup
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from confopt.selection.adaptation import DtACI

   # Initialize with default parameters
   dtaci = DtACI(alpha=0.1)

   # Custom learning rates for specific scenarios
   dtaci_custom = DtACI(
       alpha=0.2,
       gamma_values=[0.01, 0.05, 0.1]  # More aggressive adaptation
   )

Online Adaptation Loop
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from sklearn.linear_model import LinearRegression

   dtaci = DtACI(alpha=0.1)

   for t in range(len(data_stream)):
       # Get training and calibration data
       X_train, y_train = get_training_data(t)
       X_cal, y_cal = get_calibration_data(t)
       X_test, y_test = get_test_point(t)

       # Train model and get predictions
       model = LinearRegression()
       model.fit(X_train, y_train)
       y_cal_pred = model.predict(X_cal)
       y_test_pred = model.predict(X_test)

       # Calculate empirical coverage (beta)
       cal_residuals = np.abs(y_cal - y_cal_pred)
       test_residual = abs(y_test - y_test_pred)
       beta = np.mean(cal_residuals >= test_residual)

       # Update Dt-ACI and get adapted alpha
       current_alpha = dtaci.update(beta=beta)

       # Use adapted alpha for prediction interval
       quantile = np.quantile(cal_residuals, 1 - current_alpha)
       interval = [y_test_pred - quantile, y_test_pred + quantile]

Integration with Conformal Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from confopt.selection.acquisition import LocallyWeightedConformalSearcher
   from confopt.selection.sampling.bound_samplers import LowerBoundSampler

   # Create sampler with Dt-ACI adaptation
   sampler = LowerBoundSampler(
       alpha=0.1,
       adapter="DtACI"  # Enables automatic adaptation
   )

   # Create conformal searcher
   searcher = LocallyWeightedConformalSearcher(
       point_estimator_architecture="rf",
       variance_estimator_architecture="rf",
       sampler=sampler
   )

   # During optimization, adaptation happens automatically
   for config, performance in optimization_loop():
       # Searcher internally calculates beta and updates adaptation
       searcher.update(config, performance)
       next_config = searcher.search(search_space)

Expert Monitoring and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dtaci = DtACI(alpha=0.1, gamma_values=[0.001, 0.01, 0.1])

   # Track adaptation over time
   alpha_history = []
   weight_history = []

   for beta in beta_sequence:
       current_alpha = dtaci.update(beta=beta)
       alpha_history.append(current_alpha)
       weight_history.append(dtaci.get_expert_weights())

   # Analyze expert performance
   final_weights = dtaci.get_expert_weights()
   final_alphas = dtaci.get_expert_alphas()

   print(f"Expert weights: {final_weights}")
   print(f"Expert alphas: {final_alphas}")
   print(f"Best expert (highest weight): {np.argmax(final_weights)}")

Reset for New Sequences
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dtaci = DtACI(alpha=0.1)

   # Process first data sequence
   for beta in sequence_1:
       dtaci.update(beta=beta)

   # Reset for new sequence (e.g., different dataset)
   dtaci.reset()

   # Process second sequence with fresh state
   for beta in sequence_2:
       dtaci.update(beta=beta)

Performance Considerations
-------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

**Time Complexity**:
- **Initialization**: O(k) where k is number of experts
- **Update**: O(k) per time step
- **Memory**: O(k) for storing expert states

**Space Complexity**:
- **Expert weights**: O(k) floating point values
- **Expert alphas**: O(k) floating point values
- **Algorithm parameters**: O(1) constants

**Scaling Characteristics**:
- Linear scaling with number of experts
- Constant time per prediction update
- No dependence on historical data size
- Suitable for high-frequency online applications

Numerical Stability
~~~~~~~~~~~~~~~~~~~

**Robust Implementation Features**:
- Alpha values clipped to [0.01, 0.99] for numerical stability
- Weight normalization with fallback to uniform distribution
- Regularization prevents weight concentration
- Overflow protection in exponential weight updates

**Parameter Sensitivity**:
- **eta**: Auto-computed using theoretical formula
- **sigma**: Inversely proportional to interval length
- **gamma_values**: Exponential spacing provides good coverage

Best Practices
~~~~~~~~~~~~~~

**Learning Rate Selection**:
- Use default exponentially spaced gamma values for most applications
- Include both conservative (small) and aggressive (large) learning rates
- Consider problem-specific adaptation requirements

**Integration Guidelines**:
- Calculate beta accurately using proper conformal prediction setup
- Ensure sufficient calibration data for stable empirical coverage
- Monitor expert weights to understand adaptation dynamics

**Performance Optimization**:
- Limit number of experts (k) to reasonable range (5-10)
- Use consistent random seeds for reproducible expert selection
- Consider resetting after major distribution shifts

Common Pitfalls
---------------

**Incorrect Beta Calculation**

.. code-block:: python

   # INCORRECT: Using residuals directly
   beta = np.mean(y_cal - y_cal_pred >= y_test - y_test_pred)

   # CORRECT: Using absolute residuals for coverage
   cal_residuals = np.abs(y_cal - y_cal_pred)
   test_residual = abs(y_test - y_test_pred)
   beta = np.mean(cal_residuals >= test_residual)

**Insufficient Calibration Data**

.. code-block:: python

   # PROBLEMATIC: Too few calibration points
   n_cal = 5  # May lead to unstable beta estimates

   # BETTER: Ensure sufficient calibration data
   n_cal = max(int(len(data) * 0.3), 20)  # At least 20 points

**Ignoring Expert Dynamics**

.. code-block:: python

   # Monitor expert evolution for debugging
   if np.max(dtaci.get_expert_weights()) > 0.9:
       logger.warning("Single expert dominance detected")

   if np.var(dtaci.get_expert_alphas()) < 1e-6:
       logger.warning("Expert alphas have converged")

**Parameter Boundaries**

.. code-block:: python

   # INVALID: Alpha outside valid range
   dtaci = DtACI(alpha=0.0)  # Raises ValueError
   dtaci = DtACI(alpha=1.0)  # Raises ValueError

   # INVALID: Non-positive learning rates
   dtaci = DtACI(gamma_values=[0.1, 0.0, -0.1])  # Raises ValueError

**Beta Range Violations**

.. code-block:: python

   # Validate beta before updating
   if not 0 <= beta <= 1:
       logger.error(f"Invalid beta value: {beta}")
       beta = np.clip(beta, 0, 1)

   dtaci.update(beta=beta)

Integration Points
-----------------

Framework Integration
~~~~~~~~~~~~~~~~~~~~

The adaptation module integrates with several framework components:

**Sampling Infrastructure**:
- ``LowerBoundSampler``: Provides adapter parameter for automatic Dt-ACI integration
- ``ThompsonSampler``: Supports adaptive alpha updates through adapter interface
- ``ExpectedImprovementSampler``: Compatible with adaptation for improved exploration

**Acquisition Functions**:
- ``LocallyWeightedConformalSearcher``: Calculates beta values for adaptation feedback
- ``QuantileConformalSearcher``: Provides coverage feedback through beta calculation
- Base acquisition interface supports ``update_interval_width()`` for adaptation

**Conformalization Framework**:
- ``LocallyWeightedConformalEstimator``: Supplies empirical p-values as beta feedback
- ``QuantileConformalEstimator``: Provides per-alpha beta calculations
- Coverage assessment integration through ``calculate_betas()`` methods

Pipeline Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from confopt.tuning import HyperparameterOptimizer
   from confopt.selection.acquisition import LocallyWeightedConformalSearcher
   from confopt.selection.sampling.bound_samplers import LowerBoundSampler

   # Create adaptive acquisition function
   sampler = LowerBoundSampler(alpha=0.1, adapter="DtACI")
   searcher = LocallyWeightedConformalSearcher(
       point_estimator_architecture="gbm",
       variance_estimator_architecture="gbm",
       sampler=sampler
   )

   # Optimizer automatically handles adaptation
   optimizer = HyperparameterOptimizer(searcher=searcher)
   best_config = optimizer.optimize(objective_function, search_space)

Extension Points
~~~~~~~~~~~~~~~

**Custom Learning Schedules**:

.. code-block:: python

   class AdaptiveGammaDtACI(DtACI):
       def update(self, beta: float) -> float:
           # Custom logic to adjust gamma values over time
           if self.adaptation_phase == "exploration":
               self.gamma_values *= 1.1  # More aggressive
           elif self.adaptation_phase == "exploitation":
               self.gamma_values *= 0.9  # More conservative

           return super().update(beta)

**Alternative Expert Selection**:

.. code-block:: python

   class DeterministicDtACI(DtACI):
       def update(self, beta: float) -> float:
           # ... weight update logic ...

           # Use best expert instead of sampling
           best_idx = np.argmax(self.weights)
           self.alpha_t = self.alpha_t_values[best_idx]
           return self.alpha_t

See Also
--------

**Related Framework Components**:
- :doc:`acquisition` - Conformal acquisition functions that integrate adaptation
- :doc:`conformalization` - Conformal prediction estimators providing beta feedback
- :doc:`sampling` - Sampling strategies with adapter support

**External References**:
- Gibbs, I. & Candès, E. (2023). "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts"
