Acquisition Module
==================

Overview
--------

The acquisition module implements conformal prediction-based acquisition functions for Bayesian optimization. It bridges uncertainty quantification through conformal prediction with various exploration-exploitation strategies, providing theoretically grounded point selection for hyperparameter optimization.

The module serves as the primary interface between conformal prediction estimators and acquisition strategies, enabling adaptive optimization that maintains finite-sample coverage guarantees while optimizing different exploration-exploitation trade-offs.

Key Features
------------

* **Conformal prediction integration**: Maintains finite-sample coverage guarantees throughout optimization
* **Multiple acquisition strategies**: Supports UCB, Thompson sampling, Expected Improvement, Information Gain, and MES
* **Adaptive coverage control**: Dynamic alpha adjustment based on empirical coverage feedback
* **Dual conformal approaches**: Both locally weighted and quantile-based conformal prediction
* **Strategy pattern design**: Clean separation between acquisition logic and conformal prediction
* **Coverage breach tracking**: Real-time monitoring of prediction interval performance

Architecture
------------

The module implements a three-layer architecture with clear separation of concerns:

**Base Layer (BaseConformalSearcher)**
    Abstract interface defining the common acquisition function API with strategy pattern injection. Handles sampler routing, coverage tracking, and adaptive alpha updating.

**Implementation Layer**
    Two concrete implementations providing different conformal prediction approaches:

    * ``LocallyWeightedConformalSearcher``: Variance-adapted intervals using separate point and variance estimators
    * ``QuantileConformalSearcher``: Direct quantile estimation with automatic conformalization mode selection

**Integration Layer**
    Seamless integration with the framework's sampling strategies, estimation infrastructure, and optimization algorithms.

Design Patterns
~~~~~~~~~~~~~~~

The architecture leverages several key design patterns:

* **Strategy Pattern**: Acquisition behavior is delegated to interchangeable sampler implementations
* **Bridge Pattern**: Connects conformal prediction estimators with acquisition strategies
* **Template Method**: Base class defines common workflow while allowing strategy-specific implementations
* **Adapter Pattern**: Unified interface for different sampler types and conformal estimators

Locally Weighted Conformal Acquisition
---------------------------------------

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

The locally weighted approach combines point estimation with variance estimation to create adaptive prediction intervals:

.. math::

   I_\alpha(x) = \left[\hat{\mu}(x) - q_{1-\alpha}(R) \cdot \hat{\sigma}(x), \hat{\mu}(x) + q_{1-\alpha}(R) \cdot \hat{\sigma}(x)\right]

Where:
    - :math:`\hat{\mu}(x)`: Point estimate from fitted point estimator
    - :math:`\hat{\sigma}(x)`: Variance estimate from fitted variance estimator
    - :math:`R_i = \frac{|y_i - \hat{\mu}(x_i)|}{\max(\hat{\sigma}(x_i), \epsilon)}`: Normalized nonconformity scores
    - :math:`q_{1-\alpha}(R)`: :math:`(1-\alpha)`-quantile of calibration nonconformity scores

Acquisition Strategy Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different acquisition strategies utilize the locally weighted intervals in distinct ways:

**Upper Confidence Bound (UCB)**:
    .. math::

       \text{UCB}(x) = \hat{\mu}(x) - \beta_t \cdot \frac{\text{width}(I_\alpha(x))}{2}

**Thompson Sampling**:
    Random sampling from intervals with optional optimistic constraints:

    .. math::

       \text{TS}(x) = \min(\text{sample}(I_\alpha(x)), \hat{\mu}(x)) \quad \text{(if optimistic)}

**Expected Improvement**:
    Integration over locally adapted intervals accounting for heteroscedastic uncertainty.

**Information Gain**:
    Entropy reduction calculations using locally weighted uncertainty estimates.

Advantages
~~~~~~~~~~

* **Local adaptation**: Interval widths automatically adjust to heteroscedastic noise patterns
* **Separate uncertainty modeling**: Independent optimization of point and variance estimators
* **Interpretable components**: Clear separation between mean prediction and uncertainty estimation
* **Robust calibration**: Variance estimates help normalize nonconformity scores

Limitations
~~~~~~~~~~~

* **Two-stage complexity**: Requires fitting and tuning two separate estimators
* **Variance estimation quality**: Performance depends heavily on accurate conditional variance modeling
* **Computational overhead**: Additional variance estimation step increases training time

Quantile-Based Conformal Acquisition
-------------------------------------

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

The quantile approach directly estimates conditional quantiles and applies conformal adjustments:

**Conformalized Mode** (sufficient data):
    .. math::

       I_\alpha(x) = \left[\hat{q}_{\alpha/2}(x) - C_\alpha, \hat{q}_{1-\alpha/2}(x) + C_\alpha\right]

**Non-conformalized Mode** (limited data):
    .. math::

       I_\alpha(x) = \left[\hat{q}_{\alpha/2}(x), \hat{q}_{1-\alpha/2}(x)\right]

Where:
    - :math:`\hat{q}_\tau(x)`: :math:`\tau`-quantile estimate at location :math:`x`
    - :math:`C_\alpha = \text{quantile}(R^\alpha, 1-\alpha)`: Conformal adjustment
    - :math:`R^\alpha_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(x_i))`: Nonconformity scores

Mode Selection Logic
~~~~~~~~~~~~~~~~~~~~

The estimator automatically chooses between modes based on data availability:

.. code-block:: python

    if len(X_train) + len(X_val) > n_pre_conformal_trials:
        mode = "conformalized"  # Full conformal prediction with calibration
    else:
        mode = "non_conformalized"  # Direct quantile predictions

Sampler-Specific Adaptations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Conservative Samplers** (LowerBoundSampler, PessimisticLowerBoundSampler):
    Upper quantile capping at 0.5 to ensure conservative interval construction.

**Thompson Sampling with Optimism**:
    Optional point estimator integration for optimistic bias in posterior sampling.

**Information-Based Samplers**:
    Full quantile range support for comprehensive uncertainty characterization.

Advantages
~~~~~~~~~~

* **Direct quantile modeling**: No intermediate variance estimation required
* **Asymmetric intervals**: Natural handling of skewed conditional distributions
* **Automatic mode selection**: Graceful degradation when calibration data is limited
* **Quantile-specific calibration**: Alpha-dependent nonconformity score computation

Limitations
~~~~~~~~~~~

* **Quantile estimator dependency**: Performance heavily depends on base quantile estimator quality
* **Alpha-specific calibration**: Separate nonconformity scores required for each coverage level
* **Potential quantile crossing**: Risk of invalid intervals if quantile estimator lacks monotonicity constraints

Usage Examples
--------------

Basic Locally Weighted Acquisition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.acquisition import LocallyWeightedConformalSearcher
    from confopt.selection.sampling import LowerBoundSampler
    import numpy as np

    # Initialize sampler with exploration schedule
    sampler = LowerBoundSampler(
        interval_width=0.8,  # 80% coverage intervals
        beta_decay="logarithmic_decay",
        c=1.0
    )

    # Create acquisition function
    searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture="gradient_boosting",
        variance_estimator_architecture="random_forest",
        sampler=sampler
    )

    # Fit on initial data
    searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=10,
        random_state=42
    )

    # Generate acquisition values
    candidates = np.random.rand(100, X_train.shape[1])
    acquisition_values = searcher.predict(candidates)

    # Select next point
    next_idx = np.argmax(acquisition_values)
    next_point = candidates[next_idx]

Quantile-Based Acquisition with Thompson Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.acquisition import QuantileConformalSearcher
    from confopt.selection.sampling import ThompsonSampler

    # Initialize Thompson sampler with optimistic bias
    sampler = ThompsonSampler(
        n_quantiles=6,
        enable_optimistic_sampling=True,
        adapter="DtACI"  # Adaptive coverage control
    )

    # Create quantile-based acquisition function
    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="quantile_random_forest",
        sampler=sampler,
        n_pre_conformal_trials=50  # Threshold for conformal mode
    )

    # Fit with automatic mode selection
    searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=15
    )

    # Optimization loop with adaptive updates
    for iteration in range(max_iterations):
        # Get acquisition values
        acquisition_values = searcher.predict(candidates)

        # Evaluate next point
        next_point = candidates[np.argmax(acquisition_values)]
        next_value = objective_function(next_point)

        # Update with coverage adaptation
        searcher.update(next_point, next_value)

Information Gain Acquisition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.sampling import InformationGainSampler

    # Initialize information gain sampler
    sampler = InformationGainSampler(
        n_quantiles=8,
        n_X_candidates=50,
        sampling_strategy="thompson",
        adapter="DtACI"
    )

    # Use with locally weighted conformal prediction
    searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture="kernel_ridge",
        variance_estimator_architecture="gaussian_process",
        sampler=sampler
    )

    # Information gain requires fixed random state
    searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=1234  # Required for InformationGainSampler
    )

Coverage Monitoring and Adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Monitor coverage performance for interval-based samplers
    coverage_violations = []

    for iteration in range(max_iterations):
        # Generate and evaluate next point
        acquisition_values = searcher.predict(candidates)
        next_point = candidates[np.argmax(acquisition_values)]
        next_value = objective_function(next_point)

        # Check coverage breach (for compatible samplers)
        if isinstance(searcher.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)):
            breach = searcher.calculate_breach(next_point, next_value)
            coverage_violations.append(breach)

            # Compute empirical coverage rate
            empirical_coverage = 1 - np.mean(coverage_violations)
            target_coverage = 1 - searcher.sampler.fetch_alphas()[0]

            print(f"Empirical coverage: {empirical_coverage:.3f}, "
                  f"Target: {target_coverage:.3f}")

        # Update searcher state
        searcher.update(next_point, next_value)

Performance Considerations
-------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

**LocallyWeightedConformalSearcher**:
    - Training: :math:`O(n_{train} + n_{val})` for each estimator plus hyperparameter tuning overhead
    - Prediction: :math:`O(1)` per candidate point plus base estimator prediction costs
    - Memory: :math:`O(n_{val})` for storing nonconformity scores

**QuantileConformalSearcher**:
    - Training: :math:`O(|\text{quantiles}| \times n_{train})` for simultaneous quantile estimation
    - Prediction: :math:`O(|\text{quantiles}|)` per candidate point
    - Memory: :math:`O(|\text{alphas}| \times n_{val})` for alpha-specific nonconformity scores

Scaling Recommendations
~~~~~~~~~~~~~~~~~~~~~~~

* **Data splitting**: Ensure sufficient calibration data (minimum 100-200 points) for stable coverage
* **Hyperparameter tuning budget**: Balance tuning iterations with computational constraints
* **Quantile set sizing**: Limit number of alpha levels to reduce memory usage and computational overhead
* **Warm-starting**: Reuse best configurations from previous fits to reduce training time

Best Practices
~~~~~~~~~~~~~~

* **Coverage monitoring**: Track empirical coverage rates to validate theoretical guarantees
* **Sampler selection**: Choose acquisition strategy based on optimization problem characteristics
* **Data quality**: Ensure representative validation sets for proper conformal calibration
* **Alpha tuning**: Start with moderate coverage levels (80-90%) and adapt based on performance
* **Random state management**: Use consistent random seeds for reproducible Information Gain results

Integration Points
-----------------

Framework Integration
~~~~~~~~~~~~~~~~~~~~

The acquisition module integrates with several framework components:

**Conformal Prediction Infrastructure**:
    Direct dependency on ``confopt.selection.conformalization`` for uncertainty quantification.

**Sampling Strategies**:
    Leverages ``confopt.selection.sampling`` for diverse acquisition strategy implementations.

**Estimation Framework**:
    Uses ``confopt.selection.estimation`` for hyperparameter tuning and estimator initialization.

**Optimization Algorithms**:
    Provides acquisition function interface for ``confopt.tuning`` optimization methods.

Pipeline Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.tuning import BayesianOptimizer
    from confopt.selection.acquisition import LocallyWeightedConformalSearcher

    # Create complete optimization pipeline
    optimizer = BayesianOptimizer(
        acquisition_function=LocallyWeightedConformalSearcher(
            point_estimator_architecture="gradient_boosting",
            variance_estimator_architecture="random_forest",
            sampler=LowerBoundSampler(interval_width=0.85)
        ),
        n_initial_points=20,
        max_iterations=100
    )

    # Run optimization with coverage guarantees
    result = optimizer.optimize(
        objective_function=objective_function,
        parameter_space=parameter_space,
        random_state=42
    )

Common Pitfalls
---------------

**Insufficient Calibration Data**
    **Problem**: Poor coverage with small validation sets
    **Solution**: Ensure minimum 100-200 calibration points for stable coverage estimates

**Sampler-Estimator Mismatch**
    **Problem**: Suboptimal performance with incompatible sampler-estimator combinations
    **Solution**: Match sampler characteristics to estimator capabilities (e.g., conservative samplers with quantile capping)

**Alpha Adaptation Instability**
    **Problem**: Erratic coverage behavior with aggressive alpha adaptation
    **Solution**: Use conservative adaptation parameters or disable adaptation for initial optimization phases

**Information Gain Reproducibility**
    **Problem**: Non-reproducible results with InformationGainSampler
    **Solution**: Always specify random_state parameter when using information-based acquisition

**Variance Estimation Quality**
    **Problem**: Poor locally weighted performance due to inadequate variance modeling
    **Solution**: Validate variance estimator quality independently or switch to quantile-based approach

**Memory Usage with Many Alphas**
    **Problem**: Excessive memory consumption with numerous coverage levels
    **Solution**: Limit number of alpha levels or use single-alpha samplers for large-scale problems

See Also
--------

**Related Framework Components**:
    - :doc:`conformalization` - Core conformal prediction implementations
    - :doc:`sampling` - Acquisition strategy implementations
    - :doc:`estimation` - Hyperparameter tuning infrastructure
    - ``confopt.tuning`` - Optimization algorithm implementations

**External References**:
    - Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world.
    - Srinivas, N., et al. (2009). Gaussian process optimization in the bandit setting.
    - Russo, D., & Van Roy, B. (2014). Learning to optimize via information-directed sampling.
