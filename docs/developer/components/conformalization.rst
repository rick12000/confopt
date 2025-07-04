Conformalization Module
======================

Overview
--------

The conformalization module implements two distinct conformal prediction methodologies for generating prediction intervals with finite-sample coverage guarantees. Conformal prediction provides a distribution-free framework for uncertainty quantification that works with any base predictor and offers theoretical coverage guarantees under mild exchangeability assumptions.

This module serves as a core component of the confopt framework's uncertainty quantification capabilities, providing both locally adaptive and quantile-based approaches to prediction interval construction.

Key Features
------------

* **Finite-sample coverage guarantees**: Valid under exchangeability assumptions without distributional requirements
* **Two complementary approaches**: Locally weighted and quantile-based conformal prediction
* **Dynamic alpha updating**: Efficient coverage level adjustment without refitting
* **Integrated hyperparameter tuning**: Automated optimization of base estimators
* **Adaptive interval construction**: Intervals that adapt to local prediction uncertainty
* **Split conformal methodology**: Proper separation of training, calibration, and testing phases

Architecture
------------

The module implements two main estimator classes following a common interface pattern:

**LocallyWeightedConformalEstimator**
    Uses separate point and variance estimators to create locally adaptive intervals. The two-stage approach first estimates the conditional mean, then models the conditional variance using absolute residuals, and finally scales nonconformity scores by local variance estimates.

**QuantileConformalEstimator**
    Directly estimates prediction quantiles using quantile regression and applies conformal adjustments. This approach can operate in both conformalized mode (with proper calibration) and non-conformalized mode (direct quantile prediction) depending on data availability.

Both estimators support:

* **Alpha abstraction layer**: Efficient updating of coverage levels through ``update_alphas()``
* **Hyperparameter integration**: Seamless integration with the framework's tuning infrastructure
* **Performance tracking**: Built-in metrics for estimator quality assessment
* **Flexible initialization**: Support for warm-starting from previous best configurations

Locally Weighted Conformal Prediction
--------------------------------------

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

The locally weighted approach implements a heteroscedastic extension of conformal prediction that adapts interval widths to local prediction uncertainty. The method follows this process:

1. **Data Splitting**: Split training data into point estimation set :math:`(X_{pe}, y_{pe})` and variance estimation set :math:`(X_{ve}, y_{ve})`

2. **Point Estimation**: Fit point estimator :math:`\hat{\mu}(x) = \mathbb{E}[Y|X=x]`

3. **Residual Computation**: Calculate absolute residuals :math:`r_i = |y_i - \hat{\mu}(X_i)|` on variance estimation set

4. **Variance Estimation**: Fit variance estimator :math:`\hat{\sigma}^2(x) = \mathbb{E}[r^2|X=x]` using residuals

5. **Nonconformity Scores**: Compute validation scores :math:`R_i = \frac{|y_{val,i} - \hat{\mu}(X_{val,i})|}{\max(\hat{\sigma}(X_{val,i}), \epsilon)}`

6. **Interval Construction**: For coverage level :math:`1-\alpha`, prediction intervals are:

   .. math::

      C_\alpha(x) = \left[\hat{\mu}(x) - q_{1-\alpha}(R) \cdot \hat{\sigma}(x), \hat{\mu}(x) + q_{1-\alpha}(R) \cdot \hat{\sigma}(x)\right]

where :math:`q_{1-\alpha}(R)` is the :math:`(1-\alpha)`-quantile of the nonconformity scores.

Advantages
~~~~~~~~~~

* **Local adaptation**: Interval widths adapt to heteroscedastic noise patterns
* **Computational efficiency**: Single set of nonconformity scores for all alpha levels
* **Interpretable components**: Separate modeling of conditional mean and variance
* **Robust to outliers**: Variance estimates help downweight extreme residuals

Limitations
~~~~~~~~~~~

* **Two-stage complexity**: Requires optimization of two separate estimators
* **Variance estimation quality**: Performance depends on accurate conditional variance modeling
* **Data splitting overhead**: Requires sufficient data for both point and variance estimation

Quantile-Based Conformal Prediction
------------------------------------

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

The quantile approach directly estimates conditional quantiles and applies conformal adjustments when sufficient data is available for proper calibration:

1. **Quantile Set Construction**: For each :math:`\alpha`, compute required quantiles :math:`\tau_L = \alpha/2` and :math:`\tau_U = 1 - \alpha/2`

2. **Quantile Estimation**: Fit quantile estimator to predict :math:`\hat{q}_\tau(x)` for all required quantiles simultaneously

3. **Nonconformity Computation** (if conformalized): For each alpha level, calculate:

   .. math::

      R_i^\alpha = \max\left(\hat{q}_{\alpha/2}(X_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(X_i)\right)

4. **Conformal Adjustment**: Get adjustment :math:`C_\alpha = q_{1-\alpha}(R^\alpha)`

5. **Final Intervals**:

   - **Conformalized**: :math:`\left[\hat{q}_{\alpha/2}(x) - C_\alpha, \hat{q}_{1-\alpha/2}(x) + C_\alpha\right]`
   - **Non-conformalized**: :math:`\left[\hat{q}_{\alpha/2}(x), \hat{q}_{1-\alpha/2}(x)\right]`

Decision Logic
~~~~~~~~~~~~~~

The estimator automatically chooses between conformalized and non-conformalized modes:

* **Conformalized mode**: When ``len(X_train) + len(X_val) > n_pre_conformal_trials``
* **Non-conformalized mode**: When data is insufficient for proper split conformal prediction

Advantages
~~~~~~~~~~

* **Direct quantile modeling**: No intermediate variance estimation step
* **Flexible asymmetric intervals**: Natural handling of skewed conditional distributions
* **Quantile-specific calibration**: Alpha-dependent nonconformity scores
* **Automatic mode selection**: Graceful degradation when data is limited

Limitations
~~~~~~~~~~~

* **Quantile estimator dependency**: Performance heavily depends on base quantile estimator quality
* **Alpha-specific scores**: Separate calibration required for each coverage level
* **Potential refitting needs**: Changing alphas may require new quantile estimation

Usage Examples
--------------

Basic Locally Weighted Conformal Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.conformalization import LocallyWeightedConformalEstimator
    import numpy as np

    # Initialize estimator
    estimator = LocallyWeightedConformalEstimator(
        point_estimator_architecture="random_forest",
        variance_estimator_architecture="gradient_boosting",
        alphas=[0.1, 0.05]  # 90% and 95% coverage
    )

    # Fit with hyperparameter tuning
    estimator.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=20,
        random_state=42
    )

    # Generate prediction intervals
    intervals = estimator.predict_intervals(X_test)

    # Access 90% coverage intervals
    bounds_90 = intervals[0]  # corresponds to alpha=0.1
    lower_90 = bounds_90.lower_bounds
    upper_90 = bounds_90.upper_bounds

Basic Quantile Conformal Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.conformalization import QuantileConformalEstimator

    # Initialize with quantile-capable estimator
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture="quantile_random_forest",
        alphas=[0.1, 0.05],
        n_pre_conformal_trials=50  # Minimum for conformal mode
    )

    # Fit with upper quantile capping
    estimator.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        upper_quantile_cap=0.95,  # Cap extreme quantiles
        tuning_iterations=15
    )

    # Generate intervals (automatically conformalized if enough data)
    intervals = estimator.predict_intervals(X_test)

Dynamic Alpha Updating
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initial fitting with one set of alphas
    estimator.fit(X_train, y_train, X_val, y_val)

    # Later, update coverage requirements without refitting
    new_coverage_levels = [0.2, 0.1, 0.01]  # 80%, 90%, 99% coverage
    estimator.update_alphas(new_coverage_levels)

    # Predictions now use updated coverage levels
    updated_intervals = estimator.predict_intervals(X_test)

Conformity Assessment
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate empirical p-values for new observations
    x_new = np.array([1.5, 2.3, -0.7])  # Single feature vector
    y_observed = 4.2

    # Get beta values (empirical p-values)
    betas = estimator.calculate_betas(x_new, y_observed)

    # Interpret results
    for i, (alpha, beta) in enumerate(zip(estimator.alphas, betas)):
        coverage = 1 - alpha
        print(f"{coverage*100}% level: p-value = {beta:.3f}")
        if beta < alpha:
            print(f"  Observation is significantly non-conforming at {coverage*100}% level")

Performance Considerations
-------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

**LocallyWeightedConformalEstimator**:
    - Training: :math:`O(n_{train} + n_{val})` for each component estimator
    - Memory: :math:`O(n_{val})` for nonconformity scores storage
    - Prediction: :math:`O(1)` per prediction point (plus base estimator costs)

**QuantileConformalEstimator**:
    - Training: :math:`O(|\text{quantiles}| \times n_{train})` for simultaneous quantile estimation
    - Memory: :math:`O(|\text{alphas}| \times n_{val})` for alpha-specific nonconformity scores
    - Prediction: :math:`O(|\text{quantiles}|)` per prediction point

Scaling Considerations
~~~~~~~~~~~~~~~~~~~~~

* **Data splitting requirements**: Both methods require sufficient calibration data for reliable coverage
* **Hyperparameter tuning overhead**: Can dominate computation time with extensive search spaces
* **Memory usage**: Scales linearly with calibration set size and number of alpha levels
* **Warm-starting benefits**: Reusing best configurations significantly reduces retraining costs

Best Practices
~~~~~~~~~~~~~~

* **Calibration set sizing**: Use at least 100-200 observations for stable coverage estimates
* **Alpha consistency**: For quantile estimators, determine complete alpha set before fitting
* **Hyperparameter budget allocation**: Balance tuning iterations with available compute budget
* **Validation strategy**: Monitor coverage on held-out test sets for method selection

Integration Points
-----------------

Framework Integration
~~~~~~~~~~~~~~~~~~~~

The conformalization module integrates deeply with several framework components:

**Estimation Infrastructure**:
    Uses ``confopt.selection.estimation`` for hyperparameter tuning via ``PointTuner`` and ``QuantileTuner`` classes.

**Estimator Registry**:
    Leverages ``ESTIMATOR_REGISTRY`` for flexible base estimator selection and configuration.

**Data Processing**:
    Utilizes ``confopt.utils.preprocessing.train_val_split`` for proper data partitioning.

**Result Wrapping**:
    Returns predictions using ``confopt.wrapping.ConformalBounds`` for consistent interface.

Pipeline Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.conformalization import LocallyWeightedConformalEstimator
    from confopt.tuning import BayesianOptimizer

    # Integration with broader optimization pipeline
    def objective_function(hyperparams):
        estimator = LocallyWeightedConformalEstimator(**hyperparams)
        estimator.fit(X_train, y_train, X_val, y_val)

        # Return coverage quality metric
        intervals = estimator.predict_intervals(X_test)
        return compute_coverage_quality(intervals, y_test)

    # Optimize conformalization approach selection
    optimizer = BayesianOptimizer(objective_function)
    best_config = optimizer.optimize()

Extension Points
~~~~~~~~~~~~~~~

The module provides several extension points for custom implementations:

* **Custom base estimators**: Register new architectures in ``ESTIMATOR_REGISTRY``
* **Alternative nonconformity measures**: Extend calculation logic in ``calculate_betas``
* **Specialized data splitting**: Override ``train_val_split`` behavior for domain-specific requirements
* **Custom tuning strategies**: Implement domain-specific tuners extending ``RandomTuner``

Common Pitfalls
---------------

Data Leakage
~~~~~~~~~~~~

**Problem**: Using the same data for training base estimators and conformal calibration violates the split conformal assumption.

**Solution**: Ensure proper data separation:

.. code-block:: python

    # WRONG: Same data for training and calibration
    estimator.fit(X_all, y_all, X_all, y_all)  # Data leakage!

    # CORRECT: Separate training and calibration sets
    estimator.fit(X_train, y_train, X_val, y_val)

Insufficient Calibration Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Too few calibration samples lead to unreliable coverage estimates.

**Solution**: Ensure adequate calibration set size:

.. code-block:: python

    if len(X_val) < 100:
        logging.warning(f"Calibration set size {len(X_val)} may be insufficient")
        # Consider collecting more data or using direct quantile prediction

Alpha Update Inconsistency
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: For quantile estimators, updating alphas to require new quantiles without refitting.

**Solution**: Plan alpha sets comprehensively:

.. code-block:: python

    # Plan all possible alphas upfront
    all_possible_alphas = [0.1, 0.05, 0.01, 0.005]
    estimator = QuantileConformalEstimator(alphas=all_possible_alphas)
    estimator.fit(X_train, y_train, X_val, y_val)

    # Later updates are safe within the original set
    estimator.update_alphas([0.05, 0.01])  # Safe subset

Variance Estimator Overfitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Locally weighted variance estimators may overfit to residual patterns.

**Solution**: Use regularized estimators and cross-validation:

.. code-block:: python

    estimator = LocallyWeightedConformalEstimator(
        point_estimator_architecture="random_forest",
        variance_estimator_architecture="ridge_regression",  # Regularized choice
        alphas=[0.1]
    )

Quantile Crossing
~~~~~~~~~~~~~~~~

**Problem**: Estimated quantiles may cross, violating monotonicity constraints.

**Solution**: Use quantile estimators with non-crossing guarantees or post-process:

.. code-block:: python

    # Choose estimators with built-in non-crossing constraints
    estimator = QuantileConformalEstimator(
        quantile_estimator_architecture="quantile_regression_forest",  # Non-crossing
        alphas=[0.1, 0.05]
    )

See Also
--------

**Related Framework Components**:
    - :doc:`quantile_estimation` - Base quantile regression implementations
    - :doc:`ensembling` - Ensemble methods for improved base estimators
    - ``confopt.selection.estimation`` - Hyperparameter tuning infrastructure
    - ``confopt.utils.preprocessing`` - Data preprocessing utilities

**External References**:
    - Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world.
    - Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression.
    - Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002). Inductive confidence machines for regression.

**Implementation Papers**:
    The module implements methodologies from several key papers in conformal prediction, with particular emphasis on locally adaptive approaches and quantile-based methods for heteroscedastic regression problems.
