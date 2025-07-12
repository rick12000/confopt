Components Overview
===================

This page provides an overview of the key components in the ConfOpt framework. Each component plays a specific role in the conformal prediction-based hyperparameter optimization process.

Core Components
---------------

ConformalTuner
~~~~~~~~~~~~~~

The main orchestrator that coordinates the entire optimization process. It manages the two-phase optimization approach (random initialization followed by conformal prediction-guided search) and handles both maximization and minimization objectives.

**Key Responsibilities:**
- Coordinate between configuration management and conformal prediction
- Manage optimization phases and termination criteria
- Handle metric sign transformation for consistent optimization
- Provide progress tracking and result aggregation

Conformal Prediction Searchers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These components implement the conformal prediction models that guide the search process:

**LocallyWeightedConformalSearcher**
  Uses locally weighted conformal prediction to provide uncertainty estimates that adapt to local regions of the search space.

**QuantileConformalSearcher**
  Implements quantile-based conformal prediction for robust uncertainty quantification across different objective function characteristics.

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

**StaticConfigurationManager**
  Pre-generates a fixed pool of candidate configurations at initialization. Suitable for moderate-dimensional spaces with limited computational resources.

**DynamicConfigurationManager**
  Adaptively resamples configuration candidates during optimization. Ideal for high-dimensional spaces and long-running optimizations.

Estimation Components
---------------------

Quantile Estimators
~~~~~~~~~~~~~~~~~~~

The framework includes several quantile estimation methods for conformal prediction:

- **QuantileLasso**: L1-regularized quantile regression
- **QuantileGBM**: Gradient boosting for quantile estimation
- **QuantileForest**: Random forest-based quantile prediction
- **QuantileKNN**: K-nearest neighbors quantile estimation
- **GaussianProcessQuantileEstimator**: Gaussian process quantile regression

Ensemble Methods
~~~~~~~~~~~~~~~~

**QuantileEnsemble**
  Combines multiple quantile estimators to improve prediction robustness and handle diverse objective function characteristics.

Sampling Strategies
-------------------

The framework provides various sampling strategies for different optimization scenarios:

**Thompson Sampling**
  Implements Thompson sampling for exploration-exploitation balance in the conformal prediction context.

**Expected Improvement Sampling**
  Uses expected improvement criteria adapted for conformal prediction uncertainty estimates.

**Entropy-Based Sampling**
  Maximizes information gain by selecting configurations that reduce prediction uncertainty.

**Bound Sampling**
  Focuses on configurations with promising lower confidence bounds.

Utility Components
------------------

**Preprocessing**
  Handles data scaling, outlier detection, and feature transformation for conformal prediction models.

**Tracking**
  Manages experiment history, progress monitoring, and result aggregation across optimization runs.

**Optimization**
  Provides multi-armed bandit optimization for adaptive parameter tuning within the conformal prediction framework.

Integration Flow
----------------

The components work together in a coordinated flow:

1. **Configuration Management** provides candidate configurations
2. **ConformalTuner** evaluates configurations and maintains history
3. **Conformal Searchers** train on historical data to predict promising regions
4. **Sampling Strategies** select next configurations based on uncertainty estimates
5. **Utility Components** support preprocessing, tracking, and adaptive parameter tuning

This architecture ensures that each component has a clear responsibility while maintaining flexible integration points for different optimization scenarios.

For detailed implementation information, see the :doc:`architecture` documentation.
