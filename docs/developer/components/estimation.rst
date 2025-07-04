Estimation Module
=================

Overview
--------

The ``confopt.selection.estimation`` module provides automated hyperparameter tuning infrastructure for both quantile regression and point estimation models. It implements random search optimization with cross-validation support, integrating seamlessly with the estimator registry system for unified model configuration and evaluation.

Key Features
------------

* **Unified Tuning Framework**: Single interface for optimizing both point and quantile estimation models
* **Cross-Validation Support**: Flexible split strategies including K-fold and ordinal time-series splits
* **Warm-Start Optimization**: Priority evaluation of pre-specified parameter configurations
* **Robust Error Handling**: Graceful failure recovery during hyperparameter evaluation
* **Registry Integration**: Automatic parameter space discovery from estimator configurations

Architecture
------------

Class Hierarchy
~~~~~~~~~~~~~~~

::

    RandomTuner (ABC)
    ├── PointTuner
    └── QuantileTuner

The module follows a template method pattern where ``RandomTuner`` provides the optimization framework and subclasses implement model-specific fitting and evaluation logic.

**RandomTuner**
    Abstract base providing cross-validation infrastructure, parameter sampling, and optimization workflow. Subclasses implement ``_fit_model()`` and ``_evaluate_model()`` methods.

**PointTuner**
    Specialization for standard regression models using mean squared error evaluation.

**QuantileTuner**
    Specialization for quantile regression models using average pinball loss evaluation across multiple quantile levels.

Optimization Methodology
------------------------

Random Search with Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimization process follows these steps:

1. **Parameter Space Sampling**: Random configurations sampled from estimator-specific parameter grids
2. **Warm-Start Evaluation**: Pre-specified configurations evaluated first if provided
3. **Cross-Validation**: Each configuration evaluated across multiple folds using specified split strategy
4. **Score Aggregation**: Performance averaged across folds for robust estimation
5. **Best Selection**: Configuration with optimal average performance returned

**Split Strategies**

* **K-Fold**: Random stratified splits for general use cases
* **Ordinal Split**: Single time-ordered split for temporal data

**Evaluation Metrics**

* **Point Estimation**: Mean Squared Error (MSE)
* **Quantile Estimation**: Average Pinball Loss across quantile levels

Usage Examples
--------------

Point Estimation Tuning
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.estimation import PointTuner
    import numpy as np

    # Generate sample data
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    # Initialize tuner
    tuner = PointTuner(random_state=42)

    # Optimize hyperparameters
    best_config = tuner.tune(
        X=X,
        y=y,
        estimator_architecture="rf",  # Random Forest
        n_searches=20,
        split_type="k_fold"
    )

    print(f"Best configuration: {best_config}")

Quantile Estimation Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.estimation import QuantileTuner

    # Define quantile levels
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Initialize quantile tuner
    tuner = QuantileTuner(
        quantiles=quantiles,
        random_state=42
    )

    # Optimize for quantile regression
    best_config = tuner.tune(
        X=X,
        y=y,
        estimator_architecture="qgbm",  # Quantile GBM
        n_searches=15,
        split_type="k_fold"
    )

Warm-Start Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Pre-specify promising configurations
    forced_configs = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 3}
    ]

    best_config = tuner.tune(
        X=X,
        y=y,
        estimator_architecture="qrf",
        n_searches=10,
        forced_param_configurations=forced_configs
    )
    # First 2 evaluations will use forced_configs

Estimator Initialization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.estimation import initialize_estimator

    # Initialize with default parameters
    estimator = initialize_estimator(
        estimator_architecture="qgbm",
        random_state=42
    )

    # Initialize with custom parameters
    estimator = initialize_estimator(
        estimator_architecture="qgbm",
        initialization_params={
            "learning_rate": 0.05,
            "n_estimators": 200
        },
        random_state=42
    )

Performance Considerations
-------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

**Random Search Scaling**
    - Time: O(n_searches × n_folds × model_complexity)
    - Memory: O(max_model_size)

**Cross-Validation Overhead**
    - K-Fold: Requires K model fits per configuration
    - Ordinal Split: Single model fit per configuration

**Parameter Space Efficiency**
    Random search provides good coverage with relatively few evaluations compared to grid search, especially for high-dimensional parameter spaces.

Optimization Guidelines
~~~~~~~~~~~~~~~~~~~~~~

**Search Budget Allocation**
    - Small datasets (< 1K): 10-20 configurations sufficient
    - Medium datasets (1K-100K): 20-50 configurations recommended
    - Large datasets (> 100K): 50+ configurations for thorough exploration

**Split Strategy Selection**
    - Time series data: Use ``ordinal_split`` to preserve temporal ordering
    - IID data: Use ``k_fold`` for robust cross-validation
    - Small datasets: Increase fold count for better variance estimation

Integration Points
-----------------

**Estimator Registry System**
    Seamless integration with ``confopt.selection.estimator_configuration`` for automatic parameter space discovery and default value management.

**Quantile Estimators**
    Direct support for all quantile regression estimators in ``confopt.selection.estimators.quantile_estimation`` and ensemble methods.

**Conformal Prediction**
    Optimized estimators can be used directly in conformal prediction frameworks with appropriate hyperparameter configurations.

Common Pitfalls
---------------

* **Insufficient Search Budget**: Too few configurations may miss optimal regions
* **Inappropriate Split Strategy**: Using K-fold on temporal data can cause data leakage
* **Overfitting to Validation**: Excessive hyperparameter searches can overfit to cross-validation splits
* **Parameter Scale Mismatch**: Ensure parameter ranges in registry are appropriate for your data scale
* **Memory Constraints**: Large ensemble models may exceed memory during parallel evaluation

See Also
--------

* :doc:`quantile_estimation` - Base quantile regression estimators optimized by this module
* :doc:`ensembling` - Ensemble methods that can be tuned using this framework
* :doc:`../tuning` - Higher-level Bayesian optimization approaches
