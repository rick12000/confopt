Tuning Module
=============

Overview
--------

The tuning module provides the core hyperparameter optimization framework that orchestrates conformal prediction-based search strategies. As the primary orchestrator, it coordinates between configuration management, conformal prediction searchers, adaptive parameter tuning, and optimization flow control.

The module implements a sophisticated two-phase optimization approach: random search initialization followed by conformal prediction-guided exploration. It handles both maximization and minimization objectives through internal sign transformation, ensuring the underlying optimization machinery always operates in a consistent minimization framework while supporting user-specified optimization directions.

Key Features
------------

* **Bidirectional Optimization**: Supports both maximize and minimize objectives through metric sign transformation
* **Two-Phase Search Strategy**: Random initialization followed by conformal prediction-guided exploration
* **Flexible Configuration Management**: Static and dynamic configuration sampling strategies
* **Adaptive Orchestration**: Multi-armed bandit optimization for searcher parameter tuning
* **Comprehensive Flow Control**: Progress tracking, termination criteria, and resource management
* **Warm Start Integration**: Seamless incorporation of pre-evaluated configurations

Architecture and Component Interactions
---------------------------------------

The tuning framework follows a hierarchical orchestration pattern centered around the ``ConformalTuner`` class, which coordinates multiple specialized components in a well-defined optimization flow.

**Core Orchestration Components:**

``ConformalTuner`` (Main Orchestrator)
  Central coordinator managing the entire optimization lifecycle. Handles phase transitions, component initialization, and flow control between random and conformal search phases.

``Study`` (History and State Management)
  Maintains comprehensive optimization history including trial records, performance tracking, and best configuration identification. Provides metric-aware result aggregation supporting both optimization directions.

``ConfigurationManager`` (Search Space Management)
  Handles search space sampling, configuration tracking, and candidate pool management. Two variants provide different sampling strategies:

  * **StaticConfigurationManager**: Pre-samples fixed candidate pool at initialization
  * **DynamicConfigurationManager**: Adaptively resamples candidates during optimization

``BaseConformalSearcher`` (Acquisition Strategy)
  Implements conformal prediction-based configuration selection. Receives scaled features and sign-adjusted targets from the orchestrator, returning uncertainty-aware predictions for acquisition decisions.

**Integration and Data Flow:**

The architecture follows a clear data flow pattern:

1. **Initialization Phase**: ``ConformalTuner`` creates ``Study`` and ``ConfigurationManager`` instances based on sampling strategy
2. **Random Phase**: ``ConfigurationManager`` provides candidates, ``ConformalTuner`` evaluates and records in ``Study``
3. **Conformal Phase**: ``ConformalTuner`` prepares data from ``Study``, trains ``BaseConformalSearcher``, selects candidates from ``ConfigurationManager``
4. **Continuous Updates**: All components maintain state through ``Study`` while ``ConfigurationManager`` tracks evaluation status

Optimization Direction Handling
-------------------------------

The framework supports both maximization and minimization objectives through a consistent internal transformation strategy. This design allows the underlying optimization machinery to operate uniformly while providing user-friendly objective specification.

**Metric Sign Transformation:**

The tuner applies a sign transformation to convert all objectives to minimization problems:

* **Minimize objectives**: ``metric_sign = +1`` (no transformation)
* **Maximize objectives**: ``metric_sign = -1`` (negation applied)

All performance values are multiplied by ``metric_sign`` before being passed to conformal prediction models, ensuring the acquisition strategy (minimizing predicted lower bounds) correctly optimizes the user-specified direction.

**Implementation Flow:**

1. User specifies ``metric_optimization='maximize'`` or ``'minimize'``
2. Tuner sets ``metric_sign = -1`` for maximize, ``+1`` for minimize
3. Raw objective values are stored in ``Study`` with original sign
4. During conformal model training, values are transformed: ``y_transformed = y_original * metric_sign``
5. Acquisition functions operate on transformed values (always minimizing)
6. Final results maintain original objective direction for user interpretation

Configuration Management Strategies
----------------------------------

The framework provides two distinct configuration management approaches, each optimized for different search space characteristics and computational constraints.

**Static Configuration Management:**

``StaticConfigurationManager`` pre-generates a fixed pool of candidate configurations at initialization:

* **Sampling**: Uniform random sampling across the entire search space
* **Pool Size**: Fixed at ``n_candidate_configurations``
* **Updates**: Candidates marked as searched/banned but pool never refreshed
* **Memory**: Constant memory footprint throughout optimization
* **Use Cases**: Moderate-dimensional spaces, limited computational resources

**Dynamic Configuration Management:**

``DynamicConfigurationManager`` adaptively resamples configuration candidates:

* **Sampling**: Fresh sampling when candidate pool becomes depleted
* **Pool Size**: Maintains approximately ``n_candidate_configurations`` available candidates
* **Updates**: Periodic resampling to maintain candidate availability
* **Memory**: Variable memory based on current pool size
* **Use Cases**: High-dimensional spaces, long-running optimizations

**Configuration State Tracking:**

Both managers maintain detailed configuration state through the optimization lifecycle:

* **Searchable**: Available for evaluation selection
* **Searched**: Previously evaluated with recorded performance
* **Banned**: Invalid configurations producing non-numeric results

The orchestrator coordinates between managers and conformal searchers by:
1. Requesting searchable configurations from manager
2. Tabularizing configurations for conformal model input
3. Selecting next candidate using searcher predictions
4. Updating manager state after evaluation

Optimization Flow Control
------------------------

The tuning orchestrator manages a sophisticated multi-phase optimization flow with adaptive decision points and resource management.

**Phase 1: Random Search Initialization**

1. ``ConfigurationManager`` samples initial candidate pool
2. Random selection from available configurations
3. Objective evaluation and ``Study`` recording
4. Continues until random search budget exhausted or termination criteria met

**Phase 2: Conformal Prediction-Guided Search**

1. Data preparation from ``Study`` history with metric sign transformation
2. Feature scaling and train-validation splitting
3. ``BaseConformalSearcher`` training with transformed targets
4. Acquisition-guided candidate selection from ``ConfigurationManager``
5. Objective evaluation and ``Study`` update
6. Periodic searcher retraining based on adaptive frequency

**Adaptive Parameter Management:**

When searcher tuning is enabled, the orchestrator employs multi-armed bandit optimization to balance prediction improvement against computational cost:

* **Reward Signal**: Conformal model error reduction
* **Cost Signal**: Relative training time compared to objective evaluation
* **Arms**: (tuning_iterations, retraining_frequency) parameter combinations
* **Strategy**: Bayesian optimization or fixed schedule based on framework selection

**Termination and Resource Management:**

The orchestrator continuously monitors multiple termination criteria:

* **Candidate Exhaustion**: No remaining searchable configurations
* **Runtime Budget**: Maximum wall-clock time exceeded
* **Iteration Budget**: Maximum evaluation count reached

Progress tracking provides real-time optimization monitoring with metric-aware best value reporting.

Integration Points
-----------------

**Configuration Management Integration:**

* Search space sampling and discretization strategies
* Configuration deduplication and state tracking
* Banned configuration handling for evaluation failures

**Conformal Searcher Integration:**

* Feature preprocessing and scaling coordination
* Metric sign transformation for consistent optimization direction
* Acquisition function parameterization and uncertainty quantification

**Utility Component Integration:**

* Multi-armed bandit optimization for parameter tuning
* Progress tracking and resource monitoring
* Statistical preprocessing and data validation

See Also
--------

* :doc:`acquisition` - Conformal prediction searcher implementations
* :doc:`quantile_estimation` - Quantile estimation for conformal predictions
* :doc:`bound_samplers` - Lower bound sampling strategies
* ``confopt.utils.tracking`` - Configuration management and trial tracking utilities
* ``confopt.utils.optimization`` - Multi-armed bandit optimization for parameter tuning
