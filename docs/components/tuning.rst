Conformal Tuner Orchestration
=============================

The tuning module (``confopt.tuning``) contains the ``ConformalTuner`` class, which serves as the main entry point and orchestrator for the entire conformal hyperparameter optimization framework. This class coordinates all components to provide an intelligent, statistically principled approach to hyperparameter search.

Overview
--------

``ConformalTuner`` implements a sophisticated two-phase optimization strategy that combines the broad exploration capabilities of random search with the targeted efficiency of conformal prediction-guided acquisition. The tuner maintains statistical validity through proper conformal prediction procedures while adapting to the specific characteristics of each optimization problem.

**Key Responsibilities:**

- **Orchestration**: Coordinates all framework components in proper sequence
- **Phase Management**: Controls transition from random to conformal search phases
- **Configuration Management**: Handles search space sampling and candidate tracking
- **Model Training**: Manages conformal estimator training and retraining
- **Acquisition Optimization**: Selects next configurations using acquisition functions
- **Progress Tracking**: Monitors optimization progress and stopping conditions

Architecture
------------

.. mermaid::

   graph TD
       subgraph "Main Entry Point"
           CT["ConformalTuner<br/>tune()<br/>Main Orchestration"]
       end

       subgraph "Optimization Phases"
           RS["Random Search Phase<br/>random_search()<br/>Baseline Data Collection"]
           CS["Conformal Search Phase<br/>conformal_search()<br/>Guided Optimization"]
       end

       subgraph "Configuration Management"
           SCM["StaticConfigurationManager<br/>Fixed Candidate Pool"]
           DCM["DynamicConfigurationManager<br/>Adaptive Resampling"]
           CE["ConfigurationEncoder<br/>Parameter Encoding"]
       end

       subgraph "Acquisition System"
           SEARCHER["Conformal Searcher<br/>LocallyWeighted/Quantile"]
           SAMPLER["Acquisition Sampler<br/>Thompson/EI/Entropy/Bounds"]
           OPTIMIZER["Searcher Optimizer<br/>Bayesian/Fixed"]
       end

       subgraph "Progress Tracking"
           STUDY["Study<br/>Trial Management<br/>Results Storage"]
           PROGRESS["Progress Monitoring<br/>Runtime/Iterations<br/>Early Stopping"]
       end

       subgraph "Integration Components"
           OBJ["Objective Function<br/>User-Defined Target"]
           SPACE["Search Space<br/>Parameter Ranges"]
       end

       CT --> RS
       CT --> CS

       RS --> SCM
       RS --> DCM
       CS --> SCM
       CS --> DCM

       CS --> SEARCHER
       SEARCHER --> SAMPLER
       CS --> OPTIMIZER

       CT --> STUDY
       CT --> PROGRESS

       CT --> OBJ
       CT --> SPACE

       SCM --> CE
       DCM --> CE

ConformalTuner Class
--------------------

The main orchestrator class that provides the public interface for conformal hyperparameter optimization.

**Initialization Parameters:**

``objective_function`` (callable)
   Function to optimize. Must accept a single parameter named ``configuration`` of type Dict and return a numeric value. The function signature is validated during initialization.

``search_space`` (Dict[str, ParameterRange])
   Dictionary mapping parameter names to ``ParameterRange`` objects (``IntRange``, ``FloatRange``, ``CategoricalRange``). Defines the hyperparameter search space.

``metric_optimization`` (Literal["maximize", "minimize"])
   Optimization direction. Determines whether higher or lower objective values are preferred.

``n_candidate_configurations`` (int, default=10000)
   Size of the discrete configuration pool used for acquisition function optimization. Larger pools provide better optimization potential but increase computational cost.

``warm_start_configurations`` (List[Tuple[Dict, float]], optional)
   Pre-evaluated configurations to initialize optimization. Useful for incorporating prior knowledge or continuing previous optimization runs.

``dynamic_sampling`` (bool, default=False)
   Whether to dynamically resample the candidate configuration pool during optimization. Static pools are more efficient, while dynamic pools provide better exploration.

**Core Methods:**

``tune(max_searches, max_runtime, searcher, n_random_searches, ...)``
   Main optimization method that orchestrates the complete hyperparameter search process.

``get_best_params()`` / ``get_best_value()``
   Retrieve the best configuration and performance found during optimization.

``get_optimization_history()``
   Access complete optimization history for analysis and visualization.

Optimization Process
--------------------

The ``tune()`` method implements a sophisticated two-phase optimization strategy:

**Phase 1: Random Search Initialization**

``random_search(max_random_iter, max_runtime, max_searches, verbose)``
   Performs uniform random sampling to establish baseline performance understanding.

   **Algorithm Steps:**

   1. **Configuration Sampling**: Randomly select configurations from candidate pool
   2. **Evaluation**: Execute objective function for each configuration
   3. **Data Collection**: Store results for conformal model training
   4. **Progress Monitoring**: Check stopping conditions and update progress
   5. **Quality Control**: Handle NaN results and invalid configurations

   **Key Features:**

   - **Unbiased Exploration**: Uniform sampling provides unbiased data collection
   - **Robust Handling**: Graceful handling of evaluation failures
   - **Progress Tracking**: Real-time progress monitoring with optional visualization
   - **Early Stopping**: Terminates when stopping conditions are met

**Phase 2: Conformal Search Optimization**

``conformal_search(searcher, max_searches, max_runtime, ...)``
   Uses conformal prediction-guided acquisition for targeted optimization.

   **Algorithm Steps:**

   1. **Model Training**: Train conformal estimator on collected data
   2. **Acquisition Optimization**: Select next configuration using acquisition function
   3. **Configuration Evaluation**: Execute objective function on selected configuration
   4. **Model Updates**: Update conformal estimator with new data
   5. **Adaptive Retraining**: Periodically retrain models for improved performance

   **Key Features:**

   - **Statistical Validity**: Maintains coverage guarantees through conformal prediction
   - **Adaptive Learning**: Improves surrogate models with each new observation
   - **Intelligent Selection**: Uses uncertainty quantification for configuration selection
   - **Efficient Optimization**: Focuses search on promising regions

Configuration Management
-------------------------

The tuner supports two configuration management strategies:

StaticConfigurationManager
~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses a fixed pool of candidate configurations throughout optimization.

**Advantages:**

- **Computational Efficiency**: No resampling overhead
- **Reproducibility**: Consistent candidate pool across runs
- **Memory Efficiency**: Fixed memory footprint
- **Predictable Behavior**: Deterministic search progression

**Use Cases:**

- **Standard Optimization**: Most hyperparameter optimization scenarios
- **Computational Constraints**: When minimizing overhead is important
- **Reproducible Research**: When exact reproducibility is required

DynamicConfigurationManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptively resamples the candidate pool during optimization.

**Advantages:**

- **Enhanced Exploration**: Fresh candidates provide better exploration
- **Adaptive Focus**: Can focus on promising regions of search space
- **Reduced Bias**: Avoids bias from fixed initial sampling
- **Better Coverage**: Improved search space coverage over time

**Use Cases:**

- **Complex Search Spaces**: High-dimensional or complex parameter spaces
- **Long Optimizations**: Extended optimization runs benefit from fresh candidates
- **Exploration Priority**: When exploration is more important than efficiency

**Resampling Strategy:**

.. code-block:: python

   # Dynamic resampling triggers
   if should_resample(current_iteration):
       new_candidates = sample_configurations(
           search_space=self.search_space,
           n_candidates=self.n_candidate_configurations,
           exclude_searched=True
       )
       self.candidate_pool = new_candidates

Acquisition Function Integration
--------------------------------

The tuner integrates with the acquisition function framework through the ``searcher`` parameter:

**Default Searcher:**

``QuantileConformalSearcher`` with ``LowerBoundSampler`` provides robust performance across diverse optimization problems.

**Alternative Searchers:**

- **LocallyWeightedConformalSearcher**: Better for heteroscedastic objectives
- **Different Samplers**: Thompson sampling, Expected Improvement, Entropy Search
- **Custom Configurations**: User-defined searcher and sampler combinations

**Searcher Lifecycle:**

1. **Initialization**: Create searcher with appropriate architecture and sampler
2. **Training**: Fit conformal estimator on random search data
3. **Acquisition**: Generate acquisition values for candidate configurations
4. **Selection**: Choose configuration with best acquisition value
5. **Update**: Incorporate new observation and adapt coverage levels
6. **Retraining**: Periodically retrain estimator for improved performance

**Integration Example:**

.. code-block:: python

   # Custom searcher configuration
   from confopt.selection.acquisition import LocallyWeightedConformalSearcher
   from confopt.selection.sampling import ThompsonSampler

   searcher = LocallyWeightedConformalSearcher(
       point_estimator_architecture="lightgbm",
       variance_estimator_architecture="lightgbm",
       sampler=ThompsonSampler(n_quantiles=6)
   )

   tuner.tune(searcher=searcher)

Progress Monitoring and Control
-------------------------------

The tuner provides comprehensive progress monitoring and control mechanisms:

**Study Management:**

``Study`` class tracks complete optimization history:

- **Trial Records**: Configuration, performance, metadata for each evaluation
- **Best Tracking**: Maintains current best configuration and performance
- **Statistics**: Optimization statistics and performance metrics
- **Serialization**: Save/load optimization state for persistence

**Runtime Tracking:**

``RuntimeTracker`` monitors execution timing:

- **Phase Timing**: Separate tracking for random and conformal phases
- **Component Timing**: Detailed timing for each optimization component
- **Budget Management**: Runtime budget enforcement and monitoring
- **Performance Analysis**: Timing analysis for optimization efficiency

**Progress Visualization:**

Optional progress bars provide real-time feedback:

- **Phase Progress**: Current phase and completion status
- **Performance Updates**: Best performance and recent improvements
- **Timing Information**: Elapsed time and estimated completion
- **Configuration Details**: Current configuration being evaluated

**Early Stopping:**

``stop_search()`` function implements multiple stopping criteria:

- **Iteration Limits**: Maximum number of evaluations
- **Runtime Limits**: Maximum optimization time
- **Configuration Exhaustion**: All candidates evaluated
- **Convergence Detection**: No improvement over specified period

Searcher Optimization Framework
-------------------------------

The tuner supports optional meta-optimization of the acquisition function itself:

**Reward-Cost Framework:**

``BayesianSearcherOptimizer`` balances prediction improvement against computational cost:

.. math::

   \text{Utility} = \frac{\text{Expected Improvement}}{\text{Expected Cost}}

**Fixed Framework:**

``FixedSearcherOptimizer`` applies deterministic optimization schedules:

- **Interval-based**: Optimize searcher every N iterations
- **Performance-based**: Optimize when improvement stagnates
- **Resource-based**: Optimize based on available computational budget

**Optimization Targets:**

- **Searcher Architecture**: Point/variance/quantile estimator selection
- **Sampler Configuration**: Acquisition strategy and parameters
- **Hyperparameters**: Estimator-specific hyperparameters
- **Alpha Values**: Coverage levels and adaptation parameters

Error Handling and Robustness
------------------------------

The tuner implements comprehensive error handling and robustness mechanisms:

**Objective Function Validation:**

- **Signature Validation**: Ensures proper function signature and type hints
- **Return Type Checking**: Validates numeric return values
- **Exception Handling**: Graceful handling of objective function failures

**Configuration Management:**

- **Invalid Configuration Handling**: Skips configurations that cause errors
- **Banned Configuration Tracking**: Avoids re-evaluating failed configurations
- **Search Space Validation**: Ensures valid parameter ranges and types

**Model Training Robustness:**

- **Data Sufficiency Checks**: Ensures adequate data for model training
- **Convergence Monitoring**: Detects and handles training failures
- **Fallback Strategies**: Alternative approaches when primary methods fail

**Resource Management:**

- **Memory Monitoring**: Tracks memory usage and prevents exhaustion
- **Computational Budgets**: Enforces time and iteration limits
- **Graceful Degradation**: Maintains functionality under resource constraints

Performance Characteristics
---------------------------

**Computational Complexity:**

- **Random Phase**: O(n_random × objective_cost)
- **Conformal Phase**: O(n_conformal × (model_training + acquisition_optimization + objective_cost))
- **Total Complexity**: Dominated by objective function evaluations for expensive objectives

**Memory Requirements:**

- **Configuration Storage**: O(n_candidates × parameter_dimensions)
- **Trial History**: O(n_evaluations × (configuration_size + metadata))
- **Model Storage**: O(model_parameters) for conformal estimators

**Scalability Factors:**

- **Search Space Dimensionality**: Higher dimensions require more random initialization
- **Candidate Pool Size**: Larger pools provide better optimization but increase overhead
- **Objective Function Cost**: Expensive objectives benefit most from intelligent selection

Best Practices
---------------

**Initialization:**

- **Random Search Count**: Use 10-20 random searches for most problems
- **Candidate Pool Size**: 1000-10000 candidates depending on search space complexity
- **Warm Starting**: Leverage prior knowledge when available

**Searcher Selection:**

- **Default Choice**: QuantileConformalSearcher works well for most problems
- **Heteroscedastic Objectives**: Use LocallyWeightedConformalSearcher
- **Specific Needs**: Choose samplers based on exploration-exploitation preferences

**Resource Management:**

- **Time Budgets**: Set realistic runtime limits based on objective function cost
- **Iteration Limits**: Balance search thoroughness with computational constraints
- **Retraining Frequency**: Adjust based on objective function evaluation cost

**Common Pitfalls:**

- **Insufficient Random Search**: Too few random evaluations provide poor model training data
- **Excessive Candidate Pool**: Very large pools provide diminishing returns
- **Inappropriate Searcher**: Mismatched searcher for objective characteristics
- **Resource Underestimation**: Inadequate time/iteration budgets for meaningful optimization

Integration Example
-------------------

Complete example demonstrating tuner usage:

.. code-block:: python

   from confopt.tuning import ConformalTuner
   from confopt.wrapping import IntRange, FloatRange, CategoricalRange
   from confopt.selection.acquisition import LocallyWeightedConformalSearcher
   from confopt.selection.sampling import ThompsonSampler

   # Define objective function
   def objective(configuration):
       model = MyModel(
           learning_rate=configuration['lr'],
           hidden_units=configuration['units'],
           optimizer=configuration['optimizer']
       )
       return model.cross_validate()

   # Define search space
   search_space = {
       'lr': FloatRange(0.001, 0.1, log_scale=True),
       'units': IntRange(32, 512),
       'optimizer': CategoricalRange(['adam', 'sgd', 'rmsprop'])
   }

   # Optional: Custom searcher configuration
   searcher = LocallyWeightedConformalSearcher(
       point_estimator_architecture="lightgbm",
       variance_estimator_architecture="lightgbm",
       sampler=ThompsonSampler(n_quantiles=6, adapter="DtACI")
   )

   # Initialize tuner
   tuner = ConformalTuner(
       objective_function=objective,
       search_space=search_space,
       metric_optimization="maximize",
       n_candidate_configurations=5000
   )

   # Run optimization
   tuner.tune(
       max_searches=100,
       max_runtime=3600,  # 1 hour
       searcher=searcher,
       n_random_searches=20,
       conformal_retraining_frequency=2,
       random_state=42,
       verbose=True
   )

   # Retrieve results
   best_params = tuner.get_best_params()
   best_score = tuner.get_best_value()
   history = tuner.get_optimization_history()

The ``ConformalTuner`` provides a powerful, statistically principled approach to hyperparameter optimization that combines the reliability of conformal prediction with the efficiency of intelligent acquisition functions, making it suitable for a wide range of optimization challenges.
