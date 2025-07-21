Architecture
============

Module Dependency Structure
---------------------------

The following diagram shows the directional module dependencies within the confopt package.
Module paths are shown without the ``confopt.`` prefix for clarity.

.. mermaid::

   graph TD
       subgraph "Core Layer"
           tuning["tuning"]
           wrapping["wrapping"]
       end

       subgraph "Utils Layer"
           utils_preprocessing["utils.preprocessing"]
           utils_tracking["utils.tracking"]
           utils_optimization["utils.optimization"]

           subgraph "Configuration Utilities"
               utils_configurations_encoding["utils.configurations.encoding"]
               utils_configurations_sampling["utils.configurations.sampling"]
               utils_configurations_utils["utils.configurations.utils"]
           end
       end

       subgraph "Selection Layer"
           selection_acquisition["selection.acquisition"]
           selection_conformalization["selection.conformalization"]
           selection_estimation["selection.estimation"]
           selection_estimator_configuration["selection.estimator_configuration"]
           selection_adaptation["selection.adaptation"]

           subgraph "Estimator Implementations"
               selection_estimators_quantile_estimation["selection.estimators.quantile_estimation"]
               selection_estimators_ensembling["selection.estimators.ensembling"]
           end

           subgraph "Sampling Strategies"
               selection_sampling_bound_samplers["selection.sampling.bound_samplers"]
               selection_sampling_thompson_samplers["selection.sampling.thompson_samplers"]
               selection_sampling_expected_improvement_samplers["selection.sampling.expected_improvement_samplers"]
               selection_sampling_entropy_samplers["selection.sampling.entropy_samplers"]
               selection_sampling_utils["selection.sampling.utils"]
           end
       end

       %% Core Dependencies
       tuning --> wrapping
       tuning --> utils_preprocessing
       tuning --> utils_tracking
       tuning --> utils_optimization
       tuning --> selection_acquisition

       %% Utils Dependencies
       utils_tracking --> wrapping
       utils_tracking --> utils_configurations_encoding
       utils_tracking --> utils_configurations_sampling
       utils_tracking --> utils_configurations_utils

       utils_configurations_sampling --> wrapping
       utils_configurations_sampling --> utils_configurations_utils
       utils_configurations_encoding --> wrapping

       %% Selection Layer Dependencies
       selection_acquisition --> selection_conformalization
       selection_acquisition --> selection_sampling_bound_samplers
       selection_acquisition --> selection_sampling_thompson_samplers
       selection_acquisition --> selection_sampling_expected_improvement_samplers
       selection_acquisition --> selection_sampling_entropy_samplers
       selection_acquisition --> selection_estimation

       selection_conformalization --> wrapping
       selection_conformalization --> utils_preprocessing
       selection_conformalization --> selection_estimation
       selection_conformalization --> selection_estimator_configuration

       selection_estimation --> selection_estimator_configuration
       selection_estimation --> selection_estimators_quantile_estimation
       selection_estimation --> selection_estimators_ensembling
       selection_estimation --> utils_configurations_sampling

       selection_estimator_configuration --> wrapping
       selection_estimator_configuration --> selection_estimators_quantile_estimation
       selection_estimator_configuration --> selection_estimators_ensembling

       selection_estimators_ensembling --> selection_estimators_quantile_estimation

       %% Sampling Dependencies
       selection_sampling_bound_samplers --> selection_sampling_utils
       selection_sampling_thompson_samplers --> wrapping
       selection_sampling_thompson_samplers --> selection_sampling_utils
       selection_sampling_expected_improvement_samplers --> wrapping
       selection_sampling_expected_improvement_samplers --> selection_sampling_utils
       selection_sampling_entropy_samplers --> wrapping
       selection_sampling_entropy_samplers --> selection_sampling_thompson_samplers
       selection_sampling_entropy_samplers --> selection_sampling_expected_improvement_samplers
       selection_sampling_entropy_samplers --> selection_sampling_utils

       selection_sampling_utils --> selection_adaptation
       selection_sampling_utils --> wrapping

       %% Styling
       style tuning fill:#ff6b6b
       style wrapping fill:#4ecdc4
       style utils_preprocessing fill:#45b7d1
       style utils_tracking fill:#45b7d1
       style utils_optimization fill:#45b7d1
       style selection_acquisition fill:#96ceb4
       style selection_conformalization fill:#96ceb4
       style selection_estimation fill:#96ceb4

Module Organization and Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Core Orchestration**
  The ``tuning`` module contains ``ConformalTuner`` which orchestrates the entire optimization process. It depends on data structures from ``wrapping`` and coordinates all other layers.

**Utilities Layer**
  * ``utils.preprocessing``: Data splitting and outlier handling
  * ``utils.tracking``: Experiment management and progress monitoring
  * ``utils.optimization``: Bayesian optimization algorithms
  * ``utils.configurations.*``: Parameter encoding, sampling, and hashing utilities

**Selection Layer**
  * ``selection.acquisition``: Main acquisition function interface and implementations
  * ``selection.conformalization``: Conformal prediction estimators and calibration
  * ``selection.estimation``: Hyperparameter tuning and model selection
  * ``selection.estimator_configuration``: Registry and configuration for all estimators
  * ``selection.estimators.*``: Quantile regression and ensemble implementations
  * ``selection.sampling.*``: Acquisition sampling strategies and utilities
  * ``selection.adaptation``: Adaptive alpha adjustment mechanisms

**Dependency Flow Patterns**
  Data flows from ``tuning`` through ``utils`` to ``selection`` layers. The ``wrapping`` module provides shared data structures used across all layers. Configuration utilities support both experiment tracking and model selection processes.

Detailed Dependency Structure
-----------------------------

The following diagram shows the complete end-to-end flow with class and method interactions:

.. mermaid::

   graph TD
       subgraph "Main Orchestration"
           CT["ConformalTuner<br/>search()<br/>_run_trials()<br/>_evaluate_configuration()"]
           STOP["stop_search()<br/>early_stopping_check()"]
       end

       subgraph "Experiment Management"
           STUDY["Study<br/>add_trial()<br/>get_best_trial()<br/>get_trials()"]
           TRIAL["Trial<br/>configuration<br/>performance<br/>metadata"]
           RT["RuntimeTracker<br/>start_timing()<br/>stop_timing()"]
           PBM["ProgressBarManager<br/>update_progress()"]
       end

       subgraph "Configuration Management"
           SCM["StaticConfigurationManager<br/>get_configurations()"]
           DCM["DynamicConfigurationManager<br/>suggest_configuration()"]
           CE["ConfigurationEncoder<br/>encode()<br/>decode()"]
           GTC["get_tuning_configurations()<br/>uniform_sampling()<br/>sobol_sampling()"]
           CCH["create_config_hash()<br/>hash_generation()"]
       end

       subgraph "Acquisition Layer"
           BCS["BaseConformalSearcher<br/>predict()<br/>update()<br/>get_interval()"]
           LWCS["LocallyWeightedConformalSearcher<br/>fit()<br/>_predict_with_*()"]
           QCS["QuantileConformalSearcher<br/>fit()<br/>_predict_with_*()"]
       end

       subgraph "Conformal Prediction"
           LWCE["LocallyWeightedConformalEstimator<br/>fit()<br/>predict_intervals()<br/>_tune_fit_component_estimator()"]
           QCE["QuantileConformalEstimator<br/>fit()<br/>predict_intervals()<br/>calculate_betas()"]
           DTACI["DtACI<br/>update_alpha()<br/>_calculate_pinball_loss()"]
       end

       subgraph "Hyperparameter Tuning"
           RT_TUNER["RandomTuner<br/>tune()<br/>_cross_validate()"]
           PT["PointTuner<br/>tune()<br/>_evaluate_point_estimator()"]
           QT["QuantileTuner<br/>tune()<br/>_evaluate_quantile_estimator()"]
           IE["initialize_estimator()<br/>estimator_creation()"]
       end

       subgraph "Estimator Registry"
           ER["ESTIMATOR_REGISTRY<br/>estimator_configs"]
           EC["EstimatorConfig<br/>architecture<br/>param_ranges<br/>default_params"]
       end

       subgraph "Quantile Estimators"
           QL["QuantileLasso<br/>fit()<br/>predict_quantiles()"]
           QG["QuantileGBM<br/>fit()<br/>predict_quantiles()"]
           QF["QuantileForest<br/>fit()<br/>predict_quantiles()"]
           QK["QuantileKNN<br/>fit()<br/>predict_quantiles()"]
           GP["GaussianProcessQuantileEstimator<br/>fit()<br/>predict_quantiles()"]
       end

       subgraph "Ensemble Methods"
           PEE["PointEnsembleEstimator<br/>fit()<br/>predict()<br/>_fit_base_estimators()"]
           QEE["QuantileEnsembleEstimator<br/>fit()<br/>predict_quantiles()<br/>_fit_base_estimators()"]
       end

       subgraph "Sampling Strategies"
           LBS["LowerBoundSampler<br/>calculate_upper_confidence_bound()"]
           PLBS["PessimisticLowerBoundSampler<br/>calculate_lower_bound()"]
           TS["ThompsonSampler<br/>sample()<br/>_update_posterior()"]
           EIS["ExpectedImprovementSampler<br/>sample()<br/>_calculate_expected_improvement()"]
           ESS["EntropySearchSampler<br/>sample()<br/>_calculate_entropy()"]
           MVES["MaxValueEntropySearchSampler<br/>sample()<br/>_calculate_max_value_entropy()"]
       end

       subgraph "Sampling Utilities"
           IQA["initialize_quantile_alphas()<br/>alpha_generation()"]
           IMA["initialize_multi_adapters()<br/>adapter_creation()"]
           ISA["initialize_single_adapter()<br/>single_adapter_setup()"]
           UMIW["update_multi_interval_widths()<br/>width_updates()"]
           USIW["update_single_interval_width()<br/>single_width_update()"]
           FCB["flatten_conformal_bounds()<br/>bounds_flattening()"]
       end

       subgraph "Data Processing"
           TVS["train_val_split()<br/>data_splitting()"]
           RIO["remove_iqr_outliers()<br/>outlier_removal()"]
       end

       subgraph "Bayesian Optimization"
           BSO["BayesianSearcherOptimizer<br/>suggest()<br/>_fit_gp()<br/>_calculate_acquisition()"]
           FSO["FixedSearcherOptimizer<br/>suggest()<br/>_fixed_suggestions()"]
       end

       subgraph "Parameter Structures"
           PR["ParameterRange<br/>IntRange<br/>FloatRange<br/>CategoricalRange"]
           CB["ConformalBounds<br/>lower_bound<br/>upper_bound<br/>alpha"]
       end

       %% Main Flow Connections
       CT --> STUDY
       CT --> RT
       CT --> PBM
       CT --> SCM
       CT --> DCM
       CT --> LWCS
       CT --> QCS
       CT --> TVS
       CT --> RIO
       CT --> BSO
       CT --> FSO
       CT --> STOP

       %% Configuration Management Flow
       STUDY --> TRIAL
       STUDY --> CE
       STUDY --> GTC
       STUDY --> CCH
       SCM --> GTC
       DCM --> GTC
       DCM --> BSO

       %% Acquisition Flow
       LWCS --> LWCE
       QCS --> QCE
       BCS --> LBS
       BCS --> PLBS
       BCS --> TS
       BCS --> EIS
       BCS --> ESS
       BCS --> MVES

       %% Conformal Prediction Flow
       LWCE --> PT
       LWCE --> QT
       LWCE --> IE
       LWCE --> TVS
       LWCE --> DTACI
       QCE --> QT
       QCE --> IE
       QCE --> DTACI

       %% Hyperparameter Tuning Flow
       RT_TUNER --> IE
       PT --> RT_TUNER
       PT --> ER
       QT --> RT_TUNER
       QT --> ER
       IE --> ER
       IE --> EC

       %% Estimator Flow
       ER --> EC
       EC --> QL
       EC --> QG
       EC --> QF
       EC --> QK
       EC --> GP
       EC --> PEE
       EC --> QEE

       %% Ensemble Flow
       PEE --> QL
       PEE --> QG
       PEE --> QF
       QEE --> QL
       QEE --> QG
       QEE --> QF
       QEE --> QK
       QEE --> GP

       %% Sampling Utilities Flow
       LBS --> IQA
       PLBS --> IQA
       TS --> IQA
       TS --> IMA
       TS --> ISA
       EIS --> IQA
       EIS --> UMIW
       EIS --> USIW
       ESS --> IQA
       ESS --> FCB
       MVES --> IQA
       MVES --> FCB

       %% Adaptive Flow
       IMA --> DTACI
       ISA --> DTACI
       UMIW --> DTACI
       USIW --> DTACI

       %% Data Structure Flow
       CT --> PR
       LWCE --> CB
       QCE --> CB
       LBS --> CB
       PLBS --> CB
       TS --> CB
       EIS --> CB
       ESS --> CB
       MVES --> CB

       %% Styling
       style CT fill:#ff6b6b
       style LWCS fill:#4ecdc4
       style QCS fill:#4ecdc4
       style LWCE fill:#45b7d1
       style QCE fill:#45b7d1
       style BSO fill:#96ceb4
       style STUDY fill:#feca57

End-to-End Execution Flow
~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Initialization and Setup**

When ``ConformalTuner.search()`` starts, it creates a ``Study`` object to track all trials and results. The study initializes a ``RuntimeTracker`` for timing and ``ProgressBarManager`` for user feedback. Parameter spaces are defined using ``ParameterRange`` objects (``IntRange``, ``FloatRange``, ``CategoricalRange``) which specify search bounds and types.

Configuration management happens through either ``StaticConfigurationManager`` (for predefined configurations) or ``DynamicConfigurationManager`` (for adaptive suggestions). The ``ConfigurationEncoder`` handles conversion between different parameter representations, while ``get_tuning_configurations()`` generates initial parameter samples using uniform or Sobol sequences.

**Step 2: Acquisition Function Setup**

The system selects between two main acquisition approaches:

* ``LocallyWeightedConformalSearcher`` - uses variance-adaptive prediction intervals
* ``QuantileConformalSearcher`` - uses direct quantile estimation

Both inherit from ``BaseConformalSearcher`` which provides the common interface for ``predict()``, ``update()``, and ``get_interval()`` methods.

**Conformal Estimator Initialization:**

``LocallyWeightedConformalEstimator`` implements a two-stage process:

.. code-block:: text

   LocallyWeightedConformalEstimator
   ├── Point Estimator (for conditional mean)
   ├── Variance Estimator (for conditional variance)
   └── Nonconformity Score Calculation

``QuantileConformalEstimator`` uses direct quantile estimation with conformal adjustment for coverage guarantees.

**Step 3: Data Processing Pipeline**

Raw input data flows through ``train_val_split()`` which creates training, validation, and calibration sets. The ``remove_iqr_outliers()`` function filters statistical outliers. This split data structure maintains proper separation required for conformal prediction coverage guarantees.

For ``LocallyWeightedConformalEstimator``, the training data gets further split:

* Point estimation subset → trains the mean predictor
* Variance estimation subset → trains the variance predictor using residuals from point predictor
* Validation set → generates nonconformity scores for conformal calibration

**Step 4: Hyperparameter Tuning Layer**

The tuning hierarchy works as follows:

.. code-block:: text

   RandomTuner (base class)
   ├── PointTuner (for point estimation)
   └── QuantileTuner (for quantile estimation)

``_tune_fit_component_estimator()`` handles the optimization process:

1. Checks if sufficient data exists for tuning (``min_obs_for_tuning`` threshold)
2. Uses ``initialize_estimator()`` to create estimator instances from ``ESTIMATOR_REGISTRY``
3. Performs cross-validation through ``_cross_validate()``
4. Returns fitted estimator and best hyperparameters

The ``ESTIMATOR_REGISTRY`` contains ``EstimatorConfig`` objects that define:

* Architecture identifiers
* Parameter ranges for hyperparameter search
* Default parameter values
* Estimator class references

**Step 5: Estimator Implementation Layer**

The system supports multiple quantile estimator types:

**Individual Quantile Estimators:**

* ``QuantileLasso`` - L1-regularized quantile regression
* ``QuantileGBM`` - Gradient boosting for quantile estimation
* ``QuantileForest`` - Random forest with quantile prediction
* ``QuantileKNN`` - K-nearest neighbors for quantile estimation
* ``GaussianProcessQuantileEstimator`` - Gaussian process with quantile likelihood

**Ensemble Estimators:**

* ``PointEnsembleEstimator`` - combines multiple point estimators using weighted averaging
* ``QuantileEnsembleEstimator`` - combines multiple quantile estimators

Both ensemble types use ``_fit_base_estimators()`` to train component models, then learn optimal weights for combination.

**Step 6: Acquisition Strategy Execution**

The ``BaseConformalSearcher.predict()`` method routes to strategy-specific implementations:

**Acquisition Function Hierarchy:**

.. code-block:: text

   Acquisition Strategies
   ├── LowerBoundSampler (Upper Confidence Bound)
   ├── PessimisticLowerBoundSampler (Conservative Lower Bound)
   ├── ThompsonSampler (Posterior Sampling)
   ├── ExpectedImprovementSampler (Expected Improvement)
   ├── EntropySearchSampler (Information Gain)
   └── MaxValueEntropySearchSampler (Maximum Value Entropy)

Each strategy calls specific methods:

* ``LowerBoundSampler`` → ``calculate_upper_confidence_bound()``
* ``ThompsonSampler`` → ``sample()`` and ``_update_posterior()``
* ``ExpectedImprovementSampler`` → ``_calculate_expected_improvement()``
* ``EntropySearchSampler`` → ``_calculate_entropy()``

All strategies use shared utilities from ``selection.sampling.utils``:

* ``initialize_quantile_alphas()`` - sets up alpha levels
* ``initialize_multi_adapters()`` / ``initialize_single_adapter()`` - configures adaptive mechanisms
* ``update_multi_interval_widths()`` / ``update_single_interval_width()`` - adjusts interval sizes
* ``flatten_conformal_bounds()`` - converts bounds to usable format

**Step 7: Conformal Prediction and Interval Generation**

The conformal estimators generate prediction intervals:

1. ``fit()`` method trains on calibration data
2. ``predict_intervals()`` generates ``ConformalBounds`` objects containing lower_bound, upper_bound, and alpha values
3. ``calculate_betas()`` computes coverage feedback for adaptive adjustment

**Step 8: Adaptive Feedback Loop**

After each evaluation, the system updates:

1. ``get_interval()`` retrieves prediction interval bounds for storage and analysis
2. ``_calculate_betas()`` computes coverage statistics
3. ``DtACI.update_alpha()`` adjusts significance levels based on coverage feedback
4. ``_calculate_pinball_loss()`` provides loss-based adaptation signals

**Step 9: Trial Management and Optimization**

Results flow back through the trial management system:

1. ``_evaluate_configuration()`` executes the objective function
2. ``add_trial()`` records results in the study
3. ``get_best_trial()`` retrieves current optimal configuration
4. ``_run_trials()`` continues the optimization loop

**Conformal Searcher Optimization**

All conformal searchers need to train on the configuration to performance pairs accumulated during search, but how should
we tune them? (tune the tuners, sounds circular I know). Decisions about how often to tune the searchers and how many
tuning trials to perform can be handled by the optimizers:

* ``BayesianSearcherOptimizer`` - fits Gaussian processes with ``_fit_gp()`` and suggests optimal retraining interval and number of tuning trials to perform.
* ``FixedSearcherOptimizer`` - always suggests the same retraining interval and number of tuning trials to perform.

There is also an option to not tune at all.
