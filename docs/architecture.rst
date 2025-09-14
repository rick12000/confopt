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
  * ``utils.preprocessing``: Data splitting utilities
  * ``utils.tracking``: Experiment management and progress monitoring
  * ``utils.optimization``: Searcher optimization algorithms
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
           CT["ConformalTuner<br/>tune()<br/>random_search()<br/>conformal_search()<br/>_evaluate_configuration()"]
           STOP["stop_search()<br/>check_objective_function()"]
       end

       subgraph "Experiment Management"
           STUDY["Study<br/>append_trial()<br/>batch_append_trials()<br/>get_best_configuration()<br/>get_best_performance()<br/>get_searched_configurations()<br/>get_searched_performances()<br/>get_average_target_model_runtime()"]
           TRIAL["Trial<br/>iteration<br/>timestamp<br/>configuration<br/>performance<br/>acquisition_source<br/>lower_bound<br/>upper_bound<br/>searcher_runtime<br/>target_model_runtime"]
           RT["RuntimeTracker<br/>pause_runtime()<br/>resume_runtime()<br/>return_runtime()"]
           PBM["ProgressBarManager<br/>create_progress_bar()<br/>update_progress()<br/>close_progress_bar()"]
       end

       subgraph "Configuration Management"
           BCM["BaseConfigurationManager<br/>mark_as_searched()<br/>tabularize_configs()<br/>listify_configs()<br/>add_to_banned_configurations()"]
           SCM["StaticConfigurationManager<br/>get_searchable_configurations()<br/>get_searchable_configurations_count()"]
           DCM["DynamicConfigurationManager<br/>get_searchable_configurations()<br/>get_searchable_configurations_count()"]
           CE["ConfigurationEncoder<br/>transform()<br/>_build_encoding_schema()<br/>_create_feature_matrix()"]
           GTC["get_tuning_configurations()<br/>_uniform_sampling()<br/>_sobol_sampling()"]
           CCH["create_config_hash()<br/>hash_generation()"]
       end

       subgraph "Acquisition Layer"
           BCS["BaseConformalSearcher<br/>predict()<br/>update()<br/>get_interval()<br/>_calculate_betas()"]
           QCS["QuantileConformalSearcher<br/>fit()<br/>_predict_with_ucb()<br/>_predict_with_thompson()<br/>_predict_with_pessimistic_lower_bound()<br/>_predict_with_expected_improvement()"]
       end

       subgraph "Conformal Prediction"
           QCE["QuantileConformalEstimator<br/>fit()<br/>predict_intervals()<br/>calculate_betas()<br/>update_alphas()<br/>_fit_non_conformal()<br/>_fit_cv_plus()<br/>_fit_train_test_split()"]
           DTACI["DtACI<br/>update()<br/>pinball_loss()"]
           SACS["set_calibration_split()<br/>alpha_to_quantiles()"]
       end

       subgraph "Hyperparameter Tuning"
           RT_TUNER["RandomTuner<br/>tune()<br/>_create_fold_indices()<br/>_score_configurations()<br/>_fit_model()<br/>_evaluate_model()"]
           PT["PointTuner<br/>tune()<br/>_fit_model()<br/>_evaluate_model()"]
           QT["QuantileTuner<br/>_fit_model()<br/>_evaluate_model()"]
           IE["initialize_estimator()<br/>estimator_creation()"]
           ASCF["average_scores_across_folds()<br/>score_aggregation()"]
       end

       subgraph "Estimator Registry"
           ER["ESTIMATOR_REGISTRY<br/>rf, gbm, kr, knn<br/>qgbm, qrf, qknn, ql, qgp, qleaf<br/>qens1, qens2, qens3, qens4, qens5"]
           EC["EstimatorConfig<br/>estimator_name<br/>estimator_class<br/>default_params<br/>estimator_parameter_space<br/>ensemble_components<br/>is_ensemble_estimator()<br/>is_quantile_estimator()"]
       end

       subgraph "Quantile Estimators"
           BMFQE["BaseMultiFitQuantileEstimator<br/>fit()<br/>_fit_quantile_estimator()"]
           BSFQE["BaseSingleFitQuantileEstimator<br/>fit()<br/>_fit_implementation()"]
           QL["QuantileLasso<br/>fit()<br/>predict_quantiles()"]
           QG["QuantileGBM<br/>fit()<br/>predict_quantiles()"]
           QF["QuantileForest<br/>fit()<br/>predict_quantiles()"]
           QK["QuantileKNN<br/>fit()<br/>predict_quantiles()"]
           QGP["QuantileGP<br/>fit()<br/>predict_quantiles()"]
           QLeaf["QuantileLeaf<br/>fit()<br/>predict_quantiles()"]
       end

       subgraph "Ensemble Methods"
           BEE["BaseEnsembleEstimator<br/>fit()<br/>predict()"]
           PEE["PointEnsembleEstimator<br/>fit()<br/>predict()<br/>_compute_point_weights()<br/>_compute_linear_stack_weights()<br/>_get_stacking_training_data()"]
           QEE["QuantileEnsembleEstimator<br/>fit()<br/>predict()<br/>_compute_quantile_weights()<br/>_compute_linear_stack_weights()<br/>_get_stacking_training_data()"]
           QLM["QuantileLassoMeta<br/>fit()<br/>predict()<br/>_quantile_loss_objective()"]
       end

       subgraph "Sampling Strategies"
           LBS["LowerBoundSampler<br/>calculate_ucb_predictions()<br/>update_exploration_step()<br/>fetch_alphas()<br/>update_interval_width()"]
           PLBS["PessimisticLowerBoundSampler<br/>fetch_alphas()<br/>update_interval_width()"]
           TS["ThompsonSampler<br/>calculate_thompson_predictions()<br/>fetch_alphas()<br/>update_interval_width()"]
           EIS["ExpectedImprovementSampler<br/>calculate_expected_improvement()<br/>update_best_value()<br/>fetch_alphas()<br/>update_interval_width()"]
       end

       subgraph "Sampling Utilities"
           IQA["initialize_quantile_alphas()<br/>alpha_generation()"]
           IMA["initialize_multi_adapters()<br/>adapter_creation()"]
           ISA["initialize_single_adapter()<br/>single_adapter_setup()"]
           UMIW["update_multi_interval_widths()<br/>width_updates()"]
           USIW["update_single_interval_width()<br/>single_width_update()"]
           FCB["flatten_conformal_bounds()<br/>bounds_flattening()"]
           VEQ["validate_even_quantiles()<br/>quantile_validation()"]
       end

       subgraph "Data Processing"
           TVS["train_val_split()<br/>data_splitting()"]
       end

       subgraph "Searcher Optimization"
           DSO["DecayingSearcherOptimizer<br/>select_arm()<br/>update()<br/>_calculate_current_interval()"]
           FSO["FixedSearcherOptimizer<br/>select_arm()<br/>update()"]
       end

       subgraph "Parameter Structures"
           PR["ParameterRange<br/>IntRange<br/>FloatRange<br/>CategoricalRange"]
           CB["ConformalBounds<br/>lower_bounds<br/>upper_bounds"]
       end

       %% Main Flow Connections
       CT --> STUDY
       CT --> RT
       CT --> PBM
       CT --> SCM
       CT --> DCM
       CT --> QCS
       CT --> TVS
       CT --> DSO
       CT --> FSO
       CT --> STOP

       %% Configuration Management Flow
       STUDY --> TRIAL
       STUDY --> CE
       STUDY --> GTC
       STUDY --> CCH
       BCM --> SCM
       BCM --> DCM
       SCM --> GTC
       DCM --> GTC
       DCM --> DSO

       %% Acquisition Flow
       QCS --> QCE
       BCS --> LBS
       BCS --> PLBS
       BCS --> TS
       BCS --> EIS

       %% Conformal Prediction Flow
       QCE --> QT
       QCE --> IE
       QCE --> DTACI
       QCE --> SACS
       QCS --> SACS
       DTACI --> SACS

       %% Hyperparameter Tuning Flow
       RT_TUNER --> IE
       RT_TUNER --> ASCF
       PT --> RT_TUNER
       PT --> ER
       QT --> RT_TUNER
       QT --> ER
       IE --> ER
       IE --> EC

       %% Estimator Flow
       ER --> EC
       EC --> BMFQE
       EC --> BSFQE
       BMFQE --> QL
       BMFQE --> QG
       BSFQE --> QF
       BSFQE --> QK
       BSFQE --> QGP
       BSFQE --> QLeaf
       EC --> BEE
       BEE --> PEE
       BEE --> QEE

       %% Ensemble Flow
       PEE --> BMFQE
       PEE --> BSFQE
       QEE --> BMFQE
       QEE --> BSFQE
       QEE --> QLM

       %% Sampling Utilities Flow
       LBS --> IQA
       LBS --> VEQ
       PLBS --> IQA
       PLBS --> VEQ
       TS --> IQA
       TS --> IMA
       TS --> ISA
       TS --> VEQ
       EIS --> IQA
       EIS --> UMIW
       EIS --> USIW
       EIS --> VEQ

       %% Adaptive Flow
       IMA --> DTACI
       ISA --> DTACI
       UMIW --> DTACI
       USIW --> DTACI

       %% Data Structure Flow
       CT --> PR
       QCE --> CB
       LBS --> CB
       PLBS --> CB
       TS --> CB
       EIS --> CB

       %% Styling
       style CT fill:#ff6b6b
       style QCS fill:#4ecdc4
       style QCE fill:#45b7d1
       style DSO fill:#96ceb4
       style STUDY fill:#feca57

End-to-End Execution Flow
~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Initialization and Setup**

When ``ConformalTuner.tune()`` starts, it creates a ``Study`` object to track all trials and results. The study initializes a ``RuntimeTracker`` for timing and ``ProgressBarManager`` for user feedback. Parameter spaces are defined using ``ParameterRange`` objects (``IntRange``, ``FloatRange``, ``CategoricalRange``) which specify search bounds and types.

Configuration management happens through either ``StaticConfigurationManager`` (for predefined configurations) or ``DynamicConfigurationManager`` (for adaptive suggestions). The ``ConfigurationEncoder`` handles conversion between different parameter representations, while ``get_tuning_configurations()`` generates initial parameter samples using uniform or Sobol sequences.

**Step 2: Acquisition Function Setup**

The system uses quantile-based conformal prediction for acquisition:

* ``QuantileConformalSearcher`` - uses direct quantile estimation

This inherits from ``BaseConformalSearcher`` which provides the common interface for ``predict()``, ``update()``, and ``get_interval()`` methods.

**Conformal Estimator Initialization:**

``QuantileConformalEstimator`` implements quantile-based conformal prediction using direct quantile estimation with conformal adjustment for coverage guarantees.

**Step 3: Data Processing Pipeline**

Raw input data flows through ``train_val_split()`` which creates training, validation, and calibration sets. This split data structure maintains proper separation required for conformal prediction coverage guarantees.

For ``QuantileConformalEstimator``, the training data gets processed as:

* Quantile estimation → trains quantile regression models for prediction intervals
* Validation set → generates nonconformity scores for conformal calibration

**Step 4: Hyperparameter Tuning Layer**

The tuning hierarchy works as follows:

.. code-block:: text

   RandomTuner (base class)
   ├── PointTuner (for point estimation)
   └── QuantileTuner (for quantile estimation)

``tune()`` handles the optimization process:

1. Creates cross-validation folds through ``_create_fold_indices()``
2. Scores configurations using ``_score_configurations()``
3. Uses ``initialize_estimator()`` to create estimator instances from ``ESTIMATOR_REGISTRY``
4. Performs cross-validation through ``_fit_model()`` and ``_evaluate_model()``
5. Aggregates results using ``average_scores_across_folds()``
6. Returns fitted estimator and best hyperparameters

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
* ``QuantileGP`` - Gaussian process with quantile likelihood
* ``QuantileLeaf`` - Leaf-based quantile estimation

**Ensemble Estimators:**

* ``BaseEnsembleEstimator`` - abstract base class for ensemble methods
* ``PointEnsembleEstimator`` - combines multiple point estimators using weighted averaging with uniform or linear stacking strategies
* ``QuantileEnsembleEstimator`` - combines multiple quantile estimators using uniform or linear stacking approaches
* ``QuantileLassoMeta`` - specialized meta-learner for quantile ensemble optimization using Lasso regression

Ensemble implementations support multiple weighting strategies:
- Uniform weighting for simple averaging
- Linear stacking with cross-validation optimization
- Lasso-based meta-learning for optimal weight computation

**Step 6: Acquisition Strategy Execution**

The ``BaseConformalSearcher.predict()`` method routes to strategy-specific implementations:

**Acquisition Function Hierarchy:**

.. code-block:: text

   Acquisition Strategies
   ├── LowerBoundSampler (Upper Confidence Bound)
   ├── PessimisticLowerBoundSampler (Conservative Lower Bound)
   ├── ThompsonSampler (Posterior Sampling)
   └── ExpectedImprovementSampler (Expected Improvement)

Each strategy calls specific methods:

* ``LowerBoundSampler`` → ``calculate_ucb_predictions()``
* ``ThompsonSampler`` → ``calculate_thompson_predictions()``
* ``ExpectedImprovementSampler`` → ``_calculate_expected_improvement()``


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
3. ``DtACI.update()`` adjusts significance levels based on coverage feedback
4. ``pinball_loss()`` provides loss-based adaptation signals

**Step 9: Trial Management and Optimization**

Results flow back through the trial management system:

1. ``_evaluate_configuration()`` executes the objective function
2. ``append_trial()`` records results in the study
3. ``get_best_configuration()`` retrieves current optimal configuration
4. ``conformal_search()`` continues the optimization loop

**Conformal Searcher Optimization**

All conformal searchers require training on the accumulated configuration-to-performance pairs during search. The system provides different optimization strategies for determining when and how frequently to retrain the searchers:

* ``DecayingSearcherOptimizer`` - increases tuning intervals over time using linear, exponential, or logarithmic decay functions
* ``FixedSearcherOptimizer`` - maintains constant retraining intervals and tuning trial counts

The system also supports disabling searcher optimization entirely for simpler use cases.
