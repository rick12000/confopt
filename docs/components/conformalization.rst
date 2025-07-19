Conformal Prediction Estimators
===============================

The conformalization module (``confopt.selection.conformalization``) implements the core conformal prediction estimators that provide uncertainty quantification with finite-sample coverage guarantees. These estimators bridge machine learning models with statistical inference, enabling reliable prediction intervals for optimization under uncertainty.

Overview
--------

Conformal prediction provides a distribution-free framework for uncertainty quantification that maintains valid coverage guarantees regardless of the underlying data distribution. The module implements two complementary approaches:

- **LocallyWeightedConformalEstimator**: Two-stage approach with separate point and variance estimation
- **QuantileConformalEstimator**: Direct quantile estimation with optional conformal adjustment

Both estimators integrate seamlessly with the acquisition function framework, providing prediction intervals that guide optimization while maintaining statistical validity.

Mathematical Foundation
-----------------------

Conformal prediction relies on the exchangeability assumption to provide finite-sample coverage guarantees. The general framework follows these steps:

1. **Data Splitting**: Divide data into training and calibration sets
2. **Model Fitting**: Train prediction model on training set
3. **Nonconformity Computation**: Calculate nonconformity scores on calibration set
4. **Interval Construction**: Use score quantiles to build prediction intervals

**Coverage Guarantee:**

For any finite sample size n and miscoverage level α ∈ (0,1):

.. math::

   P(Y_{n+1} \in \hat{C}_{n+1}(X_{n+1})) \geq 1 - \alpha

This guarantee holds without assumptions about the data distribution, making conformal prediction particularly valuable for optimization where distributional assumptions may be violated.

Architecture
------------

.. mermaid::

   graph TD
       subgraph "Conformal Estimators"
           LWCE["LocallyWeightedConformalEstimator<br/>Two-Stage Estimation"]
           QCE["QuantileConformalEstimator<br/>Direct Quantile Estimation"]
       end

       subgraph "Component Estimators"
           PE["Point Estimator<br/>μ̂(x) = E[Y|X=x]"]
           VE["Variance Estimator<br/>σ̂²(x) = E[r²|X=x]"]
           QE["Quantile Estimator<br/>q̂_τ(x) = Q_τ[Y|X=x]"]
       end

       subgraph "Hyperparameter Tuning"
           PT["PointTuner<br/>Point Estimator Optimization"]
           QT["QuantileTuner<br/>Quantile Estimator Optimization"]
           ER["ESTIMATOR_REGISTRY<br/>Architecture Configurations"]
       end

       subgraph "Nonconformity Computation"
           NS["Nonconformity Scores<br/>R_i = f(y_i, ŷ_i, σ̂_i)"]
           QS["Quantile Scores<br/>R_i = max(q̂_α/2 - y_i, y_i - q̂_{1-α/2})"]
       end

       LWCE --> PE
       LWCE --> VE
       QCE --> QE

       PE --> PT
       VE --> PT
       QE --> QT

       PT --> ER
       QT --> ER

       LWCE --> NS
       QCE --> QS

LocallyWeightedConformalEstimator
---------------------------------

Implements locally weighted conformal prediction that adapts prediction intervals to local variance patterns in the objective function. This two-stage approach excels when the prediction uncertainty varies significantly across the input space.

**Mathematical Framework:**

The estimator implements heteroscedastic conformal prediction through variance-weighted nonconformity scores:

1. **Point Estimation**: :math:`\hat{\mu}(x) = E[Y|X=x]` using any regression algorithm
2. **Residual Computation**: :math:`r_i = |y_i - \hat{\mu}(x_i)|` for variance training data
3. **Variance Estimation**: :math:`\hat{\sigma}^2(x) = E[r^2|X=x]` using residuals as targets
4. **Nonconformity Scores**: :math:`R_i = \frac{|y_{val,i} - \hat{\mu}(x_{val,i})|}{\max(\hat{\sigma}(x_{val,i}), \epsilon)}`
5. **Interval Construction**: :math:`[\hat{\mu}(x) \pm q_{1-\alpha}(R) \times \hat{\sigma}(x)]`

**Key Features:**

- **Heteroscedastic Adaptation**: Intervals adapt to local prediction uncertainty
- **Dual Architecture**: Independent optimization of point and variance estimators
- **Warm-starting**: Reuses previous best parameters for efficient retraining
- **Robust Calibration**: Handles edge cases with minimum variance thresholds

**Implementation Details:**

``__init__(point_estimator_architecture, variance_estimator_architecture, alphas)``
   Initializes with separate architectures for point and variance estimation.

``fit(X_train, y_train, X_val, y_val, tuning_iterations, ...)``
   Implements the complete three-stage fitting process with optional hyperparameter tuning.

**Three-Stage Fitting Process:**

**Stage 1: Point Estimation**
   - Split training data: 75% for point estimation, 25% for variance estimation
   - Fit point estimator on point estimation subset
   - Optionally tune hyperparameters using cross-validation

**Stage 2: Variance Estimation**
   - Compute absolute residuals on variance estimation subset
   - Fit variance estimator using residuals as targets
   - Handle zero-variance regions with minimum threshold

**Stage 3: Conformal Calibration**
   - Compute nonconformity scores on validation set
   - Store scores for quantile computation during prediction
   - Track estimation quality metrics

**Core Methods:**

``predict_intervals(X)``
   Generates prediction intervals for new inputs using locally weighted conformal methodology.

   **Algorithm Steps:**

   1. **Point Prediction**: :math:`\hat{\mu}(x) = \text{point\_estimator.predict}(x)`
   2. **Variance Prediction**: :math:`\hat{\sigma}^2(x) = \text{variance\_estimator.predict}(x)`
   3. **Quantile Computation**: :math:`q_{1-\alpha} = \text{quantile}(\text{nonconformity\_scores}, 1-\alpha)`
   4. **Interval Construction**: :math:`[\hat{\mu}(x) - q_{1-\alpha} \hat{\sigma}(x), \hat{\mu}(x) + q_{1-\alpha} \hat{\sigma}(x)]`

``_tune_fit_component_estimator(X, y, estimator_architecture, ...)``
   Handles hyperparameter tuning for component estimators with warm-starting support.

**Data Splitting Strategy:**

The estimator uses careful data splitting to maintain coverage guarantees:

- **Training Split**: 75% for point estimation, 25% for variance estimation
- **Validation Set**: Used exclusively for conformal calibration
- **Independence**: Ensures proper separation between fitting and calibration

**Performance Characteristics:**

- **Training Complexity**: O(n_train) for each component estimator
- **Prediction Complexity**: O(1) per prediction point
- **Memory Usage**: O(n_val) for storing nonconformity scores
- **Adaptation Quality**: Excellent for heteroscedastic objectives

QuantileConformalEstimator
--------------------------

Implements quantile-based conformal prediction that directly estimates prediction quantiles and optionally applies conformal adjustments. This approach is particularly effective for asymmetric uncertainty or when limited calibration data is available.

**Mathematical Framework:**

The estimator operates in two modes depending on data availability:

**Conformalized Mode** (sufficient data):
   1. **Quantile Estimation**: :math:`\hat{q}_\tau(x)` for required quantile levels
   2. **Nonconformity Computation**: :math:`R_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(x_i))`
   3. **Conformal Adjustment**: :math:`C_\alpha = \text{quantile}(R_{\text{cal}}, 1-\alpha)`
   4. **Final Intervals**: :math:`[\hat{q}_{\alpha/2}(x) - C_\alpha, \hat{q}_{1-\alpha/2}(x) + C_\alpha]`

**Non-conformalized Mode** (limited data):
   - **Direct Quantiles**: :math:`[\hat{q}_{\alpha/2}(x), \hat{q}_{1-\alpha/2}(x)]`
   - **No Adjustment**: Uses raw quantile predictions without calibration

**Key Features:**

- **Asymmetric Intervals**: Naturally handles asymmetric prediction uncertainty
- **Automatic Mode Selection**: Switches based on data availability threshold
- **Direct Quantile Modeling**: No separate variance estimation required
- **Flexible Architectures**: Supports both multi-fit and single-fit quantile estimators

**Implementation Details:**

``__init__(quantile_estimator_architecture, alphas, n_pre_conformal_trials=32)``
   Initializes with quantile architecture and conformalization threshold.

``fit(X_train, y_train, X_val, y_val, tuning_iterations, ...)``
   Trains quantile estimator and optionally applies conformal calibration.

**Mode Selection Logic:**

.. code-block:: python

   total_samples = len(X_train) + len(X_val)
   self.conformalize_predictions = total_samples >= self.n_pre_conformal_trials

**Quantile Architecture Support:**

The estimator integrates with various quantile regression implementations:

- **Multi-fit Estimators**: Train separate models for each quantile level
- **Single-fit Estimators**: Model full conditional distribution simultaneously
- **Ensemble Methods**: Combine multiple quantile estimators for robustness

**Core Methods:**

``predict_intervals(X)``
   Generates prediction intervals using quantile-based conformal methodology.

   **Conformalized Algorithm:**

   1. **Quantile Prediction**: Get all required quantiles from fitted estimator
   2. **Conformal Adjustment**: Add/subtract stored nonconformity quantiles
   3. **Interval Construction**: Build intervals with conformal guarantees

   **Non-conformalized Algorithm:**

   1. **Direct Quantiles**: Use raw quantile predictions as interval bounds
   2. **Symmetric Pairing**: Match lower and upper quantiles by alpha level

``calculate_betas(X, y_true)``
   Computes coverage feedback (beta values) for adaptive alpha updating.

**Upper Quantile Capping:**

For conservative acquisition strategies, the estimator supports upper quantile capping:

.. code-block:: python

   if self.upper_quantile_cap is not None:
       upper_bounds = np.minimum(upper_bounds, self.upper_quantile_cap)

**Performance Characteristics:**

- **Training Complexity**: O(|quantiles| × n_train) for multi-fit, O(n_train) for single-fit
- **Prediction Complexity**: O(|quantiles|) per prediction point
- **Memory Usage**: O(|alphas| × n_val) for nonconformity scores
- **Flexibility**: Excellent for asymmetric or complex uncertainty patterns

Integration with Hyperparameter Tuning
---------------------------------------

Both conformal estimators integrate with automated hyperparameter tuning through the estimation module:

**Point Estimator Tuning:**

``PointTuner`` optimizes component estimators using:

- **Cross-validation**: K-fold validation for robust parameter selection
- **Forced Configurations**: Includes defaults and warm-start parameters
- **Architecture Registry**: Leverages ESTIMATOR_REGISTRY for parameter spaces

**Quantile Estimator Tuning:**

``QuantileTuner`` optimizes quantile estimators using:

- **Multi-quantile Evaluation**: Optimizes across all required quantile levels
- **Pinball Loss**: Uses quantile-specific loss functions for evaluation
- **Ensemble Support**: Handles both individual and ensemble quantile estimators

**Warm-starting Strategy:**

Both estimators support efficient retraining through parameter reuse:

1. **Previous Best**: Reuse last optimal parameters as starting point
2. **Default Fallback**: Use architecture defaults when no previous parameters
3. **Incremental Updates**: Minimize retraining cost during optimization

Coverage Guarantees and Validation
-----------------------------------

**Finite-Sample Validity:**

Both estimators provide exact finite-sample coverage guarantees:

.. math::

   P(Y_{n+1} \in \hat{C}_{n+1}(X_{n+1})) \geq 1 - \alpha

This holds for any sample size and any data distribution, making the methods suitable for safety-critical applications.

**Coverage Monitoring:**

The estimators support empirical coverage validation through beta calculation:

.. math::

   \beta_t = \frac{1}{n_{\text{cal}}} \sum_{i=1}^{n_{\text{cal}}} \mathbf{1}[R_i^{\text{cal}} \geq R_t]

Where high β indicates "easy" predictions (tighten intervals) and low β indicates "hard" predictions (widen intervals).

**Adaptive Coverage:**

Integration with DtACI adaptation allows dynamic coverage control:

- **Alpha Updates**: Adjust miscoverage levels based on empirical performance
- **Interval Optimization**: Balance coverage guarantees with interval efficiency
- **Non-stationarity**: Adapt to changing objective function characteristics

Best Practices
---------------

**Estimator Selection:**

- **LocallyWeighted**: Use when objective has heteroscedastic noise
- **Quantile**: Use for asymmetric uncertainty or limited calibration data
- **Architecture Choice**: Match estimator complexity to problem characteristics

**Data Splitting:**

- **Validation Size**: Use 20-30% of data for conformal calibration
- **Training Split**: LocallyWeighted uses additional internal splitting
- **Minimum Samples**: Ensure sufficient data for reliable calibration

**Hyperparameter Tuning:**

- **Tuning Iterations**: Balance search thoroughness with computational cost
- **Warm-starting**: Leverage previous parameters for efficient retraining
- **Architecture Registry**: Use registered configurations for consistent results

**Common Issues:**

- **Insufficient Calibration Data**: Leads to unreliable coverage guarantees
- **Extreme Variance**: LocallyWeighted may struggle with zero-variance regions
- **Quantile Crossing**: Some quantile estimators may produce inconsistent quantiles
- **Mode Selection**: Quantile estimator threshold affects coverage vs. efficiency trade-off

**Performance Optimization:**

- **Caching**: Reuse fitted models when possible
- **Batch Prediction**: Vectorize interval computation for efficiency
- **Memory Management**: Monitor nonconformity score storage for large datasets
- **Parallel Tuning**: Leverage parallel hyperparameter search when available

Integration with Optimization Framework
----------------------------------------

The conformal estimators integrate seamlessly with the broader optimization framework:

**Acquisition Function Interface:**

1. **Initialization**: Searcher creates estimator with appropriate architecture
2. **Fitting**: Estimator trains on accumulated optimization data
3. **Prediction**: Provides intervals for acquisition function evaluation
4. **Adaptation**: Updates alpha values based on coverage feedback

**Data Flow:**

.. mermaid::

   sequenceDiagram
       participant Tuner
       participant Searcher
       participant Estimator
       participant ComponentModel

       Tuner->>Searcher: fit(X_train, y_train, X_val, y_val)
       Searcher->>Estimator: fit() with hyperparameter tuning
       Estimator->>ComponentModel: tune and fit component models
       ComponentModel-->>Estimator: fitted models
       Estimator-->>Searcher: calibrated conformal estimator

       loop Optimization
           Tuner->>Searcher: predict(X_candidates)
           Searcher->>Estimator: predict_intervals(X_candidates)
           Estimator-->>Searcher: ConformalBounds objects
           Searcher-->>Tuner: acquisition values

           Tuner->>Searcher: update(X_selected, y_observed)
           Searcher->>Estimator: calculate_betas() for coverage feedback
           Estimator-->>Searcher: beta values for adaptation
       end

**Quality Metrics:**

Both estimators track performance metrics for monitoring:

- **Primary Estimator Error**: MSE for LocallyWeighted, mean pinball loss for Quantile
- **Coverage Rates**: Empirical coverage vs. target levels
- **Interval Widths**: Average interval width for efficiency assessment
- **Adaptation History**: Evolution of alpha values over time

This comprehensive integration enables reliable uncertainty quantification throughout the optimization process while maintaining both statistical validity and computational efficiency.
