Quantile Regression Estimators
==============================

The quantile estimation module (``confopt.selection.estimators.quantile_estimation``) provides comprehensive quantile regression implementations for distributional prediction. These estimators model conditional quantiles of the target distribution, enabling asymmetric uncertainty quantification essential for conformal prediction and robust optimization.

Overview
--------

Quantile regression extends traditional mean regression by estimating conditional quantiles :math:`Q_\tau(Y|X)` for various probability levels :math:`\tau \in (0,1)`. This approach captures the full conditional distribution rather than just the mean, providing richer uncertainty information for optimization under uncertainty.

The module implements two fundamental approaches:

- **Multi-fit Estimators**: Train separate models for each quantile level
- **Single-fit Estimators**: Model the complete conditional distribution in one step

Each approach offers different trade-offs between computational efficiency, quantile consistency, and modeling flexibility.

Mathematical Foundation
-----------------------

**Quantile Loss Function:**

Quantile regression minimizes the pinball loss (quantile loss):

.. math::

   L_\tau(y, \hat{q}) = (y - \hat{q})(\tau - \mathbf{1}[y < \hat{q}])

Where:
- :math:`y`: True target value
- :math:`\hat{q}`: Predicted quantile
- :math:`\tau`: Target quantile level
- :math:`\mathbf{1}[\cdot]`: Indicator function

**Asymmetric Penalty:**

The pinball loss provides asymmetric penalties:

- **Over-prediction** (:math:`\hat{q} > y`): Penalty of :math:`(1-\tau)(\hat{q} - y)`
- **Under-prediction** (:math:`\hat{q} < y`): Penalty of :math:`\tau(y - \hat{q})`

This asymmetry allows different costs for different types of errors, making quantile regression particularly suitable for risk-aware optimization.

Architecture
------------

.. mermaid::

   graph TD
       subgraph "Quantile Estimation Framework"
           BMQE["BaseMultiFitQuantileEstimator<br/>Separate Models per Quantile"]
           BSQE["BaseSingleFitQuantileEstimator<br/>Single Distribution Model"]
       end

       subgraph "Multi-Fit Implementations"
           QL["QuantileLasso<br/>Linear with L1 Regularization"]
           QG["QuantileGBM<br/>Gradient Boosting"]
           QLG["QuantileLightGBM<br/>LightGBM Backend"]
       end

       subgraph "Single-Fit Implementations"
           QF["QuantileForest<br/>Random Forest Distribution"]
           QK["QuantileKNN<br/>K-Nearest Neighbors"]
           GP["GaussianProcessQuantileEstimator<br/>Gaussian Process"]
           QLeaf["QuantileLeaf<br/>Leaf-based Estimation"]
       end

       subgraph "Integration Layer"
           QCE["QuantileConformalEstimator<br/>Conformal Prediction"]
           QEE["QuantileEnsembleEstimator<br/>Ensemble Methods"]
       end

       BMQE --> QL
       BMQE --> QG
       BMQE --> QLG

       BSQE --> QF
       BSQE --> QK
       BSQE --> GP
       BSQE --> QLeaf

       QL --> QCE
       QG --> QCE
       QLG --> QCE
       QF --> QCE
       QK --> QCE
       GP --> QCE

       QL --> QEE
       QG --> QEE
       QLG --> QEE
       QF --> QEE

Base Classes
------------

BaseMultiFitQuantileEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base for quantile estimators that train separate models for each quantile level. This approach provides maximum flexibility for algorithm-specific quantile optimization but requires multiple model fits.

**Key Features:**

- **Quantile-Specific Optimization**: Each model optimizes for its target quantile
- **Algorithm Flexibility**: Any regression algorithm can be adapted
- **Independent Fitting**: Quantile models are trained independently
- **Parallel Training**: Models can be trained in parallel for efficiency

**Core Methods:**

``fit(X, y, quantiles)``
   Trains separate models for each quantile level by iterating through quantiles and calling ``_fit_quantile_estimator()``.

``_fit_quantile_estimator(X, y, quantile)``
   Abstract method that subclasses must implement to fit a model for a specific quantile level.

``predict(X)``
   Generates predictions for all quantile levels by calling ``predict()`` on each trained model.

**Implementation Pattern:**

.. code-block:: python

   def _fit_quantile_estimator(self, X, y, quantile):
       # Configure algorithm for specific quantile
       model = self.create_model(quantile_level=quantile)
       model.fit(X, y)
       return model

**Advantages:**

- **Direct Optimization**: Each model directly optimizes its target quantile
- **Algorithm Agnostic**: Works with any regression algorithm
- **Robust Performance**: Poor performance at one quantile doesn't affect others
- **Interpretability**: Clear relationship between models and quantiles

**Disadvantages:**

- **Computational Cost**: Linear scaling with number of quantiles
- **Quantile Crossing**: No guarantee of monotonic quantile ordering
- **Memory Usage**: Stores multiple fitted models
- **Potential Inconsistency**: Different models may produce inconsistent results

BaseSingleFitQuantileEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base for quantile estimators that model the complete conditional distribution with a single model. Quantiles are then extracted from this distribution through sampling or analytical methods.

**Key Features:**

- **Distributional Modeling**: Captures full conditional distribution
- **Quantile Consistency**: Ensures monotonic quantile ordering
- **Computational Efficiency**: Single model training regardless of quantile count
- **Coherent Predictions**: All quantiles derived from same underlying model

**Core Methods:**

``fit(X, y, quantiles)``
   Trains a single model to capture the conditional distribution by calling ``_fit_implementation()``.

``_fit_implementation(X, y)``
   Abstract method for fitting the distributional model.

``_get_candidate_local_distribution(X)``
   Abstract method for extracting distribution samples for quantile computation.

``predict(X)``
   Generates quantile predictions by sampling from the fitted distribution and computing empirical quantiles.

**Implementation Pattern:**

.. code-block:: python

   def _fit_implementation(self, X, y):
       # Fit model to capture conditional distribution
       self.model = self.create_distributional_model()
       self.model.fit(X, y)

   def _get_candidate_local_distribution(self, X):
       # Generate samples from conditional distribution
       return self.model.sample_distribution(X)

**Advantages:**

- **Quantile Consistency**: Monotonic quantile ordering guaranteed
- **Computational Efficiency**: Single model training
- **Coherent Uncertainty**: Consistent uncertainty estimates across quantiles
- **Flexible Quantile Selection**: Can compute any quantile post-training

**Disadvantages:**

- **Distributional Assumptions**: Requires appropriate distributional model
- **Complex Implementation**: More complex than direct quantile fitting
- **Approximation Quality**: Quantile accuracy depends on distribution modeling
- **Limited Algorithm Support**: Not all algorithms support distributional modeling

Multi-Fit Implementations
-------------------------

QuantileLasso
~~~~~~~~~~~~~

Linear quantile regression with L1 regularization using statsmodels backend. Provides interpretable linear models with automatic feature selection through sparsity.

**Mathematical Framework:**

Minimizes the regularized pinball loss:

.. math::

   \min_\beta \sum_{i=1}^n L_\tau(y_i, x_i^T\beta) + \lambda \|\beta\|_1

**Key Features:**

- **Linear Interpretability**: Clear feature importance through coefficients
- **Automatic Feature Selection**: L1 penalty provides sparsity
- **Robust Convergence**: Reliable optimization through statsmodels
- **Intercept Handling**: Automatic intercept term management

**Implementation Details:**

``_fit_quantile_estimator(X, y, quantile)``
   Uses statsmodels QuantReg with automatic intercept detection and random state control.

**Use Cases:**

- **High-dimensional Problems**: Effective feature selection through sparsity
- **Interpretable Models**: Clear understanding of feature impacts
- **Linear Relationships**: When target-feature relationships are approximately linear
- **Baseline Models**: Simple and reliable quantile estimation

QuantileGBM
~~~~~~~~~~~

Gradient boosting quantile regression using scikit-learn's GradientBoostingRegressor with quantile loss. Provides non-linear quantile estimation with automatic feature selection.

**Mathematical Framework:**

Uses quantile loss in gradient boosting framework:

.. math::

   F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)

Where :math:`h_m(x)` is fitted to the negative gradient of the pinball loss.

**Key Features:**

- **Non-linear Modeling**: Captures complex feature interactions
- **Automatic Feature Selection**: Tree-based feature importance
- **Robust to Outliers**: Tree-based splits handle extreme values
- **Configurable Complexity**: Multiple hyperparameters for fine-tuning

**Implementation Details:**

``_fit_quantile_estimator(X, y, quantile)``
   Clones base GradientBoostingRegressor and sets alpha parameter to target quantile.

**Hyperparameters:**

- ``learning_rate``: Controls step size for gradient updates
- ``n_estimators``: Number of boosting stages
- ``max_depth``: Maximum tree depth for complexity control
- ``subsample``: Fraction of samples for stochastic boosting
- ``min_samples_split/leaf``: Regularization through minimum sample requirements

**Use Cases:**

- **Non-linear Relationships**: Complex feature interactions
- **Medium-sized Datasets**: Good balance of performance and interpretability
- **Robust Predictions**: Handling of outliers and noise
- **Feature Importance**: Understanding of feature contributions

QuantileLightGBM
~~~~~~~~~~~~~~~~

High-performance gradient boosting using LightGBM backend with quantile objective. Optimized for large datasets and fast training.

**Key Features:**

- **High Performance**: Optimized C++ implementation
- **Large Dataset Support**: Efficient memory usage and parallel training
- **Advanced Regularization**: Multiple regularization techniques
- **GPU Support**: Optional GPU acceleration for large-scale problems

**Implementation Details:**

Uses LightGBM's built-in quantile objective with automatic parameter management and early stopping support.

**Advantages over QuantileGBM:**

- **Speed**: 2-10x faster training on large datasets
- **Memory Efficiency**: Better memory usage for high-dimensional data
- **Advanced Features**: Built-in feature importance and validation
- **Production Ready**: Optimized for deployment scenarios

**Use Cases:**

- **Large Datasets**: > 10K samples with good performance
- **High-dimensional Data**: Efficient handling of many features
- **Production Systems**: Fast inference and reliable performance
- **Competitive Performance**: State-of-the-art quantile estimation

Single-Fit Implementations
--------------------------

QuantileForest
~~~~~~~~~~~~~~

Random forest-based quantile estimation using leaf statistics for distributional modeling. Provides robust non-parametric quantile estimation with natural uncertainty quantification.

**Mathematical Framework:**

For each leaf node, maintains statistics of training targets that fall into that leaf. Quantiles are computed from these empirical distributions:

.. math::

   \hat{q}_\tau(x) = \text{quantile}(\{y_i : x_i \text{ falls in same leaf as } x\}, \tau)

**Key Features:**

- **Non-parametric**: No distributional assumptions
- **Robust to Outliers**: Tree-based splits handle extreme values
- **Natural Uncertainty**: Leaf statistics provide uncertainty estimates
- **Consistent Quantiles**: Monotonic ordering guaranteed by empirical quantiles

**Implementation Details:**

``_fit_implementation(X, y)``
   Fits random forest and stores leaf indices and target statistics for each leaf.

``_get_candidate_local_distribution(X)``
   For each prediction point, finds corresponding leaf and returns target values from training data in that leaf.

**Advantages:**

- **Simplicity**: Straightforward implementation and interpretation
- **Robustness**: Handles complex data distributions naturally
- **Consistency**: Guaranteed monotonic quantile ordering
- **Uncertainty Quantification**: Natural confidence estimates

**Limitations:**

- **Data Requirements**: Needs sufficient samples per leaf
- **Smoothness**: Predictions can be discontinuous at leaf boundaries
- **Memory Usage**: Stores training data for leaf statistics
- **Extrapolation**: Limited ability to extrapolate beyond training data

QuantileKNN
~~~~~~~~~~~

K-nearest neighbors quantile estimation using local neighborhood statistics. Provides adaptive quantile estimation based on local data density.

**Mathematical Framework:**

For each prediction point, finds k nearest neighbors and computes empirical quantiles:

.. math::

   \hat{q}_\tau(x) = \text{quantile}(\{y_i : x_i \in \text{k-NN}(x)\}, \tau)

**Key Features:**

- **Local Adaptation**: Quantiles adapt to local data characteristics
- **Non-parametric**: No global distributional assumptions
- **Simple Implementation**: Straightforward algorithm with few hyperparameters
- **Consistent Results**: Empirical quantiles ensure monotonic ordering

**Implementation Details:**

Uses scikit-learn's NearestNeighbors for efficient neighbor search and computes empirical quantiles from neighbor targets.

**Hyperparameters:**

- ``n_neighbors``: Number of neighbors for local estimation
- ``weights``: Uniform or distance-based weighting
- ``metric``: Distance metric for neighbor search

**Use Cases:**

- **Local Patterns**: When quantiles vary significantly across input space
- **Small Datasets**: Effective with limited training data
- **Smooth Functions**: When underlying function is locally smooth
- **Baseline Method**: Simple and interpretable quantile estimation

GaussianProcessQuantileEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gaussian process-based quantile estimation using posterior distribution sampling. Provides principled uncertainty quantification with theoretical guarantees.

**Mathematical Framework:**

Models the conditional mean and uncertainty using Gaussian process:

.. math::

   f(x) \sim \mathcal{GP}(\mu(x), k(x, x'))

Quantiles are computed by sampling from the posterior distribution and adding noise.

**Key Features:**

- **Principled Uncertainty**: Theoretical foundation for uncertainty quantification
- **Flexible Kernels**: Various kernel functions for different smoothness assumptions
- **Calibrated Uncertainty**: Well-calibrated prediction intervals
- **Small Data Efficiency**: Effective with limited training data

**Implementation Details:**

``_fit_implementation(X, y)``
   Fits Gaussian process regressor with specified kernel and noise level.

``_get_candidate_local_distribution(X)``
   Samples from GP posterior and adds noise to generate distribution samples.

**Kernel Options:**

- **RBF**: Smooth functions with infinite differentiability
- **Matern**: Controlled smoothness with finite differentiability
- **RationalQuadratic**: Multi-scale patterns
- **ExpSineSquared**: Periodic patterns

**Use Cases:**

- **Small Datasets**: Excellent performance with limited data
- **Smooth Functions**: When underlying function is smooth
- **Uncertainty Quantification**: When calibrated uncertainty is crucial
- **Bayesian Framework**: When probabilistic interpretation is important

Performance Characteristics
---------------------------

**Computational Complexity:**

**Multi-fit Estimators:**
- **Training**: O(|quantiles| × base_algorithm_cost)
- **Prediction**: O(|quantiles| × base_prediction_cost)
- **Memory**: O(|quantiles| × model_size)

**Single-fit Estimators:**
- **Training**: O(base_algorithm_cost)
- **Prediction**: O(sampling_cost + quantile_computation)
- **Memory**: O(model_size + distribution_samples)

**Scalability Comparison:**

.. list-table:: Algorithm Scalability
   :header-rows: 1

   * - Algorithm
     - Training Time
     - Prediction Time
     - Memory Usage
     - Data Size Limit
   * - QuantileLasso
     - O(np)
     - O(p)
     - O(p)
     - Large
   * - QuantileGBM
     - O(n log n × trees)
     - O(trees)
     - O(trees)
     - Medium
   * - QuantileLightGBM
     - O(n × features)
     - O(trees)
     - O(trees)
     - Very Large
   * - QuantileForest
     - O(n log n × trees)
     - O(trees)
     - O(n)
     - Medium
   * - QuantileKNN
     - O(n)
     - O(k log n)
     - O(n)
     - Medium
   * - GaussianProcess
     - O(n³)
     - O(n)
     - O(n²)
     - Small

Integration with Conformal Prediction
-------------------------------------

Quantile estimators integrate seamlessly with conformal prediction through the ``QuantileConformalEstimator``:

**Conformalized Mode:**

When sufficient calibration data is available, quantile predictions are adjusted using conformal calibration:

.. math::

   \text{Final Interval} = [\hat{q}_{\alpha/2}(x) - C_\alpha, \hat{q}_{1-\alpha/2}(x) + C_\alpha]

**Non-conformalized Mode:**

With limited data, raw quantile predictions provide intervals:

.. math::

   \text{Final Interval} = [\hat{q}_{\alpha/2}(x), \hat{q}_{1-\alpha/2}(x)]

**Algorithm Selection Guidelines:**

- **QuantileLightGBM**: Default choice for most problems
- **GaussianProcess**: Small datasets (< 1000 samples)
- **QuantileForest**: When interpretability is important
- **QuantileLasso**: High-dimensional, sparse problems
- **QuantileKNN**: Local patterns, irregular distributions

Best Practices
---------------

**Algorithm Selection:**

- **Dataset Size**: GP for small, LightGBM for large datasets
- **Interpretability**: Lasso for linear, Forest for non-linear interpretability
- **Performance**: LightGBM for best predictive performance
- **Robustness**: Forest or KNN for robust non-parametric estimation

**Hyperparameter Tuning:**

- **Cross-validation**: Use quantile-aware CV with pinball loss
- **Multi-quantile Evaluation**: Optimize across all required quantiles
- **Regularization**: Balance overfitting vs. underfitting
- **Computational Budget**: Consider training time constraints

**Data Preprocessing:**

- **Feature Scaling**: Important for distance-based methods (KNN, GP)
- **Outlier Handling**: Consider robust preprocessing for extreme values
- **Missing Values**: Handle appropriately for tree-based methods
- **Feature Engineering**: Create relevant features for quantile modeling

**Common Issues:**

- **Quantile Crossing**: Multi-fit methods may produce non-monotonic quantiles
- **Insufficient Data**: Single-fit methods may struggle with sparse data
- **Computational Cost**: Multi-fit scaling with number of quantiles
- **Hyperparameter Sensitivity**: Some methods require careful tuning

**Quality Assessment:**

- **Coverage Analysis**: Check empirical coverage vs. theoretical levels
- **Pinball Loss**: Evaluate quantile-specific prediction quality
- **Interval Width**: Balance coverage with interval efficiency
- **Quantile Consistency**: Verify monotonic quantile ordering

The quantile estimation framework provides comprehensive tools for distributional modeling in conformal optimization, enabling robust uncertainty quantification and efficient optimization under uncertainty.
