Quantile Estimation Module
==========================

Overview
--------

The ``confopt.selection.estimators.quantile_estimation`` module provides comprehensive quantile regression implementations for distributional prediction and uncertainty quantification. The module offers two distinct architectural approaches: multi-fit estimators that train separate models per quantile, and single-fit estimators that model the complete conditional distribution.

Key Features
------------

* **Dual Architecture Design**: Multi-fit and single-fit approaches for different use cases and computational constraints
* **Algorithm Diversity**: Gradient boosting, random forests, linear models, k-NN, and Gaussian processes
* **Monotonic Quantiles**: Single-fit estimators ensure proper quantile ordering through distributional modeling
* **Scalability Options**: Sparse approximations and batch processing for large-scale applications
* **Robust Implementations**: Extensive error handling and fallback mechanisms for production use

Architecture
------------

Base Class Hierarchy
~~~~~~~~~~~~~~~~~~~~

::

    ABC (Abstract Base Classes)
    ├── BaseMultiFitQuantileEstimator
    │   ├── QuantileLasso
    │   ├── QuantileGBM
    │   └── QuantileLightGBM
    └── BaseSingleFitQuantileEstimator
        ├── QuantileForest
        ├── QuantileLeaf
        ├── QuantileKNN
        └── GaussianProcessQuantileEstimator

Multi-Fit vs Single-Fit Approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-Fit Estimators (BaseMultiFitQuantileEstimator)**
    Train separate models for each quantile level using quantile-specific loss functions. Provides algorithm flexibility at increased computational cost.

**Single-Fit Estimators (BaseSingleFitQuantileEstimator)**
    Train one model capturing the full conditional distribution, then extract quantiles. Ensures monotonic ordering and computational efficiency.

Quantile Estimation Strategies
------------------------------

Multi-Fit Approach
~~~~~~~~~~~~~~~~~~

Each quantile level :math:`\\tau \\in [0,1]` trains an independent model :math:`f_\\tau(\\mathbf{x})` optimizing the pinball loss:

.. math::

    L_\\tau(y, \\hat{y}) = \\tau \\max(y - \\hat{y}, 0) + (1-\\tau) \\max(\\hat{y} - y, 0)

**Advantages:**
- Direct quantile optimization
- Algorithm-specific quantile loss support
- Flexible per-quantile hyperparameters

**Disadvantages:**
- Linear scaling with number of quantiles
- No guaranteed monotonic ordering
- Higher computational overhead

Single-Fit Approach
~~~~~~~~~~~~~~~~~~~

One model captures the conditional distribution :math:`p(y|\\mathbf{x})`, then quantiles are extracted:

.. math::

    Q_\\tau(\\mathbf{x}) = F^{-1}(\\tau | \\mathbf{x})

Where :math:`F^{-1}` is the inverse cumulative distribution function.

**Advantages:**
- Constant computational cost regardless of quantile count
- Guaranteed monotonic quantile ordering
- Natural uncertainty quantification

**Disadvantages:**
- Distributional assumptions (for some methods)
- Algorithm-specific implementation complexity

Algorithm Implementations
------------------------

Linear Methods
~~~~~~~~~~~~~

**QuantileLasso**
    Implements linear quantile regression with L1 regularization using statsmodels backend. Provides interpretable coefficients and automatic feature selection through the Lasso penalty.

.. code-block:: python

    estimator = QuantileLasso(
        max_iter=1000,
        p_tol=1e-6,
        random_state=42
    )
    estimator.fit(X, y, quantiles=[0.1, 0.5, 0.9])

Tree-Based Methods
~~~~~~~~~~~~~~~~~

**QuantileGBM**
    Gradient boosting with quantile loss using scikit-learn's GradientBoostingRegressor. Provides robust non-linear modeling with automatic feature interaction detection.

**QuantileLightGBM**
    LightGBM implementation offering faster training, categorical feature support, and advanced regularization options.

**Random Forest Approaches**

The module provides two distinct random forest implementations for quantile regression:

**QuantileForest (Ensemble Predictions)**
    Uses the distribution of tree predictions to estimate quantiles. Each tree provides a point prediction, and quantiles are computed from the ensemble of these predictions. This approach is computationally efficient and provides smooth uncertainty estimates.

**QuantileLeaf (Meinshausen 2006)**
    Implements the Quantile Regression Forest methodology from Meinshausen (2006). Instead of using tree predictions, it collects all raw training target values Y_i that fall into the same leaf nodes as the prediction point across all trees. Quantiles are then computed as empirical percentiles of this combined set of training targets.

.. math::

    \\mathcal{Y}(\\mathbf{x}) = \\{ Y_i \\,|\\, \\exists b \\in \\{1,...,B\\} \\text{ s.t. } X_i \\in L_b(\\mathbf{x}) \\text{ and } \\mathbf{x} \\in L_b(\\mathbf{x}) \\}

Where :math:`L_b(\\mathbf{x})` is the leaf node containing point :math:`\\mathbf{x}` in tree :math:`b`, and :math:`B` is the total number of trees.

**Key Differences:**

* **QuantileForest**: Uses ensemble of tree predictions → smoother, computationally efficient
* **QuantileLeaf**: Uses raw training targets from matching leaves → more faithful to local data distribution, especially effective with heteroscedastic noise

.. code-block:: python

    # Gradient boosting approach
    gbm_estimator = QuantileGBM(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # Standard random forest approach
    rf_estimator = QuantileForest(
        n_estimators=100,
        max_depth=10,
        max_features=0.8,
        random_state=42
    )

    # Meinshausen (2006) leaf-based approach
    qrf_estimator = QuantileLeaf(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42
    )

Non-Parametric Methods
~~~~~~~~~~~~~~~~~~~~~

**QuantileKNN**
    K-nearest neighbors using local empirical distributions. Provides natural adaptation to local data density and non-parametric uncertainty quantification.

**GaussianProcessQuantileEstimator**
    Gaussian process regression with both analytical and sampling-based quantile extraction. Includes sparse approximations for scalability.

.. code-block:: python

    # K-NN approach
    knn_estimator = QuantileKNN(n_neighbors=10)

    # Gaussian process with sparse approximation
    gp_estimator = GaussianProcessQuantileEstimator(
        kernel="matern",
        n_inducing_points=100,
        n_samples=1000,
        use_optimized_sampling=True,
        random_state=42
    )

Advanced Features
----------------

Gaussian Process Enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Sparse Approximations**
    K-means induced point selection for scalable GP inference on large datasets.

**Analytical Quantiles**
    Direct quantile computation from Gaussian posterior distributions, ensuring monotonicity.

**Batch Processing**
    Memory-efficient prediction for large-scale applications.

**Kernel Caching**
    Performance optimization through kernel object reuse.

.. code-block:: python

    # Large-scale GP configuration
    gp_estimator = GaussianProcessQuantileEstimator(
        kernel="rbf",
        n_inducing_points=500,  # Sparse approximation
        batch_size=1000,        # Memory management
        use_optimized_sampling=True,
        random_state=42
    )


**Custom Kernel Configuration**

.. code-block:: python

    from sklearn.gaussian_process.kernels import RBF, Matern

    # Composite kernel for complex patterns
    kernel = RBF(length_scale=2.0) + Matern(length_scale=1.5, nu=0.5)

    gp = GaussianProcessQuantileEstimator(
        kernel=kernel,
        noise="gaussian",  # Automatic noise estimation
        random_state=42
    )
    gp.fit(X_train, y_train, quantiles=[0.05, 0.95])

Performance Considerations
--------------------------

**Computational Complexity**

========================== =============== =============== =================
Estimator                  Training        Prediction      Memory
========================== =============== =============== =================
QuantileGBM               O(nkd log n)     O(kd)           O(kd)
QuantileLightGBM          O(nkd log n)     O(kd)           O(kd)
QuantileForest            O(nd log n)      O(d)            O(nd)
QuantileLeaf              O(nd log n)      O(Bd)           O(nd + By)
QuantileKNN               O(n log n)       O(k log n)      O(nd)
GaussianProcess (full)    O(n³)            O(n)            O(n²)
GaussianProcess (sparse)  O(nm²)           O(m)            O(nm)
========================== =============== =============== =================

Where n=samples, d=features, k=trees/quantiles, m=inducing points, B=trees, y=targets per leaf.

**Algorithm Selection Guide**

* **Small datasets (n < 1000)**: Use full Gaussian Process for optimal uncertainty quantification
* **Medium datasets (1K-10K)**: Consider sparse GP with m=n/5 or gradient boosting
* **Large datasets (n > 10K)**: Use LightGBM for speed or sparse GP with aggressive reduction
* **High-dimensional (d > 50)**: Random forests handle interactions well; GP may need dimensionality reduction
* **Linear relationships**: QuantileLasso for interpretability
* **Many quantiles needed**: Any single-fit estimator for efficiency

Integration Points
------------------

The quantile estimation module integrates seamlessly with other confopt components:

**Conformal Prediction Integration**

.. code-block:: python

    from confopt.conformalization import QuantileConformalPredictor

    # Quantile estimator as base for conformal prediction
    base_estimator = GaussianProcessQuantileEstimator()
    conformal_predictor = QuantileConformalPredictor(base_estimator)
    conformal_predictor.fit(X_cal, y_cal, coverage=0.9)

**Ensemble Integration**

.. code-block:: python

    from confopt.ensembling import QuantileEnsemble

    # Combine multiple quantile estimators
    estimators = [
        ('gp', GaussianProcessQuantileEstimator()),
        ('gbm', QuantileGBM(n_estimators=100)),
        ('forest', QuantileForest(n_estimators=50))
    ]
    ensemble = QuantileEnsemble(estimators)

**Hyperparameter Optimization**

.. code-block:: python

    from confopt.tuning import BayesianOptimizer

    optimizer = BayesianOptimizer(
        estimator=GaussianProcessQuantileEstimator(),
        param_space={'alpha': (1e-12, 1e-3), 'kernel': ['rbf', 'matern']}
    )
    best_estimator = optimizer.optimize(X_train, y_train, quantiles=[0.1, 0.9])


Performance Considerations
-------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-Fit Estimators:**
- Training: O(M × algorithm_complexity) where M is number of quantiles
- Memory: M × model_size
- Prediction: O(M × prediction_time)

**Single-Fit Estimators:**
- Training: O(algorithm_complexity)
- Memory: model_size + distribution_samples
- Prediction: O(prediction_time + quantile_extraction)

Scalability Guidelines
~~~~~~~~~~~~~~~~~~~~~

**Small Datasets (< 1K samples):**
- Any algorithm suitable
- GP with full kernel matrices
- High-precision quantile estimation

**Medium Datasets (1K - 100K samples):**
- Tree-based methods preferred
- GP with sparse approximations
- Batch processing for predictions

**Large Datasets (> 100K samples):**
- LightGBM for speed
- Sparse GP or avoid GP entirely
- Aggressive batch processing

Algorithm Selection Guide
------------------------

Use Case Recommendations
~~~~~~~~~~~~~~~~~~~~~~~

**Linear Relationships + Interpretability**
    → QuantileLasso

**Non-linear + Speed Priority**
    → QuantileLightGBM

**Uncertainty Quantification + Small Data**
    → GaussianProcessQuantileEstimator

**Robustness + Ensemble Benefits**
    → QuantileForest

**Local Data Distribution + Heteroscedastic Noise**
    → QuantileLeaf

**Local Adaptation + Non-parametric**
    → QuantileKNN

**Many Quantiles + Computational Efficiency**
    → Any single-fit estimator

Common Usage Patterns
---------------------

Basic Quantile Regression
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.estimators.quantile_estimation import QuantileGBM

    # Define quantiles of interest
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    # Initialize and fit estimator
    estimator = QuantileGBM(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    estimator.fit(X_train, y_train, quantiles=quantiles)

    # Generate predictions
    quantile_preds = estimator.predict(X_test)  # Shape: (n_samples, 5)

Uncertainty Bands
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fit GP for smooth uncertainty bands
    gp_estimator = GaussianProcessQuantileEstimator(
        kernel="matern",
        random_state=42
    )
    gp_estimator.fit(X, y, quantiles=[0.1, 0.5, 0.9])

    predictions = gp_estimator.predict(X_test)
    lower_bound = predictions[:, 0]    # 10th percentile
    median = predictions[:, 1]         # 50th percentile (median)
    upper_bound = predictions[:, 2]    # 90th percentile

Comparing Forest Approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.estimators.quantile_estimation import (
        QuantileForest, QuantileLeaf
    )

    # Standard ensemble-based approach
    forest_ensemble = QuantileForest(
        n_estimators=100,
        max_depth=10,
        max_features=0.8,
        random_state=42
    )

    # Meinshausen (2006) leaf-based approach
    forest_leaves = QuantileLeaf(
        n_estimators=100,
        max_depth=None,  # Allow deeper trees for finer partitioning
        min_samples_leaf=5,  # Control minimum leaf size
        random_state=42
    )

    # Fit both approaches
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    forest_ensemble.fit(X_train, y_train, quantiles=quantiles)
    forest_leaves.fit(X_train, y_train, quantiles=quantiles)

    # Compare predictions
    preds_ensemble = forest_ensemble.predict(X_test)
    preds_leaves = forest_leaves.predict(X_test)

    # QuantileLeaf typically provides more faithful local uncertainty
    # especially in heteroscedastic regions

Integration Points
-----------------

The quantile estimation module integrates with:

* **Ensemble Framework**: Used as base estimators in ``QuantileEnsembleEstimator``
* **Conformal Prediction**: Provides base quantile estimates for conformal adjustment
* **Hyperparameter Tuning**: Integrated with ``confopt.tuning`` for automated optimization
* **Model Selection**: Used in ``confopt.selection`` for algorithm comparison

Common Pitfalls
---------------

* **Quantile Crossing**: Multi-fit estimators may produce non-monotonic quantiles
* **Overfitting**: High-capacity models (GP, deep trees) prone to overfitting on small datasets
* **Computational Overhead**: GP scales poorly without sparse approximations
* **Hyperparameter Sensitivity**: Tree-based methods require careful depth/complexity tuning
* **Distributional Assumptions**: GP analytical quantiles assume Gaussian posteriors

See Also
--------

* :doc:`ensembling` - Ensemble methods combining multiple quantile estimators
* :doc:`../estimation` - Higher-level conformal prediction frameworks
* :doc:`../tuning` - Hyperparameter optimization for quantile estimators
