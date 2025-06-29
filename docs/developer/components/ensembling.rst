Ensembling Module
=================

Overview
--------

The ``confopt.selection.estimators.ensembling`` module provides sophisticated ensemble methods for combining multiple regression and quantile regression estimators. The module implements cross-validation based stacking with constrained linear regression meta-learners to achieve optimal predictor combination weights.

Key Features
------------

* **Cross-validation stacking**: Prevents overfitting by using out-of-fold predictions for meta-learner training
* **Constrained linear regression**: Ensures non-negative weights that sum to 1 for interpretable combinations
* **Quantile-specific weighting**: Allows different estimator weights across quantile levels for distributional modeling
* **Uniform fallback**: Simple equal weighting option for baseline comparisons

Architecture
------------

Class Hierarchy
~~~~~~~~~~~~~~~

::

    BaseEnsembleEstimator (ABC)
    ├── PointEnsembleEstimator
    └── QuantileEnsembleEstimator

Base Classes
~~~~~~~~~~~~

**BaseEnsembleEstimator**
    Abstract base providing common initialization and interface for ensemble estimators. Enforces minimum of 2 estimators and validates weighting strategies.

**PointEnsembleEstimator**
    Concrete implementation for single-value regression predictions. Uses standard scikit-learn compatible estimators.

**QuantileEnsembleEstimator**
    Concrete implementation for quantile regression predictions. Supports both multi-fit and single-fit quantile estimators with separate weight learning per quantile level.

Stacking Methodology
-------------------

Weight Learning Process
~~~~~~~~~~~~~~~~~~~~~~

1. **Cross-validation setup**: k-fold CV splits training data
2. **Out-of-fold prediction**: Each estimator trained on k-1 folds, predicts on held-out fold
3. **Meta-learner training**: Constrained LinearRegression fits on concatenated out-of-fold predictions
4. **Weight normalization**: Coefficients clipped to minimum 1e-6 and normalized to sum to 1

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~

For point predictions:

.. math::

    \hat{y}_{ensemble} = \sum_{i=1}^{M} w_i \hat{y}_i

Where:
- :math:`w_i` are learned weights with :math:`w_i \geq 0` and :math:`\sum w_i = 1`
- :math:`\hat{y}_i` are individual estimator predictions
- :math:`M` is the number of base estimators

For quantile predictions, weights are learned separately for each quantile :math:`\tau`:

.. math::

    \hat{y}_{ensemble}^{(\tau)} = \sum_{i=1}^{M} w_i^{(\tau)} \hat{y}_i^{(\tau)}

Weighting Strategies
-------------------

Uniform Weighting
~~~~~~~~~~~~~~~~

Simple equal weighting approach:

.. code-block:: python

    weights = np.ones(n_estimators) / n_estimators

**Advantages:**
- No overfitting risk
- Computational efficiency
- Baseline for comparison

**Disadvantages:**
- Ignores individual estimator performance
- May dilute strong predictors

Linear Stacking
~~~~~~~~~~~~~~

Cross-validation based weight learning:

.. code-block:: python

    # Generate out-of-fold predictions
    cv_predictions = generate_oof_predictions(estimators, X, y, cv_folds)

    # Train constrained meta-learner
    meta_learner = LinearRegression(fit_intercept=False, positive=True)
    meta_learner.fit(cv_predictions, y_true)

    # Normalize weights
    weights = np.maximum(meta_learner.coef_, 1e-6)
    weights = weights / np.sum(weights)

**Advantages:**
- Optimal linear combination
- Accounts for estimator correlations
- Principled weight selection

**Disadvantages:**
- Higher computational cost
- Requires cross-validation
- Limited to linear combinations

Usage Examples
--------------

Point Estimation Ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from confopt.selection.estimators.ensembling import PointEnsembleEstimator

    # Define base estimators
    estimators = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        KNeighborsRegressor(n_neighbors=5)
    ]

    # Create ensemble with linear stacking
    ensemble = PointEnsembleEstimator(
        estimators=estimators,
        cv=5,
        weighting_strategy="linear_stack",
        random_state=42
    )

    # Fit and predict
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)

Quantile Estimation Ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from confopt.selection.estimators.quantile_estimation import (
        QuantileGBM, QuantileLightGBM, QuantileForest
    )
    from confopt.selection.estimators.ensembling import QuantileEnsembleEstimator

    # Define quantile estimators
    estimators = [
        QuantileGBM(learning_rate=0.1, n_estimators=100),
        QuantileLightGBM(learning_rate=0.1, n_estimators=100),
        QuantileForest(n_estimators=100)
    ]

    # Create quantile ensemble
    ensemble = QuantileEnsembleEstimator(
        estimators=estimators,
        cv=3,
        weighting_strategy="linear_stack",
        random_state=42
    )

    # Fit for specific quantiles
    quantiles = [0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles
    ensemble.fit(X_train, y_train, quantiles=quantiles)

    # Generate quantile predictions
    quantile_predictions = ensemble.predict(X_test)  # Shape: (n_samples, 3)

Performance Considerations
-------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

**Training Time:**
- Uniform: O(M × N) where M is number of estimators, N is training samples
- Linear stacking: O(M × N × K) where K is number of CV folds

**Memory Usage:**
- Stores M fitted estimators
- Stacking requires additional O(N × M) for out-of-fold predictions

**Prediction Time:**
- O(M × prediction_time_per_estimator)

Best Practices
~~~~~~~~~~~~~

1. **Estimator diversity**: Use different algorithm families (tree-based, linear, kernel methods)
2. **Hyperparameter variation**: Vary key parameters within algorithm families
3. **Cross-validation folds**: Use 3-5 folds for stacking to balance bias-variance
4. **Quantile selection**: Choose quantiles relevant to downstream uncertainty quantification needs
5. **Validation**: Always validate ensemble performance on held-out test sets

Integration Points
-----------------

The ensembling module integrates with:

* **Estimator Configuration**: Used in ``confopt.selection.estimator_configuration`` for pre-defined ensemble configurations
* **Selection Framework**: Called by ``confopt.selection.estimation`` for automated estimator selection
* **Conformal Prediction**: Ensemble predictions feed into conformal regression frameworks
* **Optimization**: Used within ``confopt.tuning`` for robust hyperparameter optimization

Common Pitfalls
---------------

* **Overfitting**: Using insufficient CV folds or highly correlated estimators
* **Weight instability**: Including too many weak estimators can lead to unstable weight learning
* **Quantile crossing**: Individual estimator quantile violations can persist in ensemble
* **Computational overhead**: Stacking significantly increases training time vs. single estimators

See Also
--------

* :doc:`quantile_estimation` - Base quantile estimator implementations
* :doc:`../estimation` - Higher-level estimation frameworks using ensembles
* :doc:`../tuning` - Hyperparameter optimization with ensemble estimators
