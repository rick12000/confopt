Advanced Usage
==============

This guide shows how to use ConfOpt's advanced features to customize and accelerate your optimization process. Each section builds on the basics, with clear code and explanations.

Custom Searchers
----------------

ConfOpt lets you define custom searchers to control how new configurations are selected.
A searcher is made up of a quantile estimator (surrogate model) and a sampler (acquisition function).

**Searcher Types**

* ``QuantileConformalSearcher``: Uses quantile regression for prediction intervals.

**Samplers**

Samplers dictate which configuration to try next.
Regardless of searcher type, you can use the following samplers:

* ``LowerBoundSampler``: Lower confidence bounds with exploration decay (good for fast convergence on simple problems)
* ``ThompsonSampler``: Posterior sampling for exploration (good for balancing exploration and exploitation)
* ``ExpectedImprovementSampler``: Expected improvement over current best (good for both fast convergence and exploration)



**Estimator Architectures**

Estimator architectures determine the framework used to build the surrogate model.
Which architectures you can choose from depends on the searcher type.

For ``QuantileConformalSearcher``, you can choose from the following architectures:

* ``"qrf"``: Quantile Random Forest
* ``"qgbm"``: Quantile Gradient Boosting Machine
* ``"qknn"``: Quantile K-Nearest Neighbors
* ``"qgp"``: Quantile Gaussian Process
* ``"ql"``: Quantile Lasso



**Example:**

Let's use a ``QuantileConformalSearcher`` with a ``LowerBoundSampler`` and a ``QuantileRandomForest`` estimator:

.. code-block:: python

   from confopt.selection.acquisition import QuantileConformalSearcher
   from confopt.selection.sampling.bound_samplers import LowerBoundSampler

   searcher = QuantileConformalSearcher(
       quantile_estimator_architecture="qrf",
       sampler=LowerBoundSampler(
           interval_width=0.8,
           adapter="DtACI",
           beta_decay="logarithmic_decay",
           c=1.0
       ),
       n_pre_conformal_trials=32
   )

To then pass the searcher to the tuner:

.. code-block:: python

   from confopt.tuning import ConformalTuner

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space=search_space,
       minimize=False,
   )

   tuner.tune(
       searcher=searcher,
       max_searches=100,
       n_random_searches=20,
       verbose=True
   )

Warm Starting
-------------

Warm starting lets you begin optimization with configurations you've already evaluated. This can speed up convergence by using prior knowledge.

**How It Works**

* Warm start configurations are evaluated first, before random search.
* They count toward the ``n_random_searches`` budget.
* They help train the initial surrogate model.

**Example:**

.. code-block:: python

   warm_start_configs = [
       ({'n_estimators': 100, 'max_depth': 8}, 0.95),
       ({'n_estimators': 150, 'max_depth': 6}, 0.93),
       ({'n_estimators': 80, 'max_depth': 10}, 0.91)
   ]

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space=search_space,
       minimize=False,
       warm_starts=warm_start_configs
   )

   tuner.tune(n_random_searches=10, max_searches=50)

Optimizers
----------

Optimizers control how the surrogate models tune their own hyperparameters.

**Optimizer Frameworks**

* ``None``: No tuning.
* ``'decaying'``: Tune parameters with increasing intervals over time, using configurable decay functions (linear, exponential, or logarithmic).
* ``'fixed'``: Tune parameters after each sampling episode, with a fixed number (10) of hyperparameter combinations.

**Which Should I Use?**

* Use ``None`` if the model you want to tune (not the surrogate model) trains very quickly (less than 10 seconds) or on little data.
* Use ``'decaying'`` if you want adaptive tuning that starts intensive and becomes less frequent over time.
* Use ``'fixed'`` if you want consistent tuning behavior throughout the optimization process.

**Example:**

.. code-block:: python

   tuner.tune(
       optimizer_framework='decaying',
       conformal_retraining_frequency=2,
       max_searches=200,
       verbose=True
   )
