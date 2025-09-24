Advanced Usage
==============

This guide shows how to use ConfOpt's advanced features to customize and accelerate your optimization process. Each section builds on the basics, with clear code and explanations.

Custom Searchers
----------------

ConfOpt lets you define custom searchers to control how new configurations are selected.
A searcher is made up of a quantile estimator (surrogate model) and a sampler (acquisition function).

A searcher can be instantied via the ``QuantileConformalSearcher`` class.

To create a custom searcher with a custom estimator architecture and sampler, select from the following:

**Estimator Architectures**

Estimator architectures determine the framework used to build the surrogate model.

You can choose from the following architectures:

* ``"qrf"``: Quantile Random Forest
* ``"qgbm"``: Quantile Gradient Boosting Machine
* ``"qknn"``: Quantile K-Nearest Neighbors
* ``"qgp"``: Quantile Gaussian Process
* ``"ql"``: Quantile Lasso
* ``"qens5"``: Quantile Ensemble of 3 models (QGBM, QGP, QL)

**Samplers**

Samplers dictate which configuration to try next, driven by some base acquisition function.

You can use the following samplers:

* ``LowerBoundSampler``: Lower confidence bounds with exploration decay (good for fast convergence on simple problems)
* ``ThompsonSampler``: Posterior sampling for exploration (good for balancing exploration and exploitation)
* ``ExpectedImprovementSampler``: Expected improvement over current best (good for both fast convergence and exploration)

**Example:**

Let's use a ``QuantileConformalSearcher`` with a ``LowerBoundSampler`` and a Quantile Random Forest surrogate (``"qrf"``) estimator:

.. code-block:: python

   from confopt.selection.acquisition import QuantileConformalSearcher
   from confopt.selection.sampling.bound_samplers import LowerBoundSampler

   searcher = QuantileConformalSearcher(
       quantile_estimator_architecture="qrf",
       sampler=LowerBoundSampler(
           interval_width=0.8, # Width of the confidence interval to use as the lower bound,
           adapter="DtACI", # Conformal adapter to use for calibration
           beta_decay="logarithmic_decay", # Lower Bound Sampling decay function
           c=1.0 # Lower Bound Sampling Decay rate
       )
   )

And pass our custom searcher to the tuner to use it:

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

* Warm start configurations are ingested before random search.
* They count toward the ``n_random_searches`` budget.
* They help train the initial surrogate model.

**Example:**

.. code-block:: python

   warm_start_configs = [
       ({'n_estimators': 100, 'max_depth': 8}, 0.95), # (hyperparameter configuration, objective value)
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

If your optimization is taking unexpectedly long on the ``'decaying'`` or ``'fixed'``optimizers, try switching to ``None``.

**Example:**

.. code-block:: python

   tuner.tune(
       optimizer_framework='decaying',
       max_searches=200,
       verbose=True
   )
