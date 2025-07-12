Advanced Usage
==============

This guide shows how to use ConfOpt's advanced features to customize and accelerate your optimization process. Each section builds on the basics, with clear code and explanations.

Custom Searchers
----------------

ConfOpt lets you define custom searchers to control how new configurations are selected. A searcher combines a quantile estimator (for prediction intervals) and a sampler (for acquisition strategy).

**Searcher Types**

* ``QuantileConformalSearcher``: Uses quantile regression for prediction intervals.
* ``LocallyWeightedConformalSearcher``: Uses separate point and variance estimators with locality weighting.

**Quantile Estimator Architectures**

Choose how prediction intervals are built:

* ``"qrf"``: Quantile Random Forest
* ``"qgbm"``: Quantile Gradient Boosting Machine
* ``"qknn"``: Quantile K-Nearest Neighbors
* ``"qlgbm"``: Quantile LightGBM
* ``"qgp"``: Quantile Gaussian Process
* ``"ql"``: Quantile Lasso

**Samplers**

Samplers decide which configuration to try next. Some options:

* ``LowerBoundSampler``: Lower confidence bounds with exploration decay
* ``PessimisticLowerBoundSampler``: Conservative, uses only lower bounds
* ``ThompsonSampler``: Posterior sampling for exploration
* ``ExpectedImprovementSampler``: Expected improvement over current best
* ``EntropySearchSampler``: Information-theoretic selection
* ``MaxValueEntropySearchSampler``: Maximum value entropy search

**Pre-Conformal Trials**

The ``n_pre_conformal_trials`` parameter sets how many random configurations are evaluated before conformal guidance starts. More trials mean better initial training, but slower start.

**Example: Custom Searcher**

.. code-block:: python

   from confopt.selection.acquisition import QuantileConformalSearcher
   from confopt.selection.sampling import LowerBoundSampler

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

You can also use ``LocallyWeightedConformalSearcher``:

.. code-block:: python

   from confopt.selection.acquisition import LocallyWeightedConformalSearcher
   from confopt.selection.sampling import LowerBoundSampler

   searcher = LocallyWeightedConformalSearcher(
       point_estimator_architecture="rf",
       variance_estimator_architecture="gbm",
       sampler=LowerBoundSampler(interval_width=0.9)
   )

**Using a Custom Searcher with the Tuner**

Pass your searcher to the tuner:

.. code-block:: python

   from confopt.tuning import ConformalTuner
   from confopt.selection.sampling import ThompsonSampler

   searcher = QuantileConformalSearcher(
       quantile_estimator_architecture="qgbm",
       sampler=ThompsonSampler(
           interval_width=0.8,
           optimistic_bias=0.1
       ),
       n_pre_conformal_trials=32
   )

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space=search_space,
       metric_optimization="maximize"
   )

   tuner.tune(
       searcher=searcher,
       max_searches=100,
       n_random_searches=20,
       verbose=True
   )

**Full Example: Custom Searcher for Classification**

.. code-block:: python

   from confopt.tuning import ConformalTuner
   from confopt.selection.acquisition import QuantileConformalSearcher
   from confopt.selection.sampling import ExpectedImprovementSampler
   from confopt.wrapping import IntRange
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import load_wine
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   X, y = load_wine(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   def objective_function(configuration):
       model = RandomForestClassifier(**configuration, random_state=42)
       model.fit(X_train, y_train)
       return accuracy_score(y_test, model.predict(X_test))

   searcher = QuantileConformalSearcher(
       quantile_estimator_architecture="qrf",
       sampler=ExpectedImprovementSampler(
           interval_width=0.85,
           xi=0.01
       ),
       n_pre_conformal_trials=32
   )

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space={'n_estimators': IntRange(min_value=50, max_value=200)},
       metric_optimization="maximize"
   )

   tuner.tune(searcher=searcher, max_searches=50, verbose=True)

Warm Starting
-------------

Warm starting lets you begin optimization with configurations you've already evaluated. This can speed up convergence by using prior knowledge.

**How It Works**

* Warm start configurations are evaluated first, before random search.
* They count toward the ``n_random_searches`` budget.
* They help train the initial conformal model.

**Example: Basic Warm Starting**

.. code-block:: python

   warm_start_configs = [
       ({'n_estimators': 100, 'max_depth': 8}, 0.95),
       ({'n_estimators': 150, 'max_depth': 6}, 0.93),
       ({'n_estimators': 80, 'max_depth': 10}, 0.91)
   ]

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space=search_space,
       metric_optimization="maximize",
       warm_start_configurations=warm_start_configs
   )

   tuner.tune(n_random_searches=10, max_searches=50)

**Continuing a Previous Optimization**

You can save results from a previous run and use the best ones as warm starts:

.. code-block:: python

   import json

   def save_results(tuner, filename):
       results = {
           'best_params': tuner.get_best_params(),
           'best_score': tuner.get_best_value(),
           'all_trials': []
       }
       for trial in tuner.study.trials:
           results['all_trials'].append({
               'configuration': trial.configuration,
               'performance': trial.performance
           })
       with open(filename, 'w') as f:
           json.dump(results, f)

   def load_warm_starts(filename, top_n=5):
       with open(filename, 'r') as f:
           data = json.load(f)
       trials = data['all_trials']
       trials.sort(key=lambda x: x['performance'], reverse=True)
       return [(trial['configuration'], trial['performance']) for trial in trials[:top_n]]

   warm_starts = load_warm_starts('previous_results.json', top_n=8)

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space={
           'n_estimators': IntRange(min_value=50, max_value=300),
           'max_depth': IntRange(min_value=3, max_value=20),
           'learning_rate': FloatRange(min_value=0.01, max_value=0.3)
       },
       metric_optimization="maximize",
       warm_start_configurations=warm_starts
   )

   tuner.tune(n_random_searches=15, max_searches=100)

   save_results(tuner, 'continued_results.json')

**Budget Tip**

Warm starts count toward your random search budget. For example, if you have 5 warm starts and set ``n_random_searches=10``, only 5 additional random configurations will be tried before conformal guidance begins.

.. code-block:: python

   warm_starts = [
       ({'param1': 1.0}, 0.8),
       ({'param1': 2.0}, 0.85),
       ({'param1': 1.5}, 0.82),
       ({'param1': 2.5}, 0.78),
       ({'param1': 1.2}, 0.83)
   ]

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space=search_space,
       metric_optimization="maximize",
       warm_start_configurations=warm_starts
   )

   tuner.tune(n_random_searches=15, max_searches=50)

Optimizers
----------

Optimizers control how the conformal models tune their own hyperparameters. This can help balance prediction quality and computational cost.

**Optimizer Frameworks**

* ``'reward_cost'``: Bayesian optimization to balance prediction improvement and cost
* ``'fixed'``: Tune parameters at fixed intervals
* ``None``: Use default parameters throughout (fastest)

**Reward-Cost Optimization**

Automatically tunes hyperparameters by weighing prediction improvement against cost.

.. code-block:: python

   tuner.tune(
       optimizer_framework='reward_cost',
       conformal_retraining_frequency=2,
       max_searches=200,
       verbose=True
   )

**Fixed Tuning Schedule**

Tune at regular intervals with a fixed schedule.

.. code-block:: python

   tuner.tune(
       optimizer_framework='fixed',
       conformal_retraining_frequency=3,
       max_searches=150,
       verbose=True
   )

**No Optimizer (Default)**

Use default parameters for the fastest runs.

.. code-block:: python

   tuner.tune(
       optimizer_framework=None,
       conformal_retraining_frequency=1,
       max_searches=100,
       verbose=True
   )

**Which Should I Use?**

* Use ``'reward_cost'`` for long or complex optimizations where performance matters most.
* Use ``'fixed'`` for medium-length runs where you want some adaptation but predictable cost.
* Use ``None`` for quick experiments or simple problems.

**Example: Comparing Optimizers**

.. code-block:: python

   import time
   from confopt.tuning import ConformalTuner

   optimizers = ['reward_cost', 'fixed', None]
   results = {}

   for opt in optimizers:
       start_time = time.time()
       tuner = ConformalTuner(
           objective_function=objective_function,
           search_space=search_space,
           metric_optimization="maximize"
       )
       tuner.tune(
           optimizer_framework=opt,
           conformal_retraining_frequency=2,
           max_searches=50,
           verbose=False
       )
       runtime = time.time() - start_time
       results[opt] = {
           'best_score': tuner.get_best_value(),
           'runtime': runtime,
           'best_params': tuner.get_best_params()
       }

   for opt, result in results.items():
       print(f"{opt}: Score={result['best_score']:.4f}, Time={result['runtime']:.1f}s")
