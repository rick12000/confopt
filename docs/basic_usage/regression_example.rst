Regression Example
==================

This example shows how to use ConfOpt to optimize hyperparameters for a regression task.

Getting Started
---------------

First, let's import everything we'll be needing:

.. code-block:: python

   from confopt.tuning import ConformalTuner
   from confopt.wrapping import IntRange, FloatRange, CategoricalRange

   from sklearn.ensemble import RandomForestRegressor

   from sklearn.datasets import load_diabetes
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error

For this tutorial, we'll be using the sklearn Diabetes dataset and trying to tune the hyperparameters of a ``RandomForestRegressor``.

Search Space
------------

Next, we need to define the hyperparameter space we want ``confopt`` to optimize over.

This is done using the :ref:`IntRange <intrange>`, :ref:`FloatRange <floatrange>`, and :ref:`CategoricalRange <categoricalrange>` classes, which specify the ranges for each hyperparameter. See :ref:`Parameter Ranges <parameter-ranges>` in the API reference for more details.

Below let's define a simple example with a few typical hyperparameters for regression:

.. code-block:: python

   search_space = {
       'n_estimators': IntRange(min_value=50, max_value=200),
       'max_depth': IntRange(min_value=3, max_value=15),
       'min_samples_split': IntRange(min_value=2, max_value=10)
   }

This tells ``confopt`` to explore the following hyperparameter ranges:

* ``n_estimators``: Number of trees in the forest (all integer values from 50 to 200)
* ``max_depth``: Maximum tree depth (all integer values from 3 to 15)
* ``min_samples_split``: Minimum samples to split a node (all integer values from 2 to 10)

Objective Function
------------------

The objective function defines how the model trains and what metric you want to optimize for during hyperparameter search:

.. code-block:: python

   def objective_function(configuration):
       X, y = load_diabetes(return_X_y=True)
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42
       )

       model = RandomForestRegressor(
           n_estimators=configuration['n_estimators'],
           max_depth=configuration['max_depth'],
           min_samples_split=configuration['min_samples_split'],
           random_state=42
       )

       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       mse = mean_squared_error(y_test, predictions)
       return mse  # Lower is better (minimize MSE)


The objective function must take a single argument called ``configuration``, which is a dictionary containing a value for each hyperparameter name specified in your ``search_space``. The values will be chosen automatically by the tuner during optimization. The ``score`` can be any metric of your choosing (e.g., MSE, R², MAE, etc.). This is the value that ``confopt`` will try to optimize for. For MSE, lower is better, so we minimize it.

In this example, the data is loaded and split inside the objective function for simplicity, but you may prefer to load the data outside (to avoid reloading it for each configuration) and either pass the training and test sets as arguments using ``partial`` from the ``functools`` library, or reference them from the global scope.

Running the Optimization
------------------------


To start optimizing, first instantiate a :ref:`ConformalTuner <conformaltuner>` by providing your objective function, search space, and the optimization direction:

.. code-block:: python

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space=search_space,
       metric_optimization="minimize"  # Minimizing MSE
   )

The ``metric_optimization`` parameter should be set to ``"minimize"`` for metrics where lower is better (e.g., MSE, MAE), or ``"maximize"`` for metrics where higher is better (e.g., R²).

To actually kickstart the hyperparameter search, call:

.. code-block:: python

   tuner.tune(
       max_searches=50,
       n_random_searches=10,
       verbose=True
   )

Where:

* ``max_searches`` controls how many different hyperparameter configurations will be tried in total.
* ``n_random_searches`` sets how many of those will be chosen randomly before the tuner switches to using smart optimization (e.g., ``max_searches=50`` and ``n_random_searches=10`` means the tuner will sample 10 random configurations, then 40 smart configurations).

Getting the Results
-------------------


After that runs, you can retrieve the best hyperparameters or the best score found using :meth:`~confopt.tuning.ConformalTuner.get_best_params` and :meth:`~confopt.tuning.ConformalTuner.get_best_value`:

.. code-block:: python

   best_params = tuner.get_best_params()
   best_mse = tuner.get_best_value()

Expected output:

.. code-block:: text

   Best MSE: 2847.32
   Best parameters: {'n_estimators': 180, 'max_depth': 12, 'min_samples_split': 2}

Which you can use to instantiate a tuned version of your model:

.. code-block:: python

   tuned_model = RandomForestRegressor(**best_params, random_state=42)


Full Example
-----------------

Here is the full tutorial code if you want to run it all together:

.. code-block:: python


   from confopt.tuning import ConformalTuner  # :class:`~confopt.tuning.ConformalTuner` in API reference
   from confopt.wrapping import IntRange, FloatRange, CategoricalRange  # See :ref:`Parameter Ranges <parameter-ranges>`
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.datasets import load_diabetes
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error, r2_score

   def objective_function(configuration):
       X, y = load_diabetes(return_X_y=True)
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42
       )

       model = RandomForestRegressor(
           n_estimators=configuration['n_estimators'],
           max_depth=configuration['max_depth'],
           min_samples_split=configuration['min_samples_split'],
           random_state=42
       )

       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
      mse = mean_squared_error(y_test, predictions)
      return mse  # Lower is better (minimize MSE)

   search_space = {
       'n_estimators': IntRange(min_value=50, max_value=200),
       'max_depth': IntRange(min_value=3, max_value=15),
       'min_samples_split': IntRange(min_value=2, max_value=10)
   }

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space=search_space,
      metric_optimization="minimize"  # Minimizing MSE
   )

   tuner.tune(
       max_searches=50,
       n_random_searches=10,
       verbose=True
   )

   best_params = tuner.get_best_params()
   best_neg_mse = tuner.get_best_value()
      best_mse = tuner.get_best_value()

   tuned_model = RandomForestRegressor(**best_params, random_state=42)
   tuned_model.fit(*train_test_split(load_diabetes(return_X_y=True)[0], load_diabetes(return_X_y=True)[1], test_size=0.3, random_state=42)[:2])

   # Compare with default
   default_model = RandomForestRegressor(random_state=42)
   default_model.fit(*train_test_split(load_diabetes(return_X_y=True)[0], load_diabetes(return_X_y=True)[1], test_size=0.3, random_state=42)[:2])

   X, y = load_diabetes(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   final_predictions = tuned_model.predict(X_test)
   default_predictions = default_model.predict(X_test)
   final_mse = mean_squared_error(y_test, final_predictions)
   default_mse = mean_squared_error(y_test, default_predictions)
   final_r2 = r2_score(y_test, final_predictions)
   default_r2 = r2_score(y_test, default_predictions)

   print(f"Optimized - MSE: {final_mse:.4f}, R²: {final_r2:.4f}")
   print(f"Default - MSE: {default_mse:.4f}, R²: {default_r2:.4f}")
   print(f"MSE improvement: {default_mse - final_mse:.4f}")


Alternative Metrics
-------------------

You can optimize for different regression metrics by changing the objective function and setting the appropriate ``metric_optimization`` parameter:

**R² Score (Coefficient of Determination):** (set ``metric_optimization='maximize'``)

.. code-block:: python

   from sklearn.metrics import r2_score

   def r2_objective(configuration):
       X, y = load_diabetes(return_X_y=True)
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42
       )
       model = RandomForestRegressor(**configuration, random_state=42)
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       return r2_score(y_test, predictions)

**Mean Absolute Error (MAE):** (set ``metric_optimization='minimize'``)

.. code-block:: python

   from sklearn.metrics import mean_absolute_error

   def mae_objective(configuration):
       X, y = load_diabetes(return_X_y=True)
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42
       )
       model = RandomForestRegressor(**configuration, random_state=42)
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       mae = mean_absolute_error(y_test, predictions)
       return mae

**Root Mean Squared Error (RMSE):** (set ``metric_optimization='minimize'``)

.. code-block:: python

   import numpy as np
   from sklearn.metrics import mean_squared_error

   def rmse_objective(configuration):
       X, y = load_diabetes(return_X_y=True)
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42
       )
       model = RandomForestRegressor(**configuration, random_state=42)
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       rmse = np.sqrt(mean_squared_error(y_test, predictions))
       return rmse
