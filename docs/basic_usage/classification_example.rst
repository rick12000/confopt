Classification Example
=======================

This example will show you how to use ConfOpt to optimize hyperparameters for a classification task.

If you already used hyperparameter tuning packages, the "Code Example" section below will give you a quick run through of how to use ConfOpt. If not, don't worry, the "Detailed Walkthrough" section will explain everything step-by-step.

Code Example
------------

1. Set up search space and objective function:

.. code-block:: python


   from confopt.tuning import ConformalTuner
   from confopt.wrapping import IntRange, FloatRange, CategoricalRange

   from sklearn.ensemble import RandomForestClassifier

   from sklearn.datasets import load_wine
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   search_space = {
       'n_estimators': IntRange(min_value=50, max_value=200),
       'max_features': FloatRange(min_value=0.1, max_value=1.0),
       'criterion': CategoricalRange(choices=['gini', 'entropy', 'log_loss'])
   }

   def objective_function(configuration):
       X, y = load_wine(return_X_y=True)
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42, stratify=y
       )

       model = RandomForestClassifier(
           n_estimators=configuration['n_estimators'],
           max_features=configuration['max_features'],
           criterion=configuration['criterion'],
           random_state=42
       )

       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       score = accuracy_score(y_test, predictions)

       return score

2. Call ConfOpt to tune hyperparameters:

.. code-block:: python

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space=search_space,
       minimize=False
   )

   tuner.tune(
       max_searches=50,
       n_random_searches=10,
       verbose=True
   )

3. Extract results:

.. code-block:: python

   best_params = tuner.get_best_params()
   best_accuracy = tuner.get_best_value()

   tuned_model = RandomForestClassifier(**best_params, random_state=42)


Detailed Walkthrough
--------------------

Imports
~~~~~~~

First, let's import everything we'll be needing:

.. code-block:: python

   from confopt.tuning import ConformalTuner
   from confopt.wrapping import IntRange, FloatRange, CategoricalRange

   from sklearn.ensemble import RandomForestClassifier

   from sklearn.datasets import load_wine
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

For this tutorial, we'll be using the sklearn Wine dataset and trying to tune the hyperparameters of a ``RandomForestClassifier``.

Search Space
~~~~~~~~~~~~

Next, we need to define the hyperparameter space we want ``confopt`` to optimize over.

This is done using the :ref:`IntRange <intrange>`, :ref:`FloatRange <floatrange>`, and :ref:`CategoricalRange <categoricalrange>` classes, which specify the ranges for each hyperparameter.
Below let's define a simple example with one of each type of hyperparameter:

.. code-block:: python

   search_space = {
       'n_estimators': IntRange(min_value=50, max_value=200),
       'max_features': FloatRange(min_value=0.1, max_value=1.0),
       'criterion': CategoricalRange(choices=['gini', 'entropy', 'log_loss'])
   }


This tells ``confopt`` to explore the following hyperparameter ranges:

* ``n_estimators``: Number of trees in the forest (all integer values from 50 to 200)
* ``max_features``: Fraction of features to consider at each split (any float between 0.1 and 1.0)
* ``criterion``: Function to measure the quality of a split (choose from 'gini', 'entropy', or 'log_loss')


Objective Function
~~~~~~~~~~~~~~~~~~

The objective function defines how the model trains and what metric you want to optimize for during hyperparameter search:

.. code-block:: python

   def objective_function(configuration):
       X, y = load_wine(return_X_y=True)
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42, stratify=y
       )

       model = RandomForestClassifier(
           n_estimators=configuration['n_estimators'],
           max_features=configuration['max_features'],
           criterion=configuration['criterion'],
           random_state=42
       )

       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       score = accuracy_score(y_test, predictions)

       return score

The objective function must take a single argument called ``configuration``, which is a dictionary containing a value for each hyperparameter name specified in your ``search_space``. The values will be chosen automatically by the tuner during optimization. The ``score`` can be any metric of your choosing (e.g., accuracy, log loss, F1 score, etc.). This is the value that ``confopt`` will try to optimize for.

In this example, the data is loaded and split inside the objective function for simplicity, but you may prefer to load the data outside (to avoid reloading it for each configuration) and
either pass the training and test sets as arguments using ``partial`` from the ``functools`` library, or reference them from the global scope.

Running the Optimization
~~~~~~~~~~~~~~~~~~~~~~~~


To start optimizing, first instantiate a :ref:`ConformalTuner <conformaltuner>` by providing your objective function, search space, and the optimization direction:

.. code-block:: python

   tuner = ConformalTuner(
       objective_function=objective_function,
       search_space=search_space,
       minimize=False  # Use True for metrics like log loss
   )

The ``minimize`` parameter should be set to ``False`` if you want to maximize your metric (e.g., accuracy), or ``True`` if you want to minimize it (e.g., log loss).

To actually kickstart the hyperparameter search, call:

.. code-block:: python

   tuner.tune(
       max_searches=50,
       n_random_searches=10,
       verbose=True
   )

Where:

* ``max_searches`` controls how many different hyperparameter configurations will be tried in total.
* ``n_random_searches`` sets how many of those will be chosen randomly before the tuner switches to using smart optimization (eg. ``max_searches=50`` and ``n_random_searches=10`` means the tuner will sample 10 random configurations, then 40 smart configurations).


Getting the Results
~~~~~~~~~~~~~~~~~~~


After that runs, you can retrieve the best hyperparameters or the best score found using ``get_best_params()`` and ``get_best_value()``:

.. code-block:: python

   best_params = tuner.get_best_params()
   best_accuracy = tuner.get_best_value()

Expected output:

.. code-block:: text

   Best accuracy: 0.9815
   Best parameters: {'n_estimators': 187, 'max_features': 0.73, 'criterion': 'entropy'}

Which you can use to instantiate a tuned version of your model:

.. code-block:: python


   tuned_model = RandomForestClassifier(**best_params, random_state=42)
