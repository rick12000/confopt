ConfOpt Documentation
=====================

ConfOpt is a Python library for hyperparameter optimization using conformal prediction. It provides a statistically principled approach to hyperparameter tuning that combines the efficiency of guided search with the reliability of uncertainty quantification.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   installation
   getting_started
   advanced_usage

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   architecture
   installation_setup

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   roadmap
   contact

Quick Start
-----------

Install ConfOpt:

.. code-block:: bash

   pip install confopt

Basic usage:

.. code-block:: python

   from confopt.tuning import ConformalTuner
   from confopt.wrapping import IntRange, FloatRange

   # Define search space
   search_space = {
       'n_estimators': IntRange(50, 200),
       'max_depth': IntRange(3, 20)
   }

   # Create tuner
   tuner = ConformalTuner(
       objective_function=your_objective_function,
       search_space=search_space,
       minimize=False
   )

   # Run optimization
   tuner.tune(max_searches=100)

   # Get results
   best_params = tuner.get_best_params()
   best_score = tuner.get_best_value()

For detailed examples and usage patterns, see the :doc:`getting_started` section.
