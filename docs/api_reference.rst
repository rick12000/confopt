
Tuner
=====

.. currentmodule:: confopt.tuning

.. _conformaltuner:

ConformalTuner
~~~~~~~~~~~~~~~
.. autoclass:: ConformalTuner
   :members:
   :exclude-members: __init__
   :noindex:

Parameter Ranges
================

.. currentmodule:: confopt.wrapping

.. _intrange:

IntRange
~~~~~~~~
.. autoclass:: IntRange
   :members:
   :noindex:

.. _floatrange:

FloatRange
~~~~~~~~~~
.. autoclass:: FloatRange
   :members:
   :noindex:

.. _categoricalrange:

CategoricalRange
~~~~~~~~~~~~~~~~
.. autoclass:: CategoricalRange
   :members:
   :noindex:

Acquisition Functions
======================

.. currentmodule:: confopt.selection.acquisition

QuantileConformalSearcher
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: QuantileConformalSearcher
   :members:
   :exclude-members: __init__
   :noindex:

Samplers
========

Bound Sampling
--------------

.. currentmodule:: confopt.selection.sampling.bound_samplers

PessimisticLowerBoundSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: PessimisticLowerBoundSampler
   :members:
   :exclude-members: __init__
   :noindex:

LowerBoundSampler
~~~~~~~~~~~~~~~~~
.. autoclass:: LowerBoundSampler
   :members:
   :exclude-members: __init__
   :noindex:

Thompson Sampling
-----------------

.. currentmodule:: confopt.selection.sampling.thompson_samplers

ThompsonSampler
~~~~~~~~~~~~~~~
.. autoclass:: ThompsonSampler
   :members:
   :exclude-members: __init__
   :noindex:

Expected Improvement Sampling
------------------------------

.. currentmodule:: confopt.selection.sampling.expected_improvement_samplers

ExpectedImprovementSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ExpectedImprovementSampler
   :members:
   :exclude-members: __init__
   :noindex:

Entropy Sampling
----------------

.. currentmodule:: confopt.selection.sampling.entropy_samplers


MaxValueEntropySearchSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MaxValueEntropySearchSampler
   :members:
   :exclude-members: __init__
   :noindex:
