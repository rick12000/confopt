API Reference
-------------

ConformalTuner
==============

.. currentmodule:: confopt.tuning

.. _conformaltuner:

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

QuantileConformalSearcher
=========================

.. currentmodule:: confopt.selection.acquisition

.. autoclass:: QuantileConformalSearcher
   :members:
   :exclude-members: __init__
   :noindex:

Samplers
========

.. currentmodule:: confopt.selection.sampling.bound_samplers

PessimisticLowerBoundSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

.. currentmodule:: confopt.selection.sampling.thompson_samplers

ThompsonSampler
~~~~~~~~~~~~~~~
.. autoclass:: ThompsonSampler
   :members:
   :exclude-members: __init__
   :noindex:

ExpectedImprovementSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: confopt.selection.sampling.expected_improvement_samplers

.. autoclass:: ExpectedImprovementSampler
   :members:
   :exclude-members: __init__
   :noindex:
