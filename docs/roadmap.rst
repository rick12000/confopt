========
Roadmap
========

Upcoming Features
================

Functionality
------------------------

* **Multi Fidelity Support**: Enable single fidelity conformal searchers to adapt to multi-fidelity settings, allowing them to be competitive in settings where models can be partially trained and lower fidelities are predictive of full fidelity performance.
* **Multi Objective Support**: Allow searchers to optimize for more than one objective (eg. accuracy and runtime).
* **Transfer Learning Support**: Allow searchers to use a pretrained model or an observation matcher as a starting point for tuning.

Resource Management
---------------------

* **Parallel Search Support**: Allow searchers to evaluate multiple configurations in parallel if compute allows.
* **Smart Resource Usage**: Auto detect best amount of parallelism based on available resources and expected load.
