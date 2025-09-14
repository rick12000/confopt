========
Roadmap
========

Upcoming Features
=================

Functionality
------------------------

* **Multi Fidelity Support**: Enable single fidelity conformal searchers to adapt to multi-fidelity settings, allowing them to be competitive in settings where models can be partially trained and lower fidelities are predictive of full fidelity performance.
* **Multi Objective Support**: Allow searchers to optimize for more than one objective (eg. accuracy and runtime).
* **Transfer Learning Support**: Allow searchers to use a pretrained model or an observation matcher as a starting point for tuning.
* **Local Search**: Expected Improvement sampler currently only performs one off configuration scoring. Local search (where a local neighbourhood around the initial EI optimum is explored as a second pass refinement) can significantly improve performance.
* **Hierarchical Hyperparameters**: Improved handling for hierarchical hyperparameter spaces (currently supported, via flattening of the hyperparameters, but potentially suboptimal for surrogate learning)

Resource Management
---------------------

* **Parallel Search Support**: Allow searchers to evaluate multiple configurations in parallel if compute allows.
* **Smart Resource Usage**: Auto detect best amount of parallelism based on available resources and expected load.
