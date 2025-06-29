========
Roadmap
========

ConfOpt Development Roadmap
========================

This document outlines the planned features and improvements for future versions of ConfOpt.

Upcoming Features
================

Features
------------------------

* **Multi Fidelity Support**: Enable single fidelity conformal searchers to adapt to multi-fidelity
   settings, allowing them to be competitive in settings where models can be partially trained and lower fidelities are
   predictive of full fidelity performance.
* **Multi Objective Support**: Allow searchers to optimizer for more than one objective (eg. accuracy and runtime).

Resource Management
---------------------

* **Parallel Search Support**: Allow searchers to evaluate multiple configurations in parallel if compute allows.
* **Smart Resource Usage**: Auto detect best amount of parallelism based on available resources and expected load.
