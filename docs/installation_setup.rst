Installation Setup
==================

This guide explains ConfOpt's installation process. ConfOpt is a pure Python package with no compiled extensions, ensuring reliable installation across all platforms and environments.

Overview
--------

ConfOpt is designed as a pure Python package that requires no compilation. The installation process is straightforward:

1. **🚀 Simple Installation**: Standard pip installation
2. **✅ Cross-Platform**: Works identically on Windows, macOS, and Linux
3. **📦 Reliable Packaging**: Pure Python ensures predictable behavior

Installation
------------

Install ConfOpt using pip:

.. code-block:: bash

   pip install confopt

Or install from source:

.. code-block:: bash

   git clone https://github.com/rick12000/confopt.git
   cd confopt
   pip install .

Build Configuration
-------------------

pyproject.toml
~~~~~~~~~~~~~~

The build configuration is minimal for pure Python packages:

.. code-block:: toml

   [build-system]
   requires = ["setuptools>=61.0", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "confopt"
   version = "1.2.4"
   description = "Conformal hyperparameter optimization tool"
   readme = "README.md"
   requires-python = ">=3.9"
   dependencies = [
       "numpy>=1.20.0",
       "scikit-learn>=1.0.0",
       "scipy>=1.7.0",
       "pandas>=1.3.0",
       "tqdm>=4.60.0",
       "pydantic>=2.0.0",
       "joblib>=1.0.0",
       "statsmodels>=0.13.0"
   ]

Dependencies
------------

ConfOpt requires the following Python packages:

- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **scipy**: Scientific computing
- **pandas**: Data manipulation
- **tqdm**: Progress bars
- **pydantic**: Data validation
- **joblib**: Parallel processing
- **statsmodels**: Statistical modeling

All dependencies are automatically installed when using pip.

Development Installation
-----------------------

To install ConfOpt with development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This installs additional packages for testing and development:

- pytest: Testing framework
- pre-commit: Code quality hooks

Testing Installation
-------------------

After installation, verify ConfOpt works correctly:

.. code-block:: python

   python -c "import confopt; print('ConfOpt installed successfully!')"

For more comprehensive testing, run the test suite:

.. code-block:: bash

   pytest tests/
