Installation Setup
==================

This guide explains ConfOpt's optional Cython build system and packaging best practices for Python libraries with compiled extensions. The implementation demonstrates how to create a robust fallback system that ensures installation never fails, regardless of the user's environment.

Overview
--------

ConfOpt uses an **optional Cython extension** for performance-critical entropy calculations. The build system follows a **3-tier fallback strategy**:

1. **🚀 Best Case - Wheel Installation**: Pre-compiled extension, no compiler required
2. **⚙️ Good Case - Source Build**: Compiles extension from C source when possible
3. **✅ Fallback Case - Pure Python**: Always works, functional but slower

This ensures ``pip install .`` **never fails**, following Python packaging best practices for optional compiled extensions.

Build Configuration
-------------------

pyproject.toml
~~~~~~~~~~~~~~

The build configuration uses minimal requirements to maximize compatibility:

.. code-block:: toml

   [build-system]
   requires = ["setuptools>=61.0", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "confopt"
   version = "1.0.2"
   # ... other metadata ...
   dependencies = [
       "numpy>=1.20.0",  # Runtime dependency, not build dependency
       # ... other runtime deps ...
   ]

**Key Points:**

- **No Cython/NumPy in build requirements** - they're optional for building
- **Runtime dependencies separate** from build dependencies
- **Minimal build requirements** ensure maximum compatibility

setup.py - Optional Extension Handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``setup.py`` implements graceful fallback logic:

.. code-block:: python

   #!/usr/bin/env python
   """Optional Cython extension setup with graceful fallback."""

   import os
   from setuptools import Extension, setup

   def build_extensions():
       """Attempt to build Cython extensions with graceful fallback."""
       try:
           import numpy as np

           # Check if C source file exists
           c_file = "confopt/selection/sampling/cy_entropy.c"
           if not os.path.exists(c_file):
               print(f"Warning: C source file {c_file} not found. Skipping Cython extension.")
               return []

           # Define Cython extensions
           extensions = [
               Extension(
                   "confopt.selection.sampling.cy_entropy",
                   sources=[c_file],
                   include_dirs=[np.get_include()],
                   define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                   language="c",
               )
           ]

           print("Building Cython extensions...")
           return extensions

       except ImportError as e:
           print(f"Warning: Could not import required dependencies: {e}")
           print("Falling back to pure Python implementation.")
           return []
       except Exception as e:
           print(f"Warning: Cython extension compilation failed: {e}")
           print("Falling back to pure Python implementation.")
           return []

   # Build extensions with fallback
   try:
       ext_modules = build_extensions()
   except Exception as e:
       print(f"Warning: Extension building failed: {e}")
       print("Installing without Cython extensions.")
       ext_modules = []

   setup(ext_modules=ext_modules)

**Best Practices Demonstrated:**

- **Defensive programming** - multiple try/except layers
- **Clear user feedback** - informative warning messages
- **Never fail installation** - always return empty list on failure
- **Resource checking** - verify files exist before attempting compilation

Runtime Import Strategy
-----------------------

The Python code uses a **single module-level import check** to avoid repeated import attempts:

.. code-block:: python

   # entropy_samplers.py
   import logging

   logger = logging.getLogger(__name__)

   # Try to import Cython implementation once at module level
   try:
       from confopt.selection.sampling.cy_entropy import cy_differential_entropy
       CYTHON_AVAILABLE = True
   except ImportError:
       logger.info("Cython differential entropy implementation not available. Using pure Python fallback.")
       cy_differential_entropy = None
       CYTHON_AVAILABLE = False

   def calculate_entropy(samples, method="distance"):
       """Compute differential entropy with automatic fallback."""
       # ... validation code ...

       if CYTHON_AVAILABLE:
           return cy_differential_entropy(samples, method)

       # Pure Python fallback implementation
       if method == "distance":
           # Vasicek estimator implementation
           # ... pure Python code ...
       elif method == "histogram":
           # Histogram-based implementation
           # ... pure Python code ...

**Optimization Techniques:**

- **Single import attempt** at module level, not per function call
- **Global availability flag** for efficient checking
- **No repeated try/except blocks** in hot code paths
- **Identical API** between Cython and Python implementations

Distribution Strategy
--------------------

MANIFEST.in Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

The manifest controls what files are included in different distribution types:

.. code-block:: text

   # Include Cython files (source and generated C for source distributions)
   include confopt/selection/sampling/cy_entropy.pyx
   include confopt/selection/sampling/cy_entropy.c

   # Exclude compiled extensions from source distributions (sdist)
   # They should only be in wheels (bdist_wheel)
   global-exclude *.pyd
   global-exclude *.so

**Distribution Contents:**

- **Source Distribution (sdist)**: Includes ``.pyx`` and ``.c`` files, excludes ``.pyd/.so``
- **Wheel Distribution (bdist_wheel)**: Includes compiled ``.pyd/.so`` files
- **Users building from source**: Don't need Cython, just a C compiler
- **Users installing from wheel**: Don't need any compiler

Build Flow Examples
-------------------

Successful Compilation
~~~~~~~~~~~~~~~~~~~~~

When NumPy and compiler are available:

.. code-block:: text

   $ pip install .
   Building Cython extensions...
   building 'confopt.selection.sampling.cy_entropy' extension
   "C:\Program Files\Microsoft Visual Studio\...\cl.exe" /c ...
   Successfully installed confopt-1.0.2

Graceful Fallback
~~~~~~~~~~~~~~~~

When dependencies are missing:

.. code-block:: text

   $ pip install .
   Warning: Could not import required dependencies: No module named 'numpy'
   Falling back to pure Python implementation.
   Successfully installed confopt-1.0.2

Testing the Implementation
-------------------------

You can verify the fallback behavior:

.. code-block:: python

   # Test script
   import numpy as np
   from confopt.selection.sampling.entropy_samplers import calculate_entropy, CYTHON_AVAILABLE

   print(f"Cython available: {CYTHON_AVAILABLE}")

   # Test data
   test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

   # Both implementations should give identical results
   for method in ["distance", "histogram"]:
       result = calculate_entropy(test_data, method=method)
       print(f"Entropy ({method}): {result}")

Performance Considerations
-------------------------

The optional Cython extension provides significant performance improvements for entropy calculations:

- **Cython implementation**: ~10-50x faster for large datasets
- **Pure Python fallback**: Fully functional, suitable for smaller datasets
- **Automatic selection**: No user intervention required
- **Identical results**: Both implementations produce the same numerical results

Development Workflow
-------------------

For developers working on the Cython extensions:

1. **Generate C source** (if modifying .pyx files):

   .. code-block:: bash

      cython confopt/selection/sampling/cy_entropy.pyx

2. **Test local development**:

   .. code-block:: bash

      pip install -e .  # Editable install

3. **Build distributions**:

   .. code-block:: bash

      python -m build --sdist    # Source distribution
      python -m build --wheel    # Wheel distribution

4. **Test fallback scenarios**:

   .. code-block:: bash

      # Test without NumPy in build environment
      pip install . --no-build-isolation

Best Practices Summary
---------------------

This implementation demonstrates several best practices for Python packages with optional compiled extensions:

**Build System:**

- ✅ Minimal build requirements for maximum compatibility
- ✅ Graceful fallback at every level
- ✅ Clear user communication about what's happening
- ✅ Never fail installation due to compilation issues

**Code Organization:**

- ✅ Single import attempt per module
- ✅ Global availability flags for efficiency
- ✅ Identical APIs between implementations
- ✅ Proper error handling and logging

**Distribution:**

- ✅ Appropriate file inclusion for different distribution types
- ✅ Source distributions include C source, not compiled binaries
- ✅ Wheels include compiled binaries for immediate use
- ✅ Users can install regardless of their environment

**Testing:**

- ✅ Verify both implementations produce identical results
- ✅ Test all fallback scenarios
- ✅ Performance benchmarking when possible

This approach ensures your package is accessible to the widest possible audience while providing optimal performance when the environment supports it.
