#!/usr/bin/env python
"""Optional Cython extension setup with graceful fallback.

This setup.py attempts to build Cython extensions but gracefully falls back
to pure Python if compilation fails. All other metadata is defined in pyproject.toml.
"""

import os
from setuptools import Extension, setup


def build_extensions():
    """Attempt to build Cython extensions with graceful fallback."""
    # Check if we're forcing Cython compilation (e.g., for cibuildwheel)
    force_cython = os.environ.get("CONFOPT_FORCE_CYTHON", "0") == "1"

    try:
        import numpy as np
        from Cython.Build import cythonize

        # Check if Cython source file exists
        pyx_file = "confopt/selection/sampling/cy_entropy.pyx"
        if not os.path.exists(pyx_file):
            msg = f"Cython source file {pyx_file} not found. Skipping Cython extension."
            if force_cython:
                raise RuntimeError(f"CONFOPT_FORCE_CYTHON=1 but {msg}")
            print(f"Warning: {msg}")
            return []

        # Define Cython extensions
        extensions = [
            Extension(
                "confopt.selection.sampling.cy_entropy",
                sources=[pyx_file],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                language="c",
            )
        ]

        # Cythonize the extensions
        cythonized_extensions = cythonize(
            extensions,
            compiler_directives={"language_level": 3},
            build_dir="build",
            annotate=False,
        )

        print("Building Cython extensions from .pyx source...")
        print("Extension module: confopt.selection.sampling.cy_entropy")
        print(f"Cython source file: {pyx_file}")
        print(f"NumPy include dir: {np.get_include()}")
        print(f"Force Cython: {force_cython}")
        print(f"Cythonized {len(cythonized_extensions)} extension(s)")
        return cythonized_extensions

    except ImportError as e:
        msg = f"Could not import required dependencies for Cython compilation: {e}"
        if force_cython:
            raise RuntimeError(f"CONFOPT_FORCE_CYTHON=1 but {msg}")
        print(f"Warning: {msg}")
        print("Falling back to pure Python implementation.")
        return []
    except Exception as e:
        msg = f"Cython extension compilation failed: {e}"
        if force_cython:
            raise RuntimeError(f"CONFOPT_FORCE_CYTHON=1 but {msg}")
        print(f"Warning: {msg}")
        print("Falling back to pure Python implementation.")
        return []


# Build extensions with fallback
try:
    ext_modules = build_extensions()
except Exception as e:
    force_cython = os.environ.get("CONFOPT_FORCE_CYTHON", "0") == "1"
    if force_cython:
        print(f"Error: Extension building failed with CONFOPT_FORCE_CYTHON=1: {e}")
        raise
    print(f"Warning: Extension building failed: {e}")
    print("Installing without Cython extensions.")
    ext_modules = []

# Use setup() with minimal configuration - pyproject.toml handles the rest
setup(
    ext_modules=ext_modules,
    zip_safe=False,  # Important for Cython extensions
)
