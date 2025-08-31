#!/usr/bin/env python
"""Optional Cython extension setup with graceful fallback.

This setup.py attempts to build Cython extensions but gracefully falls back
to pure Python if compilation fails. All other metadata is defined in pyproject.toml.
"""

import os
from setuptools import Extension, setup


def build_extensions():
    """Attempt to build Cython extensions with graceful fallback."""
    try:
        import numpy as np

        # Check if C source file exists
        c_file = "confopt/selection/sampling/cy_entropy.c"
        if not os.path.exists(c_file):
            print(
                f"Warning: C source file {c_file} not found. Skipping Cython extension."
            )
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
        print(
            f"Warning: Could not import required dependencies for Cython compilation: {e}"
        )
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

# Use setup() with minimal configuration - pyproject.toml handles the rest
setup(
    ext_modules=ext_modules,
)
