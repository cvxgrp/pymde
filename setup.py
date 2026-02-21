"""Minimal setup.py for Cython extensions (all metadata is in pyproject.toml)."""

from setuptools import setup, Extension
import numpy as np

setup(
    ext_modules=[
        Extension(
            "pymde.preprocess._graph",
            sources=["pymde/preprocess/_graph.pyx"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=["-O3"],
            include_dirs=[np.get_include()],
        )
    ],
)
