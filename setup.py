import codecs
import distutils.sysconfig
import distutils.version
import os.path

from setuptools import setup, find_packages, Extension
import numpy as np


with open("README.md", "r") as fh:
    long_description = fh.read()


_graph = Extension(
    "pymde.preprocess._graph",
    sources=["pymde/preprocess/_graph.pyx"],
    # once Cython 3.0 is released, uncomment the below line
    # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    extra_compile_args=["-O3"],
    include_dirs=[np.get_include()],
)


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="pymde",
    version=get_version("pymde/__init__.py"),
    description="Minimum-Distortion Embedding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=18.0", "cython"],
    install_requires=[
        "matplotlib",
        "numpy >= 1.17.5",
        "pynndescent",
        "scipy",
        "torch",
        "torchvision",
        # torchvision requires requests but does not list it as a dependency
        "requests",  
    ],
    packages=find_packages(),
    ext_modules=[_graph],
    license="Apache License, Version 2.0",
    license_files=["LICENSE"],
    url="https://github.com/cvxgrp/pymde",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    author="Akshay Agrawal",
    author_email="akshayka@cs.stanford.edu",
)
