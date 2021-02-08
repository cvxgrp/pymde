.. _installation:

Installation
============

PyMDE is available on PyPI. Install it with

.. code::

    pip install -U pymde

We recommend installing PyMDE in a virtual environment of your choice,
such as virtualenv, venv, or a conda environment.

PyMDE has the following requirements:

* Python >= 3.7
* numpy >= 1.17.5
* scipy
* torch
* torchvision
* pynndescent

If you have a CUDA-enabled GPU, PyMDE can use it to speed up embedding
computations. PyMDE relies on PyTorch for CUDA acceleration. For
instructions on how to install a CUDA-enabled version of PyTorch, refer to the
`PyTorch documentation <https://pytorch.org/>`_.
