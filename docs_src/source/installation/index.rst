.. _installation:

Installation
============

PyMDE is available on PyPI, and on Conda Forge. We recommend installing PyMDE
in a virtual environment of your choice, such as virtualenv, venv, or a conda
environment.

To install with Python pip, use

.. code::

    pip install -U pymde

To install with conda, use

.. code::

  conda install -c pytorch -c conda-forge pymde


Requirements
------------

PyMDE has the following requirements:

* Python >= 3.9
* numpy >= 2.0
* pynndescent
* requests
* scipy
* torch >= 1.7.1
* torchvision >= 0.8.2

GPU acceleration
----------------
If you have a CUDA-enabled GPU, PyMDE can use it to speed up embedding
computations. PyMDE relies on PyTorch for CUDA acceleration. For
instructions on how to install a CUDA-enabled version of PyTorch, refer to the
`PyTorch documentation <https://pytorch.org/>`_.
