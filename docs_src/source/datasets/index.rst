.. _datasets:

Datasets
========

The ``pymde.datasets`` module provides functions that download and return
some datasets. You can use these datasets to experiment with custom
MDE problems, or while learning how to use PyMDE.

Each function returns a :any:`pymde.datasets.Dataset` object. The ``data``
member of this object holds the raw data. The ``attributes`` member
is a dictionary whose values are (held-out) attributes associated with
the items; you can use these attributes to color your embeddings, when
visualizing them. Other data related to the dataset is in the ``other_data``
dict, and metadata about the dataset (eg, its authors) is available in
``metadata``.

The first time one of these functions is called, it will download the dataset and
cache it locally, in the current directory (change the directory with
the ``root`` keyword argument). Subsequent calls will use the cached data.

PyMDE currently provides the below datasets. If you would like to add an
additional dataset, please reach out to us on
`Github <https://github.com/cvxgrp/pymde>`_.

MNIST
-----

.. autofunction:: pymde.datasets.MNIST
   :noindex:

Fashion MNIST
-------------

.. autofunction:: pymde.datasets.FashionMNIST
   :noindex:

Google Scholar
--------------

.. autofunction:: pymde.datasets.google_scholar
   :noindex:


Academic interests
---------------------------------

.. autofunction:: pymde.datasets.google_scholar_interests
   :noindex:


scRNA transcriptomes from COVID-19 patients
-------------------------------------------

.. autofunction:: pymde.datasets.covid19_scrna_wilk
   :noindex:


Population genetics
-------------------

.. autofunction:: pymde.datasets.population_genetics
   :noindex:


US counties
-----------

.. autofunction:: pymde.datasets.counties
   :noindex:
