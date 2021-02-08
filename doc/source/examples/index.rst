.. _examples:

Examples
========

As with most software, after reading the documentation, the best way
to learn more is by playing with code.

Our Python notebooks show how to use PyMDE on real (and synthetic)
datasets. You can read these notebooks, or better yet, execute them and
experiment by modifying the cells' contents and seeing what happens.

You can run the notebooks by either downloading them locally and starting a
Jupyter server, or by opening them in Google Colab.

.. _example_mnist:

MNIST
-----
We recommend starting with our MNIST notebook, which highlights many of the
things you can do in PyMDE, using the MNIST dataset as a case study. 

- `MNIST notebook <https://github.com/cvxgrp/pymde/blob/main/examples/mnist.ipynb>`_


In this notebook, you'll see how to use the ``pymde.preserve_neighbors``
function to embed vector data, how to create MDE problems for preserving
neighbors from scratch, how to sanity-check an embedding, and how
to use an embedding to look for outliers in the original data.

.. _example_scrna:

Single-Cell Genomics
--------------------
This notebook is similar to the MNIST notebook (but with less explanatory
text). The dataset embedded here contains single-cell mRNA transcriptomes of
cells taken from human patients with severe COVID-19 infections (and also from
healthy controls). We'll see that similar cells are placed near each other in
the embedding, and cells from healthy and sick are also somewhat separated.

- `Single-cell genomics notebook <https://github.com/cvxgrp/pymde/blob/main/examples/single_cell_genomics.ipynb>`_

.. _example_google_scholar:

Google Scholar
--------------
The Google Scholar notebook uses ``pymde.preserve_distances`` to embed
an academic coauthorship network, which we collected from Google Scholar.
(This network contains most authors on Google Scholar whose h-index is at least
50.)

This example embeds a graph with roughly 40,000 nodes and (after preprocessing)
80 million edges. If you have a GPU, computing the embedding shouldn't take
much longer than a minute, but it will take longer to compute on a CPU.

- `Google Scholar notebook <https://github.com/cvxgrp/pymde/blob/main/examples/google_scholar.ipynb>`_ 

Dissimilar Edges and Negative Weights
-------------------------------------
When creating an embedding for preserving neighbors, an important hyper-parameter
is the number of dissimilar edges to include, and the size of the negative weights.
Using as many dissimilar edges as there are similar edges, and choosing
the negative weights to all be -1, usually works just fine. But different
choices do lead to different embeddings.

This notebook explores the effect these hyper-parameters have on the embedding.

- `Dissimilar edges and negative weights notebook <https://github.com/cvxgrp/pymde/blob/main/examples/dissimilar_edges_and_negative_weights.ipynb>`_

.. _example_graphs:

Drawing Graphs
--------------
PyMDE can be used to layout graphs in the Cartesian plane in an aesthetically
pleasing way. Compared to many other graph layout libraries, PyMDE can scale
to higher datasets. And, of course, PyMDE lets you design custom layouts, by
choosing your own distortion functions and constraints.

This notebook shows various ways of drawing graphs with PyMDE. It also
introduces the ``pymde.Graph`` class.

- `Drawing graphs notebook <https://github.com/cvxgrp/pymde/blob/main/examples/drawing_graphs.ipynb>`_
