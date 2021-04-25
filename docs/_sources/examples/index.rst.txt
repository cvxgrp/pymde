.. _examples:

Examples
========

After reading the documentation, the best way
to learn more is by playing with code examples.

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

.. _example_fashion_mnist:

Fashion MNIST
-------------

The Fashion MNIST notebook is analogous to the MNIST notebook, except
it uses the Fashion MNIST dataset.

- `Fashion MNIST notebook <https://github.com/cvxgrp/pymde/blob/main/examples/fashion_mnist.ipynb>`_

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

Word Embedding
--------------
This notebook shows how to make basic word embeddings. The words being embedded
are the 5000 most popular academic interests on Google Scholar.

- `Word embedding notebook <https://github.com/cvxgrp/pymde/blob/main/examples/word_embedding.ipynb>`_ 


Population Genetics
-------------------
The population genetics notebook embeds genomic data of individuals thought
to be of European ancestry, and recovers what appears to be a map of Europe.

- `Population genetics notebook <https://github.com/cvxgrp/pymde/blob/main/examples/population_genetics.ipynb>`_

US Counties
-----------
The US counties notebook embeds 3,220 US counties, described by demographic
data, into two dimensions. The resulting embedding is colored by the
fraction of voters who voted for Democratic candidates in the 2016 presidential
election (voting data was not used in computing the embedding). Moreover,
the resulting embedding vaguely resembles a map of the US (though no geographic
data was used in computing the embedding).

- `US counties notebook <https://github.com/cvxgrp/pymde/blob/main/examples/counties.ipynb>`_

Anchor Constraints
------------------
With an anchor constraint, you can pin some embedding vectors to values
that you specify ahead of time. This is useful when you have prior
knowledge of where some of the items should end up (e.g., you might
be doing semi-supervised learning, or you might be laying out a graph with
some nodes pinned in place).

This notebook gives an example of how to use an anchor constraint, using
graph drawing as an example.

- `Anchor constraint notebook <https://github.com/cvxgrp/pymde/blob/main/examples/anchor_constraints.ipynb>`_

Updating Embeddings
-------------------
With PyMDE, you can easily add new points to an existing embedding using
an anchor constraint (to pin the existing embedding in place).

This notebook gives an example of how to do this, using MNIST as an example.
We first embed half the images in the MNIST dataset. Then we augment
the embedding with vectors for the remaining images.

- `Updating embedding notebook <https://github.com/cvxgrp/pymde/blob/main/examples/updating_an_existing_embedding.ipynb>`_

Drawing Graphs
--------------
PyMDE can be used to layout graphs in the Cartesian plane in an aesthetically
pleasing way. Compared to many other graph layout libraries, PyMDE can scale
to much larger datasets. PyMDE also lets you design custom layouts, by
choosing your own distortion functions and constraints.

This notebook shows various ways of drawing graphs with PyMDE. It also
introduces the ``pymde.Graph`` class.

- `Drawing graphs notebook <https://github.com/cvxgrp/pymde/blob/main/examples/drawing_graphs.ipynb>`_

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

