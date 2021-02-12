.. _preprocess:

Preprocessing
=============

This part of the tutorial discusses ways to preprocess your original 
data in order to create the set of edges, and their associated weights or
original deviations, needed to create an MDE problem. These preprocessing
methods are analogous to feature engineering in machine learning, where
raw data are converted to feature vectors. As in feature engineering, 
the preprocessing can have a strong effect on the final result, i.e., the
embedding.

Preprocessing methods can be grouped into two types: those that 
are based on *neighbors* of the items, and those that are based on 
*distances* between items. The high-level functions
:any:`pymde.preserve_neighbors` and :any:`pymde.preserve_distances`
(which were used in the *Getting Started* guide) use neighbor-based and
distance-based preprocessing behind-the-scenes, in order to create reasonable
MDE problems. But you can just as well preprocess the data yourself, to
create custom embeddings.

PyMDE provides a few preprocessing methods for original data that come
as a data matrix (one row for each item) or a graph (one node for each item).

Graph
-----
The preprocessing methods often work with or return ``pymde.Graph`` instances,
which package up a list of edges and their associated weights.

A graph can be created from a weighted adjacency matrix (a scipy.sparse matrix,
numpy array, or torch Tensor), or a torch Tensor containing the edges and
weights. For example

.. code:: python3

    adjacency_matrix = sp.csr_matrix(np.array([
       [0, 1, 1],
       [1, 0, 1],
       [1, 1, 0],
    ]))
    graph = pymde.Graph(adjacency_matrix) 

or

.. code:: python3

    edges = torch.tensor([
         [0, 1],
         [0, 2],
         [1, 2],
    ])
    weights = torch.ones(edges.shape[0])
    graph = pymde.Graph.from_edges(edges, weights)

Given a graph, you can access the edges, weights, and adjacency matrix with

.. code:: python3

   graph.edges
   graph.weights
   graph.adjacency_matrix

The :any:`API documentation <api_graph>` describes the :any:`pymde.Graph` class
in more detail.

Neighbor-based preprocessing
----------------------------

Similar items
"""""""""""""
A *neighbor* of an item can be thought of an item that is in some sense
similar to it. One class of preprocessing methods computes some neighbors
of each item, and uses the pairs of neighbors as edges, associating these
edges with positive weights. These weights can then be used to create
a distortion function based on weights, using one of the classes
in :any:`pymde.penalties`. A common preprocessing step is to compute the
``k``-nearest neighbors of each item, where ``k`` is a parameter.

.. code:: python3

   knn_graph = pymde.preprocess.k_nearest_neighbors(data, k=10)

Preprocessing based on neighbors can be thought of as a "sparsifying"
operation: they take data and return a sparse graph (the ``knn_graph``).

.. autofunction:: pymde.preprocess.k_nearest_neighbors
    :noindex:

Distance-based preprocessing
----------------------------

Another preprocessing step is to compute some distances between pairs of
items. We can use the :any:`pymde.preprocess.distances` function to compute
distances between pairs of items.

.. code:: python3

   distance_graph = pymde.preprocess.distances(data)

When the original input is a graph, this function can be thought of as a
"densifying" operation, since it returns a denser graph. You can control
how dense the returned graph is with a keyword argument.

.. autofunction:: pymde.preprocess.distances
   :noindex:
