import numpy as np
import scipy.sparse as sp
import torch

from pymde.preprocess import data_matrix
from pymde.preprocess import graph


def _is_data_matrix(data):
    return sp.issparse(data) or isinstance(data, (np.ndarray, torch.Tensor))


def distances(data, retain_fraction=1.0, verbose=False):
    """Compute distances, given data matrix or graph.

    This function computes distances between some pairs of items, given
    either a data matrix or a ``pymde.Graph`` instance.

    When the input is a data matrix, each row is treated as the feature
    vector of an item.

    When the input is a graph, each node is an item, and the distance between
    two items is taken to be the length of the shortest-path between them.

    The ``retain_fraction`` keyword argument can be used
    to compute only a fraction of the distances. This can be useful
    when there are many items, in which case storing all
    the distances may be intractable.


    Arguments
    ---------
    data: torch.Tensor, np.ndarray, scipy.sparse matrix, or pymde.Graph
        A data matrix, shape ``(n_items, n_features)``, or a ``pymde.Graph``
    retain_fraction: float, optional
        A float between 0 and 1, specifying the fraction of all ``(n_items
        choose 2)`` to compute. For example, if ``retain_fraction`` is 0.1,
        only 10 percent of the edges will be stored.
    verbose:
        If ``True``, print verbose output.

    Returns
    -------
    pymde.Graph
        A graph object holding the distances and corresponding edges.
        Access the distances with ``graph.distances``, and the edges
        with ``graph.edges``.
    """

    if _is_data_matrix(data):
        return data_matrix.distances(
            data, retain_fraction=retain_fraction, verbose=verbose
        )
    else:
        return graph.shortest_paths(
            data, retain_fraction=retain_fraction, verbose=verbose
        )


def k_nearest_neighbors(data, k, max_distance=None, verbose=False):
    """Compute k-nearest neighbors, given data matrix or graph.

    This function computes a k-nearest neighbor graph, given either
    a data matrix or an original graph.

    When the input is a data matrix, each row is treated as the feature
    vector of an item, and the Euclidean distance is used to judge how
    close two items are to each other.

    When the input is a graph, each node is an item, and the distance between
    two items is taken to be the length of the shortest-path between them.

    In the returned graph, if i is a neighbor of j and j a neighbor of i, then
    the weight w_{ij} will be +2; if only one is a neighbor of the other, then
    w_{ij} will be +1; if neither are neighbors of each other, then (i, j) will
    not be in the graph.

    Arguments
    ---------
    data: torch.Tensor, np.ndarray, scipy.sparse matrix, or pymde.Graph
        A data matrix, shape ``(n_items, n_features)``, or a ``pymde.Graph``
    k: int
        The number of nearest neighbors per item
    max_distance: float (optional)
        If not ``None``, neighborhoods are restricted to have a radius
        no greater than `max_distance`.
    verbose: bool
        If ``True``, print verbose output.

    Returns
    -------
    pymde.Graph
        The k-nearest neighbor graph. Access the weights with
        ``graph.weights``, and the edges with ``graph.edges``
    """

    if _is_data_matrix(data):
        return data_matrix.k_nearest_neighbors(
            data, k=k, max_distance=max_distance, verbose=verbose
        )
    else:
        return graph.k_nearest_neighbors(
            data,
            k=k,
            graph_distances=True,
            max_distance=max_distance,
            verbose=verbose,
        )
