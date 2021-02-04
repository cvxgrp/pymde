import numpy as np
import torch

from pymde import problem
from pymde.preprocess.graph import Graph


def k_nearest_neighbors(data, k, max_distance=None, verbose=False):
    """Compute k-nearest neighbors for each row in data matrix.

    Computes the k-nearest neighbor graph of data matrix, under
    the Euclidean distance. Each row in the data matrix is treated as an item.

    Arguments
    ---------
    data: {torch.Tensor, np.ndarray, scipy.sparse matrix}(
            shape=(n_items, n_features))
        The data matrix
    k: int
        The number of nearest neighbors per item
    max_distance: float (optional)
        If not None, neighborhoods are restricted to have a radius
        no greater than `max_distance`.
    verbose: bool
        If True, print verbose output.

    Returns
    -------
    pymde.Graph
        a neighborhood graph
    """
    # lazy import, because importing pynndescent takes some time
    import pynndescent

    if isinstance(data, torch.Tensor):
        device = data.device
        data = data.cpu().numpy()
    else:
        device = "cpu"

    n = data.shape[0]
    if n < 10000:
        import sklearn.neighbors

        if verbose:
            problem.LOGGER.info("Exact nearest neighbors by brute force ")
        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k + 1, algorithm="brute"
        )
        nn.fit(data)
        distances, neighbors = nn.kneighbors(data)
    else:
        # TODO default params (n_trees, max_candidates)
        index = pynndescent.NNDescent(
            data,
            n_neighbors=k + 1,
            verbose=verbose,
            max_candidates=60,
        )
        neighbors, distances = index.neighbor_graph
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]

    n = data.shape[0]
    items = np.arange(n)
    items = np.repeat(items, k)
    edges = np.stack([items, neighbors.flatten()], axis=1)

    flip_idx = edges[:, 0] > edges[:, 1]
    edges[flip_idx] = np.stack(
        [edges[flip_idx][:, 1], edges[flip_idx][:, 0]], axis=1
    )

    weights = torch.ones(edges.shape[0], device=device, dtype=torch.float)
    if max_distance is not None:
        weights[
            torch.tensor(distances.ravel(), device=device, dtype=torch.float)
            > max_distance
        ] = 0.0

    # weights for duplicated edges will be summed.
    edges = torch.tensor(edges, device=device)
    return Graph.from_edges(edges, weights)


# TODO(akshayka) figure out this api ...
def _neighborhood_graph(
    data, n_neighbors=None, threshold=None, max_distances=None
):
    if n_neighbors is None and threshold is None:
        n_neighbors = 15
    elif n_neighbors is not None and threshold is not None:
        raise ValueError(
            "only one of n_neighbors and threshold can be non-none"
        )

    if n_neighbors is not None:
        return k_nearest_neighbors(data, n_neighbors)
    else:
        # TODO: move from recipes to here
        raise NotImplementedError


def _distances(index):
    pass


def _distance_matrix(data, max_distances=None):
    """Compute a distance matrix from a data matrix"""
    # TODO(akshayka): move from recipes to here
    # return vector of distances/edges, or a graph, or distance matrix?
    raise NotImplementedError
