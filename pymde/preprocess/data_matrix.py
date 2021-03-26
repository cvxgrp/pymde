import numpy as np
import scipy.sparse as sp
import torch

from pymde import problem
from pymde.preprocess.graph import Graph
from pymde.preprocess.preprocess import sample_edges
from pymde import util


def distances(data, retain_fraction=1.0, verbose=False):
    """Compute distances, given data matrix.

    This function computes distances between some pairs of items, given
    a data matrix. Each row in the data matrix is treated as an item.

    Arguments
    ---------
    data: torch.Tensor, np.ndarray, or scipy.sparse matrix
        The data matrix, shape ``(n_items, n_features)``.
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
    if not sp.issparse(data) and not isinstance(
        data, (np.ndarray, torch.Tensor)
    ):
        raise ValueError(
            "`data` must be a scipy.sparse matrix, NumPy array, "
            "or torch tensor"
        )

    n_items = int(data.shape[0])
    all_edges = n_items * (n_items - 1) / 2
    max_distances = int(retain_fraction * all_edges)

    if max_distances is None:
        max_distances = np.inf
    elif max_distances <= 0:
        raise ValueError("max_distances must be positive")

    if n_items * (n_items - 1) / 2 < max_distances:
        edges = util.all_edges(n_items)
    else:
        if verbose:
            problem.LOGGER.info(f"Sampling {int(max_distances)} edges")
        edges = sample_edges(n_items, int(max_distances))

    if sp.issparse(data):
        if not isinstance(data, sp.csr_matrix):
            data = data.tocsr()

        edges = edges.cpu().numpy()
        if verbose:
            problem.LOGGER.info(f"Computing {int(edges.shape[0])} distances")
        delta = torch.tensor(
            sp.linalg.norm(data[edges[:, 0]] - data[edges[:, 1]], axis=1),
            dtype=torch.float,
        )
        edges = torch.tensor(edges)
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float, device="cpu")

        edges = edges.to(data.device)
        if verbose:
            problem.LOGGER.info(f"Computing {int(edges.shape[0])} distances")
        # TODO(akshayka): Batch this computation when the number of edges
        # and/or the number of features is large.
        delta = (
            (data[edges[:, 0]] - data[edges[:, 1]])
            .pow(2)
            .sum(dim=1)
            .float()
            .sqrt()
        )

    return Graph.from_edges(edges, delta, n_items=n_items)


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

    duplicated_edges_mask = edges[:, 0] == edges[:, 1]
    if duplicated_edges_mask.any():
        problem.LOGGER.warning(
            "Your dataset appears to contain duplicated items (rows); "
            "when embedding, you should typically have unique items."
        )
        problem.LOGGER.warning(
            "The following items have duplicates "
            f"{edges[duplicated_edges_mask][:, 0]}"
        )
        edges = edges[~duplicated_edges_mask]

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
