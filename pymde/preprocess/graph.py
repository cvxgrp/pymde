import sys

import numpy as np
import scipy.sparse as sp
import torch


from pymde import constraints
from pymde import problem
from pymde.functions import penalties, losses
from pymde.preprocess import _graph


__this_module = sys.modules[__name__]


LOGGER = problem.LOGGER


def _to_edges(graph):
    if not sp.issparse(graph):
        n = graph.shape[0]
        edges = torch.triu_indices(n, n, 1).T
        distances = graph[edges]
        return edges, distances

    # discard the lower triangular part, which is redundant
    graph = sp.triu(graph, format="coo")
    edges = torch.stack(
        [
            torch.tensor(graph.row, dtype=torch.long),
            torch.tensor(graph.col, dtype=torch.long),
        ],
        dim=1,
    )
    data = torch.tensor(graph.data, dtype=torch.float)
    return edges, data


def _validate_adjacency_matrix(matrix):
    diagonal_nonzero = matrix.diagonal() > 0
    if diagonal_nonzero.any():
        raise ValueError(
            "Adjacency matrices must not contain self edges; "
            "the following nodes were found to have self edges: ",
            np.argwhere(diagonal_nonzero).flatten(),
        )


def _to_graph(edges, distances=None, n_items=None):
    "Distances for repeated edges are summed."
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy()

    if distances is None:
        distances = np.ones(edges.shape[0], dtype=np.float32)
    elif isinstance(distances, torch.Tensor):
        distances = distances.cpu().float().numpy()

    flip_idx = edges[:, 0] > edges[:, 1]
    edges[flip_idx] = np.stack(
        [edges[flip_idx][:, 1], edges[flip_idx][:, 0]], axis=1
    )

    if n_items is None:
        n_items = edges.max() + 1
    rows = edges[:, 0]
    cols = edges[:, 1]
    graph = sp.coo_matrix((distances, (rows, cols)), shape=(n_items, n_items))
    graph = graph + graph.T
    return Graph(graph.tocsr())


class Graph(object):
    """A weighted graph.

    This class represents a weighted graph. It is backed by a scipy.sparse
    adjacency matrix, and can be constructed from either a dense
    distance matrix, a sparse adjacency matrix, or torch.Tensor's
    of edges and weights (using the `from_edges` static method).

    It is an error for the graph to contain self-edges (i.e., non-zero
    values on the diagonal).

    This class is accepted as an argument to various preprocessing functions.
    """

    def __init__(self, adjacency_matrix):
        if isinstance(adjacency_matrix, np.ndarray):
            adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        elif isinstance(adjacency_matrix, torch.Tensor):
            adjacency_matrix = sp.csr_matrix(adjacency_matrix.cpu().numpy())
        elif sp.issparse(adjacency_matrix) and not isinstance(
            adjacency_matrix, sp.csr_matrix
        ):
            LOGGER.warning(
                "The adjacency matrix for a Graph object should be "
                "a CSR matrix; converting your matrix to a CSR "
                "matrix, which may be costly ..."
            )
            adjacency_matrix = adjacency_matrix.tocsr()

        unreachable = adjacency_matrix.data == np.inf
        adjacency_matrix.data[unreachable] = 0
        adjacency_matrix.eliminate_zeros()
        _validate_adjacency_matrix(adjacency_matrix)

        self._adjacency_matrix = adjacency_matrix
        self._edges = None
        self._distances = None

    @staticmethod
    def from_edges(edges, weights=None, n_items=None):
        """Construct a graph from edges and weights.

        Arguments
        ---------
        edges: torch.Tensor
            Tensor of edges, of shape ``(n_edges, 2)``, with each
            edge represented by a row, i.e. by two integers.
        weights: torch.Tensor, optional
            Tensor of weights, of shape ``(n_edges,)``, with each weight
            a float.
        n_items: int, optional
            The number of items in the graph; if ``None``, this is
            taken to be the maximum value in ``edges`` plus 1.

        Returns
        -------
        pymde.Graph
            A Graph object representing ``edges`` and ``weights``.
        """
        return _to_graph(edges, weights, n_items)

    @property
    def adjacency_matrix(self):
        """The scipy.sparse adjacency matrix backing the graph."""
        return self._adjacency_matrix

    @property
    def A(self):
        return self.adjacency_matrix

    @A.setter
    def A(self, value):
        self.adjacency_matrix = value

    @property
    def edges(self):
        """A torch.Tensor of the edges in the graph."""
        if self._edges is None:
            edges, distances = _to_edges(self.A)
            self._edges = edges
            self._distances = distances
        return self._edges

    @property
    def distances(self):
        """The distances associated with each edge."""
        if self._distances is None:
            edges, distances = _to_edges(self.A)
            self._edges = edges
            self._distances = distances
        return self._distances

    @property
    def weights(self):
        return self.distances

    @property
    def n_items(self):
        """The number of items in the graph."""
        return self.A.shape[0]

    @property
    def n_edges(self):
        """The number of edges in the graph."""
        return self.edges.shape[0]

    @property
    def n_all_edges(self):
        """n_items choose 2."""
        return self.n_items * (self.n_items - 1) / 2

    def neighbors(self, node: int) -> np.ndarray:
        """The indices of the neighbors of ``node``."""
        return self.A.indices[self.A.indptr[node] : self.A.indptr[node + 1]]

    def neighbor_distances(self, node) -> np.ndarray:
        """The distances associated with the edges connected to ``node``."""
        return self.A.data[self.A.indptr[node] : self.A.indptr[node + 1]]

    def __getitem__(self, key):
        return self.A[key]

    def __setitem__(self, key, value):
        raise AttributeError("Graph objects are immutable.")

    def draw(
        self, embedding_dim=2, standardized=False, device="cpu", verbose=False
    ):
        """Draw a graph in the Cartesian plane.

        This method does some basic preprocessing, constructs an MDE problem
        that is often suitable for drawing graphs, and computes/returns an
        embedding by approximately solving the MDE problem.

        Arguments
        ---------
        embedding_dim: int
            The number of dimemsions, 1, 2, or 3.
        standardized: bool
            Whether to impose a standardization constraint.
        device: str
            Device on which to compute/store embedding, 'cpu' or 'cuda'.
        verbose: bool
            Whether to print verbose output.

        Returns
        -------
        torch.Tensor
            The embedding, of shape ``(n_items, embedding_dim)``
        """
        if (self.distances < 0).any():
            raise ValueError(
                "Graphs with negative edge weights cannot be drawn."
            )

        if self.n_edges < 1e7 and self.n_all_edges > 1e7:
            retain_fraction = 1e7 / self.n_all_edges
            distance_graph = shortest_paths(
                self, retain_fraction=retain_fraction, verbose=verbose
            )
        else:
            distance_graph = self

        if not standardized:
            constraint = constraints.Centered()
            f = losses.WeightedQuadratic(distance_graph.distances)
        else:
            constraint = constraints.Standardized()
            # TODO(akshayka) better weights
            f = penalties.Cubic(1 / distance_graph.distances)
        mde = problem.MDE(
            n_items=self.n_items,
            embedding_dim=embedding_dim,
            edges=distance_graph.edges,
            distortion_function=f,
            constraint=constraint,
            device=device,
        )
        X = mde.embed(verbose=verbose)
        mde.plot(edges=self.edges)
        return X


def scale(graph, natural_length):
    """Scale graph weights to have RMS equal to `natural_length`

    Returns a new graph, leaving the old graph unmodified.

    Arguments
    ---------
    graph: pymde.Graph
        The graph whose distances to scale.
    natural_length: float
        Target RMS value.

    Returns
    -------
    pymde.Graph
        A new graph with scaled distances.
    """
    rms = graph.distances.pow(2).mean().sqrt()
    alpha = natural_length / rms
    distances = alpha * graph.distances
    return _to_graph(graph.edges, distances.cpu().numpy())


def _sparsify(graph, retain_fraction):
    raise NotImplementedError


def breadth_first_order(
    csgraph, i_start, directed=True, return_predecessors=True
):
    N = csgraph.shape[0]
    node_list = np.empty(N, dtype=np.int32)
    lengths = np.empty(N, dtype=np.int32)
    predecessors = np.empty(N, dtype=np.int32)
    node_list.fill(-9999)
    lengths.fill(-9999)
    predecessors.fill(-9999)

    if directed:
        _graph._breadth_first_directed(
            i_start,
            csgraph.indices,
            csgraph.indptr,
            node_list,
            lengths,
            predecessors,
        )
    lengths = lengths.astype(np.float32)
    lengths[lengths < 0] = np.inf
    return lengths, predecessors


def _shortest_paths(shape, node, max_length, unweighted, sample_prob):
    data = __this_module.__data
    indptr = __this_module.__indptr
    indices = __this_module.__indices

    data, indptr, indices = map(np.ctypeslib.as_array, [data, indptr, indices])
    A = sp.csr_matrix((data, indices, indptr), shape=shape)
    if unweighted:
        # fast path
        distances, _ = breadth_first_order(
            A,
            node,
        )
        distances[distances > max_length] = np.inf
    else:
        distances = sp.csgraph.dijkstra(A, indices=node, limit=max_length)

    indices = np.argwhere((distances != np.inf) * (distances > 0)).ravel()
    indices = indices[indices > node]
    if sample_prob < 1.0:
        sample_mask = (
            np.random.default_rng().uniform(size=indices.size) <= sample_prob
        )
        indices = indices[sample_mask]

    distances = distances[indices]
    return distances, indices


def __init_process(data, indptr, indices):
    global __this_module
    __this_module.__data = data
    __this_module.__indptr = indptr
    __this_module.__indices = indices


def shortest_paths(
    graph, max_length=None, retain_fraction=1.0, n_workers=None, verbose=False
):
    """Compute shortest-path distances.

    This function computes the shortest-path distances on a graph.
    It returns a new graph, with an edge for each distance that was
    computed (each edge is weighted by the shortest-path distance between
    the two nodes in the original graph). The new graph can be interpreted
    as a "filled-in" version of the input graph.

    The ``max_length`` and ``retain_fraction`` keyword arguments can be used
    to compute only a subset of the distances. This can be useful
    if the graph has a very large number of nodes, in which storing all graph
    distances may be intractable.

    If the graph is sufficiently large, multiple cores will be used to
    accelerate the computation.

    Arguments
    ---------
    graph: pymde.Graph or scipy.sparse matrix
        Graph instance, or an adjacency matrix
    max_length: float, optional
        The maximum-length path to compute; paths longer than max_length
        are not computed/explored.
    retain_fraction: float
        A float between 0 and 1, specifying the fraction of all (n choose 2)
        to compute. For example, if ``retain_fraction`` is 0.1, only 10
        percent of the edges will be stored.
    n_workers: int, optional
        The number of processes to use. Defaults to the number of available
        cores.
    verbose: bool
        Whether to print verbose output.

    Returns
    -------
    pymde.Graph
        A new graph, with an edge (and weight) for each shortest-path
        distance that was computed/stored.
    """
    if sp.issparse(graph):
        graph = Graph(graph)
    elif not isinstance(graph, Graph):
        raise ValueError(
            "`graph` must be a pymde.Graph instance or "
            "scipy.sparse adjacency matrix."
        )

    if verbose:
        LOGGER.info(
            f"Computing shortest path distances (retaining "
            f"{100*retain_fraction:.2f} percent with "
            f"max_distance={max_length}) ..."
        )

    if n_workers is None and max_length is None and retain_fraction == 1.0:
        # obtain a dense distance matrix
        distance_matrix = sp.csgraph.shortest_path(graph.A, directed=False)
        return Graph(sp.csr_matrix(distance_matrix))
    elif max_length is None:
        max_length = np.inf
    unweighted = (graph.distances == 1.0).all()
    if verbose and not unweighted:
        LOGGER.info("Graph is weighted ... using slow path.")

    # call dijkstra n times and compress the output in each iteration
    # (Calling just once, with indices = [1, ..., n], would return a dense
    # n times n matrix.)
    import multiprocessing
    from multiprocessing import sharedctypes

    def to_shared_memory(array):
        array = np.ctypeslib.as_ctypes(array)
        return sharedctypes.RawArray(array._type_, array)

    global __this_module
    data = __this_module.__data = to_shared_memory(graph.A.data)
    indptr = __this_module.__indptr = to_shared_memory(graph.A.indptr)
    indices = __this_module.__indices = to_shared_memory(graph.A.indices)

    nodes = np.arange(graph.n_items)
    print_every = max(1, nodes.shape[0] // 10)
    n_workers = multiprocessing.cpu_count() if n_workers is None else n_workers
    with multiprocessing.Pool(
        n_workers, initializer=__init_process, initargs=(data, indptr, indices)
    ) as pool:
        try:
            async_ret = [
                pool.apply_async(
                    _shortest_paths,
                    (
                        graph.A.shape,
                        node,
                        max_length,
                        unweighted,
                        retain_fraction,
                    ),
                )
                for node in nodes
            ]
            values = []
            for idx, ret in enumerate(async_ret):
                data, columns = ret.get()
                values.append((data, columns))
                if verbose and idx % print_every == 0:
                    LOGGER.info(f"processed node {idx + 1}/{nodes.shape[0]}")
        except:  # noqa: E722
            del __this_module.__data
            del __this_module.__indptr
            del __this_module.__indices
            raise

    del __this_module.__data
    del __this_module.__indptr
    del __this_module.__indices

    data, columns = zip(*values)
    graph.A.eliminate_zeros()

    indptr = [0]
    for row in range(graph.n_items):
        indptr.append(indptr[-1] + columns[row].size)
    data = np.concatenate(data)
    indices = np.concatenate(columns)
    adj_matrix = sp.csr_matrix((data, indices, indptr), shape=graph.A.shape)

    adj_matrix = adj_matrix + adj_matrix.T.tocsr()
    return Graph(adj_matrix)


def _minimum_spanning_tree(graph):
    return Graph(sp.csgraph.minimum_spanning_tree(graph.A))


def _thresholded_neighborhood_graph(graph, threshold, max_length):
    graph = shortest_paths(graph, max_length)

    data = []
    indices = []
    indptr = [0]
    n = graph.n_items
    for node in range(n):
        distances = graph.neighbor_distances(node)
        columns = graph.neighbors(node)
        thresholded_indices = distances[distances <= threshold]

        data.append(distances[thresholded_indices])
        indices.append(columns[thresholded_indices])
        indptr.append(indptr[-1] + thresholded_indices.size)

    data = np.concatenate(data)
    indices = np.concatenate(indices)
    indptr = np.array(indptr)
    return Graph(sp.csr_matrix((data, indices, indptr), shape=graph.A.shape))


def k_nearest_neighbors(
    graph,
    k,
    graph_distances=False,
    max_distance=None,
    verbose=False,
):
    """Compute k-nearest neighbors for each node in graph.

    This function computes a k-nearest neighbor graph.

    By default, the input graph is interpreted as a distance matrix, with
    ``graph.adjacency_matrix[i, j]`` giving the distance between i and j. If
    `graph_distances` is True, then the shortest-path metric is used to
    compute neighbors.

    In the returned graph, if i is a neighbor of j and j a neighbor of i, then
    the weight w_{ij} will be +2; if only one is a neighbor of the other, then
    w_{ij} will be +1; if neither are neighbors of each other, then (i, j) will
    not be in the graph.

    Arguments
    ---------
    graph: pymde.Graph
        The graph, representing a distance metric on the items.
    k: int
        The number of nearest neighbors per item.
    graph_distances: bool
        If ``True``, computes shortest-path distances on the graph; otherwise,
        interprets the graph as a distance matrix.
    max_distance: float (optional)
        If not ``None``, neighborhoods are restricted to have a radius
        no greater than max_distance.
    verbose: bool
        If ``True``, print verbose output.

    Returns
    -------
    pymde.Graph
        The k-nearest neighbor graph.
    """

    if sp.issparse(graph):
        graph = Graph(graph)
    elif not isinstance(graph, Graph):
        raise ValueError(
            "`graph` must be a pymde.Graph instance or "
            "scipy.sparse adjacency matrix."
        )

    if graph_distances:
        graph = shortest_paths(graph, max_length=max_distance, verbose=verbose)

    if max_distance is None:
        max_distance = np.inf

    coo_data = []
    coo_rows = []
    coo_cols = []
    for node in range(graph.n_items):
        dist = graph.neighbor_distances(node)
        cols = graph.neighbors(node)
        if cols.size == 0:
            continue

        dist[dist > max_distance] = np.inf
        knn_indices = np.argsort(dist)[:k]

        coo_rows.extend([node for _ in range(knn_indices.size)])
        coo_cols.extend([c for c in cols[knn_indices]])
        coo_data.extend([d for d in dist[knn_indices]])
    coo_graph = sp.coo_matrix(
        (coo_data, (coo_rows, coo_cols)), shape=graph.A.shape
    )

    # weights will be 0, 1, or 2
    coo_graph.data = np.ones(coo_graph.data.shape)
    coo_graph = coo_graph + coo_graph.T
    csr_graph = coo_graph.tocsr()
    return Graph(csr_graph)


# TODO(akshayka) including both threshold and max_length is confusing. fix
# this api
def _neighborhood_graph(
    graph, n_neighbors=None, threshold=None, max_length=None
):
    if n_neighbors is None and threshold is None:
        raise ValueError("either n_neighbors or threshold must be non-None")
    if n_neighbors is not None and threshold is not None:
        raise ValueError(
            "only one of n_neighbors and threshold can be non-none"
        )

    if n_neighbors is not None:
        return k_nearest_neighbors(graph, n_neighbors, max_length)
    return _thresholded_neighborhood_graph(graph, threshold, max_length)
