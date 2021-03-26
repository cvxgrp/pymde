"""Recipes for constructing MDE problems.
"""
import numpy as np
import scipy.sparse
import torch


from pymde import problem
from pymde import quadratic
from pymde import constraints
from pymde.functions import penalties, losses
from pymde import preprocess


def preserve_distances(
    data,
    embedding_dim=2,
    loss=losses.Absolute,
    constraint=None,
    max_distances=5e7,
    device="cpu",
    verbose=False,
) -> problem.MDE:
    """Construct an MDE problem based on original distances.

    This function constructs an MDE problem for preserving pairwise
    distances between items. This can be useful for preserving the global
    structure of the data.

    The data can be specified with either a data matrix (a NumPy array, torch
    Tensor, or sparse matrix), or a ``pymde.Graph`` instance encoding the
    distances:

        A NumPy array, torch tensor, or sparse matrix is interpreted as a
        collection of feature vectors: each row gives the feature vector for an
        item. The original distances are the Euclidean distances between the
        feature vectors.

        A ``pymde.Graph`` instance is interpreted as encoding all (n_items
        choose 2) distances: the distance between i and j is taken to be the
        length of the shortest path connecting i and j.

    When the number of items n_items is large, the total number of pairs will
    be very large. When this happens, instead of computing all pairs of
    distances, this function will sample a subset uniformly at random. The
    maximum number of distances to compute is specified by the parameter
    ``max_distances``. Depending on how many items you have (and how much
    memory your machine has), you may need to adjust this parameter.

    To obtain an embedding, call the ``embed`` method on the returned object.
    To plot it, use ``pymde.plot``.

    For example:

    .. code:: python3

        embedding = pymde.preserve_distances(data).embed()
        pymde.plot(embedding)

    Arguments
    ---------
    data: {np.ndarray, torch.Tensor, scipy.sparse matrix}(
            shape=(n_items, n_features)), or pymde.Graph
        A data matrix or a ``pymde.Graph`` instance.
    embedding_dim: int
        The embedding dimension.
    loss: pymde.Function class (or factory)
        Callable that constructs a distortion function, given
        original distances. Typically one of the classes defined in
        ``pymde.losses``, such as ``pymde.losses.Absolute``, or
        ``pymde.losses.WeightedQuadratic``.
    constraint: pymde.constraints.Constraint (optional)
        Embedding constraint, such as ``pymde.Standardized()`` or
        ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
        constraint. Note: when the constraint is ``pymde.Standardized()``,
        the original distances will be scaled by a constant (because the
        standardization constraint puts a limit on how large any one
        distance can be).
    max_distances: int
        Maximum number of distances to compute.
    device: str (optional)
        Device for the embedding (eg, 'cpu', 'cuda').
    verbose: bool
        If ``True``, print verbose output.

    Returns
    -------
    pymde.MDE
        A ``pymde.MDE`` instance, based on preserving the original distances.
    """
    if not isinstance(
        data, (np.ndarray, torch.Tensor, preprocess.graph.Graph)
    ) and not scipy.sparse.issparse(data):
        raise ValueError(
            "`data` must be a np.ndarray/torch.Tensor/scipy.sparse matrix"
            ", or a pymde.Graph."
        )

    if isinstance(data, preprocess.graph.Graph):
        n_items = data.n_items
    else:
        n_items = data.shape[0]
    n_all_edges = (n_items) * (n_items - 1) / 2
    retain_fraction = max_distances / n_all_edges

    graph = preprocess.generic.distances(
        data, retain_fraction=retain_fraction, verbose=verbose
    )
    edges = graph.edges.to(device)
    deviations = graph.distances.to(device)

    if constraint is None:
        constraint = constraints.Centered()
    elif isinstance(constraint, constraints._Standardized):
        deviations = preprocess.scale(
            deviations, constraint.natural_length(n_items, embedding_dim)
        )

    return problem.MDE(
        n_items=n_items,
        embedding_dim=embedding_dim,
        edges=edges,
        distortion_function=loss(deviations),
        constraint=constraint,
        device=device,
    )


def preserve_neighbors(
    data,
    embedding_dim=2,
    attractive_penalty=penalties.Log1p,
    repulsive_penalty=penalties.Log,
    constraint=None,
    n_neighbors=None,
    repulsive_fraction=None,
    max_distance=None,
    init="quadratic",
    device="cpu",
    verbose=False,
) -> problem.MDE:
    """Construct an MDE problem designed to preserve local structure.

    This function constructs an MDE problem for preserving the
    local structure of original data. This MDE problem is well-suited for
    visualization (using ``embedding_dim`` 2 or 3), but can also be used to
    generate features for machine learning tasks (with ``embedding_dim`` = 10,
    50, or 100, for example). It yields embeddings in which similar items
    are near each other, and dissimilar items are not near each other.

    The original data can either be a data matrix, or a graph.
    Data matrices should be torch Tensors, NumPy arrays, or scipy sparse
    matrices; graphs should be instances of ``pymde.Graph``.

    The MDE problem uses distortion functions derived from weights (i.e.,
    penalties).

    To obtain an embedding, call the ``embed`` method on the returned ``MDE``
    object. To plot it, use ``pymde.plot``.

    .. code:: python3

        embedding = pymde.preserve_neighbors(data).embed()
        pymde.plot(embedding)

    Arguments
    ---------
    data: {torch.Tensor, numpy.ndarray, scipy.sparse matrix}(
            shape=(n_items, n_features)) or pymde.Graph
        The original data, a data matrix or a graph. Neighbors are
        computed using Euclidean distance if the data is a matrix,
        or the shortest-path metric if the data is a graph.
    embedding_dim: int
        The embedding dimension. Use 2 or 3 for visualization.
    attractive_penalty: pymde.Function class (or factory)
        Callable that constructs a distortion function, given positive
        weights. Typically one of the classes from ``pymde.penalties``,
        such as ``pymde.penalties.log1p``, ``pymde.penalties.Huber``, or
        ``pymde.penalties.Quadratic``.
    repulsive_penalty: pymde.Function class (or factory)
        Callable that constructs a distortion function, given negative
        weights. (If ``None``, only positive weights are used.) For example,
        ``pymde.penalties.Log`` or ``pymde.penalties.InversePower``.
    constraint: pymde.constraints.Constraint (optional)
        Embedding constraint, like ``pymde.Standardized()`` or
        ``pymde.Anchored(anchors, values)`` (or ``None``). Defaults to no
        constraint when a repulsive penalty is provided, otherwise defaults to
        ``pymde.Standardized()``.
    n_neighbors: int (optional)
        The number of nearest neighbors to compute for each row (item) of
        ``data``. A sensible value is chosen by default, depending on the
        number of items.
    repulsive_fraction: float (optional)
        How many repulsive edges to include, relative to the number
        of attractive edges. ``1`` means as many repulsive edges as attractive
        edges. The higher this number, the more uniformly spread out the
        embedding will be. Defaults to ``0.5`` for standardized embeddings, and
        ``1`` otherwise. (If ``repulsive_penalty`` is ``None``, this argument
        is ignored.)
    max_distance: float (optional)
        If not None, neighborhoods are restricted to have a radius
        no greater than ``max_distance``.
    init: str
        Initialization strategy; 'quadratic' or 'random'.
    device: str (optional)
        Device for the embedding (eg, 'cpu', 'cuda').
    verbose: bool
        If ``True``, print verbose output.

    Returns
    -------
    pymde.MDE
        A ``pymde.MDE`` object, based on the original data.
    """
    if isinstance(data, preprocess.graph.Graph):
        n = data.n_items
    elif data.shape[0] <= 1:
        raise ValueError("The data matrix must have at least two rows.")
    else:
        n = data.shape[0]

    if n_neighbors is None:
        # target included edges to be ~1% of total number of edges
        n_choose_2 = n * (n - 1) / 2
        n_neighbors = int(max(min(15, n_choose_2 * 0.01 / n), 5))
    if n_neighbors > n:
        problem.LOGGER.warning(
            (
                "Requested n_neighbors {0} > number of items {1}."
                " Setting n_neighbors to {2}"
            ).format(n_neighbors, n, n - 1)
        )
        n_neighbors = n - 1

    if constraint is None and repulsive_penalty is not None:
        constraint = constraints.Centered()
    elif constraint is None and repulsive_penalty is None:
        constraint = constraints.Standardized()

    if isinstance(data, preprocess.graph.Graph):
        # enforce a max distance, otherwise may very well run out of memory
        # when n_items is large
        if max_distance is None:
            max_distance = (3 * torch.quantile(data.distances, 0.75)).item()

    if verbose:
        problem.LOGGER.info(
            f"Computing {n_neighbors}-nearest neighbors, with "
            f"max_distance={max_distance}"
        )

    knn_graph = preprocess.generic.k_nearest_neighbors(
        data,
        k=n_neighbors,
        max_distance=max_distance,
        verbose=verbose,
    )
    edges = knn_graph.edges.to(device)
    weights = knn_graph.weights.to(device)

    if init == "quadratic":
        if verbose:
            problem.LOGGER.info("Computing quadratic initialization.")
        X_init = quadratic.spectral(
            n, embedding_dim, edges, weights, device=device
        )
    elif init == "random":
        X_init = constraint.initialization(n, embedding_dim, device)
    else:
        raise ValueError(
            f"Unsupported value '{init}' for keyword argument `init`; "
            "the supported values are 'quadratic' and 'random'."
        )

    if repulsive_penalty is not None:
        if repulsive_fraction is None:
            if isinstance(constraint, constraints._Standardized):
                repulsive_fraction = 0.5
            else:
                repulsive_fraction = 1

        n_repulsive = int(repulsive_fraction * edges.shape[0])
        n_choose_2 = int(n * (n - 1) / 2)
        if n_repulsive > n_choose_2:
            problem.LOGGER.info(
                f"The number of repulsive edges requested ({n_repulsive}) "
                f"is larger than the total number of edges ({n_choose_2}). "
                f"Sampling at most ({n_choose_2}) edges ..."
            )
            n_repulsive = n_choose_2

        negative_edges = preprocess.sample_edges(
            n, n_repulsive, exclude=edges
        ).to(device)
        edges = torch.cat([edges, negative_edges])

        negative_weights = -torch.ones(
            negative_edges.shape[0], dtype=X_init.dtype, device=device
        )
        weights = torch.cat([weights, negative_weights])

        f = penalties.PushAndPull(
            weights,
            attractive_penalty=attractive_penalty,
            repulsive_penalty=repulsive_penalty,
        )
    else:
        f = attractive_penalty(weights)

    mde = problem.MDE(
        n_items=n,
        embedding_dim=embedding_dim,
        edges=edges,
        distortion_function=f,
        constraint=constraint,
        device=device,
    )
    mde._X_init = X_init

    # TODO cache the graph for subsequent calls / constructor for MDE from graph

    distances = mde.distances(mde._X_init)
    if (distances == 0).any():
        # pathological scenario in which at least two points overlap can yield
        # non-differentiable average distortion. perturb the initialization to
        # mitigate.
        mde._X_init += 1e-4 * torch.randn(
            mde._X_init.shape,
            device=mde._X_init.device,
            dtype=mde._X_init.dtype,
        )
    return mde
