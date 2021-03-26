"""Miscellenous preprocessing utilities.
"""
import numpy as np
import scipy.sparse as sp
import torch


def sample_edges(n, num_edges, exclude=None, seed=None):
    """Randomly sample num_edges edges (optionally) not in excluded edges

    Arguments
    ---------
    n : int
        The number of items
    num_edges : int
        The (maximum) number of edges to sample. Must be at most (n choose 2).
    exclude: torch.Tensor, shape (-1, 2) (optional)
        edges to exclude from the sampled edges
    seed: int (optional)
        Random seed.

    Returns
    -------
    torch.Tensor
        randomly sampled edges; the edges are de-duplicated and ordered
        so that e[0] < e[1] for sampled edge e, so the number of rows
        in this tensor may be less than num_edges
    """
    if isinstance(exclude, torch.Tensor):
        exclude = exclude.cpu().numpy()

    if num_edges > int((n * (n - 1) / 2.0)):
        raise ValueError(
            f"Cannot sample more than ({n} choose 2)="
            f"{int(n * (n - 1) / 2.0)} edges. "
            f"(requested: {num_edges} edges)"
        )

    if exclude is not None and (exclude.shape[0] / (n * (n - 1) / 2.0)) >= 0.2:
        A = sp.coo_matrix(
            (np.ones(exclude.shape[0]), (exclude[:, 0], exclude[:, 1])),
            shape=(n, n),
        ).todense()
        A[np.tril_indices(A.shape[0])] = 1.0
        A_comp = sp.coo_matrix(1 - A)
        sampled_edges = np.stack([A_comp.row, A_comp.col], axis=1)
        idx = np.random.choice(sampled_edges.shape[0], num_edges, replace=False)
        return torch.tensor(sampled_edges[idx])

    # randomly sample edges (i, j) with i < j via a bijection to
    # triangular numbers
    randomstate = np.random.default_rng(seed)
    edge_idx = randomstate.choice(
        int(n * (n - 1) / 2), num_edges, replace=False, shuffle=False
    )
    u = (
        n
        - 2
        - np.floor(np.sqrt(-8 * edge_idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    )
    v = edge_idx + u + 1 - n * (n - 1) / 2 + (n - u) * ((n - u) - 1) / 2
    sampled_edges = np.stack([u, v], axis=1).astype(np.int64)

    if exclude is not None:
        # remove duplicated edges
        all_edges = np.concatenate([exclude, sampled_edges])
        neg_edge_offset = exclude.shape[0]
        unique, idx = np.unique(all_edges, axis=0, return_index=True)
        idx = np.sort(idx)
        cutoff = np.searchsorted(idx, neg_edge_offset)
        sampled_edges = all_edges[idx[cutoff:]]
    return torch.tensor(sampled_edges)


def dissimilar_edges(n_items, similar_edges, num_edges=None, seed=None):
    """Sample edges not in ``similar_edges``.

    Given a number of items, and a ``torch.Tensor`` containing pairs of
    items known to be similar, this function samples (uniformly at random) some
    edges in the complement of ``similar_edges``. The returned number
    of edges will be approximately equal to (and no greater than) the
    number of edges in ``similar_edges`` (or ``num_edges``, if provided).

    Arguments
    ---------
    n_items: int
        The number of items.
    similar_edges: torch.Tensor
        Edges to exclude when sampling.
    num_edges: int (optional)
        Number of edges to sample, defaults to approximately
        the same number as ``similar_edges``.
    seed: int (optional)
        Random seed for the sampling.

    Returns
    -------
    torch.Tensor
        Edges not in ``similar_edges``.
    """
    if num_edges is None:
        num_edges = similar_edges.shape[0]
    return sample_edges(n_items, num_edges, exclude=similar_edges, seed=seed)


def deduplicate_edges(edges):
    if isinstance(edges, torch.Tensor):
        device = edges.device
        dtype = edges.dtype
        edges = edges.cpu().numpy()
    else:
        device = None
        dtype = None

    flip_idx = edges[:, 0] > edges[:, 1]
    if flip_idx.any():
        edges[flip_idx] = np.stack(
            [edges[flip_idx][:, 1], edges[flip_idx][:, 0]], axis=1
        )
    unique_edges = np.unique(edges, axis=0)
    return torch.tensor(unique_edges, dtype=dtype, device=device)


def _rms(distances):
    return distances.pow(2).mean().sqrt()


def scale(distances, natural_length):
    alpha = natural_length / _rms(distances)
    return alpha * distances
