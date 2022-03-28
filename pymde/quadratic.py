"""Standardized quadratic embeddings from weights

Historical embeddings that reduce to eigenproblems, like PCA and spectral
embedding.
"""
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

from pymde import problem
from pymde import util
from pymde.functions import penalties


@util.tensor_arguments
def pca(Y, embedding_dim):
    """PCA embedding of a data matrix.

    Arguments
    ---------
    Y: torch.Tensor(shape=(n, k))
        data matrix, with n >= k
    embedding_dim: int
        the number of eigenvectors to retrieve, the embedding dimension;
        must be <= k

    Returns
    -------
    torch.Tensor(shape=(n, embedding_dim))
        The top embedding_dim eigenvectors of YY^T, scaled by sqrt(n)
    """
    n = Y.shape[0]
    embedding_dim = embedding_dim.int()
    min_dim = min([n, Y.shape[1]])
    if embedding_dim > min_dim:
        raise ValueError(
            "Embedding dimension must be at most minimum dimension of Y"
        )

    # PCA requires the data to be centered.
    Y = Y - Y.mean(axis=0)[None, :]
    U, _, _ = torch.svd(Y)
    return np.sqrt(float(n)) * U[:, :embedding_dim]


def _laplacian(n, m, edges, weights, use_scipy=True):
    A = util.adjacency_matrix(n, m, edges, weights, use_scipy=use_scipy)
    L = -A
    if use_scipy:
        L.setdiag(np.asarray(A.sum(axis=1)).squeeze())
    else:
        # diag is currently 0
        diag_vals = torch.sparse.sum(A, dim=1).to_dense()
        diag_inds = torch.tensor(
            [[i, i] for i in range(len(A))], device=L.device, dtype=L.dtype
        )
        diag_inds = diag_inds.transpose(0, 1)
        diag_matrix = torch.sparse_coo_tensor(
            diag_inds,
            diag_vals,
            size=(n, n),
            dtype=torch.float32,
            device=L.device,
        )
        L = L + diag_matrix
        L = L.coalesce()
    return L


def _spectral(
    L,
    m,
    cg=False,
    max_iter=400,
    edges=None,
    weights=None,
    warm_start=False,
    device=None,
):
    n = L.shape[0]
    k = m + 1
    if not cg:
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(L.shape[0])))
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            L,
            k,
            which="SM",
            ncv=num_lanczos_vectors,
            tol=1e-4,
            v0=np.ones(L.shape[0]),
            maxiter=L.shape[0] * 5,
        )
        order = np.argsort(eigenvalues)[1:k]
    else:
        if warm_start:
            mde = problem.MDE(
                n,
                k,
                edges,
                distortion_function=penalties.Quadratic(weights),
                device=device,
            )
            X_init = mde.embed(max_iter=40)
        else:
            X_init = util.proj_standardized(
                torch.randn((n, k), device=device),
                demean=True,
            )
        eigenvalues, eigenvectors = torch.lobpcg(
            A=L,
            X=X_init,
            tol=None,
            # largest: find the smallest eigenvalues
            largest=False,
            niter=max_iter,
        )
        order = torch.argsort(eigenvalues)[1:k]
    return eigenvectors[:, order]


# TODO(akshayka): Deprecate device argument, infer from edges/weights
def spectral(
    n_items, embedding_dim, edges, weights, cg=False, max_iter=40, device="cpu"
):
    """Compute a spectral embedding


    Solves the quadratic MDE problem

    .. math::

        \\begin{array}{ll}
        \\mbox{minimize} & \\sum_{(i, j) in \\text{edges}} w_{ij} d_{ij}^2 \\\\
        \\mbox{subject to} & (1/n) X^T X = I, \quad d_{ij} = |x_i - x_j|_2.
        \\end{array}

    By default, the problem is solved using a Lanczos method. If cg=True,
    LOBPCG is used; LOBPCG is warm-started by running a projected quasi-newton
    method for a small number of iterations. Use cg=True when the number
    of edges is very large, and when an approximate solution is satisfactory
    (the Lanczos method typically gives much more accurate solutions, but can
    be slower).

    Arguments
    ---------
    n_items: int
        The number of items
    embedding_dim: int
        The embedding dimension
    edges: torch.Tensor(shape=(n_edges, 2))
        A list of edges (i, j), 0 <= i < j < n_items
    weights: torch.Tensor(shape=(n_edges,))
        A list of nonnegative weights associated with each edge
    cg: bool
        If True, uses a preconditioned CG method to find the embedding,
        which requires that the Laplacian matrix plus the identity is
        positive definite; otherwise, a Lanczos method is used. Use True when
        the Lanczos method is too slow (which might happen when the number of
        edges is very large).
    max_iter: int
        max iteration count for the CG method
    device: str (optional)
        The device on which to allocate the embedding

    Returns
    -------
    torch.Tensor(shape=(n_items, embedding_dim))
        A spectral embedding, projected onto the standardization constraint
    """
    # use torch sparse and linalg for lobpcg
    use_scipy = not cg
    L = _laplacian(n_items, embedding_dim, edges, weights, use_scipy=use_scipy)
    emb = _spectral(L, embedding_dim, cg=cg, device=device, max_iter=max_iter)
    if use_scipy:
        emb -= emb.mean(axis=0)
        emb = torch.tensor(emb, dtype=weights.dtype, device=device)
    else:
        emb -= emb.mean(dim=0)
    return util.proj_standardized(emb)
