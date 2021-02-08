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


def _laplacian(n, m, edges, weights):
    A = util.adjacency_matrix(n, m, edges, weights)
    L = -A
    L.setdiag((np.array(A.sum(axis=1)).squeeze()))
    return L


def _spectral(
    L,
    m,
    cg=False,
    max_iter=40,
    edges=None,
    weights=None,
    warm_start=False,
    device=None,
):
    n = L.shape[0]
    if not cg:
        k = m + 1
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
        k = m
        if warm_start:
            mde = problem.MDE(
                n, m, edges, f=penalties.Quadratic(weights), device=device
            )
            X_init = mde.fit(max_iter=40, use_line_search=False)
        else:
            X_init = util.proj_standardized(
                torch.tensor(np.random.randn(n, m), device=device), demean=True
            )
        eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
            A=L,
            X=X_init.cpu().numpy(),
            # Y: search in the orthogonal complement of the ones vector
            Y=np.ones((L.shape[0], 1)),
            tol=None,
            # largest: find the smallest eigenvalues
            largest=False,
            maxiter=max_iter,
        )
        order = np.argsort(eigenvalues)[0:k]
    return eigenvectors[:, order]


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

    The weights may be negative.

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
        A list of weights associated with each edge
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
    L = _laplacian(n_items, embedding_dim, edges, weights)
    emb = _spectral(L, embedding_dim, cg=cg, device=device, max_iter=max_iter)
    emb -= emb.mean(axis=0)
    return util.proj_standardized(
        torch.tensor(emb, dtype=weights.dtype, device=device)
    )
