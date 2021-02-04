import numpy as np
import pytest
import torch
import scipy.sparse.linalg

from pymde import quadratic, util
import pymde.testing as testing


def test_pca():
    torch.random.manual_seed(0)
    n = 5
    Y = np.random.randn(n, n).astype(np.float32)
    Y -= Y.mean(axis=0)
    for m in range(1, 5):
        X = quadratic.pca(Y, m)
        testing.assert_allclose(1.0 / n * X.T @ X, np.eye(m))
        U, _, _ = np.linalg.svd(Y)
        X_unscaled = 1.0 / np.sqrt(n) * X
        U = U[:, :m]
        U = util.align(source=U, target=X_unscaled)
        for col in range(m):
            testing.assert_allclose(
                X_unscaled[:, col], U[:, col], up_to_sign=True
            )

    n = 5
    k = 4
    Y = np.random.randn(n, k).astype(np.float32)
    Y -= Y.mean(axis=0)
    for m in range(1, k):
        X = quadratic.pca(Y, m)
        testing.assert_allclose(1.0 / n * X.T @ X, np.eye(m))
        U, _, _ = np.linalg.svd(Y)
        X_unscaled = 1.0 / np.sqrt(n) * X
        U = U[:, :m]
        U = util.align(source=U, target=X_unscaled)
        for col in range(m):
            testing.assert_allclose(
                X_unscaled[:, col], U[:, col], up_to_sign=True
            )
    with pytest.raises(
        ValueError, match=r"Embedding dimension must be at most.*"
    ):
        X = quadratic.pca(Y, k + 1)

    n = 4
    k = 5
    Y = np.random.randn(n, k).astype(np.float32)
    Y -= Y.mean(axis=0)
    for m in range(1, n):
        X = quadratic.pca(Y, m)
        testing.assert_allclose(1.0 / n * X.T @ X, np.eye(m))
        U, _, _ = np.linalg.svd(Y)
        X_unscaled = 1.0 / np.sqrt(n) * X
        U = U[:, :m]
        for col in range(m):
            testing.assert_allclose(
                X_unscaled[:, col], U[:, col], up_to_sign=True
            )
    with pytest.raises(
        ValueError, match=r"Embedding dimension must be at most.*"
    ):
        X = quadratic.pca(Y, n + 1)


def test_spectral():
    np.random.seed(0)
    torch.random.manual_seed(0)
    n = 5
    m = 3
    L = -np.abs(np.random.randn(n, n).astype(np.float32))
    L += L.T
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(L, -L.sum(axis=1))
    offdiag = np.triu_indices(n, 1)
    edges = np.column_stack(offdiag)
    weights = -L[offdiag]
    X = quadratic.spectral(n, m, edges, torch.tensor(weights))
    testing.assert_allclose(1.0 / n * X.T @ X, np.eye(m))
    X *= 1.0 / np.sqrt(n)

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        L, k=m + 1, which="SM", return_eigenvectors=True
    )
    eigenvectors = eigenvectors[:, 1:]
    for col in range(m):
        testing.assert_allclose(
            eigenvectors[:, col], X[:, col], up_to_sign=True
        )
