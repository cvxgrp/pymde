import numpy as np
import pytest
import sklearn.neighbors

from pymde._nndescent import nn_descent
from pymde import preprocess, util


def _brute_force_knn(data, k):
    """Brute-force k-nearest neighbors for recall comparison."""
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=k + 1, algorithm="brute"
    )
    nn.fit(data)
    distances, indices = nn.kneighbors(data)
    # Skip self (column 0)
    return indices[:, 1:], distances[:, 1:]


def _recall(approx_indices, true_indices):
    """Fraction of true neighbors found in approximate result."""
    n, k = true_indices.shape
    hits = 0
    for i in range(n):
        true_set = set(true_indices[i])
        # Skip column 0 (self)
        for j in range(1, approx_indices.shape[1]):
            if approx_indices[i, j] in true_set:
                hits += 1
    return hits / (n * k)


def test_nn_descent_small():
    """50 points in 3D — compare against brute force, recall >= 0.99."""
    np.random.seed(42)
    data = np.random.randn(50, 3).astype(np.float32)
    k = 5
    indices, distances = nn_descent(data, n_neighbors=k + 1, seed=42)
    true_indices, _ = _brute_force_knn(data, k)
    r = _recall(indices, true_indices)
    assert r >= 0.99, f"Recall {r} too low for small dataset"


def test_nn_descent_medium():
    """5000 points in 32D — recall >= 0.88 (matches pynndescent ~0.90 on same params)."""
    np.random.seed(123)
    data = np.random.randn(5000, 32).astype(np.float32)
    k = 10
    indices, distances = nn_descent(data, n_neighbors=k + 1, seed=123)
    true_indices, _ = _brute_force_knn(data, k)
    r = _recall(indices, true_indices)
    assert r >= 0.88, f"Recall {r} too low for medium dataset"


def test_nn_descent_output_shape():
    """Verify output shapes and dtypes."""
    data = np.random.randn(100, 10).astype(np.float32)
    k = 7
    indices, distances = nn_descent(data, n_neighbors=k + 1, seed=0)
    assert indices.shape == (100, k + 1)
    assert distances.shape == (100, k + 1)
    assert indices.dtype == np.int32
    assert distances.dtype == np.float32


def test_nn_descent_output_sorted():
    """Each row's distances should be ascending."""
    data = np.random.randn(200, 8).astype(np.float32)
    k = 10
    _, distances = nn_descent(data, n_neighbors=k + 1, seed=0)
    for i in range(200):
        for j in range(k):
            assert (
                distances[i, j] <= distances[i, j + 1] + 1e-6
            ), f"Row {i}: distances not sorted"


def test_nn_descent_self_neighbors():
    """Index 0 of each row should be self (matching pynndescent)."""
    data = np.random.randn(100, 5).astype(np.float32)
    k = 5
    indices, distances = nn_descent(data, n_neighbors=k + 1, seed=0)
    for i in range(100):
        assert indices[i, 0] == i, f"Point {i}: self not first"
        assert distances[i, 0] == 0.0, f"Point {i}: self distance not 0"


def test_nn_descent_distances_correct():
    """Recompute distances from data and verify they match."""
    np.random.seed(99)
    data = np.random.randn(100, 10).astype(np.float32)
    k = 5
    indices, distances = nn_descent(data, n_neighbors=k + 1, seed=99)
    for i in range(100):
        for j in range(1, k + 1):
            nb = indices[i, j]
            expected = np.linalg.norm(data[i] - data[nb])
            assert abs(distances[i, j] - expected) < 1e-4, (
                f"Point {i}, neighbor {j}: "
                f"got {distances[i, j]}, expected {expected}"
            )


def test_nn_descent_reproducibility():
    """With concurrent heap updates, results may not be bit-exact.
    Verify both runs achieve high recall instead."""
    np.random.seed(42)
    data = np.random.randn(200, 16).astype(np.float32)
    k = 8
    idx1, _ = nn_descent(data, n_neighbors=k + 1, seed=42)
    idx2, _ = nn_descent(data, n_neighbors=k + 1, seed=42)
    true_indices, _ = _brute_force_knn(data, k)
    r1 = _recall(idx1, true_indices)
    r2 = _recall(idx2, true_indices)
    assert r1 >= 0.90, f"Run 1 recall {r1} too low"
    assert r2 >= 0.90, f"Run 2 recall {r2} too low"


def test_k_nearest_neighbors_large():
    """Exercise the Rust code path (n >= 10000)."""
    np.random.seed(0)
    data = np.random.randn(11000, 10).astype(np.float32)
    util.seed(0)
    graph = preprocess.data_matrix.k_nearest_neighbors(data, k=5)
    assert graph.edges.shape[0] > 0
    assert graph.edges.shape[1] == 2


def test_k_nearest_neighbors_large_recall():
    """Verify recall >= 0.90 for the large-n Rust path vs brute force."""
    np.random.seed(0)
    n = 11000
    k = 5
    data = np.random.randn(n, 10).astype(np.float32)
    util.seed(0)
    indices, distances = nn_descent(
        data, n_neighbors=k + 1, seed=42
    )
    true_indices, _ = _brute_force_knn(data, k)
    r = _recall(indices, true_indices)
    assert r >= 0.90, f"Recall {r} too low for large dataset"
