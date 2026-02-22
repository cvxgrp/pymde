"""Tests for Rust NN-Descent implementation."""

import numpy as np
import pytest

from pymde._native import nn_descent, knn_l2


def _brute_force_knn(data, k):
    """Ground truth using exact Rust knn_l2."""
    neighbors, sq_distances = knn_l2(data, k)
    return neighbors, np.sqrt(np.maximum(sq_distances, 0.0))


def _recall(approx_neighbors, gt_neighbors, k):
    """Fraction of true neighbors found, averaged over all queries."""
    n = gt_neighbors.shape[0]
    # Both arrays have k+1 columns (column 0 = self); compare columns 1:
    hits = 0
    for i in range(n):
        approx_set = set(approx_neighbors[i, 1 : k + 1])
        gt_set = set(gt_neighbors[i, 1 : k + 1])
        hits += len(approx_set & gt_set)
    return hits / (n * k)


class TestNNDescentSmall:
    """Small dataset tests (exact or near-exact results)."""

    def test_small_recall(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 8)).astype(np.float32)
        k = 5
        neighbors, distances = nn_descent(data, k, seed=42)
        gt_neighbors, _ = _brute_force_knn(data, k)
        r = _recall(neighbors, gt_neighbors, k)
        assert r >= 0.95, f"recall {r:.4f} < 0.95 for n=50"

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 4)).astype(np.float32)
        k = 5
        neighbors, distances = nn_descent(data, k, seed=42)
        assert neighbors.shape == (50, k + 1)
        assert distances.shape == (50, k + 1)

    def test_output_dtypes(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 4)).astype(np.float32)
        neighbors, distances = nn_descent(data, 5, seed=42)
        assert neighbors.dtype == np.int64
        assert distances.dtype == np.float32

    def test_self_neighbor(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 4)).astype(np.float32)
        neighbors, distances = nn_descent(data, 5, seed=42)
        # Column 0 should be self
        np.testing.assert_array_equal(neighbors[:, 0], np.arange(50))
        np.testing.assert_array_equal(distances[:, 0], 0.0)

    def test_distances_sorted(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 8)).astype(np.float32)
        neighbors, distances = nn_descent(data, 10, seed=42)
        for i in range(50):
            assert all(
                distances[i, j] <= distances[i, j + 1]
                for j in range(distances.shape[1] - 1)
            ), f"row {i}: distances not sorted"

    def test_reproducibility(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 8)).astype(np.float32)
        nb1, d1 = nn_descent(data, 10, seed=123)
        nb2, d2 = nn_descent(data, 10, seed=123)
        np.testing.assert_array_equal(nb1, nb2)
        np.testing.assert_array_equal(d1, d2)


class TestNNDescentMedium:
    """Medium dataset tests."""

    def test_medium_recall(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5000, 32)).astype(np.float32)
        k = 15
        neighbors, distances = nn_descent(data, k, seed=42)
        gt_neighbors, _ = _brute_force_knn(data, k)
        r = _recall(neighbors, gt_neighbors, k)
        assert r >= 0.95, f"recall {r:.4f} < 0.95 for n=5000"

    def test_distances_match_recomputed(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 16)).astype(np.float32)
        neighbors, distances = nn_descent(data, 10, seed=42)
        # Recompute L2 distances for the reported neighbors
        for i in range(200):
            for j in range(1, 11):
                nb = neighbors[i, j]
                expected = np.sqrt(np.sum((data[i] - data[nb]) ** 2))
                np.testing.assert_allclose(
                    distances[i, j],
                    expected,
                    rtol=1e-4,
                    err_msg=f"row {i} col {j}: distance mismatch",
                )


class TestNNDescentLarge:
    """Larger dataset to exercise the auto-switching threshold."""

    def test_large_recall(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((11000, 32)).astype(np.float32)
        k = 15
        neighbors, distances = nn_descent(data, k, seed=42, verbose=True)
        gt_neighbors, _ = _brute_force_knn(data, k)
        r = _recall(neighbors, gt_neighbors, k)
        assert r >= 0.95, f"recall {r:.4f} < 0.95 for n=11000"


class TestKNearestNeighborsIntegration:
    """Integration test: nn_descent can be called from Python directly."""

    def test_knn_integration(self):
        from pymde.preprocess.data_matrix import k_nearest_neighbors

        rng = np.random.default_rng(42)
        data = rng.standard_normal((5000, 32)).astype(np.float32)
        graph = k_nearest_neighbors(data, k=15, verbose=True)
        # Should return a valid Graph object
        assert graph.edges.shape[0] > 0
        assert graph.edges.shape[1] == 2
