"""Tests for the Rust knn_l2 extension."""

import numpy as np
import pytest

from pymde._knn import knn_l2


def _brute_force_knn(data, k):
    """NumPy brute-force kNN for reference."""
    n = data.shape[0]
    # Pairwise squared distances via broadcasting
    diff = data[:, None, :] - data[None, :, :]
    sq_dists = (diff * diff).sum(axis=2)
    neighbors = np.argsort(sq_dists, axis=1)[:, : k + 1]
    sq_distances = np.take_along_axis(sq_dists, neighbors, axis=1)
    return neighbors.astype(np.int64), sq_distances.astype(np.float32)


class TestKnnL2:
    def test_basic_correctness(self):
        """4 points on a line: [0], [1], [2], [3]."""
        data = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        neighbors, sq_dists = knn_l2(data, k=2)

        assert neighbors.shape == (4, 3)
        assert sq_dists.shape == (4, 3)

        # Column 0 should be self (distance 0)
        np.testing.assert_array_equal(neighbors[:, 0], [0, 1, 2, 3])
        np.testing.assert_allclose(sq_dists[:, 0], 0.0, atol=1e-6)

        # Point 0: nearest are 1 (dist=1), 2 (dist=4)
        assert set(neighbors[0, 1:]) == {1, 2}
        # Point 1: nearest are 0 (dist=1), 2 (dist=1)
        assert set(neighbors[1, 1:]) == {0, 2}

    def test_matches_brute_force(self):
        """Random data: match against numpy brute-force."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 10)).astype(np.float32)
        k = 15

        neighbors, sq_dists = knn_l2(data, k)
        gt_neighbors, gt_sq_dists = _brute_force_knn(data, k)

        np.testing.assert_array_equal(neighbors, gt_neighbors)
        np.testing.assert_allclose(sq_dists, gt_sq_dists, rtol=1e-4, atol=1e-5)

    def test_k_equals_n_minus_1(self):
        """Edge case: k = n-1 returns all other points."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((20, 5)).astype(np.float32)
        k = 19

        neighbors, sq_dists = knn_l2(data, k)
        assert neighbors.shape == (20, 20)
        assert sq_dists.shape == (20, 20)

        # Each row should be a permutation of 0..19
        for i in range(20):
            assert set(neighbors[i]) == set(range(20))
            assert neighbors[i, 0] == i

    def test_invalid_k_raises(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            knn_l2(data, k=0)
        with pytest.raises(ValueError):
            knn_l2(data, k=2)

    def test_high_dimensional(self):
        """High-dimensional data (d=784) should work correctly."""
        rng = np.random.default_rng(123)
        data = rng.standard_normal((50, 784)).astype(np.float32)
        k = 5

        neighbors, sq_dists = knn_l2(data, k)
        gt_neighbors, gt_sq_dists = _brute_force_knn(data, k)

        np.testing.assert_array_equal(neighbors, gt_neighbors)
        np.testing.assert_allclose(sq_dists, gt_sq_dists, rtol=1e-4, atol=1e-5)
