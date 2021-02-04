import numpy as np

from pymde import preprocess
import pymde.testing as testing


def test_k_nearest_neighbors():
    data_matrix = np.array([[0.0], [1.0], [1.5], [1.75]])
    graph = preprocess.data_matrix.k_nearest_neighbors(data_matrix, k=2)
    edges = set(tuple(e) for e in graph.edges.cpu().numpy().tolist())
    # [2, 1], [3, 2], [3, 1], omitted because they are duplicates
    expected = set(
        tuple(e)
        for e in np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]).tolist()
    )
    assert edges == expected

    edges = graph.edges.cpu().numpy()
    # 2 if i and j are neighbors of each other,
    # 1 if i is neighbor of j but not vice versa
    testing.assert_allclose(np.array([1., 1., 2., 2., 2.]), graph.weights)
