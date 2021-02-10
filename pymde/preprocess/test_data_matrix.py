import numpy as np
import scipy.sparse as sp
import torch

from pymde import preprocess
import pymde.testing as testing


@testing.cpu_and_cuda
def test_all_distances_numpy(device):
    del device
    np.random.seed(0)
    data_matrix = np.random.randn(4, 2)
    graph = preprocess.data_matrix.distances(data_matrix)

    assert graph.n_items == data_matrix.shape[0]
    assert graph.n_edges == 6
    testing.assert_all_equal(
        graph.edges,
        torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
    )

    for e, d in zip(graph.edges, graph.distances):
        e = e.cpu().numpy()
        d = d.item()
        true_distance = np.linalg.norm(data_matrix[e[0]] - data_matrix[e[1]])
        testing.assert_allclose(true_distance, d)


@testing.cpu_and_cuda
def test_all_distances_torch(device):
    np.random.seed(0)
    data_matrix = torch.tensor(
        np.random.randn(4, 2), dtype=torch.float, device=device
    )
    graph = preprocess.data_matrix.distances(data_matrix)

    assert graph.n_items == data_matrix.shape[0]
    assert graph.n_edges == 6
    testing.assert_all_equal(
        graph.edges,
        torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
    )

    for e, d in zip(graph.edges, graph.distances):
        e = e
        d = d
        true_distance = (data_matrix[e[0]] - data_matrix[e[1]]).norm()
        testing.assert_allclose(true_distance, d)


@testing.cpu_and_cuda
def test_all_distances_sparse(device):
    del device
    np.random.seed(0)
    data_matrix = sp.csr_matrix(np.random.randn(4, 2))
    graph = preprocess.data_matrix.distances(data_matrix)
    data_matrix = data_matrix.todense()

    assert graph.n_items == data_matrix.shape[0]
    assert graph.n_edges == 6
    testing.assert_all_equal(
        graph.edges,
        torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
    )

    for e, d in zip(graph.edges, graph.distances):
        e = e.cpu().numpy()
        d = d.item()
        true_distance = np.linalg.norm(data_matrix[e[0]] - data_matrix[e[1]])
        testing.assert_allclose(true_distance, d)


@testing.cpu_and_cuda
def test_some_distances_numpy(device):
    del device
    np.random.seed(0)
    max_distances = 50
    retain_fraction = max_distances / int(500 * (499) / 2)
    data_matrix = np.random.randn(500, 2)
    graph = preprocess.data_matrix.distances(
        data_matrix, retain_fraction=retain_fraction
    )

    assert graph.n_items == data_matrix.shape[0]
    assert graph.n_edges == max_distances
    for e, d in zip(graph.edges, graph.distances):
        e = e.cpu().numpy()
        d = d.item()
        true_distance = np.linalg.norm(data_matrix[e[0]] - data_matrix[e[1]])
        testing.assert_allclose(true_distance, d)


@testing.cpu_and_cuda
def test_some_distances_torch(device):
    np.random.seed(0)
    max_distances = 50
    retain_fraction = max_distances / int(500 * (499) / 2)
    data_matrix = torch.tensor(
        np.random.randn(500, 2), dtype=torch.float, device=device
    )
    graph = preprocess.data_matrix.distances(
        data_matrix, retain_fraction=retain_fraction
    )
    data_matrix = data_matrix.cpu().numpy()

    assert graph.n_items == data_matrix.shape[0]
    assert graph.n_edges == max_distances
    for e, d in zip(graph.edges, graph.distances):
        e = e.cpu().numpy()
        d = d.item()
        true_distance = np.linalg.norm(data_matrix[e[0]] - data_matrix[e[1]])
        testing.assert_allclose(true_distance, d)


@testing.cpu_and_cuda
def test_some_distances_sparse(device):
    del device
    np.random.seed(0)
    max_distances = 50
    retain_fraction = max_distances / int(500 * (499) / 2)
    data_matrix = sp.csr_matrix(np.random.randn(500, 2))
    graph = preprocess.data_matrix.distances(
        data_matrix, retain_fraction=retain_fraction
    )
    data_matrix = data_matrix.todense()

    assert graph.n_items == data_matrix.shape[0]
    assert graph.n_edges == max_distances
    for e, d in zip(graph.edges, graph.distances):
        e = e.cpu().numpy()
        d = d.item()
        true_distance = np.linalg.norm(data_matrix[e[0]] - data_matrix[e[1]])
        testing.assert_allclose(true_distance, d)
