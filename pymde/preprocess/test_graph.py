import numpy as np
import pytest
import scipy.sparse as sp
import torch

from pymde import preprocess
import pymde.testing as testing


@testing.cpu_and_cuda
def test_from_edges(device):
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], device=device)
    graph = preprocess.Graph.from_edges(edges)
    assert graph.n_items == 4
    assert graph.n_edges == edges.shape[0]

    expected_edges = torch.tensor([[0, 1], [0, 3], [1, 2], [2, 3]])
    testing.assert_all_equal(expected_edges, graph.edges)

    expected_distances = torch.ones(graph.n_edges)
    testing.assert_all_equal(expected_distances, graph.distances)


@testing.cpu_and_cuda
def test_adjacency_matrix(device):
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], device=device)
    graph = preprocess.Graph.from_edges(edges)
    expected = np.array(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
    )
    testing.assert_all_equal(expected, graph.A.todense())


@testing.cpu_and_cuda
def test_neighbors(device):
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], device=device)
    graph = preprocess.Graph.from_edges(edges)

    neighbors = graph.neighbors(0)
    testing.assert_all_equal(np.array([1, 3]), neighbors)

    neighbors = graph.neighbors(1)
    testing.assert_all_equal(np.array([0, 2]), neighbors)

    neighbors = graph.neighbors(2)
    testing.assert_all_equal(np.array([1, 3]), neighbors)

    neighbors = graph.neighbors(3)
    testing.assert_all_equal(np.array([0, 2]), neighbors)


@testing.cpu_and_cuda
def test_distances(device):
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], device=device)
    graph = preprocess.Graph.from_edges(edges, torch.arange(edges.shape[0]) + 1)

    distances = graph.neighbor_distances(0)
    testing.assert_all_equal(np.array([1.0, 4.0]), distances)

    distances = graph.neighbor_distances(1)
    testing.assert_all_equal(np.array([1.0, 2.0]), distances)

    distances = graph.neighbor_distances(2)
    testing.assert_all_equal(np.array([2.0, 3.0]), distances)

    distances = graph.neighbor_distances(3)
    testing.assert_all_equal(np.array([4.0, 3.0]), distances)


@testing.cpu_and_cuda
def test_distances_repeated_edges_summed(device):
    edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 3]], device=device
    )
    graph = preprocess.Graph.from_edges(edges, torch.arange(edges.shape[0]) + 1)

    # edge (0, 3) repeated, so distance is 4 + 5 = 9
    distances = graph.neighbor_distances(0)
    testing.assert_all_equal(np.array([1.0, 9.0]), distances)

    distances = graph.neighbor_distances(1)
    testing.assert_all_equal(np.array([1.0, 2.0]), distances)

    distances = graph.neighbor_distances(2)
    testing.assert_all_equal(np.array([2.0, 3.0]), distances)

    distances = graph.neighbor_distances(3)
    testing.assert_all_equal(np.array([9.0, 3.0]), distances)


def test_from_matrix():
    A = sp.csr_matrix(
        np.array([[0, 2, 0, 3], [2, 0, 1, 1], [0, 1, 0, 0], [3, 1, 0, 0]])
    )
    graph = preprocess.graph.Graph(A)

    expected_edges = torch.tensor([(0, 1), (0, 3), (1, 2), (1, 3)])
    assert graph.n_items == 4
    assert graph.n_edges == 4

    testing.assert_all_equal(expected_edges, graph.edges)
    testing.assert_all_equal(np.array([2.0, 3.0, 1.0, 1.0]), graph.distances)


@testing.cpu_and_cuda
def test_shortest_paths_unweighted(device):
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], device=device)
    graph = preprocess.Graph.from_edges(edges)

    shortest_path_graph = preprocess.graph.shortest_paths(graph)

    assert shortest_path_graph.n_items == 4
    assert shortest_path_graph.n_edges == 6

    assert shortest_path_graph[0, 1] == 1.0
    assert shortest_path_graph[1, 2] == 1.0
    assert shortest_path_graph[2, 3] == 1.0
    assert shortest_path_graph[3, 0] == 1.0

    assert shortest_path_graph[0, 2] == 2.0
    assert shortest_path_graph[1, 3] == 2.0


@testing.cpu_and_cuda
def test_shortest_paths_unweighted_max_length(device):
    n = 1000
    edges = torch.tensor(
        [[i, i + 1] for i in range(n - 1)] + [[n - 1, 0]], device=device
    )
    graph = preprocess.Graph.from_edges(edges)

    shortest_path_graph = preprocess.graph.shortest_paths(graph, max_length=2)

    assert shortest_path_graph.n_items == n
    assert shortest_path_graph.n_edges == (n - 1 + (4 + n - 3))

    for i in range(n - 1):
        assert shortest_path_graph[i, i + 1] == 1.0
        assert shortest_path_graph[i, (i + 2) % n] == 2.0
        assert shortest_path_graph[i, (i - 2) % n] == 2.0


@testing.cpu_and_cuda
def test_shortest_paths_weighted_max_length(device):
    n = 1000
    edges = torch.tensor(
        [[i, i + 1] for i in range(n - 1)] + [[n - 1, 0]], device=device
    )
    distances = 2.0 * torch.ones(
        edges.shape[0], dtype=torch.float, device=device
    )
    graph = preprocess.Graph.from_edges(edges, distances)

    shortest_path_graph = preprocess.graph.shortest_paths(graph, max_length=4)

    assert shortest_path_graph.n_items == n
    assert shortest_path_graph.n_edges == (n - 1 + (4 + n - 3))

    for i in range(n - 1):
        assert shortest_path_graph[i, i + 1] == 2.0
        assert shortest_path_graph[i, (i + 2) % n] == 4.0
        assert shortest_path_graph[i, (i - 2) % n] == 4.0


@testing.cpu_and_cuda
def test_shortest_paths_weighted(device):
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], device=device)
    distances = torch.tensor([1.0, 2.0, 3.0, 3.0], device=device)
    graph = preprocess.Graph.from_edges(edges, distances)

    shortest_path_graph = preprocess.graph.shortest_paths(graph)

    assert shortest_path_graph.n_items == 4
    assert shortest_path_graph.n_edges == 6

    assert shortest_path_graph[0, 1] == 1.0
    assert shortest_path_graph[1, 2] == 2.0
    assert shortest_path_graph[2, 3] == 3.0
    assert shortest_path_graph[3, 0] == 3.0

    # 0 -> 1 -> 2
    assert shortest_path_graph[0, 2] == 3.0

    # 1 -> 0 -> 3
    assert shortest_path_graph[1, 3] == 4.0


@testing.cpu_and_cuda
def test_shortest_paths_retain_fraction(device):
    pytest.skip("This test is flaky.")

    # graph on 10 nodes: 10(9)/2 = 45 edges
    n = 10
    edges = torch.tensor(
        [[i, i + 1] for i in range(n - 1)] + [[n - 1, 0]], device=device
    )
    graph = preprocess.Graph.from_edges(edges)

    # 0.2 * 45 = approx 9 edges expected
    n_edges = 0
    nnz = 0
    n_trials = 100
    for _ in range(n_trials):
        shortest_path_graph = preprocess.graph.shortest_paths(
            graph, retain_fraction=0.2
        )
        n_edges += shortest_path_graph.n_edges
        nnz += shortest_path_graph.A.nnz
    mean_n_edges = n_edges / n_trials
    mean_nnz = nnz / n_trials

    assert shortest_path_graph.n_items == n
    testing.assert_allclose(mean_n_edges, 9.0, atol=1.5)
    testing.assert_allclose(mean_nnz, 18.0, atol=1.5)

    assert not (
        shortest_path_graph.A != shortest_path_graph.A.T
    ).todense().all()


def test_graph_from_numpy():
    X = np.ones(16).reshape(4, 4)
    np.fill_diagonal(X, 0.0)
    graph = preprocess.graph.Graph(X)
    assert graph.n_items == 4
    assert graph.n_edges == 6
    testing.assert_all_equal(torch.ones(graph.n_edges), graph.distances)

    expected_edges = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    )
    testing.assert_all_equal(expected_edges, graph.edges)


def test_graph_from_torch():
    X = torch.ones(16).reshape(4, 4)
    X.fill_diagonal_(0.0)
    graph = preprocess.graph.Graph(X)
    assert graph.n_items == 4
    assert graph.n_edges == 6
    testing.assert_all_equal(torch.ones(graph.n_edges), graph.distances)

    expected_edges = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    )
    testing.assert_all_equal(expected_edges, graph.edges)


def test_graph_from_csc_matrix():
    X = np.ones(16).reshape(4, 4)
    np.fill_diagonal(X, 0.0)
    X = sp.csc_matrix(X)
    with testing.disable_logging():
        # suppress a warning re: non-CSR matrix.
        graph = preprocess.graph.Graph(X)
    assert graph.n_items == 4
    assert graph.n_edges == 6
    testing.assert_all_equal(torch.ones(graph.n_edges), graph.distances)

    expected_edges = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    )
    testing.assert_all_equal(expected_edges, graph.edges)
