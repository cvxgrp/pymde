import numpy as np
import torch

from pymde import constraints
from pymde import preprocess
from pymde import recipes
from pymde import util
from pymde.functions import penalties
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
    testing.assert_allclose(np.array([1.0, 1.0, 2.0, 2.0, 2.0]), graph.weights)


@testing.cpu_and_cuda
def test_laplacian_embedding(device):
    torch.manual_seed(0)
    data_matrix = torch.randn(100, 10)

    util.seed(0)
    laplacian_emb = recipes.laplacian_embedding(
        data_matrix, device=device
    ).embed()

    util.seed(0)
    also_laplacian = recipes.preserve_neighbors(
        data_matrix,
        attractive_penalty=penalties.Quadratic,
        repulsive_penalty=None,
        device=device,
    ).embed()

    also_laplacian = util.align(source=also_laplacian, target=laplacian_emb)

    testing.assert_allclose(
        laplacian_emb.cpu().numpy(), also_laplacian.cpu().numpy()
    )


@testing.cpu_and_cuda
def test_anchor_initialization(device):
    n_items = 10

    util.seed(0)
    data_matrix = torch.randn(n_items, 5)

    anchors = torch.tensor([0, 1, 3])
    values = torch.tensor([2.0, 1.0, 3.0]).reshape(3, 1)
    constraint = constraints.Anchored(anchors, values)

    # preserve neighbors
    mde = recipes.preserve_neighbors(
        data_matrix, embedding_dim=1, constraint=constraint, init="random"
    )
    testing.assert_allclose(mde._X_init[anchors], values)

    mde = recipes.preserve_neighbors(
        data_matrix, embedding_dim=1, constraint=constraint, init="quadratic"
    )
    testing.assert_allclose(mde._X_init[anchors], values)


@testing.cpu_and_cuda
def test_no_anchor_anchor_edges(device):
    util.seed(0)
    data_matrix = torch.randn(3, 2)

    anchors = torch.tensor([0, 1])
    values = torch.tensor([2.0, 3.0]).reshape(2, 1)
    constraint = constraints.Anchored(anchors, values)

    mde = recipes.preserve_distances(
        data_matrix, embedding_dim=1, constraint=constraint
    )
    expected_edges = torch.tensor([[0, 2], [1, 2]])
    testing.assert_all_equal(expected_edges, mde.edges)

    mde = recipes.preserve_neighbors(
        data_matrix, embedding_dim=1, constraint=constraint
    )
    testing.assert_all_equal(expected_edges, mde.edges)


@testing.cpu_and_cuda
def test_neighbor_reproducibility(device):
    def _run_test(n_items):
        torch.manual_seed(0)
        Y = torch.rand((n_items, 128))

        prev_edges = None
        prev_weights = None
        for i in range(5):
            util.seed(0)
            mde = recipes.preserve_neighbors(Y)
            edges = mde.edges
            weights = mde.distortion_function.weights
            if prev_edges is not None:
                assert (edges == prev_edges).all()
                assert (weights == prev_weights).all()
            prev_edges = edges
            prev_weights = weights

    _run_test(36)
    _run_test(1001)


@testing.cpu_and_cuda
def test_distances_reproducibility(device):
    def _run_test(n_items):
        torch.manual_seed(0)
        Y = torch.rand((n_items, 128))

        prev_edges = None
        prev_deviations = None
        for i in range(5):
            util.seed(0)
            mde = recipes.preserve_distances(Y, max_distances=1e5)
            edges = mde.edges
            deviations = mde.distortion_function.deviations
            if prev_edges is not None:
                assert (edges == prev_edges).all()
                assert (deviations == prev_deviations).all()
            prev_edges = edges
            prev_deviations = deviations

    _run_test(36)
    _run_test(1001)
