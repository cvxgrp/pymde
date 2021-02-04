import numpy as np
import pytest
import torch

from pymde import problem
from pymde import quadratic
from pymde import util
from pymde.constraints import Standardized
from pymde.functions import penalties
import pymde.testing as testing


@testing.cpu_and_cuda
def test_initialization(device):
    torch.random.manual_seed(0)
    constraint = Standardized()
    X = constraint.initialization(5, 3, device=device)
    testing.assert_allclose(1.0 / 5 * X.T @ X, np.eye(3))


@testing.cpu_and_cuda
def test_differences(device):
    torch.random.manual_seed(0)
    edges = np.array([(0, 1), (0, 2), (1, 2)])
    X = torch.randn((3, 3), dtype=torch.float32, device=device)
    mde = problem.MDE(
        3,
        3,
        edges,
        penalties.Quadratic(torch.ones(3)),
        constraint=Standardized(),
        device=device,
    )
    diff = mde.differences(X)
    testing.assert_allclose(X[edges[:, 0]] - X[edges[:, 1]], diff)


@testing.cpu_and_cuda
def test_norms(device):
    torch.random.manual_seed(0)
    edges = np.array([(0, 1), (0, 2), (1, 2)])
    X = torch.randn((3, 3), dtype=torch.float32, device=device)
    mde = problem.MDE(
        3,
        3,
        edges,
        penalties.Quadratic(torch.ones(3)),
        constraint=Standardized(),
        device=device,
    )
    diff = X[edges[:, 0]] - X[edges[:, 1]]
    norms = diff.norm(dim=1)
    testing.assert_allclose(norms, mde.distances(X))


@testing.cpu_and_cuda
def test_norm_grad_zero(device):
    torch.random.manual_seed(0)
    edges = np.array([(0, 1)])
    mde = problem.MDE(
        3,
        3,
        edges,
        penalties.Quadratic(torch.ones(3)),
        constraint=Standardized(),
        device=device,
    )
    X = torch.ones((3, 3), requires_grad=True, device=device)
    norms = mde.distances(X)
    norms.backward()
    testing.assert_allclose(X.grad, 0.0)


@testing.cpu_and_cuda
def test_average_distortion(device):
    torch.random.manual_seed(0)
    edges = np.array([(0, 1), (0, 2), (1, 2)])
    mde = problem.MDE(
        3,
        2,
        edges,
        penalties.Quadratic(torch.tensor([1.0, 2.0, 3.0])),
        constraint=Standardized(),
        device=device,
    )
    X = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]],
        dtype=torch.float32,
        device=device,
    )
    average_distortion = mde.average_distortion(X)
    # (1*2 + 2*18 + 3*8)/3 = (2 + 36 + 24)/3 = 62/3
    testing.assert_allclose(average_distortion.detach().cpu().numpy(), 62.0 / 3)


@testing.cpu_and_cuda
def test_average_distortion_grad(device):
    torch.random.manual_seed(0)
    edges = np.array([(0, 1), (0, 2), (1, 2)])
    f = penalties.Quadratic(torch.tensor([1.0, 2.0, 3.0], device=device))
    mde = problem.MDE(3, 2, edges, f, Standardized(), device=device)
    X = torch.randn(
        (3, 2),
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )
    average_distortion = mde.average_distortion(X)
    average_distortion.backward()
    A = torch.tensor(
        [[1, 1, 0], [-1, 0, 1], [0, -1, -1]],
        device=device,
    ).float()
    auto_grad = X.grad
    X.grad = None
    util._distortion(X, f, A, mde._lhs, mde._rhs).backward()
    manual_grad = X.grad
    testing.assert_allclose(auto_grad, manual_grad)


@testing.cpu_and_cuda
def test_fit_spectral(device):
    # TODO deflake this test
    pytest.skip("This test is flaky on macOS.")
    np.random.seed(0)
    torch.random.manual_seed(0)
    n = 200
    m = 3
    max_iter = 1000

    edges = util.all_edges(n)
    weights = torch.ones(edges.shape[0])
    f = penalties.Quadratic(weights)

    mde = problem.MDE(
        n,
        m,
        edges=edges,
        distortion_function=f,
        constraint=Standardized(),
        device=device,
    )
    X = mde.embed(max_iter=max_iter, eps=1e-10, memory_size=10)

    assert id(X) == id(mde.X)
    X_spectral = quadratic.spectral(
        n, m, edges=edges, weights=weights, device=device
    )

    testing.assert_allclose(
        mde.average_distortion(X).detach().cpu().numpy(),
        mde.average_distortion(X_spectral).detach().cpu().numpy(),
        atol=1e-4,
    )


@testing.cpu_and_cuda
def test_self_edges_raises_error(device):
    torch.random.manual_seed(0)
    edges = np.array([(0, 1), (0, 0), (0, 2), (1, 2), (1, 1)])
    with pytest.raises(
        ValueError, match=r"The edge list must not contain self edges.*"
    ):
        problem.MDE(
            3,
            3,
            edges,
            penalties.Quadratic(torch.ones(edges.shape[0])),
            constraint=Standardized(),
            device=device,
        )
