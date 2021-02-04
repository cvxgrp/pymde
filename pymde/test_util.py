import numpy as np
import pytest
import torch

from pymde import util
import pymde.testing as testing


def test_to_tensor():
    args = [0.0, np.array(1.0), torch.tensor(2.0)]
    tensors = util.to_tensor(args)
    for t, arg in zip(tensors, args):
        assert isinstance(t, torch.Tensor)
        assert t.item() == float(arg)
        assert str(t.device) == "cpu"
    assert id(tensors[-1]) == id(args[-1])


@testing.cpu_and_cuda
def test_proj_standardized(device):
    X = torch.eye(2, dtype=torch.float32, device=device)
    proj = util.proj_standardized(X)
    testing.assert_allclose(1 / 2.0 * proj.T @ proj, np.eye(2))

    n = 10
    m = 3
    X = torch.randn((n, m), dtype=torch.float32, device=device)
    proj = util.proj_standardized(X)
    testing.assert_allclose(1.0 / n * proj.T @ proj, np.eye(m))

    n = 100
    m = 3
    X = torch.randn((n, m), dtype=torch.float32, device=device)
    proj = util.proj_standardized(X)
    testing.assert_allclose(1.0 / n * proj.T @ proj, np.eye(m))

    n = 100
    m = 3
    X = torch.randn((n, m), dtype=torch.float32, device=device)
    X -= X.mean(axis=0)
    proj = util.proj_standardized(X)
    testing.assert_allclose(1.0 / n * proj.T @ proj, np.eye(m))
    testing.assert_allclose(proj.mean(axis=0), np.zeros(m))

    n = 100
    m = 3
    X = torch.randn((n, m), dtype=torch.float32, device=device)
    proj = util.proj_standardized(X, demean=True)
    testing.assert_allclose(1.0 / n * proj.T @ proj, np.eye(m))
    testing.assert_allclose(proj.mean(axis=0), np.zeros(m))

    n = 1000
    m = 2
    X = torch.randn((n, m), dtype=torch.float32, device=device)
    proj = util.proj_standardized(X, demean=True)
    testing.assert_allclose(1.0 / n * proj.T @ proj, np.eye(m))
    testing.assert_allclose(proj.mean(axis=0), np.zeros(m))

    n = 1000
    m = 3
    X = torch.randn((n, m), dtype=torch.float32, device=device)
    proj = util.proj_standardized(X, demean=True)
    testing.assert_allclose(1.0 / n * proj.T @ proj, np.eye(m))
    testing.assert_allclose(proj.mean(axis=0), np.zeros(m))

    n = 1000
    m = 250
    X = torch.randn((n, m), dtype=torch.float32, device=device)
    proj = util.proj_standardized(X, demean=True)
    testing.assert_allclose(1.0 / n * proj.T @ proj, np.eye(m))
    testing.assert_allclose(proj.mean(axis=0), np.zeros(m))


def test_align():
    pytest.skip("Unimplemented.")


def test_adjacency_matrix():
    pytest.skip("Unimplemented.")
