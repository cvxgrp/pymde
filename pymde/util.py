"""Internal utilities."""
import functools
import numbers

import numpy as np
import scipy.sparse
import torch

from pymde.average_distortion import _project_gradient


_DEVICE = torch.device("cpu")


class SolverError(Exception):
    pass


def get_default_device():
    return str(_DEVICE)


def _canonical_device(device):
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise ValueError("device must be a str or a torch.device object.")

    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def set_default_device(device):
    global _DEVICE
    _DEVICE = _canonical_device(device)


def _module_device(module):
    data = list(module.buffers())
    if not data:
        return None
    device = str(data[0].device)
    if any(str(datum.device) != device for datum in data):
        return None
    return device


def _is_numeric(arg):
    return (
        isinstance(arg, numbers.Number)
        or isinstance(arg, np.ndarray)
        or isinstance(arg, np.matrix)
        or isinstance(arg, torch.Tensor)
    )


def to_tensor(args, device=None):
    """Convert an arg or sequence of args to torch Tensors
    """
    singleton = not isinstance(args, (list, tuple))
    if singleton:
        args = [args]

    tensor_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_args.append(arg)
        elif _is_numeric(arg):
            if isinstance(arg, np.ndarray) and arg.dtype == np.float64:
                tensor_args.append(
                    torch.tensor(arg, dtype=torch.float32, device=device)
                )
            else:
                tensor_args.append(torch.tensor(arg, device=device))
        else:
            raise ValueError("Received non-numeric argument ", arg)
    return tensor_args[0] if singleton else tensor_args


def tensor_arguments(func):
    """Cast numeric args and kwargs of func to Tensors."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tensor_args = to_tensor(args)
        tensor_kwargs = {}
        for key, arg in kwargs.items():
            if isinstance(arg, torch.Tensor):
                tensor_kwargs[key] = arg
            elif _is_numeric(arg):
                tensor_kwargs[key] = torch.tensor(arg, device=_DEVICE)
            else:
                raise ValueError(
                    "Received non-numeric argument (name %s, value %s)"
                    % (key, arg)
                )
        return func(*tensor_args, **tensor_kwargs)

    return wrapper


def all_edges(n):
    """Return a tensor of all (n choose 2) edges

    Constructs all possible edges among n items. For example, if ``n`` is 4,
    the return value will be equal to

    .. code:: python3

        torch.Tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    """
    return torch.triu_indices(n, n, 1).T


@tensor_arguments
def natural_length(n, m):
    return (2.0 * n * m / (n - 1)).sqrt()


def in_stdemb(X):
    cov = (1.0 / X.shape[0]) * X.T @ X
    eye = torch.eye(2, dtype=X.dtype, device=X.device)
    mean = X.mean(axis=0)
    zero = torch.tensor(0.0, dtype=X.dtype, device=X.device)
    return torch.isclose(cov, eye).all() and torch.isclose(mean, zero).all()


def proj_standardized(X, demean=False, inplace=False):
    if demean:
        if inplace:
            X.sub_(X.mean(axis=0))
        else:
            X -= X.mean(axis=0)

    # pytorch 1.8.0 has a bug in which torch.svd fails when requires_grad
    # is true on the input (even if called under torch.no_grad)
    requires_grad = X.requires_grad
    X.requires_grad_(False)

    n = torch.tensor(X.shape[0], dtype=X.dtype, device=X.device)
    m = X.shape[1]
    # TODO: Gracefully handle the rare svd failure
    # TODO: debug alternative eigenvec approach ...
    # (evals, V = torch.eig(X.T @ X, eigenvectors=True)
    # proj = X @ V @ torch.diag(evals[:, 0].sqrt().pow(-1)) ...
    # proj *= torch.sqrt(n)
    if inplace:
        s = torch.zeros(m, device=X.device, dtype=X.dtype)
        V = torch.zeros((m, m), device=X.device, dtype=X.dtype)
        try:
            U, _, V = torch.svd(X, out=(X, s, V))
        except RuntimeError as e:
            X.requires_grad_(requires_grad)
            raise SolverError(str(e))
        torch.matmul(U[:, :m], V.T[:, :m], out=X)
        X.mul_(torch.sqrt(n))
        X.requires_grad_(requires_grad)
        return X
    else:
        try:
            U, _, V = torch.svd(X)
        except RuntimeError as e:
            X.requires_grad_(requires_grad)
            raise SolverError(str(e))
        proj = torch.sqrt(n) * U[:, :m] @ V.T[:, :m]
        X.requires_grad_(requires_grad)
        return proj


def adjacency_matrix(n, m, edges, weights):
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    if isinstance(edges, torch.Tensor):
        edges = edges.detach().cpu().numpy()
    A = scipy.sparse.coo_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32
    )
    A = A + A.T
    return A.tocoo()


@tensor_arguments
def procrustes(X_source, X_target):
    """min |X_source Q - X_target|_F s.t. Q^TQ = I"""
    U, _, V = torch.svd(X_target.T @ X_source)
    return V @ U.T


@tensor_arguments
def _rotate_2d(X, degrees):
    theta = torch.deg2rad(degrees)
    rot = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ],
        device=X.device,
    )
    return X @ rot


@tensor_arguments
def _rotate_3d(X, alpha, beta, gamma):
    alpha = torch.deg2rad(alpha.float())
    beta = torch.deg2rad(beta.float())
    gamma = torch.deg2rad(gamma.float())
    rot_x = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(alpha), torch.sin(alpha)],
            [0, -torch.sin(alpha), torch.cos(alpha)],
        ],
        device=X.device,
    )
    rot_y = torch.tensor(
        [
            [torch.cos(beta), 0.0, -torch.sin(beta)],
            [0, 1, 0],
            [torch.sin(beta), 0.0, torch.cos(beta)],
        ],
        device=X.device,
    )
    rot_z = torch.tensor(
        [
            [torch.cos(gamma), torch.sin(gamma), 0.0],
            [-torch.sin(gamma), torch.cos(gamma), 0.0],
            [0, 0, 1],
        ],
        device=X.device,
    )
    rot_3d = rot_x @ rot_y @ rot_z
    return X @ rot_3d


@tensor_arguments
def rotate(X, degrees):
    """Rotate a 2 or 3D embedding

    Rotates a 2/3D embedding by ``degrees``. If ``X`` is a 2D embedding,
    ``degrees`` should be a scalar; if it is 3D, ``degrees`` should be
    a length-3 ``torch.Tensor``, with one angle for each axis (the embedding
    will be rotated along the x-axis first, then the y-axis, then the z-axis).

    Arguments
    ---------
    X : torch.Tensor
        The embedding to rotate.
    degrees: torch.Tensor
        The angles of rotation.

    Returns
    -------
    torch.Tensor
        The rotated embedding
    """
    if X.shape[1] not in [2, 3]:
        raise ValueError(
            "Only 2 or 3 dimensional embeddings can be "
            "rotated using this method."
        )

    if X.shape[1] == 2:
        if degrees.numel() != 1:
            raise ValueError("`degrees` must be a scalar.")
        return _rotate_2d(X, degrees)
    else:
        if degrees.numel() != 3:
            raise ValueError("`degrees` must be a length-3 tensor.")
        return _rotate_3d(X, degrees[0], degrees[1], degrees[2])


@tensor_arguments
def center(X):
    """Center an embedding

    Returns a new embedding, equal to the given embedding minus the mean
    of its rows.
    """
    return X - X.mean(dim=0)[None, :]


@tensor_arguments
def align(source, target):
    """Align an source embedding to a target embedding

    Align the source embedding to another target embedding, via
    rotation. The alignment is done by solving an
    orthogonal Procrustes problem.

    Arguments
    ---------
    source: torch.Tensor
        The embedding to be aligned.
    target: torch.Tensor
        The target embedding, of the same shape as source.

    Returns
    -------
    torch.Tensor
        The rotation of source best aligned to the target.
    """
    source_mean = source.mean(dim=0)
    source = source - source_mean[None, :]
    source_col_rms = source.norm(dim=0)
    source = source / source_col_rms[None, :]

    target = center(target)
    target = target / target.norm(dim=0)

    Q = procrustes(source, target)
    rotated = source @ Q
    return (rotated * source_col_rms[None, :]) + source_mean


@tensor_arguments
def scale_delta(delta, d_nat):
    # scale delta so RMS(delta) == d_nat
    N = delta.nelement()
    rms = torch.sqrt(1 / N * torch.sum(delta ** 2))
    return delta * d_nat / rms


class LinearOperator(object):
    def __init__(self, matvec, device):
        self._matvec = matvec
        self.device = device

    def matvec(self, vecs):
        return self._matvec(vecs)


def make_hvp(f, edges, X, constraint):
    X_shape = X.shape

    def avg_dist_flat(X_flat):
        X_reshaped = X_flat.view(X_shape)
        if constraint is not None:
            # a noop in the forward pass, but projects the gradient onto
            # the tangent space of the constraint in the backward pass
            X_reshaped = _project_gradient(X_reshaped, constraint)
        # using custom average distortion yields a zero for hvp, since
        # gradient graph is disconnected
        differences = X_reshaped[edges[:, 0]] - X_reshaped[edges[:, 1]]
        norms = differences.pow(2).sum(dim=1).sqrt()
        return f(norms).mean()

    X_flat = X.view(-1).detach()

    def hvp(vecs):
        vecs = torch.split(vecs, 1, dim=1)
        products = []
        for v in vecs:
            _, product = torch.autograd.functional.vhp(
                avg_dist_flat, X_flat, v.squeeze()
            )
            products.append(product)
        return torch.stack(products, dim=1)

    return hvp


def hutchpp(linear_operator, dimension, n_queries):
    A = linear_operator
    d = dimension
    m = n_queries
    S = torch.randn(d, m // 3, device=A.device)
    G = torch.randn(d, m // 3, device=A.device)
    Q, _ = torch.qr(A.matvec(S))
    proj = G - Q @ (Q.T @ G)
    return torch.trace(Q.T @ A.matvec(Q)) + (3.0 / m) * torch.trace(
        proj.T @ A.matvec(proj)
    )


def random_edges(n, p, seed=0):
    randomstate = np.random.default_rng(seed)
    edge_idx = randomstate.choice(
        int(n * (n - 1) / 2), p, replace=False, shuffle=False
    )
    u = (
        n
        - 2
        - np.floor(np.sqrt(-8 * edge_idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    )
    v = edge_idx + u + 1 - n * (n - 1) / 2 + (n - u) * ((n - u) - 1) / 2
    return torch.tensor(np.stack([u, v], axis=1).astype(np.int64))


class Distortion(torch.autograd.Function):
    """Manual implementation of the average distortion gradient, for testing"""

    @staticmethod
    def forward(ctx, X, f, A, lhs, rhs):
        distances = A.T @ X
        norms = distances.norm(dim=1)

        with torch.enable_grad():
            X.requires_grad_(False)
            norms.requires_grad_(True)
            norms.grad = None
            distortion = f(norms).mean()
            distortion.backward()
            g = norms.grad / norms
        X.requires_grad_(True)
        D = g.diag()

        grad_E = A @ (D @ (A.T @ X))
        ctx.grad_E = grad_E
        return distortion

    def backward(ctx, grad_output):
        return ctx.grad_E * grad_output, None, None, None, None


_distortion = Distortion.apply
