import torch
from pymde import util


class Constraint(object):
    def name(self):
        raise NotImplementedError

    def initialization(self, n, m, device=None):
        raise NotImplementedError

    def project_tspace(self, X, Z, inplace=True):
        raise NotImplementedError

    def project(self, Z, inplace=True):
        raise NotImplementedError


class _Centered(Constraint):
    def name(self):
        return "centered"

    def initialization(self, n, m, device=None):
        X = torch.randn((n, m), device=device)
        return X - X.mean(axis=0)

    def project_tspace(self, X, Z, inplace=True):
        del X
        return Z

    def project(self, Z, inplace=True):
        if inplace:
            Z.sub_(Z.mean(axis=0))
            return Z
        else:
            return Z - Z.mean(axis=0)


class Anchored(Constraint):
    def __init__(self, anchors, values):
        """
        Constructs an anchor constraint, in which some embedding vectors
        (the anchors) are fixed to specific values.

        Arguments
        ---------
            anchors: torch.Tensor, shape (n_anchors)
                a Tensor in which each entry gives the index of an anchored
                vertex
            values: torch.Tensor, shape (n_anchors, m)
                a Tensor which gives the value to which each anchor should be
                fixed
        """
        super(Anchored, self).__init__()
        self.anchors = anchors
        self.values = values

    def name(self):
        return "anchored"

    def initialization(self, n, m, device=None):
        X = torch.randn((n, m), device=device)
        X[self.anchors] = self.values
        return X

    def project_tspace(self, X, Z, inplace=True):
        del X
        if inplace:
            Z[self.anchors, :] = 0.0
        else:
            Z = Z.detach().clone()
            Z[self.anchors, :] = 0.0
        return Z

    def project(self, Z, inplace=True):
        return Z


class _Standardized(Constraint):
    """Standardization constraint.

    Constrains an embedding X to be centered (the rows have mean zero)
    and to satisfy (1/n) X.T @ X == I
    """
    def name(self):
        return "standardized"

    def initialization(self, n, m, device=None):
        X = torch.randn((n, m), device=device)
        X -= X.mean(axis=0)
        if not isinstance(n, torch.Tensor):
            n = torch.tensor(n, dtype=X.dtype, device=device)
        lmbda, Q = torch.symeig(X.T @ X, eigenvectors=True)
        X = X @ Q @ torch.diag(lmbda ** (-1 / 2)) * (n.type(X.dtype).sqrt())
        return X

    def project_tspace(self, X, Z, inplace=True):
        n = torch.tensor(X.shape[0], dtype=X.dtype, device=X.device)
        gtx = Z.T @ X
        if inplace:
            return Z.sub_((1.0 / n) * X @ gtx)
        else:
            return Z - (1.0 / n) * X @ gtx

    def project(self, Z, inplace=True):
        return util.proj_standardized(Z, demean=True, inplace=inplace)

    def natural_length(self, n, m):
        return (torch.tensor(2.0) * n * m / (n - 1)).sqrt()


class _Sphere(Constraint):
    def __init__(self, radius):
        self.radius = radius
        super(_Sphere, self).__init__()

    def name(self):
        return "sphere"

    def initialization(self, n, m, device=None):
        X = torch.randn((n, m), device=device)
        return self.radius * (X / X.norm(dim=1)[:, None])

    def project_tspace(self, X, Z, inplace=True):
        # get the diagonal of Z @ X.T efficiently
        dual_variables = torch.bmm(
            Z.view(Z.shape[0], 1, Z.shape[1]),
            X.view(X.shape[0], X.shape[1], 1),
        ).squeeze()
        offset = (1.0 / self.radius) * dual_variables[:, None] * X
        if inplace:
            return Z.sub_(offset)
        return Z - offset

    def project(self, Z, inplace=True):
        if inplace:
            Z.div_(Z.norm(dim=1)[:, None])
            Z.mul_(self.radius)
            return Z
        return self.radius * Z / Z.norm(dim=1)[:, None]


__Centered = _Centered()
__Standardized = _Standardized()


def Centered():
    """Centering constraint.

    Requires the embedding vectors to be centered around 0.
    """
    return __Centered


def Standardized():
    """Standardization constraint.

    Constrains an embedding X to be centered (the rows have mean zero)
    and to satisfy (1/n) X.T @ X == I
    """
    return __Standardized
