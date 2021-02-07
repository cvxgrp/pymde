import abc

import torch
from pymde import util


class Constraint(abc.ABC):
    """A generic constraint.

    To create a custom constraint, create a subclass of this class,
    and implement its abstract methods.
    """

    @abc.abstractmethod
    def name(self) -> str:
        """The name of the constraint."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialization(
        self, n_items: int, embedding_dim: int, device=None
    ) -> torch.Tensor:
        """Return a random embedding in the constraint set.

        Arguments
        ---------
        n_items: int
            The number of items.
        embedding_dim: int
            The embedding dimension.
        device: str
            Device on which to store the returned embedding.

        Returns
        -------
        torch.Tensor
            a tensor of shape ``(n_items, embedding_dim)`` satisfying the
            constraints.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def project_onto_constraint(
        self, Z: torch.Tensor, inplace=True
    ) -> torch.Tensor:
        """Project ``Z`` onto the constraint set.

        Returns a projection of ``Z`` onto the constraint set.


        Arguments
        ---------
        Z: torch.Tensor
            The point to project.
        inplace: bool
            If True, stores the projection in ``Z``.

        Returns
        -------
        The projection of ``Z`` onto the constraints.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def project_onto_tangent_space(
        self, X: torch.Tensor, Z: torch.Tensor, inplace=True
    ) -> torch.Tensor:
        """Project ``Z`` onto the tangent space of the constraint set at ``X``.

        Returns the Euclidean projection of ``Z`` onto
        the tangent space of the constraint set at ``X`` (where ``X`` is
        some matrix satisfying the constraints).

        ``X`` and ``Z`` should have the same shape.

        Arguments
        ---------
        X: torch.Tensor
            A point satisfying the constraints.
        Z: torch.Tensor
            The point to project.
        inplace: bool
            If True, stores the projection in ``Z``.

        Return
        ------
        The projection of ``Z`` onto the tangent space of the constraint
        set at ``X``.
        """
        raise NotImplementedError


class _Centered(Constraint):
    def name(self):
        return "centered"

    def initialization(self, n_items, embedding_dim, device=None):
        X = torch.randn((n_items, embedding_dim), device=device)
        return X - X.mean(axis=0)

    def project_onto_tangent_space(self, X, Z, inplace=True):
        del X
        return Z

    def project_onto_constraint(self, Z, inplace=True):
        if inplace:
            Z.sub_(Z.mean(axis=0))
            return Z
        else:
            return Z - Z.mean(axis=0)


class Anchored(Constraint):
    """Anchor some vectors to specific values."""

    def __init__(self, anchors, values):
        """
        Constructs an anchor constraint, in which some embedding vectors
        (the anchors) are fixed to specific values.

        Arguments
        ---------
        anchors: torch.Tensor, shape (n_anchors)
            a Tensor in which each entry gives the index of an anchored
            vertex
        values: torch.Tensor, shape (n_anchors, embedding_dim)
            a Tensor which gives the value to which each anchor should be
            fixed
        """
        super(Anchored, self).__init__()
        self.anchors = anchors
        self.values = values

    def name(self):
        return "anchored"

    def initialization(self, n_items, embedding_dim, device=None):
        X = torch.randn((n_items, embedding_dim), device=device)
        X[self.anchors] = self.values
        return X

    def project_onto_tangent_space(self, X, Z, inplace=True):
        del X
        if inplace:
            Z[self.anchors, :] = 0.0
        else:
            Z = Z.detach().clone()
            Z[self.anchors, :] = 0.0
        return Z

    def project_onto_constraint(self, Z, inplace=True):
        # When called by our algorithms, `Z` will always be in the
        # constraint set, since we only update the non-anchors ...
        # (project_onto_tangent_space zeros out the gradient for the anchors)
        #
        # This operation is so cheap that it likely doesn't matter that we're
        # doing redundant work.
        if inplace:
            Z[self.anchors, :] = self.values
        else:
            Z = Z.detach().clone()
            Z[self.anchors, :] = self.values
        return Z


class _Standardized(Constraint):
    """Standardization constraint.

    Constrains an embedding X to be centered (the rows have mean zero)
    and to satisfy (1/n) X.T @ X == I
    """

    def name(self):
        return "standardized"

    def initialization(self, n_items, embedding_dim, device=None):
        X = torch.randn((n_items, embedding_dim), device=device)
        X -= X.mean(axis=0)
        if not isinstance(n_items, torch.Tensor):
            n_items = torch.tensor(n_items, dtype=X.dtype, device=device)
        lmbda, Q = torch.symeig(X.T @ X, eigenvectors=True)
        X = X @ Q @ torch.diag(lmbda ** (-0.5)) * (n_items.type(X.dtype).sqrt())
        return X

    def project_onto_tangent_space(self, X, Z, inplace=True):
        n = torch.tensor(X.shape[0], dtype=X.dtype, device=X.device)
        gtx = Z.T @ X
        if inplace:
            return Z.sub_((1.0 / n) * X @ gtx)
        else:
            return Z - (1.0 / n) * X @ gtx

    def project_onto_constraint(self, Z, inplace=True):
        return util.proj_standardized(Z, demean=True, inplace=inplace)

    def natural_length(self, n_items, embedding_dim):
        return (
            torch.tensor(2.0) * n_items * embedding_dim / (n_items - 1)
        ).sqrt()


class _Sphere(Constraint):
    def __init__(self, radius):
        self.radius = radius
        super(_Sphere, self).__init__()

    def name(self):
        return "sphere"

    def initialization(self, n_items, embedding_dim, device=None):
        X = torch.randn((n_items, embedding_dim), device=device)
        return self.radius * (X / X.norm(dim=1)[:, None])

    def project_onto_tangent_space(self, X, Z, inplace=True):
        # get the diagonal of Z @ X.T efficiently
        dual_variables = torch.bmm(
            Z.view(Z.shape[0], 1, Z.shape[1]),
            X.view(X.shape[0], X.shape[1], 1),
        ).squeeze()
        offset = (1.0 / self.radius) * dual_variables[:, None] * X
        if inplace:
            return Z.sub_(offset)
        return Z - offset

    def project_onto_constraint(self, Z, inplace=True):
        if inplace:
            Z.div_(Z.norm(dim=1)[:, None])
            Z.mul_(self.radius)
            return Z
        return self.radius * Z / Z.norm(dim=1)[:, None]


__Centered = _Centered()
__Standardized = _Standardized()


def Centered():
    """Centering constraint.

    This function returns a centering constraint, which requires the embedding
    vectors to be centered around 0.
    """
    return __Centered


def Standardized():
    """Standardization constraint.

    This function returns a standardization constraint, which constrains an
    embedding :math:`X` to be centered (the rows have mean zero) and to satisfy
    :math:`(1/n) X^T  X = I`.
    """
    return __Standardized
