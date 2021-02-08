"""Losses: distortion functions derived from original deviations.

A vector distortion function :math:`f : \\mathbf{R}^{p} \\to \\mathbf{R}^p`
derived from original deviations has component functions

.. math::

    f_k(d_k) = \\ell(d_k, \delta_k), \\quad k=1, \\ldots, p,

where
:math:`\\ell` is a loss function, :math:`\\delta_k` is a nonnegative deviation
or dissimilarity score, :math:`d_k` is an embedding distance,

When an MDE problem calls a distortion function, :math:`d_k` is the Euclidean
distance between the items paired by the k-th edge, so :math:`\\delta_k` should
be the original deviation associated with the k-th edge, and :math:`f_k(d_k)`
is the distortion associated with the edge.

The deviations can be interpreted as targets for the embedding distances:
the loss function is 0 when :math:`d_k = \\delta_k`, and positive otherwise.
So a deviation :math:`\\delta_k`` of 0 means that the items in the k-th edge
are the same, and the larger the deviation, the more dissimilar the items are.

Distortion functions are created in a vectorized or elementwise fashion. The
constructor takes a sequence (torch.Tensor) of deviations (target distances),
returning a callable object. The object takes a sequence of distances of the
same length as the weights, and returns a sequence of distortions, one for each
distance.

Some examples of losses inlcude:

.. code:: python3

    pymde.losses.Absolute
    pymde.losses.Quadratic
    pymde.losses.SoftFractional

**Example.**

.. code:: python3

    deviations = torch.tensor([1., 2., 3.])
    f = pymde.losses.Quadratic(weights)

    distances = torch.tensor([2., 5., 4.])
    distortions = f(distances)
    # the distortions are (2 - 1)**2 == 1, (5 - 2)**2 == 9, (4 - 3)**2 = 1
    print(distortions)

prints

.. code:: python3

    torch.tensor([1., 9., 1.])
"""
from pymde import util
from pymde.functions.function import Function
import torch


class Quadratic(Function):
    """:math:`\\ell(d, \\delta) = (d - \\delta)^2`"""

    def __init__(self, deviations):
        super(Quadratic, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        return (self.deviations - distances).pow(2)


class WeightedQuadratic(Function):
    """:math:`\\ell(d, \\delta) = \\frac{1}{\\delta^2} (d - \\delta)^2`

    If ``weights`` is not None, the coefficient then the
    coefficient :math:`1/\\delta^2` is replaced by the weights.
    """

    def __init__(self, deviations, weights=None):
        super(WeightedQuadratic, self).__init__()
        self.deviations = util.to_tensor(deviations)
        if weights is None:
            weights = 1.0 / deviations.pow(2)
        self.weights = util.to_tensor(weights, device=self.deviations.device)

    def forward(self, distances):
        return self.weights * (self.deviations - distances).pow(2)


class _ClippedQuadratic(Function):
    def __init__(self, deviations, threshold):
        super(_ClippedQuadratic, self).__init__()
        self.deviations = util.to_tensor(deviations)
        self.threshold = threshold

    def forward(self, distances):
        diff = self.deviations - distances
        return torch.min(diff ** 2, (self.threshold + 1) ** 2)


class Huber(Function):
    """
    .. math::

        \ell(d, \\delta) = \\begin{cases}
            \\cdot (d - \\delta)^2 & d < \\text{threshold} \\\\
            \\text{threshold}(2(d - \\delta) - \\cdot \\text{threshold})
            & d \\geq \\text{threshold}
        \\end{cases}
    """

    def __init__(self, deviations, threshold):
        super(Huber, self).__init__()
        self.deviations = util.to_tensor(deviations)
        self.threshold = threshold

    def forward(self, distances):
        output = torch.zeros(
            distances.shape, dtype=distances.dtype, device=distances.device
        )
        diff = (self.deviations - distances).abs()
        lt_idx = diff < self.threshold
        output[lt_idx] = diff[lt_idx].pow(2)
        output[~lt_idx] = self.threshold * (2 * diff[~lt_idx] - self.threshold)
        return output


class Cubic(Function):
    """:math:`\\ell(d, \\delta) = (d - \\delta)^3`"""

    def __init__(self, deviations):
        super(Cubic, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        return (self.deviations - distances).abs().pow(3)


class Power(Function):
    """:math:`\\ell(d, \\delta) = (d - \\delta)^{\\text{exponent}}`"""

    def __init__(self, deviations, exponent):
        super(Power, self).__init__()
        self.deviations = util.to_tensor(deviations)
        self.exponent = util.to_tensor(exponent, device=self.deviations.device)

    def forward(self, distances):
        return (self.deviations - distances).abs().pow(self.exponent)


class _WeightedPower(Function):
    def __init__(self, deviations, exponent, weights=None):
        super(_WeightedPower, self).__init__()
        self.deviations = util.to_tensor(deviations)
        if weights is None:
            weights = 1.0 / deviations.pow(2)
        self.exponent = util.to_tensor(exponent, device=self.deviations.device)
        self.weights = util.to_tensor(weights, device=self.deviations.device)

    def forward(self, distances):
        return self.weights * (self.deviations - distances).abs().pow(
            self.exponent
        )


class Absolute(Function):
    """:math:`\\ell(d, \\delta) = |d - \\delta|`"""

    def __init__(self, deviations):
        super(Absolute, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        return (self.deviations - distances).abs()


class Logistic(Function):
    """:math:`\\ell(d, \\delta) = \\log(1 + \\exp(|d - \\delta|))`"""

    def __init__(self, deviations):
        super(Logistic, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        diff = self.deviations - distances
        return torch.log(1.0 + torch.exp(diff.abs()))


class Fractional(Function):
    """:math:`\\ell(d, \\delta) = \\max(\\delta / d, d / \\delta)`"""

    def __init__(self, deviations):
        super(Fractional, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        return (
            torch.max(self.deviations / distances, distances / self.deviations)
            - 1.0
        )


class SoftFractional(Function):
    """:math:`\\ell(d, \\delta) = \\frac{1}{\\gamma}\\log\\left( \\frac{\\exp(\\gamma \\delta/d) + \\exp(\\gamma d/\\delta)}{2\\exp(\\gamma)} \\right)`

    The parameter ``gamma`` controls how close this loss is to the
    fractional loss. The larger ``gamma`` is, the closer to the fractional
    loss.
    """  # noqa: E501

    def __init__(self, deviations, gamma=10.0):
        super(SoftFractional, self).__init__()
        self.deviations = util.to_tensor(deviations)
        self.gamma = util.to_tensor(gamma, device=self.deviations.device)
        if gamma <= 0.0:
            raise ValueError("gamma must be positive, received ", float(gamma))

    def forward(self, distances):
        stacked = torch.stack(
            (self.deviations / distances, distances / self.deviations)
        ).T
        return (
            1.0
            / self.gamma
            * (
                torch.logsumexp(self.gamma * stacked, dim=1)
                - (torch.log(torch.tensor(2.0)) + self.gamma)
            )
        )


class _Log1p(Function):
    def __init__(self, deviations, exponent):
        super(_Log1p, self).__init__()
        self.deviations = util.to_tensor(deviations)
        self.exponent = util.to_tensor(exponent, device=self.deviations.device)

    def forward(self, distances):
        return torch.log(1 + (distances - self.deviations).pow(self.exponent))
