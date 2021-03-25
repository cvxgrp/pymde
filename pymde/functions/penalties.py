"""Penalties: distortion functions derived from weights.

A vector distortion function  :math:`f : \\mathbf{R}^{p} \\to \\mathbf{R}^p`
derived from weights has component functions

.. math::

    f_k(d_k) = w_kp(d_k), \\quad k=1, \\ldots, p,

where :math:`w_k` is a scalar weight, :math:`p` is a penalty function, and
:math:`d_k` is an embedding distance. The penalty encourages distances to be
small when the weights are positive, and encourages them to be not small
when the weights are negative.

When an MDE problem calls a distortion function, :math:`d_k` is the Euclidean
distance between the items paired by the :math:`k`-th edge, so :math:`w_k`
should be the weight associated with the :math:`k`-th edge, and
:math:`f_k(d_k)` is the distortion associated with the edge.

Every penalty can be used with positive or negative weights. When :math:`w_k`
is positive, :math:`f_k` is attractive, meaning it encourages the embedding
distances to be small; when :math:`w_k` is negative, :math:`f_k` is repulsive,
meaning it encourages the distances to be large. Some penalties are better
suited to attracting points, while others are better suited to repelling them.

**Negative weights.**
For negative weights, it is recommended to only use one of the
following penalties:

.. code:: python3

    pymde.penalties.Log
    pymde.penalties.InvPower
    pymde.penalties.LogRatio

These penalties go to negative infinity as the input approaches zero,
and to zero as the input approaches infinity. With a negative weight,
that means the distortion function goes to infinity at 0, and to 0 at infinity.

Using other penalties with negative weights is possible, but it can lead to
pathological MDE problems if you are not careful.

**Positive weights.**
Penalties that work well in attracting points are those that are :math:`0`
when the distance is :math:`0`, grows when the distance is larger than
:math:`0`. All the penalties in this module, other than the ones listed above
(and the function described below), can be safely used with attractive
penalties. Some examples inlcude:

.. code:: python3

    pymde.penalties.Log1p
    pymde.penalties.Linear
    pymde.penalties.Quadratic
    pymde.penalties.Cubic
    pymde.penalties.Huber

**Combining penalties.**
The ``PushAndPull`` function can be used to combine two penalties, an attractive
penalty for use with positive weights, and a repulsive penalty for use with
negative weights. This leads to a distortion function of the form

.. math::

    f_k(d) = \\begin{cases}
        w_k p_{\\text{attractive}}(d_k) & w_k > 0 \\\\
        w_k p_{\\text{repulsive}}(d_k) & w_k < 0 \\\\
    \\end{cases}.

For example:

.. code:: python3

    weights = torch.tensor([1., 1., -1., 1., -1.])
    attractive_penalty = pymde.penalties.Log1p
    repulsive_penalty = pymde.penalties.Log

    distortion_function = pymde.PushAndPull(
        weights,
        attractive_penalty,
        repulsive_penalty)

**Example.**
Distortion functions are created in a vectorized or elementwise fashion. The
constructor takes a sequence (torch.Tensor) of weights, returning a callable
object. The object takes a sequence of distances of the same length as the
weights, and returns a sequence of distortions, one for each distance.

For example:

.. code:: python3

    weights = torch.tensor([1., 2., 3.])
    f = pymde.penalties.Quadratic(weights)

    distances = torch.tensor([2., 1., 4.])
    distortions = f(distances)
    # the distortions are 1 * 2**2 == 4, 2 * 1**2 == 2, 3 * 4**2 = 48
    print(distortions)

prints

.. code:: python3

    torch.tensor([4., 2., 48.])
"""
from pymde import util
from pymde.functions.function import Function
import torch


class Linear(Function):
    """:math:`p(d) = d`"""

    def __init__(self, weights):
        super(Linear, self).__init__()
        self.weights = util.to_tensor(weights)

    def forward(self, distances):
        return self.weights * distances


class Quadratic(Function):
    """:math:`p(d) = d^2`"""

    def __init__(self, weights):
        super(Quadratic, self).__init__()
        self.weights = util.to_tensor(weights)

    def forward(self, distances):
        return self.weights * distances.pow(2)


class _DeadzoneQuadratic(Function):
    def __init__(self, weights, threshold):
        super(_DeadzoneQuadratic, self).__init__()
        self.weights = util.to_tensor(weights)
        self.threshold = threshold

    def forward(self, distances):
        output = torch.zeros(
            distances.shape, dtype=distances.dtype, device=distances.device
        )
        lt_thresh = distances < self.threshold
        gt_thresh = ~lt_thresh
        output[lt_thresh] = 0.0
        output[gt_thresh] = distances[gt_thresh].pow(2)
        return self.weights * output


class _ClippedQuadratic(Function):
    def __init__(self, weights, threshold):
        super(_ClippedQuadratic, self).__init__()
        self.weights = util.to_tensor(weights)
        self.threshold = threshold

    def forward(self, distances):
        return self.weights * torch.min(
            distances ** 2, (self.threshold + 1) ** 2
        )


class Cubic(Function):
    """:math:`p(d) = d^3`"""

    def __init__(self, weights):
        super(Cubic, self).__init__()
        self.weights = util.to_tensor(weights)

    def forward(self, distances):
        return self.weights * distances.pow(3)


class _DeadzoneCubic(Function):
    def __init__(self, weights, threshold):
        super(_DeadzoneCubic, self).__init__()
        self.weights = util.to_tensor(weights)
        self.threshold = threshold

    def forward(self, distances):
        output = torch.zeros(
            distances.shape, dtype=distances.dtype, device=distances.device
        )
        lt_thresh = distances < self.threshold
        gt_thresh = ~lt_thresh
        output[lt_thresh] = 0.0
        output[gt_thresh] = distances[gt_thresh].pow(3)
        return self.weights * output


class Power(Function):
    """:math:`p(d) = d^\\text{exponent}`"""

    def __init__(self, weights, exponent):
        super(Power, self).__init__()
        self.weights = util.to_tensor(weights)
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent, device=self.weights.device)
        self.exponent = exponent

    def forward(self, distances):
        return self.weights * distances.pow(self.exponent)


class Huber(Function):
    """
    .. math::

        p(d) = \\begin{cases}
            0.5 \\cdot d^2 & d < \\text{threshold} \\\\
            \\text{threshold}(d - 0.5 \\cdot \\text{threshold})
            & d \\geq \\text{threshold}
        \\end{cases}

    """

    def __init__(self, weights, threshold=0.5):
        if threshold < 0:
            raise ValueError(
                "Threshold must be nonnegative, received ", threshold
            )
        super(Huber, self).__init__()
        self.weights = util.to_tensor(weights)
        self.threshold = threshold

    def forward(self, distances):
        output = torch.zeros(
            distances.shape, dtype=distances.dtype, device=distances.device
        )
        lt_idx = distances < self.threshold
        if self.weights.nelement() == 1:
            lt_weights = self.weights
            gt_weights = self.weights
        else:
            lt_weights = self.weights[lt_idx]
            gt_weights = self.weights[~lt_idx]
        output[lt_idx] = lt_weights * 0.5 * distances[lt_idx].pow(2)
        output[~lt_idx] = (
            gt_weights
            * self.threshold
            * (distances[~lt_idx] - 0.5 * self.threshold)
        )
        return output


class Logistic(Function):
    """:math:`p(d) = \\log(1 + e^{\\alpha(d - \\text{threshold})})`"""

    def __init__(self, weights, threshold=0.0, alpha=3.0):
        if threshold < 0:
            raise ValueError(
                "Threshold must be nonnegative, received ", threshold
            )
        super(Logistic, self).__init__()
        self.weights = util.to_tensor(weights)
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, distances):
        zeros = torch.zeros(
            distances.shape, dtype=distances.dtype, device=distances.device
        )
        stacked = torch.stack(
            (zeros, self.alpha * (distances - self.threshold))
        ).T
        return self.weights * torch.logsumexp(stacked, dim=1)


class Sigmoid(Function):
    def __init__(self, weights, threshold, alpha=1.0):
        if threshold < 0:
            raise ValueError(
                "Threshold must be nonnegative, received ", threshold
            )
        super(Sigmoid, self).__init__()
        self.weights = util.to_tensor(weights)
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, distances):
        return self.weights * torch.sigmoid(
            self.alpha * (distances - self.threshold)
        )


class Hinge(Function):
    def __init__(self, weights, threshold, sigma=None):
        if threshold < 0:
            raise ValueError(
                "Threshold must be nonnegative, received ", threshold
            )
        if sigma is None:
            sigma = threshold / 2
        super(Hinge, self).__init__()
        self.weights = util.to_tensor(weights)
        self.threshold = threshold
        self.sigma = sigma

    def forward(self, distances):
        return torch.max(
            torch.tensor(0.0, device=distances.device, dtype=distances.dtype),
            self.weights
            * (
                distances
                - (self.threshold - torch.sign(self.weights) * (self.sigma))
            ),
        )


class Log1p(Function):
    """:math:`p(d) = \log(1 + d^{\\text{exponent}})`"""

    def __init__(self, weights, exponent=1.5):
        super(Log1p, self).__init__()
        self.weights = util.to_tensor(weights)
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent, device=self.weights.device)
        self.exponent = exponent

    def forward(self, distances):
        return self.weights * torch.log1p(distances.pow(self.exponent))


class Log(Function):
    """:math:`p(d) = \log(1 - \\exp(-d^\\text{exponent}))`"""

    def __init__(self, weights, exponent=1.0):
        super(Log, self).__init__()
        self.weights = util.to_tensor(weights)
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent, device=weights.device)
        self.exponent = exponent

    def forward(self, distances):
        return self.weights * torch.log(
            -torch.expm1(-(distances ** self.exponent))
        )


class InvPower(Function):
    """:math:`p(d) = 1/d^\\text{exponent}`"""

    def __init__(self, weights, exponent=1):
        if not (weights <= 0).all():
            raise ValueError("Weights must be negative.")
        super(InvPower, self).__init__()
        self.weights = util.to_tensor(weights)
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent)
        self.exponent = exponent

    def forward(self, distances):
        return self.weights.abs() * 1 / (distances ** self.exponent)


class LogRatio(Function):
    """:math:`p(d) = \\log\\left(\\frac{d^\\text{exponent}}{1 + d^{\\text{exponent}}}\\right)`"""  # noqa: E501

    def __init__(self, weights, exponent=2):
        super(LogRatio, self).__init__()
        self.weights = util.to_tensor(weights)
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent)
        self.exponent = exponent

    def forward(self, distances):
        return self.weights * torch.log(
            distances ** self.exponent / (1 + distances ** self.exponent)
        )


class PushAndPull(Function):
    """Combine an attractive and repulsive penalty.

        .. math::

            f_k(d) = \\begin{cases}
                w_k p_{\\text{attractive}}(d_k) & w_k > 0 \\\\
                w_k p_{\\text{repulsive}}(d_k) & w_k < 0 \\\\
            \\end{cases}
    """

    def __init__(
        self, weights, attractive_penalty=Log1p, repulsive_penalty=LogRatio
    ):
        super(PushAndPull, self).__init__()
        self.weights = util.to_tensor(weights)
        if weights.nelement() == 1:
            raise ValueError("`PushAndPull` requires at least two weights.")
        self.pos_idx = weights >= 0
        self.attractive_penalty = attractive_penalty(weights[self.pos_idx])
        self.repulsive_penalty = repulsive_penalty(weights[~self.pos_idx])

    def forward(self, distances):
        output = torch.empty(
            distances.shape, dtype=distances.dtype, device=distances.device
        )
        output[self.pos_idx] = self.attractive_penalty(distances[self.pos_idx])
        output[~self.pos_idx] = self.repulsive_penalty(distances[~self.pos_idx])
        return output
