"""Penalties: distortion functions derived from weights.

A penalty discouarages distances from being large when the weights are
positive, and encourages them to be small. When the weights are negative, a
penalty discouarges distances from being small, and encourages them to be
large.

Every penalty can be used with positive or negative weights. But when
using negative weights, it is recommended to only use one of the following
penalties, which we refer to as repulsive penalties:

    pymde.penalties.Log
    pymde.penalties.InvPower
    pymde.penalties.LogRatio

Using other penalties with negative weights can lead to pathological
MDE problems if one is not careful.

Penalties that are 0 when the distance is 0, and grow when the distance
is larger than 0, are called attractive penalties. All the penalties in
this module, other than the ones listed above, are attractive penalties.

The `PushAndPull` penalty can be used to combine two penalties, an attractive
penalty for use with positive weights, and a repulsive penalty for use with
negative weights.
"""
from pymde import util
from pymde.functions.function import Function
import torch


class Linear(Function):
    def __init__(self, weights):
        super(Linear, self).__init__()
        self.weights = util.to_tensor(weights)

    def forward(self, distances):
        return self.weights * distances


class Quadratic(Function):
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
    def __init__(self, weights, exponent):
        super(Power, self).__init__()
        self.weights = util.to_tensor(weights)
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent, device=self.weights.device)
        self.exponent = exponent

    def forward(self, distances):
        return self.weights * distances.pow(self.exponent)


class Huber(Function):
    def __init__(self, weights, threshold=0.5, slope=1.0):
        if threshold < 0:
            raise ValueError(
                "Threshold must be nonnegative, received ", threshold
            )
        if slope < 0:
            raise ValueError("Slope must be nonnegative, received ", slope)
        super(Huber, self).__init__()
        self.weights = util.to_tensor(weights)
        self.threshold = threshold
        self.slope = slope

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
        output[lt_idx] = (
            lt_weights * 0.5 * self.slope * distances[lt_idx].pow(2)
        )
        output[~lt_idx] = (
            gt_weights
            * self.threshold
            * self.slope
            * (distances[~lt_idx] - 0.5 * self.threshold)
        )
        return output


class Logistic(Function):
    def __init__(self, weights, threshold=0., alpha=3.0):
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
    def __init__(self, weights, exponent=1.5):
        super(Log1p, self).__init__()
        self.weights = util.to_tensor(weights)
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent, device=self.weights.device)
        self.exponent = exponent

    def forward(self, distances):
        return self.weights * torch.log1p(distances.pow(self.exponent))


class Log(Function):
    def __init__(self, weights, exponent=1.0):
        super(Log, self).__init__()
        self.weights = util.to_tensor(weights)
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent, device=weights.device)
        self.exponent = exponent

    def forward(self, distances):
        return self.weights * torch.log(
            1 - torch.exp(-(distances ** self.exponent))
        )


class InvPower(Function):
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
