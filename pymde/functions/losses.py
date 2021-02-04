"""Losses: Distortion functions derived from original deviations.

A deviation is a nonnegative scalar that quantifies how different two
things are. The larger the deviation, the more different the things are.

A deviation of 0 means that two things are the same.
"""
from pymde import util
from pymde.functions.function import Function
import torch


class Quadratic(Function):
    def __init__(self, deviations):
        super(Quadratic, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        return (self.deviations - distances).pow(2)


class WeightedQuadratic(Function):
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
    def __init__(self, deviations):
        super(Cubic, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        return (self.deviations - distances).abs().pow(3)


class Power(Function):
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
    def __init__(self, deviations):
        super(Absolute, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        return (self.deviations - distances).abs()


class Logistic(Function):
    def __init__(self, deviations):
        super(Logistic, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        diff = self.deviations - distances
        return torch.log(1.0 + torch.exp(diff.abs()))


class Fractional(Function):
    def __init__(self, deviations):
        super(Fractional, self).__init__()
        self.deviations = util.to_tensor(deviations)

    def forward(self, distances):
        return (
            torch.max(self.deviations / distances, distances / self.deviations)
            - 1.0
        )


class SoftFractional(Function):
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
