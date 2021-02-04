"""Distortion function base class

`Function` is just a torch.nn.Module that has a device property.
"""
from pymde import util
import torch


class Function(torch.nn.Module):
    def __init__(self):
        """Distortion function.

        Subclasses should implement the `forward` function, with signature

        forward(self, distances : torch.Tensor) -> distortion : torch.Tensor,

        which maps a vector of embedding distances to a vector of distortions,
        one for each distance.
        """
        super(Function, self).__init__()

    def __setattr__(self, name, value):
        if isinstance(value, torch.Tensor):
            self.register_buffer(name, value)
        else:
            super(Function, self).__setattr__(name, value)

    @property
    def device(self):
        return util._module_device(self)


class StochasticFunction(torch.nn.Module):
    def __init__(self, function, sampler, batch_size, device="cpu", p=None):
        super(StochasticFunction, self).__init__()
        self.function_factory = function
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = util.canonical_device(device)
        self.p = p

    def sample(self):
        edges, args = self.sampler(self.batch_size)
        return self.function_factory(*args), edges
