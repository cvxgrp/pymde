from contextlib import contextmanager
import functools
import logging

import numpy as np
import pytest
import torch

from pymde import util


def assert_allclose(x, y, up_to_sign=False, rtol=1e-4, atol=1e-5):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if up_to_sign:
        try:
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
        except AssertionError:
            np.testing.assert_allclose(-x, y, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def assert_all_equal(x, y):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    np.testing.assert_array_equal(x, y)


def cpu(func):

    @functools.wraps(func)
    def wrapper(self):
        current_device = util.get_default_device()
        util.set_default_device('cpu')
        func(self)
        util.set_default_device(current_device)
    return wrapper


def cpu_and_cuda(func):
    if torch.cuda.is_available():
        return pytest.mark.parametrize("device", ['cpu', 'cuda'])(func)
    else:
        return pytest.mark.parametrize("device", ['cpu'])(func)


@contextmanager
def disable_logging(up_to=logging.CRITICAL):
    previous_level = logging.root.manager.disable

    logging.disable(up_to)

    try:
        yield
    finally:
        logging.disable(previous_level)
