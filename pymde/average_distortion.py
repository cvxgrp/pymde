import logging


import torch


LOGGER = logging.getLogger("__pymde__")


# PyTorch has an inneficient implementation of the 2-norm; this
# implementation is about 2x faster on a GPU.
class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        norm_x = x.pow(2).sum(dim=1).sqrt()
        ctx.save_for_backward(x, norm_x)
        return norm_x

    @staticmethod
    def backward(ctx, grad_output):
        x, norm_x = ctx.saved_tensors
        grad_output = grad_output.unsqueeze(1)
        norm_x = norm_x.unsqueeze(1)
        grad_input = x.mul(grad_output).div(norm_x)
        grad_input[torch.isnan(grad_input)] = 0.0
        return grad_input


_norm = _Norm.apply


def _gather_indices(idx, m):
    return idx[:, None].expand(idx.shape[0], m)


class _AverageDistortion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, f, lhs, rhs):
        ctx.shape = X.shape

        # indexing appears to be faster on cpu and cuda
        differences = X[lhs[:, 0]] - X[rhs[:, 0]]
        # differences.pow(2) allocates another p*m*4 bytes,
        # which can OOM a GPU when p is large, so square it in place and
        # sqrt root it when not needed
        norms = differences.pow(2).sum(dim=1).sqrt()
        if X.requires_grad:
            with torch.enable_grad():
                norms.requires_grad_(True)
                norms.grad = None
                distortion = f(norms).mean()
                distortion.backward()
                norms.requires_grad_(False)
            # norms can be zero (rarely)
            g = norms.grad / norms
            # TODO(akshayka): Log a warning on first occurrence during solve.
            if (norms == 0).any():
                pass
            if torch.isnan(g).any():
                g[torch.isnan(g)] = 1.0
            if torch.isinf(g).any():
                g[torch.isinf(g)] = 1.0
            ctx.save_for_backward(differences, g, lhs, rhs)
        else:
            distortion = f(norms).mean()
        return distortion

    @staticmethod
    def backward(ctx, grad_output):
        out = torch.zeros(
            ctx.shape, dtype=grad_output.dtype, device=grad_output.device
        )
        differences, g, lhs, rhs = ctx.saved_tensors
        DATX = g[:, None] * differences
        # scatter add is faster than index_put on cuda, and on CPU
        # (index_put is serial on CPU) ... but scatter_add is nondeterministic
        out.scatter_add_(0, lhs, DATX)
        out.scatter_add_(0, rhs, -DATX)
        out.mul_(grad_output)
        return out, None, None, None


_average_distortion = _AverageDistortion.apply


class _ProjectGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, constraint):
        ctx.constraint = constraint
        ctx.X = X
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return (
            ctx.constraint.project_onto_tangent_space(
                ctx.X, grad_output, inplace=False
            ),
            None,
        )


_project_gradient = _ProjectGradient.apply
