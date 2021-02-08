import math
import time

import torch

from pymde.average_distortion import _average_distortion, _gather_indices
from pymde.lbfgs import LBFGS
from pymde import util


class SolveStats(object):
    """Summary statistics for a solve.

    Attributes
    ----------
    average_distortions: sequence
        The average distortion at each iteration.
    residual_norms: sequence
        The residual norm at each iteration.
    step_size_percents: sequence
        The relative size of each step.
    solve_time: float
        The time it took to create the embedding, in seconds.
    snapshots: sequence
        Snapshots of the embedding.
    snapshot_every: int
        The number of iterations between snapshots.
    """
    def __init__(
        self,
        average_distortions,
        residual_norms,
        step_size_percents,
        solve_time,
        times,
        snapshots,
        snapshot_every,
    ):
        self.average_distortions = average_distortions
        self.residual_norms = residual_norms
        self.step_size_percents = step_size_percents
        self.solve_time = solve_time
        self.iterations = len(average_distortions)
        self.times = times
        self.snapshots = snapshots
        self.snapshot_every = snapshot_every

    def __str__(self):
        return (
            "SolveStats:\n"
            "\taverage distortion {0:.3g}\n"
            "\tresidual norm {1:.3g}\n"
            "\tsolve_time (s) {2:.3g}\n"
            "\titerations {3}".format(
                self.average_distortions[-1],
                self.residual_norms[-1],
                self.solve_time,
                self.iterations,
            )
        )

    def _repr_pretty_(self, p, cycle):
        del cycle
        text = self.__str__()
        p.text(text)


def lbfgs(
    X,
    objective_fn,
    constraint,
    eps,
    max_iter,
    memory_size,
    use_line_search,
    use_cached_loss,
    verbose,
    print_every,
    snapshot_every,
    logger,
):
    start_time = time.time()
    average_distortions = []
    grad_norms = []
    step_size_percents = []
    times = []
    snapshots = []

    # LBFGS set-up
    #
    # This callback logs the average distortion and gradient norm
    # per teration
    def callback(loss, grad):
        average_distortions.append(loss.detach().cpu().item())
        grad_norms.append(grad.detach().norm(p="fro").cpu().item())

    # closure for LBFGS optimization step; projects gradient onto
    # onto the tangent space of the iterate
    def value_and_grad():
        opt.zero_grad()
        average_distortion = objective_fn(X)
        average_distortion.backward()
        constraint.project_onto_tangent_space(X, X.grad, inplace=True)
        return average_distortion

    ls = "strong_wolfe" if use_line_search else None
    opt = LBFGS(
        params=[X],
        max_iter=1,
        history_size=memory_size,
        line_search_fn=ls,
        callback=callback,
        tolerance_grad=-torch.tensor(
            float("inf"), dtype=X.dtype, device=X.device
        ),
        tolerance_change=-torch.tensor(
            float("inf"), dtype=X.dtype, device=X.device
        ),
        project_callback=constraint.project_onto_constraint,
        use_cached_loss=use_cached_loss,
    )

    digits = len(str(max_iter))
    start = time.time()
    for iteration in range(0, max_iter):
        if snapshot_every is not None and iteration % snapshot_every == 0:
            snapshots.append(X.detach().cpu().clone())
        with torch.no_grad():
            norm_X = X.norm(p="fro")
        X.requires_grad_(True)
        opt.step(value_and_grad)
        X.requires_grad_(False)

        with torch.no_grad():
            constraint.project_onto_constraint(X, inplace=True)

        times.append(time.time() - start)
        try:
            h = opt.state[opt._params[0]]["t"]
            d = opt.state[opt._params[0]]["d"]
            percent_change = 100 * h * d.norm() / norm_X
        except KeyError:
            h = 0.0
            d = 0.0
            percent_change = 0.0
        step_size_percents.append(float(percent_change))

        norm_grad = grad_norms[-1]
        if verbose and (
            ((iteration % print_every == 0)) or (iteration == max_iter - 1)
        ):
            logger.info(
                "iteration %0*d | distortion %6f | residual norm %g | "
                "step length %g | percent change %g"
                % (
                    digits,
                    iteration,
                    average_distortions[-1],
                    norm_grad,
                    h,
                    percent_change,
                )
            )
        if norm_grad <= eps:
            if verbose:
                logger.info(
                    "Converged in %03d iterations, with residual norm %g"
                    % (iteration + 1, norm_grad)
                )
            break
        elif h == 0:
            opt.reset()
    tot_time = time.time() - start_time
    solve_stats = SolveStats(
        average_distortions,
        grad_norms,
        step_size_percents,
        tot_time,
        times,
        snapshots,
        snapshot_every,
    )
    return X, solve_stats


def spi(
    batch_size,
    stochastic_function,
    X,
    constraint,
    eps,
    max_iter,
    memory_size,
    use_line_search,
    verbose,
    print_every,
    snapshot_every,
    logger,
):
    average_distortions = []
    step_size_percents = []
    times = []
    snapshots = []

    X_target = torch.empty(X.shape, dtype=X.dtype, device=X.device)
    X_curr = X
    X_curr.requires_grad_(False)
    del X

    start_time = time.time()
    for iteration in range(0, max_iter):
        if snapshot_every is not None and iteration % snapshot_every == 0:
            snapshots.append(X_curr.detach().cpu().clone())
        f_batch, edges_batch = stochastic_function.sample(batch_size)
        lhs = _gather_indices(edges_batch[:, 0], X_curr.shape[1])
        rhs = _gather_indices(edges_batch[:, 1], X_curr.shape[1])
        average_distortion_batch = lambda X: _average_distortion(
            X, f_batch, lhs, rhs
        )

        X_target[:] = X_curr

        def solve_batch(X, fn, max_iter):
            try:
                _, solve_stats = lbfgs(
                    X=X,
                    objective_fn=fn,
                    constraint=constraint,
                    eps=eps,
                    max_iter=max_iter,
                    memory_size=memory_size,
                    use_line_search=use_line_search,
                    use_cached_loss=False,
                    verbose=False,
                    print_every=print_every,
                    snapshot_every=None,
                    logger=logger,
                )
            except util.SolverError as e:
                # NaN encountered, svd failed ...
                # skip this iteration and hope that the next batch of edges is
                # okay ...
                logger.warning("error in batch: %s" % str(e))
                solve_stats = None
            X.requires_grad_(False)
            return solve_stats

        if iteration == 0:
            objective_fn = average_distortion_batch
        else:

            def objective_fn(X):
                avg_dist = average_distortion_batch(X)
                prox_term = c * iteration / 2.0 * (X - X_target).pow(2).sum()
                return avg_dist + prox_term

        start = time.time()
        solve_stats = solve_batch(X_curr, objective_fn, 20)
        times.append(time.time() - start)

        if torch.isnan(X_curr).any():
            X_curr[:] = X_target
            break

        with torch.no_grad():
            average_distortions.append(
                average_distortion_batch(X_curr).cpu().item()
            )
            dX = X_curr - X_target
            step_size_percents.append(
                (100 * dX.norm() / X_target.norm()).cpu().item()
            )
            del dX

        if verbose and iteration % print_every == 0:
            iterations = 0 if solve_stats is None else solve_stats.iterations
            logger.info(
                "round %d | distortion %6f | "
                "percent change %g | LBFGS iterations %d"
                % (
                    iteration,
                    average_distortions[-1],
                    step_size_percents[-1],
                    iterations,
                )
            )

        if iteration == 0:
            del X_target
            hvp = util.make_hvp(f_batch, edges_batch, X_curr, constraint)
            linop = util.LinearOperator(hvp, device=X_curr.device)
            trace_estimate = util.hutchpp(linop, X_curr.nelement(), 10)
            c = trace_estimate / (X_curr.nelement())
            c /= 10.0
            if verbose:
                logger.info("Using c = %f" % c)
            X_target = X_curr.detach().clone()

    tot_time = time.time() - start_time
    return (
        X_curr,
        SolveStats(
            average_distortions,
            [math.inf],
            step_size_percents,
            tot_time,
            times,
            snapshots,
            snapshot_every,
        ),
    )
