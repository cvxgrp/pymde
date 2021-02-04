import argparse
import cProfile
import functools
import re
import time

import numpy as np
import torch
from pymde import constraints
from pymde import problem
from pymde import quadratic
from pymde import util
from pymde.functions import penalties


METHODS = []


def register_benchmark(n_rounds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self):
            for i in range(n_rounds):
                np.random.seed(i)
                torch.manual_seed(i)
                func(self)

        METHODS.append(wrapper)
        return wrapper

    return decorator


def print_name(func):

    name = func.__name__[func.__name__.find("_") + 1 :]

    @functools.wraps(func)
    def wrapper(self):
        print(name, end=",")
        func(self)

    return wrapper


class Benchmark(object):
    def __init__(self, eps, max_iter, memory, regex, profile):
        self.eps = eps
        self.max_iter = max_iter
        self.memory = memory
        self.regex = regex
        self.profile = profile

    def run(self):
        print("name,device,n,m,p,seconds,iters,residual norm")
        if self.profile is not None:
            test_name = self.profile[: self.profile.rfind("_")]
            method = None
            for m in METHODS:
                if m.__name__ == test_name:
                    method = m
                    break
            if method is None:
                raise ValueError("test not found")
            device = self.profile[self.profile.rfind("_") + 1 :]
            util.set_default_device(device)
            np.random.seed(0)
            torch.manual_seed(0)
            while hasattr(method, "__wrapped__"):
                method = method.__wrapped__
            method(self)
            return

        for method in METHODS:
            name = method.__name__
            name = name[name.find("_") + 1 :]
            if re.match(self.regex, name):
                method(self)

    def _sec_iters_err(self, mde, sec):
        residual_norm = mde.solve_stats.residual_norms[-1]
        return sec, mde.solve_stats.iterations, residual_norm

    def _fit(self, mde):
        if self.profile is not None:
            pr = cProfile.Profile()
            pr.enable()
        mde.embed(
            max_iter=self.max_iter,
            eps=self.eps,
            memory_size=self.memory,
        )
        if self.profile is not None:
            pr.disable()
            pr.dump_stats(self.profile + ".prof")

    @register_benchmark(3)
    @print_name
    def test_spectral_tiny(self):
        for device in ["cpu", "cuda"]:
            n = 50
            m = 2
            p = 10 * n

            edges = util.random_edges(n, p).to(device)
            weights = np.abs(np.random.randn(p))
            f = penalties.Quadratic(
                torch.tensor(weights, dtype=torch.float)
            ).to(device)

            mde = problem.MDE(
                n_items=n,
                embedding_dim=m,
                edges=edges,
                distortion_function=f,
                constraint=constraints.Standardized(),
                device=device,
            )
            start = time.time()
            self._fit(mde)
            end = time.time()

            sec, iters, err = self._sec_iters_err(mde, end - start)
            print(
                ",".join(
                    [
                        device,
                        str(n),
                        str(m),
                        str(edges.shape[0]),
                        str(sec),
                        str(iters),
                        str(err),
                    ]
                )
            )

    @register_benchmark(3)
    @print_name
    def test_spectral_small(self):
        for device in ["cpu", "cuda"]:
            n = 500
            m = 2
            p = 10 * n

            edges = util.random_edges(n, p)
            weights = np.abs(np.random.randn(p))
            f = penalties.Quadratic(torch.tensor(weights, dtype=torch.float))

            mde = problem.MDE(
                n_items=n,
                embedding_dim=m,
                edges=edges,
                distortion_function=f,
                constraint=constraints.Standardized(),
            )
            start = time.time()
            self._fit(mde)
            end = time.time()

            sec, iters, err = self._sec_iters_err(mde, end - start)
            print(
                ",".join(
                    [
                        str(device),
                        str(n),
                        str(m),
                        str(edges.shape[0]),
                        str(sec),
                        str(iters),
                        str(err),
                    ]
                )
            )

    @register_benchmark(3)
    @print_name
    def test_spectral_medium(self):
        for device in ["cpu", "cuda"]:
            n = 5000
            m = 2
            p = 10 * n

            edges = util.random_edges(n, p).to(device)
            weights = np.abs(np.random.randn(p))
            f = penalties.Quadratic(
                torch.tensor(weights, dtype=torch.float)
            ).to(device)

            mde = problem.MDE(
                n_items=n,
                embedding_dim=m,
                edges=edges,
                distortion_function=f,
                constraint=constraints.Standardized(),
            )
            start = time.time()
            self._fit(mde)
            end = time.time()

            sec, iters, err = self._sec_iters_err(mde, end - start)
            print(
                ",".join(
                    [
                        str(device),
                        str(n),
                        str(m),
                        str(edges.shape[0]),
                        str(sec),
                        str(iters),
                        str(err),
                    ]
                )
            )

    @register_benchmark(2)
    @print_name
    def test_spectral_large(self):
        for device in ["cpu", "cuda"]:
            n = 50000
            m = 2
            p = 10 * n

            edges = util.random_edges(n, p).to(device)
            weights = np.abs(np.random.randn(p))
            f = penalties.Quadratic(
                torch.tensor(weights, dtype=torch.float)
            ).to(device)

            mde = problem.MDE(
                n_items=n,
                embedding_dim=m,
                edges=edges,
                distortion_function=f,
                constraint=constraints.Standardized(),
            )
            start = time.time()
            self._fit(mde)
            end = time.time()

            sec, iters, err = self._sec_iters_err(mde, end - start)
            print(
                ",".join(
                    [
                        str(device),
                        str(n),
                        str(m),
                        str(edges.shape[0]),
                        str(sec),
                        str(iters),
                        str(err),
                    ]
                )
            )

    @register_benchmark(1)
    @print_name
    def test_spectral_tiny_scipy(self):
        n = 50
        m = 2
        p = 10 * n

        edges = util.random_edges(n, p)
        weights = np.abs(np.random.randn(p))
        L = quadratic._laplacian(n, m, edges, weights)
        start = time.time()
        quadratic._spectral(L, m)
        end = time.time()

        print(
            ",".join(
                [
                    "cpu",
                    str(n),
                    str(m),
                    str(edges.shape[0]),
                    str(end - start),
                    str(-1),
                ]
            )
        )

    @register_benchmark(1)
    @print_name
    def test_spectral_small_scipy(self):
        n = 500
        m = 2
        p = 10 * n

        edges = util.random_edges(n, p)
        weights = np.abs(np.random.randn(p))
        L = quadratic._laplacian(n, m, edges, weights)
        start = time.time()
        quadratic._spectral(L, m)
        end = time.time()

        device = "cpu"
        print(
            ",".join(
                [
                    str(device),
                    str(n),
                    str(m),
                    str(edges.shape[0]),
                    str(end - start),
                    str(-1),
                ]
            )
        )

    @register_benchmark(1)
    @print_name
    def test_spectral_medium_scipy(self):
        n = 5000
        m = 2
        p = 10 * n

        edges = util.random_edges(n, p)
        weights = np.abs(np.random.randn(p))
        L = quadratic._laplacian(n, m, edges, weights)
        start = time.time()
        quadratic._spectral(L, m)
        end = time.time()

        device = "cpu"
        print(
            ",".join(
                [
                    str(device),
                    str(n),
                    str(m),
                    str(edges.shape[0]),
                    str(end - start),
                    str(-1),
                ]
            )
        )

    @register_benchmark(1)
    @print_name
    def test_spectral_large_scipy(self):
        n = 50000
        m = 2
        p = 10 * n

        edges = util.random_edges(n, p)
        weights = np.abs(np.random.randn(p))
        L = quadratic._laplacian(n, m, edges, weights)
        start = time.time()
        quadratic._spectral(L, m)
        end = time.time()

        device = "cpu"
        print(
            ",".join(
                [
                    str(device),
                    str(n),
                    str(m),
                    str(edges.shape[0]),
                    str(end - start),
                    str(-1),
                ]
            )
        )


def driver(eps, max_iter, memory, regex, profile):
    benchmark = Benchmark(
        eps=eps,
        max_iter=max_iter,
        memory=memory,
        regex=regex,
        profile=profile,
    )
    benchmark.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimization benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eps",
        dest="eps",
        metavar="<eps>",
        default=1e-5,
        type=float,
        help="solver tolerance",
    )
    parser.add_argument(
        "--maxiters",
        metavar="<maxiters>",
        default=300,
        type=int,
        help="max iterations",
    )
    parser.add_argument(
        "--memory",
        metavar="<memory>",
        default=10,
        type=int,
        help="LBFGS memory size",
    )
    parser.add_argument(
        "--profile",
        metavar="<testname>",
        type=str,
        help=(
            "name of test method to profile; "
            "profile will be saved to <testname>.prof; "
            "eg, test_spectral_medium_cpu or "
            "test_spectral_medium_cuda"
        ),
    )
    parser.add_argument(
        "--r",
        dest="regex",
        metavar="<regex>",
        default="",
        type=str,
        help="only run benchmarks matching regex",
    )
    args = parser.parse_args()
    driver(
        args.eps,
        args.maxiters,
        args.memory,
        args.regex,
        args.profile,
    )
