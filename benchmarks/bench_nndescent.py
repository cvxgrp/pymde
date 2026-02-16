"""Benchmark: Rust NN-Descent vs pynndescent.

Usage:
    python benchmarks/bench_nndescent.py [--sizes 10000,50000] [--dims 32,128] [--k 15] [--runs 3]

Requirements:
    pip install pynndescent   # for the baseline comparison
"""

import argparse
import subprocess
import sys
import time

import numpy as np


def _check_pynndescent():
    try:
        import pynndescent  # noqa: F401

        return True
    except ImportError:
        return False


def brute_force_knn(data, k):
    """Brute-force KNN for recall computation (small datasets only)."""
    import sklearn.neighbors

    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=k + 1, algorithm="brute"
    )
    nn.fit(data)
    _, indices = nn.kneighbors(data)
    return indices[:, 1:]  # exclude self


def recall(approx_indices, true_indices):
    """Fraction of true neighbors found in approximate result."""
    n, k = true_indices.shape
    hits = 0
    for i in range(n):
        true_set = set(true_indices[i])
        for j in range(1, approx_indices.shape[1]):
            if approx_indices[i, j] in true_set:
                hits += 1
    return hits / (n * k)


def bench_rust(data, k, n_runs):
    """Benchmark the Rust nn_descent implementation."""
    from pymde._nndescent import nn_descent

    data_f32 = np.ascontiguousarray(data, dtype=np.float32)
    times = []
    last_indices = None
    for i in range(n_runs):
        start = time.perf_counter()
        indices, distances = nn_descent(
            data_f32, n_neighbors=k + 1, max_candidates=60, seed=42
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        last_indices = indices
    return times, last_indices


def bench_pynndescent_cold(data, k):
    """Benchmark pynndescent in a fresh subprocess (includes JIT burn-in).

    Spawns a child process so numba's JIT cache doesn't carry over.
    Returns wall-clock time in seconds and the neighbor indices.
    """
    script = f"""\
import time, sys, numpy as np, json
data = np.load(sys.argv[1])
k = {k}
start = time.perf_counter()
import pynndescent
index = pynndescent.NNDescent(data, n_neighbors=k+1, verbose=False, max_candidates=60)
neighbors, distances = index.neighbor_graph
elapsed = time.perf_counter() - start
np.save(sys.argv[2], neighbors)
print(f"{{elapsed:.4f}}")
"""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.npy")
        idx_path = os.path.join(tmpdir, "indices.npy")
        np.save(data_path, data)

        result = subprocess.run(
            [sys.executable, "-c", script, data_path, idx_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  pynndescent subprocess failed:\n{result.stderr}")
            return None, None

        elapsed = float(result.stdout.strip())
        indices = np.load(idx_path)
    return elapsed, indices


def bench_pynndescent_warm(data, k, n_runs):
    """Benchmark pynndescent in-process (JIT already compiled)."""
    import pynndescent

    # Warm-up JIT (this run is NOT included in timing)
    tiny = np.random.randn(100, data.shape[1]).astype(np.float32)
    pynndescent.NNDescent(tiny, n_neighbors=min(k + 1, 50))

    times = []
    last_indices = None
    for _ in range(n_runs):
        start = time.perf_counter()
        index = pynndescent.NNDescent(
            data, n_neighbors=k + 1, verbose=False, max_candidates=60
        )
        neighbors, distances = index.neighbor_graph
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        last_indices = neighbors
    return times, last_indices


def fmt_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def fmt_speedup(baseline, candidate):
    """Format speedup of candidate vs baseline. >1 means candidate is faster."""
    ratio = baseline / candidate
    if ratio >= 1:
        return f"Rust is {ratio:.1f}x faster"
    else:
        return f"pynndescent is {1 / ratio:.1f}x faster"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Rust NN-Descent vs pynndescent"
    )
    parser.add_argument(
        "--sizes",
        default="10000,50000",
        help="Comma-separated dataset sizes (default: 10000,50000)",
    )
    parser.add_argument(
        "--dims",
        default="32,128",
        help="Comma-separated dimensions (default: 32,128)",
    )
    parser.add_argument("--k", type=int, default=15, help="Number of neighbors (default: 15)")
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of timed runs per config (default: 3)"
    )
    parser.add_argument(
        "--recall",
        action="store_true",
        help="Compute recall vs brute force (slow for large n)",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    k = args.k
    n_runs = args.runs
    has_pynn = _check_pynndescent()

    print("=" * 72)
    print("NN-Descent Benchmark: Rust vs pynndescent")
    print("=" * 72)
    if not has_pynn:
        print(
            "\nWARNING: pynndescent not installed. "
            "Install it to compare: pip install pynndescent\n"
        )
    print(f"  k={k}, runs={n_runs}, sizes={sizes}, dims={dims}")
    print(f"  recall={'yes' if args.recall else 'no'}")
    print()

    for n in sizes:
        for dim in dims:
            print("-" * 72)
            print(f"n={n:,}  dim={dim}  k={k}")
            print("-" * 72)

            np.random.seed(0)
            data = np.random.randn(n, dim).astype(np.float32)

            # --- Rust ---
            rust_times, rust_indices = bench_rust(data, k, n_runs)
            rust_median = np.median(rust_times)
            print(
                f"  Rust:               {fmt_time(rust_median):>8s} median  "
                f"({', '.join(fmt_time(t) for t in rust_times)})"
            )

            # --- pynndescent (cold = fresh process, includes JIT) ---
            if has_pynn:
                pynn_cold_time, pynn_cold_indices = bench_pynndescent_cold(
                    data, k
                )
                if pynn_cold_time is not None:
                    print(
                        f"  pynndescent (cold): {fmt_time(pynn_cold_time):>8s}          "
                        f"{fmt_speedup(pynn_cold_time, rust_median)}"
                    )

                # --- pynndescent (warm = JIT cached in-process) ---
                pynn_warm_times, pynn_warm_indices = bench_pynndescent_warm(
                    data, k, n_runs
                )
                pynn_warm_median = np.median(pynn_warm_times)
                print(
                    f"  pynndescent (warm): {fmt_time(pynn_warm_median):>8s} median  "
                    f"{fmt_speedup(pynn_warm_median, rust_median)}  "
                    f"({', '.join(fmt_time(t) for t in pynn_warm_times)})"
                )

            # --- Recall ---
            if args.recall and n <= 20000:
                true_indices = brute_force_knn(data, k)
                rust_r = recall(rust_indices, true_indices)
                print(f"  Recall (Rust):      {rust_r:.4f}")
                if has_pynn and pynn_warm_indices is not None:
                    pynn_r = recall(pynn_warm_indices, true_indices)
                    print(f"  Recall (pynndescent): {pynn_r:.4f}")
            elif args.recall:
                print(
                    f"  Recall: skipped (n={n:,} too large for brute force)"
                )

            print()

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
