"""Benchmark k-nearest neighbor methods.

Usage:
    python benchmarks/bench_knn.py [--output FILE] [--skip-pynndescent]
                                   [--sizes ...] [--dims ...] [--k K]
"""

import argparse
import csv
import io
import sys
import time
import tracemalloc

import numpy as np


def _ground_truth(data, k):
    """Brute-force exact kNN using Rust extension for recall computation."""
    from pymde._native import knn_l2

    neighbors, _ = knn_l2(data, k)
    return neighbors[:, 1:]  # drop self


def recall_at_k(neighbors, gt_neighbors):
    """Fraction of true neighbors found, averaged over all queries."""
    n, k = gt_neighbors.shape
    neighbors = neighbors[:, :k]
    hits = 0
    for i in range(n):
        hits += len(set(neighbors[i]) & set(gt_neighbors[i]))
    return hits / (n * k)


def bench_rust_knn(data, k):
    from pymde._native import knn_l2

    tracemalloc.start()
    t0 = time.perf_counter()
    neighbors, sq_distances = knn_l2(data, k)
    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    distances = np.sqrt(np.maximum(sq_distances[:, 1:], 0.0))
    return neighbors[:, 1:], distances, elapsed, peak_mem


def bench_rust_nndescent(data, k):
    from pymde._native import nn_descent

    tracemalloc.start()
    t0 = time.perf_counter()
    neighbors, distances = nn_descent(data, k)
    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return neighbors[:, 1:], distances[:, 1:], elapsed, peak_mem


def bench_pynndescent(data, k):
    import pynndescent

    tracemalloc.start()
    t0 = time.perf_counter()
    index = pynndescent.NNDescent(
        data, n_neighbors=k + 1, verbose=False, max_candidates=60
    )
    neighbors, distances = index.neighbor_graph
    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return neighbors[:, 1:], distances[:, 1:], elapsed, peak_mem


def bench_faiss_flat(data, k):
    import faiss

    d = data.shape[1]
    tracemalloc.start()
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(d)
    index.add(data)
    sq_distances, neighbors = index.search(data, k + 1)
    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    distances = np.sqrt(np.maximum(sq_distances[:, 1:], 0.0))
    return neighbors[:, 1:], distances, elapsed, peak_mem


def bench_faiss_knn(data, k):
    import faiss
    from faiss.contrib.exhaustive_search import knn

    tracemalloc.start()
    t0 = time.perf_counter()
    sq_distances, neighbors = knn(data, data, k + 1)
    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    distances = np.sqrt(np.maximum(sq_distances[:, 1:], 0.0))
    return neighbors[:, 1:], distances, elapsed, peak_mem


def bench_faiss_ivf(data, k):
    import faiss

    n, d = data.shape
    nlist = max(1, int(np.sqrt(n)))
    tracemalloc.start()
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(data)
    index.add(data)
    index.nprobe = min(10, nlist)
    sq_distances, neighbors = index.search(data, k + 1)
    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    distances = np.sqrt(np.maximum(sq_distances[:, 1:], 0.0))
    return neighbors[:, 1:], distances, elapsed, peak_mem


# Always-available methods
METHODS = {
    "rust-knn": bench_rust_knn,
    "rust-nndescent": bench_rust_nndescent,
}

# Conditionally add faiss methods
try:
    import faiss  # noqa: F401

    METHODS["faiss-flat"] = bench_faiss_flat
    METHODS["faiss-knn"] = bench_faiss_knn
    METHODS["faiss-ivf"] = bench_faiss_ivf
except ImportError:
    pass

try:
    import pynndescent

    METHODS["pynndescent"] = bench_pynndescent
except ImportError:
    pass


def run_benchmarks(sizes, dims, k, skip_pynndescent=False):
    methods = {
        name: fn
        for name, fn in METHODS.items()
        if not (skip_pynndescent and name == "pynndescent")
    }

    results = []
    for n in sizes:
        for d in dims:
            print(f"\n--- n={n}, d={d}, k={k} ---")
            rng = np.random.default_rng(42)
            data = rng.standard_normal((n, d)).astype(np.float32)

            # Compute ground truth once
            gt_neighbors = _ground_truth(data, k)

            for name, fn in methods.items():
                print(f"  {name:20s} ... ", end="", flush=True)
                try:
                    neighbors, distances, elapsed, peak_mem = fn(data, k)
                    rec = recall_at_k(neighbors, gt_neighbors)
                    peak_mb = peak_mem / (1024 * 1024)
                    print(
                        f"time={elapsed:7.2f}s  "
                        f"mem={peak_mb:7.1f}MB  "
                        f"recall={rec:.4f}"
                    )
                    results.append(
                        {
                            "n": n,
                            "d": d,
                            "k": k,
                            "method": name,
                            "time_s": round(elapsed, 4),
                            "peak_mem_mb": round(peak_mb, 2),
                            "recall": round(rec, 6),
                        }
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
                    results.append(
                        {
                            "n": n,
                            "d": d,
                            "k": k,
                            "method": name,
                            "time_s": None,
                            "peak_mem_mb": None,
                            "recall": None,
                        }
                    )
    return results


def format_table(results):
    if not results:
        return ""
    buf = io.StringIO()
    header = f"{'method':20s} {'n':>8s} {'d':>5s} {'k':>4s} {'time(s)':>9s} {'mem(MB)':>9s} {'recall':>8s}"
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    for r in results:
        t = f"{r['time_s']:9.2f}" if r["time_s"] is not None else "    FAIL"
        m = (
            f"{r['peak_mem_mb']:9.1f}"
            if r["peak_mem_mb"] is not None
            else "    FAIL"
        )
        rec = f"{r['recall']:8.4f}" if r["recall"] is not None else "    FAIL"
        buf.write(
            f"{r['method']:20s} {r['n']:8d} {r['d']:5d} {r['k']:4d} {t} {m} {rec}\n"
        )
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Benchmark kNN methods")
    parser.add_argument(
        "--output", type=str, default=None, help="CSV output file"
    )
    parser.add_argument(
        "--skip-pynndescent", action="store_true", help="Skip pynndescent"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 10000, 50000, 100000, 500000],
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[32, 128, 784])
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    results = run_benchmarks(
        args.sizes, args.dims, args.k, skip_pynndescent=args.skip_pynndescent
    )

    print("\n\n=== Summary ===\n")
    print(format_table(results))

    if args.output:
        fieldnames = [
            "method",
            "n",
            "d",
            "k",
            "time_s",
            "peak_mem_mb",
            "recall",
        ]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
