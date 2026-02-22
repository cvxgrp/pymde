# Rust extension for pymde

This directory contains Rust implementations of core pymde algorithms,
exposed to Python via [PyO3](https://pyo3.rs):

- **Exact KNN** (`knn_l2`): brute-force L2 k-nearest neighbor search using
  platform-native BLAS (Accelerate on macOS, OpenBLAS on Linux).
- **Approximate KNN** (`nn_descent`): NN-Descent algorithm for building
  approximate k-nearest neighbor graphs. Uses RP-tree initialization and
  iterative local joins to converge on high-recall neighbor graphs, much
  faster than exact search for large datasets.
- **BFS** (`breadth_first_directed`): breadth-first search on directed CSR
  graphs.

## Prerequisites

Install the Rust toolchain via [rustup](https://rustup.rs):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Verify it works:

```sh
rustc --version
cargo --version
```

No other Rust-specific setup is needed — `setuptools-rust` handles invoking
`cargo` as part of the normal Python build.

## Building

From the **project root** (not this directory), build the full package
including the Rust extension:

```sh
pip install -e '.[dev]'
```

This compiles the Rust code in release mode and places the resulting shared
library (`_native.*.so` / `_native.*.pyd`) into `pymde/`.

To rebuild after editing Rust code, run the same command again. Only changed
files are recompiled.

## Testing

### Rust unit tests

```sh
cd rust && cargo test
```

This runs the native Rust tests without needing Python. No extra setup
beyond the Rust toolchain is required.

### Python integration tests

```sh
pytest pymde/test_knn.py -v                     # exact KNN
pytest pymde/preprocess/test_nndescent.py -v    # approximate KNN
```

## Project layout

```
rust/
├── Cargo.toml          # Package metadata and dependencies
├── Cargo.lock          # Pinned dependency versions (committed for reproducibility)
└── src/
    ├── lib.rs          # PyO3 module definition and exports
    ├── knn.rs          # Exact KNN (BLAS-accelerated brute force)
    ├── blas.rs         # BLAS FFI bindings (sgemm)
    ├── nndescent.rs    # NN-Descent approximate KNN algorithm
    ├── heap.rs         # Thread-safe neighbor heaps with AtomicBool try-locks
    ├── candidates.rs   # Candidate tracking for NN-Descent iterations
    ├── distance.rs     # L2 distance kernels (with NEON intrinsics on aarch64)
    ├── rng.rs          # Fast deterministic PRNG (SplitMix64)
    └── bfs.rs          # Breadth-first search on directed CSR graphs
```

## How it works

### Exact KNN (`knn_l2`)

`pymde._native.knn_l2(data, k)` — brute-force exact search.

1. Precompute squared norms `||x_i||^2` for every row.
2. Tile the data matrix into query blocks and database blocks.
3. For each tile pair, compute pairwise inner products using BLAS `sgemm`.
4. Recover squared distances via `||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a · b`.
5. Maintain a sorted top-k list per query row, keeping only the closest neighbors.

Query tiles are processed in parallel using [rayon](https://docs.rs/rayon).

### Approximate KNN (`nn_descent`)

`pymde._native.nn_descent(data, n_neighbors)` — approximate search via
NN-Descent, much faster than exact search for large datasets.

1. **RP-Tree Init**: Build random projection trees to get an initial neighbor
   graph. Points in the same leaf node become candidate neighbors.
2. **NN-Descent Loop**: Iteratively refine the graph using local joins — for
   each point, compare its neighbors' neighbors as potential new neighbors.
   Repeat until convergence (few updates per iteration).
3. **Finalize**: Sort heaps, apply sqrt to distances, return
   `(neighbors, distances)`.

Thread safety uses per-point `AtomicBool` try-locks for concurrent heap
updates, skipping on contention rather than blocking.

## Key dependencies

| Crate | Purpose |
|-------|---------|
| [pyo3](https://pyo3.rs) | Rust ↔ Python bindings (function signatures, type conversions, GIL management) |
| [numpy](https://docs.rs/numpy) | Zero-copy access to NumPy arrays from Rust |
| [rayon](https://docs.rs/rayon) | Data-parallel iteration (parallelizes across query tiles and NN-Descent joins) |
| [rand](https://docs.rs/rand) | Random number generation (RP-tree construction) |
| [rand_chacha](https://docs.rs/rand_chacha) | Deterministic seeded RNG for reproducibility |

BLAS is linked directly via `extern "C"` — no Rust BLAS crate is used.

## Common tasks

### Adding a new Python-visible function

1. Write your function in `src/lib.rs` with the `#[pyfunction]` attribute:

   ```rust
   #[pyfunction]
   fn my_function(py: Python<'_>, arg: i64) -> PyResult<i64> {
       Ok(arg * 2)
   }
   ```

2. Export it from the module in `lib.rs`:

   ```rust
   #[pymodule]
   mod _native {
       #[pymodule_export]
       use super::my_function;
   }
   ```

3. Rebuild with `pip install -e '.[dev]'`.

### Checking your code compiles without a full rebuild

```sh
cd rust && cargo check
```

This is much faster than a full build and catches type errors and borrow
checker issues.

### Formatting and linting

```sh
cd rust && cargo fmt       # auto-format
cd rust && cargo clippy    # lint (like ruff for Rust)
```

### Useful Rust concepts for reading this code

- **`unsafe`**: Rust normally guarantees memory safety at compile time. The
  `unsafe` blocks here are used to call C functions (BLAS `cblas_sgemm`)
  where Rust can't verify safety — the programmer asserts correctness.
- **`Vec<f32>`**: a growable array, similar to a Python list but typed.
  Backed by a contiguous heap allocation, like a NumPy array.
- **`&[f32]`** (a "slice"): a borrowed view into contiguous memory — similar
  to a NumPy view. No copying, no ownership transfer.
- **`par_chunks_mut`**: rayon's parallel version of splitting a slice into
  fixed-size chunks. Each chunk is processed on a separate thread.
- **`PyReadonlyArray2`**: a read-only borrow of a 2D NumPy array. Zero-copy —
  Rust reads the same memory Python allocated.
