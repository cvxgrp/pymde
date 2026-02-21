# Rust extension for pymde

This directory contains a Rust implementation of brute-force exact L2
k-nearest neighbor search, exposed to Python via [PyO3](https://pyo3.rs).
It is a self-contained, solution that uses platform-native BLAS
(Accelerate on macOS, OpenBLAS on Linux) for accelerated matrix operations.

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
library (`_knn.*.so` / `_knn.*.pyd`) into `pymde/`.

To rebuild after editing Rust code, run the same command again. Only changed
files are recompiled.

## Testing

### Rust unit tests

```sh
cd rust && cargo test
```

This runs the native Rust tests (for `insert_topk`, `sgemm_nn_t`, and
`knn_blas_tiled`) without needing Python. No extra setup beyond the Rust
toolchain is required.

### Python integration tests

```sh
pytest pymde/test_knn.py -v
```

## Project layout

```
rust/
├── Cargo.toml      # Package metadata and dependencies
├── Cargo.lock      # Pinned dependency versions (committed for reproducibility)
└── src/
    └── lib.rs      # All Rust source code (single file)
```

## How it works

The module exposes one Python function: `pymde._knn.knn_l2(data, k)`.

The algorithm:

1. Precompute squared norms `||x_i||^2` for every row.
2. Tile the data matrix into query blocks and database blocks.
3. For each tile pair, compute pairwise inner products using BLAS `sgemm`
   (the fastest way to do dense matrix multiply).
4. Recover squared distances via `||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a · b`.
5. Maintain a sorted top-k list per query row, keeping only the closest neighbors.

Query tiles are processed in parallel using [rayon](https://docs.rs/rayon).

## Key dependencies

| Crate | Purpose |
|-------|---------|
| [pyo3](https://pyo3.rs) | Rust ↔ Python bindings (function signatures, type conversions, GIL management) |
| [numpy](https://docs.rs/numpy) | Zero-copy access to NumPy arrays from Rust |
| [rayon](https://docs.rs/rayon) | Data-parallel iteration (parallelizes across query tiles) |

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

2. Export it from the module at the bottom of `lib.rs`:

   ```rust
   #[pymodule]
   mod _knn {
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
