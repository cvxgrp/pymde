use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::os::raw::c_int;

// ---------------------------------------------------------------------------
// Platform-native BLAS sgemm
// ---------------------------------------------------------------------------

const ROW_MAJOR: c_int = 101;
const NO_TRANS: c_int = 111;
const TRANS: c_int = 112;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
#[link(name = "openblas")]
extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

#[cfg(target_os = "windows")]
#[link(name = "openblas")]
extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

/// Call sgemm: C = alpha * A @ B^T + beta * C
/// A is (m, k) row-major, B is (n, k) row-major, C is (m, n) row-major.
#[inline]
unsafe fn sgemm_nn_t(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
) {
    cblas_sgemm(
        ROW_MAJOR,
        NO_TRANS,
        TRANS,
        m as c_int,
        n as c_int,
        k as c_int,
        alpha,
        a,
        k as c_int, // lda = number of columns of A
        b,
        k as c_int, // ldb = number of columns of B
        beta,
        c,
        n as c_int, // ldc = number of columns of C
    );
}

// ---------------------------------------------------------------------------
// Tiled kNN
// ---------------------------------------------------------------------------

const QUERY_BS: usize = 1024;
const DB_BS: usize = 4096;

/// Brute-force exact L2 k-nearest neighbor search.
///
/// Uses tiled BLAS sgemm for the distance computation
/// (||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b) and rayon for parallelism
/// across query tiles.
///
/// Parameters
/// ----------
/// data : numpy.ndarray, shape (n, d), dtype float32
///     Row-major data matrix.
/// k : int
///     Number of neighbors (excluding self). Must satisfy 1 <= k < n.
///
/// Returns
/// -------
/// (neighbors, sq_distances) : tuple of numpy arrays
///     neighbors : int64, shape (n, k+1) — column 0 is self
///     sq_distances : float32, shape (n, k+1) — squared L2 distances
#[pyfunction]
fn knn_l2<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    k: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let data = data.as_array();
    let n = data.nrows();
    let d = data.ncols();

    if k == 0 || k >= n {
        return Err(PyValueError::new_err(format!(
            "k must satisfy 1 <= k < n, got k={k}, n={n}"
        )));
    }

    let cols = k + 1;

    // Flatten data to contiguous f32 slice
    let flat: Vec<f32> = if let Some(s) = data.as_slice() {
        s.to_vec()
    } else {
        data.iter().copied().collect()
    };

    let (neighbors, sq_distances) = py.detach(|| knn_blas_tiled(&flat, n, d, k));

    let neighbors =
        Array2::from_shape_vec((n, cols), neighbors).expect("shape mismatch for neighbors");
    let sq_distances =
        Array2::from_shape_vec((n, cols), sq_distances).expect("shape mismatch for sq_distances");

    Ok((
        neighbors.into_pyarray(py).unbind(),
        sq_distances.into_pyarray(py).unbind(),
    ))
}

/// Insert (dist, idx) into a sorted slice, maintaining ascending order by
/// (distance, index). Drops the worst (last) element when a better candidate
/// is found.
#[inline(always)]
fn insert_topk(row: &mut [(f32, i64)], dist: f32, idx: i64) {
    let cols = row.len();
    let (worst_d, worst_i) = row[cols - 1];
    if dist > worst_d || (dist == worst_d && idx >= worst_i) {
        return;
    }
    let mut pos = cols - 1;
    while pos > 0 {
        let (d, i) = row[pos - 1];
        if d < dist || (d == dist && i < idx) {
            break;
        }
        row[pos] = row[pos - 1];
        pos -= 1;
    }
    row[pos] = (dist, idx);
}

fn knn_blas_tiled(flat: &[f32], n: usize, d: usize, k: usize) -> (Vec<i64>, Vec<f32>) {
    let cols = k + 1;

    // Precompute squared norms: ||x_i||^2
    let norms: Vec<f32> = (0..n)
        .map(|i| flat[i * d..(i + 1) * d].iter().map(|&v| v * v).sum())
        .collect();

    // Top-k state: n rows of (dist, idx) pairs, sorted ascending.
    let mut best = vec![(f32::INFINITY, i64::MAX); n * cols];

    // Process query tiles in parallel.
    best.par_chunks_mut(QUERY_BS * cols)
        .enumerate()
        .for_each(|(tile_idx, tile_best)| {
            let i0 = tile_idx * QUERY_BS;
            let tm = tile_best.len() / cols;

            // Thread-local scratch for sgemm output
            let mut ip_block = vec![0.0f32; tm * DB_BS];

            for j0 in (0..n).step_by(DB_BS) {
                let j1 = (j0 + DB_BS).min(n);
                let tn = j1 - j0;

                // ip_block[tm × tn] = -2 * data[i0..i0+tm] @ data[j0..j1]^T
                unsafe {
                    sgemm_nn_t(
                        tm,
                        tn,
                        d,
                        -2.0,
                        flat.as_ptr().add(i0 * d),
                        flat.as_ptr().add(j0 * d),
                        0.0,
                        ip_block.as_mut_ptr(),
                    );
                }

                // Update top-k for each query row in this tile
                for bi in 0..tm {
                    let qi = i0 + bi;
                    let norm_qi = norms[qi];
                    let row = &mut tile_best[bi * cols..(bi + 1) * cols];
                    let mut threshold = row[cols - 1].0;

                    let ip_row = &ip_block[bi * tn..(bi + 1) * tn];

                    for bj in 0..tn {
                        let dj = j0 + bj;
                        // Self-distance is exactly 0 (avoid FP noise from
                        // the ||a||^2 + ||b||^2 - 2*a·b expansion)
                        let dist = if qi == dj {
                            0.0
                        } else {
                            (norm_qi + norms[dj] + ip_row[bj]).max(0.0)
                        };
                        if dist < threshold {
                            insert_topk(row, dist, dj as i64);
                            threshold = row[cols - 1].0;
                        }
                    }
                }
            }
        });

    // Extract into separate arrays
    let mut neighbors = vec![0i64; n * cols];
    let mut sq_dists = vec![0.0f32; n * cols];
    for (i, &(dist, idx)) in best.iter().enumerate() {
        sq_dists[i] = dist;
        neighbors[i] = idx;
    }

    (neighbors, sq_dists)
}

#[pymodule]
mod _knn {
    #[pymodule_export]
    use super::knn_l2;
}
