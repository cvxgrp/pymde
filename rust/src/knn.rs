use crate::blas::sgemm_nn_t;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use core::ffi::c_int;

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
pub(crate) fn knn_l2<'py>(
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

    if n > c_int::MAX as usize || d > c_int::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "dimensions exceed c_int::MAX ({}): n={n}, d={d}",
            c_int::MAX
        )));
    }

    let cols = k + 1;

    // Copy into a contiguous Vec so the data is owned and can cross into
    // py.detach() (which releases the GIL — no NumPy references allowed).
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
    let norms: Vec<f32> = flat
        .chunks_exact(d)
        .map(|row| row.iter().map(|&v| v * v).sum())
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
                        if dist <= threshold {
                            insert_topk(row, dist, dj as i64);
                            threshold = row[cols - 1].0;
                        }
                    }
                }
            }
        });

    // Extract into separate arrays
    let (sq_dists, neighbors): (Vec<f32>, Vec<i64>) = best.into_iter().unzip();

    (neighbors, sq_dists)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // insert_topk
    // -----------------------------------------------------------------------

    #[test]
    fn insert_topk_better_candidate_replaces_worst() {
        let mut row = vec![(1.0, 0i64), (2.0, 1), (f32::INFINITY, i64::MAX)];
        insert_topk(&mut row, 3.0, 2);
        assert_eq!(row, [(1.0, 0), (2.0, 1), (3.0, 2)]);
    }

    #[test]
    fn insert_topk_inserts_at_front() {
        let mut row = vec![(2.0, 1i64), (3.0, 2), (4.0, 3)];
        insert_topk(&mut row, 1.0, 0);
        assert_eq!(row, [(1.0, 0), (2.0, 1), (3.0, 2)]);
    }

    #[test]
    fn insert_topk_inserts_in_middle() {
        let mut row = vec![(1.0, 0i64), (3.0, 2), (4.0, 3)];
        insert_topk(&mut row, 2.0, 1);
        assert_eq!(row, [(1.0, 0), (2.0, 1), (3.0, 2)]);
    }

    #[test]
    fn insert_topk_rejects_worse_candidate() {
        let mut row = vec![(1.0, 0i64), (2.0, 1), (3.0, 2)];
        insert_topk(&mut row, 5.0, 99);
        assert_eq!(row, [(1.0, 0), (2.0, 1), (3.0, 2)]);
    }

    #[test]
    fn insert_topk_rejects_equal_dist_worse_idx() {
        let mut row = vec![(1.0, 0i64), (2.0, 1), (3.0, 2)];
        // Same distance as worst, but index >= worst index → rejected
        insert_topk(&mut row, 3.0, 2);
        assert_eq!(row, [(1.0, 0), (2.0, 1), (3.0, 2)]);
        insert_topk(&mut row, 3.0, 5);
        assert_eq!(row, [(1.0, 0), (2.0, 1), (3.0, 2)]);
    }

    #[test]
    fn insert_topk_tiebreak_by_lower_idx() {
        let mut row = vec![(1.0, 5i64), (2.0, 10), (3.0, 15)];
        // Same distance as worst, but lower index → accepted
        insert_topk(&mut row, 3.0, 7);
        assert_eq!(row, [(1.0, 5), (2.0, 10), (3.0, 7)]);
    }

    #[test]
    fn insert_topk_single_element() {
        let mut row = vec![(5.0, 10i64)];
        insert_topk(&mut row, 3.0, 2);
        assert_eq!(row, [(3.0, 2)]);
        // Worse → no change
        insert_topk(&mut row, 4.0, 1);
        assert_eq!(row, [(3.0, 2)]);
    }

    #[test]
    fn insert_topk_multiple_inserts_maintain_order() {
        let mut row = vec![(f32::INFINITY, i64::MAX); 4];
        insert_topk(&mut row, 5.0, 50);
        insert_topk(&mut row, 1.0, 10);
        insert_topk(&mut row, 3.0, 30);
        insert_topk(&mut row, 2.0, 20);
        assert_eq!(row, [(1.0, 10), (2.0, 20), (3.0, 30), (5.0, 50)]);
        // Inserting something worse than all 4 does nothing
        insert_topk(&mut row, 6.0, 60);
        assert_eq!(row, [(1.0, 10), (2.0, 20), (3.0, 30), (5.0, 50)]);
        // Inserting something that displaces the worst
        insert_topk(&mut row, 4.0, 40);
        assert_eq!(row, [(1.0, 10), (2.0, 20), (3.0, 30), (4.0, 40)]);
    }

    // -----------------------------------------------------------------------
    // knn_blas_tiled  (end-to-end on small data)
    // -----------------------------------------------------------------------

    /// Helper: compute squared L2 distance between two d-dimensional vectors.
    fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    #[test]
    fn knn_blas_tiled_basic() {
        // 4 points in 2D, k=2 → output has 3 columns (self + 2 neighbors)
        //   p0=(0,0)  p1=(1,0)  p2=(3,0)  p3=(10,0)
        // Distances:
        //   d(0,1)=1  d(0,2)=9  d(0,3)=100
        //   d(1,2)=4  d(1,3)=81
        //   d(2,3)=49
        let flat: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 10.0, 0.0];
        let (neighbors, sq_dists) = knn_blas_tiled(&flat, 4, 2, 2);

        // Each row: [self, nn1, nn2]
        let cols = 3;
        for i in 0..4 {
            // Column 0 is self
            assert_eq!(neighbors[i * cols], i as i64);
            assert_eq!(sq_dists[i * cols], 0.0);
        }
        // p0's neighbors: p1 (d=1), p2 (d=9)
        assert_eq!(neighbors[0 * cols + 1], 1);
        assert_eq!(neighbors[0 * cols + 2], 2);
        assert!((sq_dists[0 * cols + 1] - 1.0).abs() < 1e-5);
        assert!((sq_dists[0 * cols + 2] - 9.0).abs() < 1e-5);

        // p1's neighbors: p0 (d=1), p2 (d=4)
        assert_eq!(neighbors[1 * cols + 1], 0);
        assert_eq!(neighbors[1 * cols + 2], 2);

        // p3's neighbors: p2 (d=49), p1 (d=81)
        assert_eq!(neighbors[3 * cols + 1], 2);
        assert_eq!(neighbors[3 * cols + 2], 1);
    }

    #[test]
    fn knn_blas_tiled_k1() {
        // k=1: each point should find its nearest neighbor
        // Triangle: (0,0), (1,0), (0.5, 0.866)
        let flat: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.866];
        let (neighbors, sq_dists) = knn_blas_tiled(&flat, 3, 2, 1);
        let cols = 2; // self + 1 neighbor

        // All self-distances are 0
        for i in 0..3 {
            assert_eq!(neighbors[i * cols], i as i64);
            assert_eq!(sq_dists[i * cols], 0.0);
        }

        // Brute-force check the nearest neighbor for each point
        let points = [[0.0f32, 0.0], [1.0, 0.0], [0.5, 0.866]];
        for i in 0..3 {
            let nn = neighbors[i * cols + 1] as usize;
            let nn_dist = sq_dists[i * cols + 1];
            // Verify this is indeed the closest
            for j in 0..3 {
                if j == i {
                    continue;
                }
                let d = sq_l2(&points[i], &points[j]);
                assert!(
                    nn_dist <= d + 1e-4,
                    "point {i}: reported nn={nn} dist={nn_dist}, but d({i},{j})={d}"
                );
            }
        }
    }

    #[test]
    fn knn_blas_tiled_distances_correct() {
        // Verify reported squared distances match brute-force computation
        let flat: Vec<f32> = vec![
            1.0, 2.0, 3.0, // p0
            4.0, 5.0, 6.0, // p1
            7.0, 8.0, 9.0, // p2
            1.5, 2.5, 3.5, // p3
            10.0, 0.0, 0.0, // p4
        ];
        let n = 5;
        let d = 3;
        let k = 3;
        let (neighbors, sq_dists) = knn_blas_tiled(&flat, n, d, k);
        let cols = k + 1;

        for i in 0..n {
            for c in 1..cols {
                let j = neighbors[i * cols + c] as usize;
                let reported = sq_dists[i * cols + c];
                let expected = sq_l2(&flat[i * d..(i + 1) * d], &flat[j * d..(j + 1) * d]);
                assert!(
                    (reported - expected).abs() < 1e-3,
                    "row {i} col {c}: neighbor {j}, reported={reported}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn knn_blas_tiled_sorted_ascending() {
        // Verify each row's distances are in non-decreasing order
        let flat: Vec<f32> = vec![
            0.0, 0.0, 3.0, 4.0, 1.0, 1.0, 6.0, 0.0, 2.0, 3.0,
        ];
        let n = 5;
        let d = 2;
        let k = 4;
        let (_, sq_dists) = knn_blas_tiled(&flat, n, d, k);
        let cols = k + 1;

        for i in 0..n {
            for c in 1..cols {
                assert!(
                    sq_dists[i * cols + c - 1] <= sq_dists[i * cols + c],
                    "row {i}: distances not sorted at columns {}-{}",
                    c - 1,
                    c
                );
            }
        }
    }
}
