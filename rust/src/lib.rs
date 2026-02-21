use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

#[cfg(not(target_os = "macos"))]
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
                        if dist <= threshold {
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

// ---------------------------------------------------------------------------
// BFS on CSR graph
// ---------------------------------------------------------------------------

const NULL_IDX: i32 = -9999;

/// Breadth-first search on a directed CSR graph.
///
/// Mutates `node_list`, `lengths`, and `predecessors` in place.
/// Returns the number of nodes visited.
///
/// Parameters
/// ----------
/// head_node : int
///     Starting node index.
/// indices : numpy.ndarray[int32]
///     CSR column indices.
/// indptr : numpy.ndarray[int32]
///     CSR row pointers.
/// node_list : numpy.ndarray[int32]
///     Output: BFS traversal order.
/// lengths : numpy.ndarray[int32]
///     Output: BFS depth of each node.
/// predecessors : numpy.ndarray[int32]
///     Output: predecessor of each node in BFS tree.
///     Should be initialized to NULL_IDX (-9999).
#[pyfunction]
fn breadth_first_directed<'py>(
    _py: Python<'py>,
    head_node: u32,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    mut node_list: numpy::PyReadwriteArray1<'py, i32>,
    mut lengths: numpy::PyReadwriteArray1<'py, i32>,
    mut predecessors: numpy::PyReadwriteArray1<'py, i32>,
) -> PyResult<usize> {
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    let node_list = node_list.as_slice_mut()?;
    let lengths = lengths.as_slice_mut()?;
    let predecessors = predecessors.as_slice_mut()?;

    Ok(bfs_directed(
        head_node as usize,
        indices,
        indptr,
        node_list,
        lengths,
        predecessors,
    ))
}

/// Pure-Rust BFS on a CSR graph. Mirrors the Cython `_breadth_first_directed`.
fn bfs_directed(
    head_node: usize,
    indices: &[i32],
    indptr: &[i32],
    node_list: &mut [i32],
    lengths: &mut [i32],
    predecessors: &mut [i32],
) -> usize {
    node_list[0] = head_node as i32;
    lengths[head_node] = 0;

    let mut i_nl: usize = 0;
    let mut i_nl_end: usize = 1;

    while i_nl < i_nl_end {
        let pnode = node_list[i_nl] as usize;
        let curr_length = lengths[pnode];

        let start = indptr[pnode] as usize;
        let end = indptr[pnode + 1] as usize;
        for i in start..end {
            let cnode = indices[i] as usize;
            if cnode == head_node {
                continue;
            }
            if predecessors[cnode] == NULL_IDX {
                node_list[i_nl_end] = cnode as i32;
                predecessors[cnode] = pnode as i32;
                lengths[cnode] = curr_length + 1;
                i_nl_end += 1;
            }
        }

        i_nl += 1;
    }

    i_nl
}

#[pymodule]
mod _native {
    #[pymodule_export]
    use super::knn_l2;
    #[pymodule_export]
    use super::breadth_first_directed;
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // bfs_directed
    // -----------------------------------------------------------------------

    /// Helper: run BFS and return (visited_count, node_list, lengths, predecessors).
    fn run_bfs(
        head: usize,
        indices: &[i32],
        indptr: &[i32],
        n: usize,
    ) -> (usize, Vec<i32>, Vec<i32>, Vec<i32>) {
        let mut node_list = vec![NULL_IDX; n];
        let mut lengths = vec![NULL_IDX; n];
        let mut predecessors = vec![NULL_IDX; n];
        let count = bfs_directed(
            head,
            indices,
            indptr,
            &mut node_list,
            &mut lengths,
            &mut predecessors,
        );
        (count, node_list, lengths, predecessors)
    }

    #[test]
    fn bfs_linear_graph() {
        // 0 -> 1 -> 2 -> 3
        let indptr = vec![0, 1, 2, 3, 3]; // node 3 has no outgoing edges
        let indices = vec![1, 2, 3];
        let (count, node_list, lengths, predecessors) = run_bfs(0, &indices, &indptr, 4);
        assert_eq!(count, 4);
        assert_eq!(&node_list[..4], &[0, 1, 2, 3]);
        assert_eq!(lengths[0], 0);
        assert_eq!(lengths[1], 1);
        assert_eq!(lengths[2], 2);
        assert_eq!(lengths[3], 3);
        assert_eq!(predecessors[0], NULL_IDX); // head has no predecessor
        assert_eq!(predecessors[1], 0);
        assert_eq!(predecessors[2], 1);
        assert_eq!(predecessors[3], 2);
    }

    #[test]
    fn bfs_star_graph() {
        // 0 -> 1, 0 -> 2, 0 -> 3
        let indptr = vec![0, 3, 3, 3, 3];
        let indices = vec![1, 2, 3];
        let (count, _node_list, lengths, predecessors) = run_bfs(0, &indices, &indptr, 4);
        assert_eq!(count, 4);
        assert_eq!(lengths[0], 0);
        for i in 1..4 {
            assert_eq!(lengths[i], 1);
            assert_eq!(predecessors[i], 0);
        }
    }

    #[test]
    fn bfs_disconnected() {
        // 0 -> 1, nodes 2 and 3 unreachable
        let indptr = vec![0, 1, 1, 2, 2];
        let indices = vec![1, 3]; // edge 2->3
        let (count, node_list, lengths, predecessors) = run_bfs(0, &indices, &indptr, 4);
        assert_eq!(count, 2); // only 0 and 1 reachable
        assert_eq!(node_list[0], 0);
        assert_eq!(node_list[1], 1);
        assert_eq!(lengths[2], NULL_IDX);
        assert_eq!(lengths[3], NULL_IDX);
        assert_eq!(predecessors[2], NULL_IDX);
    }

    #[test]
    fn bfs_cycle() {
        // Undirected cycle: 0-1-2-3-0 (as symmetric CSR)
        // indptr: [0, 2, 4, 6, 8]
        // indices: [1,3, 0,2, 1,3, 2,0]
        let indptr = vec![0, 2, 4, 6, 8];
        let indices = vec![1, 3, 0, 2, 1, 3, 2, 0];
        let (count, _node_list, lengths, _preds) = run_bfs(0, &indices, &indptr, 4);
        assert_eq!(count, 4);
        assert_eq!(lengths[0], 0);
        assert_eq!(lengths[1], 1);
        assert_eq!(lengths[3], 1);
        assert_eq!(lengths[2], 2);
    }

    #[test]
    fn bfs_single_node() {
        let indptr = vec![0, 0];
        let indices: Vec<i32> = vec![];
        let (count, node_list, lengths, _) = run_bfs(0, &indices, &indptr, 1);
        assert_eq!(count, 1);
        assert_eq!(node_list[0], 0);
        assert_eq!(lengths[0], 0);
    }

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
    // sgemm_nn_t
    // -----------------------------------------------------------------------

    #[test]
    fn sgemm_nn_t_identity() {
        // A = [[1,0],[0,1]], B = [[1,0],[0,1]]
        // C = -2 * A @ B^T = -2 * I
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        unsafe { sgemm_nn_t(2, 2, 2, -2.0, a.as_ptr(), b.as_ptr(), 0.0, c.as_mut_ptr()) };
        assert_eq!(c, [-2.0, 0.0, 0.0, -2.0]);
    }

    #[test]
    fn sgemm_nn_t_simple() {
        // A = [[1,2,3]], B = [[4,5,6],[7,8,9]]
        // A @ B^T = [1*4+2*5+3*6, 1*7+2*8+3*9] = [32, 50]
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = [0.0f32; 2];
        unsafe { sgemm_nn_t(1, 2, 3, 1.0, a.as_ptr(), b.as_ptr(), 0.0, c.as_mut_ptr()) };
        assert_eq!(c, [32.0, 50.0]);
    }

    #[test]
    fn sgemm_nn_t_beta_accumulate() {
        // Check that beta * C is accumulated
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [10.0f32, 20.0, 30.0, 40.0];
        // C = 1.0 * A @ B^T + 0.5 * C = I + [5,10,15,20]
        unsafe { sgemm_nn_t(2, 2, 2, 1.0, a.as_ptr(), b.as_ptr(), 0.5, c.as_mut_ptr()) };
        assert_eq!(c, [6.0, 10.0, 15.0, 21.0]);
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
