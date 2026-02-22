/// NN-Descent approximate k-nearest neighbor graph construction.
///
/// Three phases:
/// 1. RP-Tree Init: Build random projection trees, use leaf co-occurrence to
///    initialize neighbor heaps.
/// 2. NN-Descent Loop: Iteratively refine using local joins on new/old
///    candidates until convergence.
/// 3. Finalize: Sort heaps, apply sqrt, return (neighbors, distances).

use crate::candidates::CandidateHeap;
use crate::distance::{dot_product, squared_euclidean, squared_euclidean_bounded};
use crate::heap::NeighborHeap;
use crate::rng::Rng64;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ───────────────────────────────────────────────────────────────────
// Parameters
// ───────────────────────────────────────────────────────────────────

fn n_trees(n: usize) -> usize {
    let v = (2.0 * (n as f64).log10()).round() as usize;
    v.clamp(3, 12)
}

fn leaf_size(n_neighbors: usize) -> usize {
    let v = 5 * n_neighbors;
    v.clamp(60, 256)
}

fn n_iters(n: usize) -> usize {
    let v = (n as f64).log2().round() as usize;
    v.max(5)
}

const DELTA: f64 = 0.001;

// ───────────────────────────────────────────────────────────────────
// PyO3 entry point
// ───────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (data, n_neighbors, max_candidates=60, seed=42, verbose=false))]
pub fn nn_descent<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    n_neighbors: usize,
    max_candidates: usize,
    seed: u64,
    verbose: bool,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let data = data.as_array();
    let n = data.nrows();
    let d = data.ncols();

    if n_neighbors == 0 || n_neighbors >= n {
        return Err(PyValueError::new_err(format!(
            "n_neighbors must satisfy 1 <= n_neighbors < n, got n_neighbors={n_neighbors}, n={n}"
        )));
    }

    if verbose {
        eprintln!(
            "nn_descent: building {n_neighbors}-neighbor graph for {n} points in {d} dimensions"
        );
    }

    // Copy data into contiguous Vec for owned access across threads
    let flat: Vec<f32> = if let Some(s) = data.as_slice() {
        s.to_vec()
    } else {
        data.iter().copied().collect()
    };

    let start = Instant::now();
    let (neighbors, distances) =
        py.detach(|| nn_descent_impl(&flat, n, d, n_neighbors, max_candidates, seed, verbose));
    let elapsed = start.elapsed();

    if verbose {
        eprintln!("nn_descent: done in {:.2}s", elapsed.as_secs_f64());
    }

    let cols = n_neighbors + 1;
    let neighbors =
        Array2::from_shape_vec((n, cols), neighbors).expect("shape mismatch for neighbors");
    let distances =
        Array2::from_shape_vec((n, cols), distances).expect("shape mismatch for distances");

    Ok((
        neighbors.into_pyarray(py).unbind(),
        distances.into_pyarray(py).unbind(),
    ))
}

// ───────────────────────────────────────────────────────────────────
// Core implementation
// ───────────────────────────────────────────────────────────────────

fn nn_descent_impl(
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    max_candidates: usize,
    seed: u64,
    verbose: bool,
) -> (Vec<i64>, Vec<f32>) {
    let num_trees = n_trees(n);
    let ls = leaf_size(k);
    let iters = n_iters(n);
    // Scale max_candidates for larger datasets to maintain recall
    let max_candidates = max_candidates.max((1.5 * (n as f64).sqrt()) as usize).min(256);

    if verbose {
        eprintln!(
            "  params: n_trees={num_trees}, leaf_size={ls}, n_iters={iters}, max_candidates={max_candidates}"
        );
    }

    // Phase 1: RP-Tree initialization
    let mut heap = NeighborHeap::new(n, k);
    rp_tree_init(data, n, d, num_trees, ls, seed, &mut heap, verbose);

    // Fill remaining empty slots with random neighbors
    random_fill(data, n, d, k, seed + 0xDEAD, &mut heap);

    if verbose {
        eprintln!("  rp-tree init complete");
    }

    // Phase 2: NN-Descent loop
    let threshold = (DELTA * k as f64 * n as f64) as usize;
    let mut new_candidates = CandidateHeap::new(n, max_candidates);
    let mut old_candidates = CandidateHeap::new(n, max_candidates);
    let mut rng = Rng64::new(seed + 0xBEEF);

    for iter in 0..iters {
        new_candidates.clear();
        old_candidates.clear();

        build_candidates(
            &mut heap,
            &mut new_candidates,
            &mut old_candidates,
            &mut rng,
        );

        let updates = local_join(data, n, d, &heap, &new_candidates, &old_candidates);

        if verbose {
            eprintln!("  iter {}: {} updates (threshold={})", iter + 1, updates, threshold);
        }

        if updates <= threshold {
            if verbose {
                eprintln!("  converged at iter {}", iter + 1);
            }
            break;
        }
    }

    // Phase 3: Finalize
    finalize(n, k, heap)
}

// ───────────────────────────────────────────────────────────────────
// Phase 1: RP-Tree initialization
// ───────────────────────────────────────────────────────────────────

struct RpTree {
    /// Each leaf is a list of point indices
    leaves: Vec<Vec<usize>>,
}

fn rp_tree_init(
    data: &[f32],
    n: usize,
    d: usize,
    num_trees: usize,
    leaf_size: usize,
    seed: u64,
    heap: &mut NeighborHeap,
    verbose: bool,
) {
    // Build all trees in parallel
    let trees: Vec<RpTree> = (0..num_trees)
        .into_par_iter()
        .map(|t| {
            let mut rng = Rng64::new(seed.wrapping_add((t as u64).wrapping_mul(0x9E3779B97F4A7C15)));
            let indices: Vec<usize> = (0..n).collect();
            let mut leaves = Vec::new();
            // Pre-allocate buffer for hyperplane computation
            let mut diff = vec![0.0f32; d];
            rp_tree_split(data, d, &indices, leaf_size, &mut rng, &mut leaves, &mut diff);
            RpTree { leaves }
        })
        .collect();

    // Process trees one at a time (apply tree T's results before T+1
    // so thresholds tighten progressively). Within each tree, process
    // leaves in parallel using push_unlocked.
    for (t, tree) in trees.iter().enumerate() {
        tree.leaves.par_iter().for_each(|leaf| {
            // Compute all pairwise distances within the leaf
            for i in 0..leaf.len() {
                let pi = leaf[i];
                let row_i = &data[pi * d..(pi + 1) * d];
                for j in (i + 1)..leaf.len() {
                    let pj = leaf[j];
                    let row_j = &data[pj * d..(pj + 1) * d];

                    let bound_i = heap.largest_distance(pi);
                    let bound_j = heap.largest_distance(pj);
                    let bound = bound_i.max(bound_j);

                    let dist = squared_euclidean_bounded(row_i, row_j, bound);
                    if dist < bound_i {
                        heap.push_unlocked(pi, dist, pj as i32);
                    }
                    if dist < bound_j {
                        heap.push_unlocked(pj, dist, pi as i32);
                    }
                }
            }
        });

        if verbose {
            eprintln!("  rp-tree {}/{} processed ({} leaves)", t + 1, num_trees, tree.leaves.len());
        }
    }
}

fn rp_tree_split(
    data: &[f32],
    d: usize,
    indices: &[usize],
    leaf_size: usize,
    rng: &mut Rng64,
    leaves: &mut Vec<Vec<usize>>,
    diff: &mut [f32],
) {
    if indices.len() <= leaf_size {
        leaves.push(indices.to_vec());
        return;
    }

    // Pick two random pivot points
    let i = rng.rand_int(indices.len());
    let mut j = rng.rand_int(indices.len());
    if j == i {
        j = (i + 1) % indices.len();
    }
    let pivot_a = indices[i];
    let pivot_b = indices[j];

    let row_a = &data[pivot_a * d..(pivot_a + 1) * d];
    let row_b = &data[pivot_b * d..(pivot_b + 1) * d];

    // Compute hyperplane: diff = a - b, offset = dot(diff, midpoint)
    let mut offset = 0.0f32;
    for dim in 0..d {
        diff[dim] = row_a[dim] - row_b[dim];
        offset += diff[dim] * (row_a[dim] + row_b[dim]);
    }
    offset *= 0.5;

    // Project all points and partition
    let mut left = Vec::new();
    let mut right = Vec::new();

    for &idx in indices {
        let row = &data[idx * d..(idx + 1) * d];
        let p = dot_product(&diff[..d], row);
        if p > offset {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    // If one side is empty, split evenly (degenerate case)
    if left.is_empty() || right.is_empty() {
        let mid = indices.len() / 2;
        left = indices[..mid].to_vec();
        right = indices[mid..].to_vec();
    }

    rp_tree_split(data, d, &left, leaf_size, rng, leaves, diff);
    rp_tree_split(data, d, &right, leaf_size, rng, leaves, diff);
}

fn random_fill(data: &[f32], n: usize, d: usize, k: usize, seed: u64, heap: &mut NeighborHeap) {
    let mut rng = Rng64::new(seed);
    for i in 0..n {
        // Count how many empty slots this point has
        let base = i * k;
        let mut empty = 0;
        for j in 0..k {
            if heap.indices()[base + j] == -1 {
                empty += 1;
            }
        }
        // Try to fill empty slots with random neighbors (with real distances)
        let mut attempts = 0;
        let row_i = &data[i * d..(i + 1) * d];
        while empty > 0 && attempts < k * 5 {
            let j = rng.rand_int(n);
            if j != i {
                let row_j = &data[j * d..(j + 1) * d];
                let dist = squared_euclidean(row_i, row_j);
                if heap.push(i, dist, j as i32) {
                    empty -= 1;
                }
            }
            attempts += 1;
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Phase 2: NN-Descent loop
// ───────────────────────────────────────────────────────────────────

fn build_candidates(
    heap: &mut NeighborHeap,
    new_candidates: &mut CandidateHeap,
    old_candidates: &mut CandidateHeap,
    rng: &mut Rng64,
) {
    let n = heap.n;
    let k = heap.k;

    for i in 0..n {
        let base = i * k;
        for j in 0..k {
            let neighbor = heap.indices()[base + j];
            if neighbor < 0 {
                continue;
            }
            let nb = neighbor as usize;
            let priority = rng.rand_float();

            if heap.is_new()[base + j] {
                // Add to new_candidates: forward (i→neighbor) and reverse (neighbor→i)
                let sampled = new_candidates.push(i, neighbor, priority);
                new_candidates.push(nb, i as i32, priority);
                // Clear is_new inline when the forward direction was sampled
                if sampled {
                    heap.is_new_mut()[base + j] = false;
                }
            } else {
                old_candidates.push(i, neighbor, priority);
                old_candidates.push(nb, i as i32, priority);
            }
        }
    }
}

fn local_join(
    data: &[f32],
    n: usize,
    d: usize,
    heap: &NeighborHeap,
    new_candidates: &CandidateHeap,
    old_candidates: &CandidateHeap,
) -> usize {
    let total_updates = AtomicUsize::new(0);

    (0..n).into_par_iter().for_each(|v| {
        let new_list = new_candidates.get(v);
        let old_list = old_candidates.get(v);
        let mut updates = 0usize;

        // new-new pairs
        for i in 0..new_list.len() {
            let p = new_list[i];
            if p < 0 {
                continue;
            }
            let pu = p as usize;
            let row_p = &data[pu * d..(pu + 1) * d];

            for j in (i + 1)..new_list.len() {
                let q = new_list[j];
                if q < 0 || p == q {
                    continue;
                }
                let qu = q as usize;

                let row_q = &data[qu * d..(qu + 1) * d];
                let bound = heap.largest_distance(pu).max(heap.largest_distance(qu));
                let dist = squared_euclidean_bounded(row_p, row_q, bound);

                if dist < heap.largest_distance(pu) {
                    if heap.push_concurrent(pu, dist, q) {
                        updates += 1;
                    }
                }
                if dist < heap.largest_distance(qu) {
                    if heap.push_concurrent(qu, dist, p) {
                        updates += 1;
                    }
                }
            }
        }

        // new-old pairs
        for &p in new_list {
            if p < 0 {
                continue;
            }
            let pu = p as usize;
            let row_p = &data[pu * d..(pu + 1) * d];

            for &q in old_list {
                if q < 0 || p == q {
                    continue;
                }
                let qu = q as usize;

                let row_q = &data[qu * d..(qu + 1) * d];
                let bound = heap.largest_distance(pu).max(heap.largest_distance(qu));
                let dist = squared_euclidean_bounded(row_p, row_q, bound);

                if dist < heap.largest_distance(pu) {
                    if heap.push_concurrent(pu, dist, q) {
                        updates += 1;
                    }
                }
                if dist < heap.largest_distance(qu) {
                    if heap.push_concurrent(qu, dist, p) {
                        updates += 1;
                    }
                }
            }
        }

        total_updates.fetch_add(updates, Ordering::Relaxed);
    });

    total_updates.load(Ordering::Relaxed)
}

// ───────────────────────────────────────────────────────────────────
// Phase 3: Finalize
// ───────────────────────────────────────────────────────────────────

fn finalize(
    n: usize,
    k: usize,
    mut heap: NeighborHeap,
) -> (Vec<i64>, Vec<f32>) {
    heap.sort_by_distance();

    let cols = k + 1;
    let mut neighbors = vec![0i64; n * cols];
    let mut distances = vec![0.0f32; n * cols];

    let heap_indices = heap.indices();
    let heap_distances = heap.distances();

    // Parallel finalization: each point fills its row
    let chunk_size = cols;
    neighbors
        .par_chunks_mut(chunk_size)
        .zip(distances.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(i, (nb_row, dist_row))| {
            // Column 0: self
            nb_row[0] = i as i64;
            dist_row[0] = 0.0;

            let base = i * k;
            for j in 0..k {
                let idx = heap_indices[base + j];
                let sq_dist = heap_distances[base + j];

                nb_row[j + 1] = idx as i64;
                if idx >= 0 && sq_dist < f32::INFINITY {
                    dist_row[j + 1] = sq_dist.max(0.0).sqrt();
                } else {
                    dist_row[j + 1] = f32::INFINITY;
                }
            }
        });

    (neighbors, distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng64::new(seed);
        (0..n * d).map(|_| rng.rand_float() * 10.0).collect()
    }

    fn brute_force_knn(data: &[f32], n: usize, d: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = vec![vec![]; n];
        for i in 0..n {
            let mut dists: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let row_i = &data[i * d..(i + 1) * d];
                    let row_j = &data[j * d..(j + 1) * d];
                    (squared_euclidean(row_i, row_j), j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            result[i] = dists[..k].iter().map(|&(_, j)| j).collect();
        }
        result
    }

    fn recall(approx: &[(Vec<i64>, Vec<f32>)], gt: &[Vec<usize>], k: usize) -> f64 {
        let (ref nb, _) = approx[0];
        let n = gt.len();
        let cols = k + 1;
        let mut hits = 0;
        for i in 0..n {
            let approx_set: std::collections::HashSet<i64> =
                (1..cols).map(|j| nb[i * cols + j]).collect();
            for &g in &gt[i] {
                if approx_set.contains(&(g as i64)) {
                    hits += 1;
                }
            }
        }
        hits as f64 / (n * k) as f64
    }

    #[test]
    fn small_exact_recall() {
        let n = 20;
        let d = 8;
        let k = 5;
        let data = make_grid(n, d, 42);
        let gt = brute_force_knn(&data, n, d, k);
        let (nb, dist) = nn_descent_impl(&data, n, d, k, 60, 42, false);
        let r = recall(&[(nb.clone(), dist.clone())], &gt, k);
        assert!(
            r >= 0.99,
            "recall {r} < 0.99 for n={n} small exact test"
        );
    }

    #[test]
    fn medium_recall() {
        let n = 1000;
        let d = 32;
        let k = 15;
        let data = make_grid(n, d, 123);
        let gt = brute_force_knn(&data, n, d, k);
        let (nb, dist) = nn_descent_impl(&data, n, d, k, 60, 42, false);
        let r = recall(&[(nb.clone(), dist.clone())], &gt, k);
        assert!(
            r >= 0.95,
            "recall {r} < 0.95 for n={n} medium test"
        );
    }

    #[test]
    fn output_shape() {
        let n = 50;
        let d = 4;
        let k = 5;
        let data = make_grid(n, d, 42);
        let (nb, dist) = nn_descent_impl(&data, n, d, k, 60, 42, false);
        assert_eq!(nb.len(), n * (k + 1));
        assert_eq!(dist.len(), n * (k + 1));
    }

    #[test]
    fn self_neighbors() {
        let n = 50;
        let d = 4;
        let k = 5;
        let data = make_grid(n, d, 42);
        let (nb, dist) = nn_descent_impl(&data, n, d, k, 60, 42, false);
        let cols = k + 1;
        for i in 0..n {
            assert_eq!(nb[i * cols], i as i64, "row {i} column 0 should be self");
            assert_eq!(dist[i * cols], 0.0, "row {i} self-distance should be 0");
        }
    }

    #[test]
    fn determinism() {
        let n = 100;
        let d = 8;
        let k = 10;
        let data = make_grid(n, d, 42);
        let (nb1, dist1) = nn_descent_impl(&data, n, d, k, 60, 42, false);
        let (nb2, dist2) = nn_descent_impl(&data, n, d, k, 60, 42, false);
        assert_eq!(nb1, nb2);
        assert_eq!(dist1, dist2);
    }

    #[test]
    fn distances_sorted_ascending() {
        let n = 100;
        let d = 8;
        let k = 10;
        let data = make_grid(n, d, 42);
        let (_, dist) = nn_descent_impl(&data, n, d, k, 60, 42, false);
        let cols = k + 1;
        for i in 0..n {
            for j in 1..cols {
                assert!(
                    dist[i * cols + j - 1] <= dist[i * cols + j],
                    "row {i}: distances not sorted at columns {}-{}",
                    j - 1,
                    j
                );
            }
        }
    }
}
