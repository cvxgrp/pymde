use std::sync::atomic::{AtomicUsize, Ordering};

use crate::distance::{dot_product, squared_euclidean, squared_euclidean_bounded};
use crate::heap::NeighborHeap;
use crate::rng::NnRng;
use rayon::prelude::*;

const DELTA: f64 = 0.001;

/// Flat contiguous candidate list: `n` points, each with up to `width` candidates.
/// Avoids Vec<Vec<i32>> pointer chasing for better cache locality in local_join.
struct FlatCandidates {
    data: Vec<i32>,
    counts: Vec<u32>,
    width: usize,
}

impl FlatCandidates {
    fn new(n: usize, width: usize) -> Self {
        Self {
            data: vec![-1; n * width],
            counts: vec![0; n],
            width,
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.counts.fill(0);
    }

    /// Push a candidate. Skips if full (first-arrival, matches pynndescent).
    #[inline]
    fn push(&mut self, point: usize, neighbor: i32) {
        let count = self.counts[point] as usize;
        if count < self.width {
            self.data[point * self.width + count] = neighbor;
            self.counts[point] = (count + 1) as u32;
        }
    }

    #[inline]
    fn get(&self, point: usize) -> &[i32] {
        let base = point * self.width;
        let count = self.counts[point] as usize;
        &self.data[base..base + count]
    }

    /// Subsample to `max` entries using partial Fisher-Yates shuffle.
    fn subsample(&mut self, point: usize, max: usize, rng: &mut NnRng) {
        let count = self.counts[point] as usize;
        if count <= max {
            return;
        }
        let base = point * self.width;
        for j in 0..max {
            let swap_idx = j + rng.rand_int(count - j);
            self.data.swap(base + j, base + swap_idx);
        }
        self.counts[point] = max as u32;
    }
}

/// Run NN-Descent and return (indices, distances) arrays each of shape [n, n_neighbors].
/// `n_neighbors` includes self as the first neighbor (index=self, distance=0.0).
pub fn nn_descent(
    data: &[f32],
    n: usize,
    dim: usize,
    n_neighbors: usize,
    max_candidates: usize,
    seed: u64,
    verbose: bool,
) -> (Vec<i32>, Vec<f32>) {
    let k = n_neighbors - 1;
    assert!(k > 0, "n_neighbors must be at least 2");
    assert!(k < n, "n_neighbors must be less than n + 1");

    let max_iters = std::cmp::max(10, ((n as f64).log2().round() as usize) + 5);
    let mut rng = NnRng::new(seed);
    let mut heap = NeighborHeap::new(n, k);

    // Phase 1: RP-tree initialization (small fixed number of trees)
    let t0 = std::time::Instant::now();
    rp_tree_init(data, n, dim, k, &mut heap, &mut rng, verbose);
    if verbose {
        eprintln!("NNDescent: rp_tree_init {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }

    // Phase 2: NN-Descent iterations
    // Flat candidate arrays. Width = max_candidates: excess reverse neighbors
    // are dropped (first-arrival, matches pynndescent's checked_heap_push).
    let cand_width = max_candidates;
    let mut new_candidates = FlatCandidates::new(n, cand_width);
    let mut old_candidates = FlatCandidates::new(n, cand_width);

    for iter in 0..max_iters {
        let t1 = std::time::Instant::now();
        build_candidates(
            n,
            max_candidates,
            &heap,
            &mut rng,
            &mut new_candidates,
            &mut old_candidates,
        );
        let bc_ms = t1.elapsed().as_secs_f64() * 1000.0;

        heap.mark_all_old();

        let t2 = std::time::Instant::now();
        let updates = local_join(data, dim, n, &new_candidates, &old_candidates, &mut heap);
        let lj_ms = t2.elapsed().as_secs_f64() * 1000.0;

        if verbose {
            eprintln!("NNDescent: iter {}, updates={}, build_cand={:.1}ms, local_join={:.1}ms", iter, updates, bc_ms, lj_ms);
        }

        if (updates as f64) < DELTA * (n as f64) * (k as f64) {
            if verbose {
                eprintln!("NNDescent: converged after {} iterations", iter + 1);
            }
            break;
        }
    }

    finalize(n, n_neighbors, &heap)
}

/// Initialize neighbor graph using random partition trees.
/// For each tree: build tree (sequential, uses RNG), compute leaf pairwise
/// distances in parallel (rayon), apply updates to heap. Processing one tree
/// at a time allows the heap to tighten after each tree, so subsequent trees
/// benefit from early rejection (most pairs are filtered out).
fn rp_tree_init(
    data: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    heap: &mut NeighborHeap,
    rng: &mut NnRng,
    verbose: bool,
) {
    // Match pynndescent: leaf_size = clamp(60, 256, 5 * n_neighbors)
    let n_neighbors = k + 1;
    let leaf_size = std::cmp::max(60, std::cmp::min(256, 5 * n_neighbors));
    // n_trees: balance between init cost and iteration savings.
    // More trees = better initialization but higher init cost.
    let n_trees = std::cmp::max(3, std::cmp::min(12, (2.0 * (n as f64).log10()).round() as usize));

    // Phase 1: Build all RP trees in parallel (each tree gets its own RNG,
    // indices array, and reusable buffers). Tree construction is the
    // bottleneck — parallelizing across trees gives ~n_trees/n_cores speedup.
    let tb = std::time::Instant::now();
    let base_seed = rng.next_u64();
    let trees: Vec<(Vec<usize>, Vec<(usize, usize)>)> = (0..n_trees)
        .into_par_iter()
        .map(|t| {
            let mut tree_rng = NnRng::new(base_seed.wrapping_add(t as u64));
            let mut indices: Vec<usize> = (0..n).collect();
            let mut leaf_ranges: Vec<(usize, usize)> = Vec::new();
            let mut diff_buf = vec![0.0f32; dim];
            let mut proj_buf = vec![0.0f32; n];
            rp_tree_split(
                data, dim, leaf_size, &mut indices, 0,
                &mut leaf_ranges, &mut diff_buf, &mut proj_buf, &mut tree_rng,
            );
            (indices, leaf_ranges)
        })
        .collect();
    let tree_build_ms = tb.elapsed().as_secs_f64() * 1000.0;

    // Advance main RNG past all tree construction (for reproducibility
    // of subsequent phases regardless of thread scheduling).
    for _ in 0..n_trees {
        rng.next_u64();
    }

    // Phase 2: Process leaf pairwise distances tree by tree. Sequential
    // tree ordering preserves inter-tree tightening (later trees benefit
    // from earlier trees' results). Within each tree, leaves are processed
    // in parallel with lock-free push_unlocked.
    let ld = std::time::Instant::now();
    for (indices, leaf_ranges) in &trees {
        let heap_ref: &NeighborHeap = &*heap;
        leaf_ranges.par_iter().for_each(|&(start, end)| {
            for i in start..end {
                let a = indices[i];
                let a_data = &data[a * dim..(a + 1) * dim];
                for j in (i + 1)..end {
                    let b = indices[j];
                    // SAFETY: a and b are both in this leaf, and each point
                    // appears in exactly one leaf per tree.
                    unsafe {
                        let ld_a = *heap_ref.distances.as_ptr().add(a * heap_ref.k);
                        let ld_b = *heap_ref.distances.as_ptr().add(b * heap_ref.k);
                        let bound = ld_a.max(ld_b);
                        let dist = squared_euclidean_bounded(
                            a_data,
                            &data[b * dim..(b + 1) * dim],
                            bound,
                        );
                        if dist < ld_a {
                            heap_ref.push_unlocked(a, b as i32, dist);
                        }
                        if dist < ld_b {
                            heap_ref.push_unlocked(b, a as i32, dist);
                        }
                    }
                }
            }
        });
    }
    let leaf_dist_ms = ld.elapsed().as_secs_f64() * 1000.0;

    if verbose {
        eprintln!("NNDescent: rp_tree_init breakdown: tree_build={:.1}ms, leaf_dist={:.1}ms",
            tree_build_ms, leaf_dist_ms);
    }

    // Ensure all points have at least k neighbors with a random fallback
    for i in 0..n {
        let base = i * k;
        let mut filled = 0;
        for s in 0..k {
            if heap.indices[base + s] >= 0 {
                filled += 1;
            }
        }
        if filled < k {
            let needed = k - filled;
            let max_attempts = needed * 5 + 10;
            for _ in 0..max_attempts {
                let j = rng.rand_int(n);
                if j == i {
                    continue;
                }
                let dist = squared_euclidean(
                    &data[i * dim..(i + 1) * dim],
                    &data[j * dim..(j + 1) * dim],
                );
                heap.push(i, j as i32, dist);
            }
        }
    }
}

/// Recursively split indices using random projection until leaf_size is reached.
/// Stores leaf ranges (start, end) into `leaf_ranges` referencing positions in
/// the rearranged `indices` array. Uses pre-allocated `diff` and `proj` buffers
/// to avoid per-level allocations. Separates projection computation (batch
/// vectorizable) from partitioning (branch-heavy) for better autovectorization.
fn rp_tree_split(
    data: &[f32],
    dim: usize,
    leaf_size: usize,
    indices: &mut [usize],
    offset: usize,
    leaf_ranges: &mut Vec<(usize, usize)>,
    diff: &mut [f32],
    proj: &mut [f32],
    rng: &mut NnRng,
) {
    let n = indices.len();
    if n <= leaf_size {
        leaf_ranges.push((offset, offset + n));
        return;
    }

    let a_idx = rng.rand_int(n);
    let mut b_idx = rng.rand_int(n);
    while b_idx == a_idx {
        b_idx = rng.rand_int(n);
    }
    let a = indices[a_idx];
    let b = indices[b_idx];

    // Precompute hyperplane into reusable buffer
    let a_off = a * dim;
    let b_off = b * dim;
    let mut hp_offset = 0.0f32;
    for d in 0..dim {
        let ad = data[a_off + d];
        let bd = data[b_off + d];
        let dd = bd - ad;
        diff[d] = dd;
        hp_offset += (ad + bd) * 0.5 * dd;
    }

    // Batch-compute all projections using SIMD-friendly dot product.
    let diff_ref: &[f32] = &diff[..dim];
    for i in 0..n {
        let p = indices[i];
        let p_data = &data[p * dim..(p + 1) * dim];
        proj[i] = dot_product(p_data, diff_ref);
    }

    // Partition based on precomputed projections (no dim-loop, fast)
    let mut left = 0;
    let mut right = n - 1;
    while left < right {
        if proj[left] <= hp_offset {
            left += 1;
        } else {
            proj.swap(left, right);
            indices.swap(left, right);
            right -= 1;
        }
    }

    if left == 0 {
        left = 1;
    }
    if left == n {
        left = n - 1;
    }

    let (left_idx, right_idx) = indices.split_at_mut(left);
    let (left_proj, right_proj) = proj.split_at_mut(left);
    rp_tree_split(data, dim, leaf_size, left_idx, offset, leaf_ranges, diff, left_proj, rng);
    rp_tree_split(data, dim, leaf_size, right_idx, offset + left, leaf_ranges, diff, right_proj, rng);
}

/// Build new and old candidate lists for each point by iterating the heap
/// flat arrays directly. Uses FlatCandidates for contiguous memory layout.
fn build_candidates(
    n: usize,
    max_candidates: usize,
    heap: &NeighborHeap,
    rng: &mut NnRng,
    new_candidates: &mut FlatCandidates,
    old_candidates: &mut FlatCandidates,
) {
    let k = heap.k;
    new_candidates.clear();
    old_candidates.clear();

    // Collect forward and reverse neighbors directly from heap arrays
    for i in 0..n {
        let base = i * k;
        for s in 0..k {
            let nb = heap.indices[base + s];
            if nb < 0 {
                continue;
            }
            if heap.is_new[base + s] {
                new_candidates.push(i, nb);
                new_candidates.push(nb as usize, i as i32);
            } else {
                old_candidates.push(i, nb);
                old_candidates.push(nb as usize, i as i32);
            }
        }
    }

    // Subsample to max_candidates via partial Fisher-Yates shuffle
    for i in 0..n {
        new_candidates.subsample(i, max_candidates, rng);
        old_candidates.subsample(i, max_candidates, rng);
    }
}

/// Single-phase parallel local join: compute distances and apply heap updates
/// in-place using per-point try-locks. Uses bounded distance computation with
/// early exit to skip non-neighbor pairs quickly, and pre-filters push calls
/// using cached thresholds. Per-point local counters reduce atomic contention.
#[inline(never)]
fn local_join(
    data: &[f32],
    dim: usize,
    n: usize,
    new_candidates: &FlatCandidates,
    old_candidates: &FlatCandidates,
    heap: &mut NeighborHeap,
) -> usize {
    let update_count = AtomicUsize::new(0);
    let heap_ref: &NeighborHeap = &*heap;

    (0..n).into_par_iter().for_each(|i| {
        let new_i = new_candidates.get(i);
        let old_i = old_candidates.get(i);
        let mut local_updates = 0usize;

        // new-new pairs: cache v1 data and threshold across inner loop
        for ni in 0..new_i.len() {
            let v1 = new_i[ni] as usize;
            let v1_data = &data[v1 * dim..(v1 + 1) * dim];
            let mut ld1 = heap_ref.largest_distance(v1);
            for nj in (ni + 1)..new_i.len() {
                let v2 = new_i[nj] as usize;
                if v1 == v2 {
                    continue;
                }
                let ld2 = heap_ref.largest_distance(v2);
                let bound = ld1.max(ld2);
                let dist = squared_euclidean_bounded(
                    v1_data,
                    &data[v2 * dim..(v2 + 1) * dim],
                    bound,
                );
                if dist < ld1 {
                    if heap_ref.push_concurrent(v1, v2 as i32, dist) {
                        local_updates += 1;
                        ld1 = heap_ref.largest_distance(v1);
                    }
                }
                if dist < ld2 {
                    if heap_ref.push_concurrent(v2, v1 as i32, dist) {
                        local_updates += 1;
                    }
                }
            }
        }

        // new-old pairs: cache v1 data and threshold
        for &v1_raw in new_i.iter() {
            let v1 = v1_raw as usize;
            let v1_data = &data[v1 * dim..(v1 + 1) * dim];
            let mut ld1 = heap_ref.largest_distance(v1);
            for &v2_raw in old_i.iter() {
                let v2 = v2_raw as usize;
                if v1 == v2 {
                    continue;
                }
                let ld2 = heap_ref.largest_distance(v2);
                let bound = ld1.max(ld2);
                let dist = squared_euclidean_bounded(
                    v1_data,
                    &data[v2 * dim..(v2 + 1) * dim],
                    bound,
                );
                if dist < ld1 {
                    if heap_ref.push_concurrent(v1, v2 as i32, dist) {
                        local_updates += 1;
                        ld1 = heap_ref.largest_distance(v1);
                    }
                }
                if dist < ld2 {
                    if heap_ref.push_concurrent(v2, v1 as i32, dist) {
                        local_updates += 1;
                    }
                }
            }
        }

        if local_updates > 0 {
            update_count.fetch_add(local_updates, Ordering::Relaxed);
        }
    });

    update_count.load(Ordering::Relaxed)
}

/// Sort neighbors, convert squared distances to Euclidean, prepend self.
/// Parallelized with rayon — each point's output chunk is independent.
fn finalize(
    n: usize,
    n_neighbors: usize,
    heap: &NeighborHeap,
) -> (Vec<i32>, Vec<f32>) {
    let total = n * n_neighbors;
    let mut out_indices = vec![0i32; total];
    let mut out_distances = vec![0.0f32; total];
    let k = heap.k;

    // Split output into per-point chunks and process in parallel
    out_indices
        .par_chunks_mut(n_neighbors)
        .zip(out_distances.par_chunks_mut(n_neighbors))
        .enumerate()
        .for_each(|(i, (idx_chunk, dist_chunk))| {
            let heap_base = i * k;

            // Collect valid neighbors and sort by distance (stack array, no heap alloc)
            let mut pairs = [(f32::INFINITY, -1i32); 64];
            let mut count = 0;
            for s in 0..k {
                let idx = heap.indices[heap_base + s];
                if idx >= 0 && count < 64 {
                    pairs[count] = (heap.distances[heap_base + s], idx);
                    count += 1;
                }
            }
            pairs[..count].sort_by(|a, b| a.0.total_cmp(&b.0));

            // Self as first neighbor
            idx_chunk[0] = i as i32;
            dist_chunk[0] = 0.0;

            // Fill sorted neighbors
            for s in 0..count.min(n_neighbors - 1) {
                idx_chunk[s + 1] = pairs[s].1;
                dist_chunk[s + 1] = pairs[s].0.sqrt();
            }
        });

    (out_indices, out_distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force_knn(data: &[f32], n: usize, dim: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = vec![Vec::new(); n];
        for i in 0..n {
            let mut dists: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d = squared_euclidean(
                        &data[i * dim..(i + 1) * dim],
                        &data[j * dim..(j + 1) * dim],
                    );
                    (d, j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            result[i] = dists.iter().take(k).map(|&(_, j)| j).collect();
        }
        result
    }

    fn recall(
        approx_indices: &[i32],
        true_neighbors: &[Vec<usize>],
        n: usize,
        n_neighbors: usize,
    ) -> f64 {
        let k = n_neighbors - 1;
        let mut hits = 0usize;
        let total = n * k;
        for i in 0..n {
            let true_set: std::collections::HashSet<usize> =
                true_neighbors[i].iter().copied().collect();
            for s in 1..n_neighbors {
                let idx = approx_indices[i * n_neighbors + s] as usize;
                if true_set.contains(&idx) {
                    hits += 1;
                }
            }
        }
        hits as f64 / total as f64
    }

    #[test]
    fn test_small_exact() {
        let n = 20;
        let dim = 2;
        let k = 5;
        let n_neighbors = k + 1;
        let mut rng = NnRng::new(42);
        let data: Vec<f32> = (0..n * dim).map(|_| (rng.rand_int(1000) as f32) / 100.0).collect();

        let (indices, distances) = nn_descent(&data, n, dim, n_neighbors, 60, 42, false);

        assert_eq!(indices.len(), n * n_neighbors);
        assert_eq!(distances.len(), n * n_neighbors);

        for i in 0..n {
            assert_eq!(indices[i * n_neighbors], i as i32);
            assert_eq!(distances[i * n_neighbors], 0.0);
        }

        let true_nn = brute_force_knn(&data, n, dim, k);
        let r = recall(&indices, &true_nn, n, n_neighbors);
        assert!(r >= 0.99, "Recall {r} too low for small dataset");
    }

    #[test]
    fn test_medium_recall() {
        let n = 1000;
        let dim = 32;
        let k = 10;
        let n_neighbors = k + 1;
        let mut rng = NnRng::new(123);
        let data: Vec<f32> = (0..n * dim).map(|_| (rng.rand_int(10000) as f32) / 100.0).collect();

        let (indices, _distances) = nn_descent(&data, n, dim, n_neighbors, 60, 123, false);

        let true_nn = brute_force_knn(&data, n, dim, k);
        let r = recall(&indices, &true_nn, n, n_neighbors);
        assert!(r >= 0.95, "Recall {r} too low for medium dataset");
    }

    #[test]
    fn test_high_recall() {
        // With concurrent heap updates, results are not bit-exact deterministic
        // (thread scheduling affects tie-breaking). Verify both runs achieve
        // high recall instead.
        let n = 100;
        let dim = 10;
        let n_neighbors = 6;
        let k = n_neighbors - 1;
        let mut rng = NnRng::new(77);
        let data: Vec<f32> = (0..n * dim).map(|_| (rng.rand_int(1000) as f32) / 100.0).collect();

        let true_nn = brute_force_knn(&data, n, dim, k);
        let (idx1, _) = nn_descent(&data, n, dim, n_neighbors, 60, 42, false);
        let (idx2, _) = nn_descent(&data, n, dim, n_neighbors, 60, 42, false);
        let r1 = recall(&idx1, &true_nn, n, n_neighbors);
        let r2 = recall(&idx2, &true_nn, n, n_neighbors);
        assert!(r1 >= 0.95, "Run 1 recall {r1} too low");
        assert!(r2 >= 0.95, "Run 2 recall {r2} too low");
    }

    #[test]
    fn test_different_k() {
        let n = 50;
        let dim = 5;
        let mut rng = NnRng::new(99);
        let data: Vec<f32> = (0..n * dim).map(|_| (rng.rand_int(1000) as f32) / 100.0).collect();

        for k in [2, 5, 10, 20] {
            let n_neighbors = k + 1;
            let (indices, distances) = nn_descent(&data, n, dim, n_neighbors, 60, 42, false);
            assert_eq!(indices.len(), n * n_neighbors);
            assert_eq!(distances.len(), n * n_neighbors);
            for i in 0..n {
                for s in 1..n_neighbors - 1 {
                    assert!(
                        distances[i * n_neighbors + s] <= distances[i * n_neighbors + s + 1] + 1e-6,
                        "Distances not sorted for point {i}, k={k}"
                    );
                }
            }
        }
    }
}
