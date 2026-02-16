use crate::distance::squared_euclidean;
use crate::heap::NeighborHeap;
use crate::rng::NnRng;
use rayon::prelude::*;

const DELTA: f64 = 0.001;

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
    rp_tree_init(data, n, dim, k, &mut heap, &mut rng);

    // Phase 2: NN-Descent iterations
    // Pre-allocate candidate lists (reused across iterations)
    let mut new_candidates: Vec<Vec<i32>> = (0..n).map(|_| Vec::with_capacity(max_candidates)).collect();
    let mut old_candidates: Vec<Vec<i32>> = (0..n).map(|_| Vec::with_capacity(max_candidates)).collect();

    for iter in 0..max_iters {
        build_candidates(
            n,
            max_candidates,
            &heap,
            &mut rng,
            &mut new_candidates,
            &mut old_candidates,
        );

        heap.mark_all_old();

        let updates = local_join(data, dim, n, &new_candidates, &old_candidates, &mut heap, verbose);

        if verbose {
            eprintln!("NNDescent: iteration {}, updates = {}", iter, updates);
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
) {
    // Match pynndescent: leaf_size = clamp(60, 256, 5 * n_neighbors)
    let n_neighbors = k + 1;
    let leaf_size = std::cmp::max(60, std::cmp::min(256, 5 * n_neighbors));
    // Match pynndescent: n_trees = clamp(3, 12, round(2 * log10(n)))
    let n_trees = std::cmp::max(3, std::cmp::min(12, (2.0 * (n as f64).log10()).round() as usize));

    for _ in 0..n_trees {
        // Build tree structure, collect leaf index arrays (sequential — uses RNG)
        let mut indices: Vec<usize> = (0..n).collect();
        let mut leaves: Vec<Vec<usize>> = Vec::new();
        rp_tree_collect_leaves(data, dim, leaf_size, &mut indices, &mut leaves, rng);

        // Compute pairwise distances within leaves in parallel, pre-filtering
        // against the current heap's farthest distances.
        let heap_ref: &NeighborHeap = &*heap;
        let updates: Vec<(usize, i32, f32)> = leaves
            .par_iter()
            .fold(
                Vec::new,
                |mut acc, leaf| {
                    for i in 0..leaf.len() {
                        let a = leaf[i];
                        for j in (i + 1)..leaf.len() {
                            let b = leaf[j];
                            let dist = squared_euclidean(
                                &data[a * dim..(a + 1) * dim],
                                &data[b * dim..(b + 1) * dim],
                            );
                            if dist < heap_ref.largest_distance(a) {
                                acc.push((a, b as i32, dist));
                            }
                            if dist < heap_ref.largest_distance(b) {
                                acc.push((b, a as i32, dist));
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(Vec::new, |mut a, b| {
                a.extend(b);
                a
            });

        // Apply this tree's updates to the heap
        for &(point, neighbor, dist) in &updates {
            heap.push(point, neighbor, dist);
        }
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
/// Collects leaf index arrays into `leaves` instead of computing distances inline.
fn rp_tree_collect_leaves(
    data: &[f32],
    dim: usize,
    leaf_size: usize,
    indices: &mut [usize],
    leaves: &mut Vec<Vec<usize>>,
    rng: &mut NnRng,
) {
    let n = indices.len();
    if n <= leaf_size {
        leaves.push(indices.to_vec());
        return;
    }

    let a_idx = rng.rand_int(n);
    let mut b_idx = rng.rand_int(n);
    while b_idx == a_idx {
        b_idx = rng.rand_int(n);
    }
    let a = indices[a_idx];
    let b = indices[b_idx];

    let mut left = 0;
    let mut right = n - 1;
    while left < right {
        let p = indices[left];
        let mut proj = 0.0f32;
        for d in 0..dim {
            let mid = (data[a * dim + d] + data[b * dim + d]) * 0.5;
            proj += (data[p * dim + d] - mid) * (data[b * dim + d] - data[a * dim + d]);
        }
        if proj <= 0.0 {
            left += 1;
        } else {
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

    let (left_slice, right_slice) = indices.split_at_mut(left);
    rp_tree_collect_leaves(data, dim, leaf_size, left_slice, leaves, rng);
    rp_tree_collect_leaves(data, dim, leaf_size, right_slice, leaves, rng);
}

/// Build new and old candidate lists for each point by iterating the heap
/// flat arrays directly (no per-point Vec allocations from get_new/get_old).
fn build_candidates(
    n: usize,
    max_candidates: usize,
    heap: &NeighborHeap,
    rng: &mut NnRng,
    new_candidates: &mut [Vec<i32>],
    old_candidates: &mut [Vec<i32>],
) {
    let k = heap.k;

    // Clear previous iteration's candidates
    for i in 0..n {
        new_candidates[i].clear();
        old_candidates[i].clear();
    }

    // Collect forward and reverse neighbors directly from heap arrays
    for i in 0..n {
        let base = i * k;
        for s in 0..k {
            let nb = heap.indices[base + s];
            if nb < 0 {
                continue;
            }
            if heap.is_new[base + s] {
                new_candidates[i].push(nb);
                new_candidates[nb as usize].push(i as i32);
            } else {
                old_candidates[i].push(nb);
                old_candidates[nb as usize].push(i as i32);
            }
        }
    }

    // Subsample to max_candidates via partial Fisher-Yates shuffle
    for i in 0..n {
        let nc = &mut new_candidates[i];
        let len = nc.len();
        if len > max_candidates {
            for j in 0..max_candidates {
                let swap_idx = j + rng.rand_int(len - j);
                nc.swap(j, swap_idx);
            }
            nc.truncate(max_candidates);
        }

        let oc = &mut old_candidates[i];
        let olen = oc.len();
        if olen > max_candidates {
            for j in 0..max_candidates {
                let swap_idx = j + rng.rand_int(olen - j);
                oc.swap(j, swap_idx);
            }
            oc.truncate(max_candidates);
        }
    }
}

/// Parallel local join: compute distances in parallel across points, then
/// apply updates sequentially. The heap is read-only during the parallel phase
/// (for `largest_distance` early-reject checks) so the filter is conservative
/// (stale thresholds may admit a few extra candidates, but never miss valid ones).
/// The sequential apply phase ensures the final heap state is identical to any
/// serial insertion order since the max-heap keeps the k closest neighbors.
#[inline(never)]
/// Parallel local join: compute distances across all points in parallel,
/// then apply updates sequentially. The heap is read-only during the parallel
/// phase (stale `largest_distance` checks are conservative — they admit a few
/// extra candidates but never miss valid ones).
#[inline(never)]
fn local_join(
    data: &[f32],
    dim: usize,
    n: usize,
    new_candidates: &[Vec<i32>],
    old_candidates: &[Vec<i32>],
    heap: &mut NeighborHeap,
    _verbose: bool,
) -> usize {
    let heap_ref: &NeighborHeap = &*heap;

    let updates: Vec<(usize, i32, f32)> = (0..n)
        .into_par_iter()
        .fold(
            Vec::new,
            |mut acc, i| {
                let new_i = &new_candidates[i];
                let old_i = &old_candidates[i];

                // new-new pairs
                for ni in 0..new_i.len() {
                    let v1 = new_i[ni] as usize;
                    for nj in (ni + 1)..new_i.len() {
                        let v2 = new_i[nj] as usize;
                        if v1 == v2 {
                            continue;
                        }
                        let dist = squared_euclidean(
                            &data[v1 * dim..(v1 + 1) * dim],
                            &data[v2 * dim..(v2 + 1) * dim],
                        );
                        if dist < heap_ref.largest_distance(v1) {
                            acc.push((v1, v2 as i32, dist));
                        }
                        if dist < heap_ref.largest_distance(v2) {
                            acc.push((v2, v1 as i32, dist));
                        }
                    }
                }

                // new-old pairs
                for &v1_raw in new_i.iter() {
                    let v1 = v1_raw as usize;
                    for &v2_raw in old_i.iter() {
                        let v2 = v2_raw as usize;
                        if v1 == v2 {
                            continue;
                        }
                        let dist = squared_euclidean(
                            &data[v1 * dim..(v1 + 1) * dim],
                            &data[v2 * dim..(v2 + 1) * dim],
                        );
                        if dist < heap_ref.largest_distance(v1) {
                            acc.push((v1, v2 as i32, dist));
                        }
                        if dist < heap_ref.largest_distance(v2) {
                            acc.push((v2, v1 as i32, dist));
                        }
                    }
                }

                acc
            },
        )
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        });

    // Sequential phase: apply all collected updates to the heap.
    let mut update_count = 0usize;
    for &(point, neighbor, dist) in &updates {
        if heap.push(point, neighbor, dist) {
            update_count += 1;
        }
    }

    update_count
}

/// Sort neighbors, convert squared distances to Euclidean, prepend self.
fn finalize(
    n: usize,
    n_neighbors: usize,
    heap: &NeighborHeap,
) -> (Vec<i32>, Vec<f32>) {
    let total = n * n_neighbors;
    let mut out_indices = vec![0i32; total];
    let mut out_distances = vec![0.0f32; total];

    for i in 0..n {
        let (nb_indices, nb_dists) = heap.get_sorted_neighbors(i);
        let base = i * n_neighbors;

        out_indices[base] = i as i32;
        out_distances[base] = 0.0;

        for (s, (&idx, &sq_dist)) in nb_indices.iter().zip(nb_dists.iter()).enumerate() {
            if s + 1 >= n_neighbors {
                break;
            }
            out_indices[base + s + 1] = idx;
            out_distances[base + s + 1] = sq_dist.sqrt();
        }
    }

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
    fn test_determinism() {
        let n = 100;
        let dim = 10;
        let n_neighbors = 6;
        let mut rng = NnRng::new(77);
        let data: Vec<f32> = (0..n * dim).map(|_| (rng.rand_int(1000) as f32) / 100.0).collect();

        let (idx1, dist1) = nn_descent(&data, n, dim, n_neighbors, 60, 42, false);
        let (idx2, dist2) = nn_descent(&data, n, dim, n_neighbors, 60, 42, false);
        assert_eq!(idx1, idx2);
        assert_eq!(dist1, dist2);
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
