use rayon::prelude::*;

use crate::distance::squared_euclidean;
use crate::heap::NeighborHeap;
use crate::rng::NnRng;

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

    // Phase 1: Random partition tree initialization
    rp_tree_init(data, n, dim, k, &mut heap, &mut rng);

    // Phase 2: NN-Descent iterations
    for iter in 0..max_iters {
        let (new_candidates, old_candidates) =
            build_candidates(n, max_candidates, &heap, &mut rng);

        heap.mark_all_old();

        let updates = local_join(data, dim, n, &new_candidates, &old_candidates, &mut heap);

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
/// Builds multiple RP trees and considers all points in the same leaf as candidate neighbors.
fn rp_tree_init(
    data: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    heap: &mut NeighborHeap,
    rng: &mut NnRng,
) {
    let leaf_size = std::cmp::max(k * 2, 30);
    // Build enough trees to get good coverage
    // Scale trees with both n and dim for better coverage in high-dimensional spaces
    let base_trees = (n as f64).log2() as usize;
    let dim_factor = std::cmp::max(1, (dim + 4) / 5);
    let n_trees = std::cmp::max(8, base_trees * dim_factor);

    for _ in 0..n_trees {
        let mut indices: Vec<usize> = (0..n).collect();
        rp_tree_split(data, dim, leaf_size, &mut indices, heap, rng);
    }

    // Ensure all points have at least k neighbors with a random fallback
    for i in 0..n {
        let (existing, _) = heap.get_sorted_neighbors(i);
        let filled = existing.len();
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
/// At each leaf, insert all pairwise distances into the heap.
fn rp_tree_split(
    data: &[f32],
    dim: usize,
    leaf_size: usize,
    indices: &mut [usize],
    heap: &mut NeighborHeap,
    rng: &mut NnRng,
) {
    let n = indices.len();
    if n <= leaf_size {
        // Leaf node: insert all pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let a = indices[i];
                let b = indices[j];
                let dist = squared_euclidean(
                    &data[a * dim..(a + 1) * dim],
                    &data[b * dim..(b + 1) * dim],
                );
                heap.push(a, b as i32, dist);
                heap.push(b, a as i32, dist);
            }
        }
        return;
    }

    // Pick two random points and split by which is closer
    let a_idx = rng.rand_int(n);
    let mut b_idx = rng.rand_int(n);
    while b_idx == a_idx {
        b_idx = rng.rand_int(n);
    }
    let a = indices[a_idx];
    let b = indices[b_idx];

    // Partition: compute dot product with (b - a) for each point
    // Points closer to a go left, points closer to b go right
    let mut left = 0;
    let mut right = n - 1;
    while left < right {
        let p = indices[left];
        // Project onto (b - a): if sum((p - midpoint) * (b - a)) > 0, point is closer to b
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

    // Ensure neither side is empty (force at least one element on each side)
    if left == 0 {
        left = 1;
    }
    if left == n {
        left = n - 1;
    }

    let (left_slice, right_slice) = indices.split_at_mut(left);
    rp_tree_split(data, dim, leaf_size, left_slice, heap, rng);
    rp_tree_split(data, dim, leaf_size, right_slice, heap, rng);
}

/// Build new and old candidate lists for each point.
fn build_candidates(
    n: usize,
    max_candidates: usize,
    heap: &NeighborHeap,
    rng: &mut NnRng,
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let mut new_candidates: Vec<Vec<i32>> = vec![Vec::new(); n];
    let mut old_candidates: Vec<Vec<i32>> = vec![Vec::new(); n];

    for i in 0..n {
        for &nb in &heap.get_new(i) {
            new_candidates[i].push(nb);
            new_candidates[nb as usize].push(i as i32);
        }
        for &nb in &heap.get_old(i) {
            old_candidates[i].push(nb);
            old_candidates[nb as usize].push(i as i32);
        }
    }

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

    (new_candidates, old_candidates)
}

/// Local join: collect candidate pairs, compute distances in parallel,
/// then apply updates to the heap sequentially.
fn local_join(
    data: &[f32],
    dim: usize,
    n: usize,
    new_candidates: &[Vec<i32>],
    old_candidates: &[Vec<i32>],
    heap: &mut NeighborHeap,
) -> usize {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        let new_i = &new_candidates[i];
        let old_i = &old_candidates[i];

        for ni in 0..new_i.len() {
            let v1 = new_i[ni] as usize;
            for nj in (ni + 1)..new_i.len() {
                let v2 = new_i[nj] as usize;
                if v1 != v2 {
                    pairs.push((v1, v2));
                }
            }
        }

        for &v1 in new_i.iter() {
            let v1 = v1 as usize;
            for &v2 in old_i.iter() {
                let v2 = v2 as usize;
                if v1 != v2 {
                    pairs.push((v1, v2));
                }
            }
        }
    }

    // Compute distances in parallel
    let distances: Vec<(usize, usize, f32)> = pairs
        .par_iter()
        .map(|&(v1, v2)| {
            let dist = squared_euclidean(
                &data[v1 * dim..(v1 + 1) * dim],
                &data[v2 * dim..(v2 + 1) * dim],
            );
            (v1, v2, dist)
        })
        .collect();

    let mut update_count = 0;
    for (v1, v2, dist) in distances {
        if heap.push(v1, v2 as i32, dist) {
            update_count += 1;
        }
        if heap.push(v2, v1 as i32, dist) {
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
