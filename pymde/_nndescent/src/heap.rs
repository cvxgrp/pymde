use ordered_float::OrderedFloat;

/// A flat max-heap of neighbors for all points.
///
/// Layout: `n` points each with `k` neighbor slots.
/// For point `i`, neighbors are at indices `[i*k .. (i+1)*k]`.
/// Max-heap keyed on distance (farthest neighbor at root = index 0 within each point's block).
pub struct NeighborHeap {
    pub k: usize,
    /// Distances, shape [n * k]. f32::INFINITY for empty slots.
    pub distances: Vec<f32>,
    /// Neighbor indices, shape [n * k]. -1 for empty slots.
    pub indices: Vec<i32>,
    /// Whether this neighbor is "new" (inserted since last iteration).
    pub is_new: Vec<bool>,
}

impl NeighborHeap {
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            k,
            distances: vec![f32::INFINITY; n * k],
            indices: vec![-1; n * k],
            is_new: vec![false; n * k],
        }
    }

    /// Offset for point `p`, slot `s`.
    #[inline]
    fn idx(&self, point: usize, slot: usize) -> usize {
        point * self.k + slot
    }

    /// Distance of the farthest (root) neighbor for a point.
    #[inline]
    pub fn largest_distance(&self, point: usize) -> f32 {
        self.distances[self.idx(point, 0)]
    }

    /// Try to insert `neighbor` with `dist` into `point`'s neighbor list.
    /// Returns `true` if inserted (i.e., it was closer than the farthest current neighbor).
    /// Rejects duplicates and self-loops.
    pub fn push(&mut self, point: usize, neighbor: i32, dist: f32) -> bool {
        if dist >= self.largest_distance(point) {
            return false;
        }
        // Reject self-loops
        if neighbor == point as i32 {
            return false;
        }
        // Reject duplicates
        let base = point * self.k;
        for s in 0..self.k {
            if self.indices[base + s] == neighbor {
                return false;
            }
        }
        // Replace root (farthest) with new neighbor
        let root = self.idx(point, 0);
        self.distances[root] = dist;
        self.indices[root] = neighbor;
        self.is_new[root] = true;
        // Sift down to restore max-heap property
        self.sift_down(point, 0);
        true
    }

    /// Standard max-heap sift-down within point's block.
    fn sift_down(&mut self, point: usize, mut pos: usize) {
        let base = point * self.k;
        loop {
            let left = 2 * pos + 1;
            let right = 2 * pos + 2;
            let mut largest = pos;
            if left < self.k
                && OrderedFloat(self.distances[base + left])
                    > OrderedFloat(self.distances[base + largest])
            {
                largest = left;
            }
            if right < self.k
                && OrderedFloat(self.distances[base + right])
                    > OrderedFloat(self.distances[base + largest])
            {
                largest = right;
            }
            if largest == pos {
                break;
            }
            self.distances.swap(base + pos, base + largest);
            self.indices.swap(base + pos, base + largest);
            self.is_new.swap(base + pos, base + largest);
            pos = largest;
        }
    }

    /// Get indices of "new" neighbors for a point.
    pub fn get_new(&self, point: usize) -> Vec<i32> {
        let base = point * self.k;
        (0..self.k)
            .filter(|&s| self.is_new[base + s] && self.indices[base + s] >= 0)
            .map(|s| self.indices[base + s])
            .collect()
    }

    /// Get indices of "old" (not new) neighbors for a point.
    pub fn get_old(&self, point: usize) -> Vec<i32> {
        let base = point * self.k;
        (0..self.k)
            .filter(|&s| !self.is_new[base + s] && self.indices[base + s] >= 0)
            .map(|s| self.indices[base + s])
            .collect()
    }

    /// Mark all neighbors as old.
    pub fn mark_all_old(&mut self) {
        self.is_new.fill(false);
    }

    /// Get sorted (by distance, ascending) neighbor list for a point.
    /// Returns (indices, distances) excluding empty slots.
    pub fn get_sorted_neighbors(&self, point: usize) -> (Vec<i32>, Vec<f32>) {
        let base = point * self.k;
        let mut pairs: Vec<(f32, i32)> = (0..self.k)
            .filter(|&s| self.indices[base + s] >= 0)
            .map(|s| (self.distances[base + s], self.indices[base + s]))
            .collect();
        pairs.sort_by_key(|&(d, _)| OrderedFloat(d));
        let indices: Vec<i32> = pairs.iter().map(|&(_, idx)| idx).collect();
        let distances: Vec<f32> = pairs.iter().map(|&(d, _)| d).collect();
        (indices, distances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_overflow() {
        let mut heap = NeighborHeap::new(1, 3);
        // Insert 3 neighbors
        assert!(heap.push(0, 1, 1.0));
        assert!(heap.push(0, 2, 2.0));
        assert!(heap.push(0, 3, 3.0));
        // Now full; inserting farther should fail
        assert!(!heap.push(0, 4, 4.0));
        // Inserting closer should succeed (replaces farthest)
        assert!(heap.push(0, 4, 0.5));
        let (indices, dists) = heap.get_sorted_neighbors(0);
        assert_eq!(indices.len(), 3);
        assert!(*dists.last().unwrap() <= 3.0);
        assert!(indices.contains(&4));
    }

    #[test]
    fn test_reject_duplicates() {
        let mut heap = NeighborHeap::new(1, 3);
        assert!(heap.push(0, 1, 1.0));
        assert!(!heap.push(0, 1, 0.5)); // duplicate, rejected
        let (indices, _) = heap.get_sorted_neighbors(0);
        assert_eq!(indices.iter().filter(|&&x| x == 1).count(), 1);
    }

    #[test]
    fn test_reject_self() {
        let mut heap = NeighborHeap::new(2, 3);
        assert!(!heap.push(0, 0, 0.0)); // self-loop, rejected
        assert!(heap.push(0, 1, 1.0));
    }

    #[test]
    fn test_sorted_output() {
        let mut heap = NeighborHeap::new(1, 5);
        heap.push(0, 10, 5.0);
        heap.push(0, 20, 1.0);
        heap.push(0, 30, 3.0);
        heap.push(0, 40, 2.0);
        heap.push(0, 50, 4.0);
        let (indices, dists) = heap.get_sorted_neighbors(0);
        assert_eq!(indices, vec![20, 40, 30, 50, 10]);
        for w in dists.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_new_old_flags() {
        let mut heap = NeighborHeap::new(1, 3);
        heap.push(0, 1, 1.0);
        heap.push(0, 2, 2.0);
        assert_eq!(heap.get_new(0).len(), 2);
        assert_eq!(heap.get_old(0).len(), 0);
        heap.mark_all_old();
        assert_eq!(heap.get_new(0).len(), 0);
        assert_eq!(heap.get_old(0).len(), 2);
        // Inserting new one should be "new"
        heap.push(0, 3, 3.0);
        assert_eq!(heap.get_new(0).len(), 1);
        assert!(heap.get_new(0).contains(&3));
    }
}
