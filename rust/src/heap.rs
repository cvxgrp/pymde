/// Max-heap based neighbor storage with concurrent access support.
///
/// Flat contiguous arrays: distances[n*k], indices[n*k], is_new[n*k].
/// Each point's block is a max-heap (root = farthest neighbor).
/// Per-point AtomicBool locks for concurrent access.

use std::sync::atomic::{AtomicBool, Ordering};

pub struct NeighborHeap {
    pub n: usize,
    pub k: usize,
    pub distances: Vec<f32>,
    pub indices: Vec<i32>,
    pub is_new: Vec<bool>,
    locks: Vec<AtomicBool>,
}

impl NeighborHeap {
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            n,
            k,
            distances: vec![f32::INFINITY; n * k],
            indices: vec![-1; n * k],
            is_new: vec![false; n * k],
            locks: (0..n).map(|_| AtomicBool::new(false)).collect(),
        }
    }

    /// Largest distance (heap root) for point `row`.
    #[inline]
    pub fn largest_distance(&self, row: usize) -> f32 {
        self.distances[row * self.k]
    }

    /// Push with exclusive (mutable) access. For sequential initialization.
    pub fn push(&mut self, row: usize, dist: f32, idx: i32) -> bool {
        let k = self.k;
        let base = row * k;

        // Reject if worse than root
        if dist >= self.distances[base] {
            return false;
        }
        // Reject self-loops
        if idx == row as i32 {
            return false;
        }
        // Reject duplicates
        for j in 0..k {
            if self.indices[base + j] == idx {
                return false;
            }
        }
        // Replace root and sift down
        self.distances[base] = dist;
        self.indices[base] = idx;
        self.is_new[base] = true;
        self.sift_down(base, k);
        true
    }

    /// Push without any lock. For RP-tree leaf processing where each point
    /// appears in exactly one leaf per tree (no concurrent writes to same row).
    ///
    /// Safety: caller must ensure no concurrent writes to the same `row`.
    pub fn push_unlocked(&self, row: usize, dist: f32, idx: i32) -> bool {
        let k = self.k;
        let base = row * k;

        // Fast-path rejection
        if dist >= self.distances[base] {
            return false;
        }
        if idx == row as i32 {
            return false;
        }
        // Duplicate check
        for j in 0..k {
            if self.indices[base + j] == idx {
                return false;
            }
        }

        // SAFETY: Caller guarantees no concurrent writes to this row.
        // We use raw pointers to mutate through shared reference.
        unsafe {
            let dist_ptr = self.distances.as_ptr().add(base) as *mut f32;
            let idx_ptr = self.indices.as_ptr().add(base) as *mut i32;
            let new_ptr = self.is_new.as_ptr().add(base) as *mut bool;

            *dist_ptr = dist;
            *idx_ptr = idx;
            *new_ptr = true;
            Self::sift_down_raw(dist_ptr, idx_ptr, new_ptr, k);
        }
        true
    }

    /// Push with TTAS spinlock per point. For local_join concurrent access.
    pub fn push_concurrent(&self, row: usize, dist: f32, idx: i32) -> bool {
        let k = self.k;
        let base = row * k;

        // Fast-path rejection before acquiring lock (~90% rejected here)
        if dist >= self.distances[base] {
            return false;
        }

        let lock = &self.locks[row];

        // TTAS (Test-and-Test-and-Set) spinlock
        loop {
            // Test phase: spin on read (cache-friendly)
            while lock.load(Ordering::Relaxed) {
                std::hint::spin_loop();
            }
            // Test-and-set phase
            if lock
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Re-check after acquiring lock (dist may have changed)
        let result = if dist >= self.distances[base] {
            false
        } else if idx == row as i32 {
            false
        } else {
            // Duplicate check
            let mut dup = false;
            for j in 0..k {
                if self.indices[base + j] == idx {
                    dup = true;
                    break;
                }
            }
            if dup {
                false
            } else {
                unsafe {
                    let dist_ptr = self.distances.as_ptr().add(base) as *mut f32;
                    let idx_ptr = self.indices.as_ptr().add(base) as *mut i32;
                    let new_ptr = self.is_new.as_ptr().add(base) as *mut bool;

                    *dist_ptr = dist;
                    *idx_ptr = idx;
                    *new_ptr = true;
                    Self::sift_down_raw(dist_ptr, idx_ptr, new_ptr, k);
                }
                true
            }
        };

        lock.store(false, Ordering::Release);
        result
    }

    /// Sort each row ascending by distance for final output.
    pub fn sort_by_distance(&mut self) {
        let k = self.k;
        for row in 0..self.n {
            let base = row * k;
            // Collect into sortable tuples
            let mut entries: Vec<(f32, i32)> = (0..k)
                .map(|j| (self.distances[base + j], self.indices[base + j]))
                .collect();
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for j in 0..k {
                self.distances[base + j] = entries[j].0;
                self.indices[base + j] = entries[j].1;
            }
        }
    }

    /// Sift-down on mutable slice (for `push`).
    fn sift_down(&mut self, base: usize, k: usize) {
        let mut pos = 0;
        loop {
            let left = 2 * pos + 1;
            if left >= k {
                break;
            }
            let right = left + 1;
            let mut largest = pos;
            if self.distances[base + left] > self.distances[base + largest] {
                largest = left;
            }
            if right < k && self.distances[base + right] > self.distances[base + largest] {
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

    /// Sift-down via raw pointers (for lock-free / concurrent push).
    unsafe fn sift_down_raw(
        distances: *mut f32,
        indices: *mut i32,
        is_new: *mut bool,
        k: usize,
    ) {
        unsafe {
            let mut pos = 0;
            loop {
                let left = 2 * pos + 1;
                if left >= k {
                    break;
                }
                let right = left + 1;
                let mut largest = pos;
                if *distances.add(left) > *distances.add(largest) {
                    largest = left;
                }
                if right < k && *distances.add(right) > *distances.add(largest) {
                    largest = right;
                }
                if largest == pos {
                    break;
                }
                // Swap pos and largest
                let tmp_d = *distances.add(pos);
                *distances.add(pos) = *distances.add(largest);
                *distances.add(largest) = tmp_d;

                let tmp_i = *indices.add(pos);
                *indices.add(pos) = *indices.add(largest);
                *indices.add(largest) = tmp_i;

                let tmp_n = *is_new.add(pos);
                *is_new.add(pos) = *is_new.add(largest);
                *is_new.add(largest) = tmp_n;

                pos = largest;
            }
        }
    }
}

// Send + Sync for use with rayon. The AtomicBool locks ensure thread safety
// for push_concurrent, and push_unlocked requires caller guarantees.
unsafe impl Send for NeighborHeap {}
unsafe impl Sync for NeighborHeap {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_overflow() {
        let mut heap = NeighborHeap::new(1, 3);
        // Fill all 3 slots
        assert!(heap.push(0, 5.0, 10));
        assert!(heap.push(0, 3.0, 11));
        assert!(heap.push(0, 4.0, 12));
        // Reject worse
        assert!(!heap.push(0, 6.0, 13));
        // Accept better (replaces worst)
        assert!(heap.push(0, 2.0, 14));
        // Now heap should contain 2.0, 3.0, 4.0
        heap.sort_by_distance();
        assert_eq!(heap.indices[0..3], [14, 11, 12]);
        assert!((heap.distances[0] - 2.0).abs() < 1e-6);
        assert!((heap.distances[1] - 3.0).abs() < 1e-6);
        assert!((heap.distances[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn reject_duplicates() {
        let mut heap = NeighborHeap::new(1, 3);
        assert!(heap.push(0, 5.0, 10));
        assert!(!heap.push(0, 3.0, 10)); // duplicate index
    }

    #[test]
    fn reject_self_loop() {
        let mut heap = NeighborHeap::new(2, 3);
        assert!(!heap.push(0, 1.0, 0)); // self-loop for row 0
        assert!(heap.push(0, 1.0, 1)); // different point is fine
    }

    #[test]
    fn is_new_flag() {
        let mut heap = NeighborHeap::new(1, 3);
        heap.push(0, 5.0, 10);
        // Find where index 10 ended up and check is_new
        let pos = heap.indices[0..3]
            .iter()
            .position(|&x| x == 10)
            .unwrap();
        assert!(heap.is_new[pos]);
    }

    #[test]
    fn sorted_output() {
        let mut heap = NeighborHeap::new(1, 5);
        heap.push(0, 10.0, 1);
        heap.push(0, 3.0, 2);
        heap.push(0, 7.0, 3);
        heap.push(0, 1.0, 4);
        heap.push(0, 5.0, 5);
        heap.sort_by_distance();
        for j in 1..5 {
            assert!(heap.distances[j - 1] <= heap.distances[j]);
        }
    }

    #[test]
    fn push_unlocked_basic() {
        let heap = NeighborHeap::new(1, 3);
        assert!(heap.push_unlocked(0, 5.0, 10));
        assert!(heap.push_unlocked(0, 3.0, 11));
        // Duplicate rejected
        assert!(!heap.push_unlocked(0, 2.0, 10));
    }

    #[test]
    fn concurrent_push_multithreaded() {
        use std::sync::Arc;
        use std::thread;

        let heap = Arc::new(NeighborHeap::new(1, 10));
        let mut handles = vec![];

        for t in 0..4 {
            let heap = Arc::clone(&heap);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let idx = (t * 100 + i + 1) as i32; // avoid 0 (self-loop)
                    let dist = idx as f32;
                    heap.push_concurrent(0, dist, idx);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Should have 10 smallest distances (1..=10)
        // The heap root should be the largest of the 10 smallest
        assert!(heap.largest_distance(0) <= 10.0);

        // Verify no duplicates
        let indices: Vec<i32> = heap.indices[0..10].to_vec();
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
    }

    #[test]
    fn multiple_rows() {
        let mut heap = NeighborHeap::new(3, 2);
        heap.push(0, 1.0, 1);
        heap.push(0, 2.0, 2);
        heap.push(1, 3.0, 0);
        heap.push(1, 4.0, 2);
        heap.push(2, 5.0, 0);
        heap.push(2, 6.0, 1);

        heap.sort_by_distance();
        assert_eq!(heap.indices[0..2], [1, 2]);
        assert_eq!(heap.indices[2..4], [0, 2]);
        assert_eq!(heap.indices[4..6], [0, 1]);
    }
}
