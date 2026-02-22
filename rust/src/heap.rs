/// Max-heap based neighbor storage with concurrent access support.
///
/// Flat contiguous arrays: distances[n*k], indices[n*k], is_new[n*k].
/// Each point's block is a max-heap (root = farthest neighbor).
/// Per-point AtomicBool locks for concurrent access.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

pub struct NeighborHeap {
    pub n: usize,
    pub k: usize,
    /// Mutable data behind UnsafeCell for sound interior mutation.
    data: UnsafeCell<HeapData>,
    /// Per-point root distance cached as AtomicU32 (bit-cast f32) for
    /// data-race-free fast-path rejection in push_concurrent.
    root_dists: Vec<AtomicU32>,
    locks: Vec<AtomicBool>,
}

struct HeapData {
    distances: Vec<f32>,
    indices: Vec<i32>,
    is_new: Vec<bool>,
}

// SAFETY: All concurrent mutation goes through either:
// - push_concurrent: protected by per-point TTAS spinlocks
// - push_unlocked: caller guarantees exclusive per-row access
// The AtomicU32 root_dists and AtomicBool locks handle cross-thread visibility.
unsafe impl Send for NeighborHeap {}
unsafe impl Sync for NeighborHeap {}

impl NeighborHeap {
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            n,
            k,
            data: UnsafeCell::new(HeapData {
                distances: vec![f32::INFINITY; n * k],
                indices: vec![-1; n * k],
                is_new: vec![false; n * k],
            }),
            root_dists: (0..n)
                .map(|_| AtomicU32::new(f32::INFINITY.to_bits()))
                .collect(),
            locks: (0..n).map(|_| AtomicBool::new(false)).collect(),
        }
    }

    // ── Accessors ─────────────────────────────────────────────────

    #[inline]
    pub fn distances(&self) -> &[f32] {
        // SAFETY: No mutable aliases exist; callers use this read-only
        // outside of concurrent push regions.
        unsafe { &(*self.data.get()).distances }
    }

    #[inline]
    pub fn indices(&self) -> &[i32] {
        unsafe { &(*self.data.get()).indices }
    }

    #[inline]
    pub fn is_new(&self) -> &[bool] {
        unsafe { &(*self.data.get()).is_new }
    }

    #[inline]
    pub fn is_new_mut(&mut self) -> &mut [bool] {
        &mut self.data.get_mut().is_new
    }

    /// Largest distance (heap root) for point `row`.
    /// Uses AtomicU32 for data-race-free access from concurrent threads.
    #[inline]
    pub fn largest_distance(&self, row: usize) -> f32 {
        f32::from_bits(self.root_dists[row].load(Ordering::Relaxed))
    }

    /// Push with exclusive (mutable) access. For sequential initialization.
    pub fn push(&mut self, row: usize, dist: f32, idx: i32) -> bool {
        let k = self.k;
        let base = row * k;
        let data = self.data.get_mut();

        // Reject if worse than root
        if dist >= data.distances[base] {
            return false;
        }
        // Reject self-loops
        if idx == row as i32 {
            return false;
        }
        // Reject duplicates
        for j in 0..k {
            if data.indices[base + j] == idx {
                return false;
            }
        }
        // Replace root and sift down
        data.distances[base] = dist;
        data.indices[base] = idx;
        data.is_new[base] = true;
        Self::sift_down_data(data, base, k);
        self.root_dists[row].store(data.distances[base].to_bits(), Ordering::Relaxed);
        true
    }

    /// Push without any lock. For RP-tree leaf processing where each point
    /// appears in exactly one leaf per tree (no concurrent writes to same row).
    ///
    /// SAFETY: caller must ensure no concurrent writes to the same `row`.
    pub fn push_unlocked(&self, row: usize, dist: f32, idx: i32) -> bool {
        let k = self.k;
        let base = row * k;

        // SAFETY: Caller guarantees exclusive access to this row.
        let data = unsafe { &mut *self.data.get() };

        // Fast-path rejection
        if dist >= data.distances[base] {
            return false;
        }
        if idx == row as i32 {
            return false;
        }
        // Duplicate check
        for j in 0..k {
            if data.indices[base + j] == idx {
                return false;
            }
        }

        data.distances[base] = dist;
        data.indices[base] = idx;
        data.is_new[base] = true;
        Self::sift_down_data(data, base, k);
        self.root_dists[row].store(data.distances[base].to_bits(), Ordering::Relaxed);
        true
    }

    /// Push with TTAS spinlock per point. For local_join concurrent access.
    pub fn push_concurrent(&self, row: usize, dist: f32, idx: i32) -> bool {
        let k = self.k;
        let base = row * k;

        // Fast-path rejection before acquiring lock (~90% rejected here).
        // Uses AtomicU32 root_dists to avoid data races on non-atomic f32.
        if dist >= self.largest_distance(row) {
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

        // SAFETY: We hold the per-row lock; no other thread can mutate this row.
        let data = unsafe { &mut *self.data.get() };

        // Re-check after acquiring lock (dist may have changed)
        let result = if dist >= data.distances[base] {
            false
        } else if idx == row as i32 {
            false
        } else {
            // Duplicate check
            let mut dup = false;
            for j in 0..k {
                if data.indices[base + j] == idx {
                    dup = true;
                    break;
                }
            }
            if dup {
                false
            } else {
                data.distances[base] = dist;
                data.indices[base] = idx;
                data.is_new[base] = true;
                Self::sift_down_data(data, base, k);
                self.root_dists[row].store(data.distances[base].to_bits(), Ordering::Relaxed);
                true
            }
        };

        lock.store(false, Ordering::Release);
        result
    }

    /// Sort each row ascending by distance for final output (in-place insertion sort).
    pub fn sort_by_distance(&mut self) {
        let k = self.k;
        let data = self.data.get_mut();
        for row in 0..self.n {
            let base = row * k;
            // Insertion sort — k is small (typically 10-30)
            for i in 1..k {
                let d = data.distances[base + i];
                let idx = data.indices[base + i];
                let mut j = i;
                while j > 0 && data.distances[base + j - 1] > d {
                    data.distances[base + j] = data.distances[base + j - 1];
                    data.indices[base + j] = data.indices[base + j - 1];
                    j -= 1;
                }
                data.distances[base + j] = d;
                data.indices[base + j] = idx;
            }
        }
    }

    /// Sift-down on HeapData (shared by all push variants).
    fn sift_down_data(data: &mut HeapData, base: usize, k: usize) {
        let mut pos = 0;
        loop {
            let left = 2 * pos + 1;
            if left >= k {
                break;
            }
            let right = left + 1;
            let mut largest = pos;
            if data.distances[base + left] > data.distances[base + largest] {
                largest = left;
            }
            if right < k && data.distances[base + right] > data.distances[base + largest] {
                largest = right;
            }
            if largest == pos {
                break;
            }
            data.distances.swap(base + pos, base + largest);
            data.indices.swap(base + pos, base + largest);
            data.is_new.swap(base + pos, base + largest);
            pos = largest;
        }
    }
}

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
        assert_eq!(heap.indices()[0..3], [14, 11, 12]);
        assert!((heap.distances()[0] - 2.0).abs() < 1e-6);
        assert!((heap.distances()[1] - 3.0).abs() < 1e-6);
        assert!((heap.distances()[2] - 4.0).abs() < 1e-6);
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
        let pos = heap.indices()[0..3]
            .iter()
            .position(|&x| x == 10)
            .unwrap();
        assert!(heap.is_new()[pos]);
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
            assert!(heap.distances()[j - 1] <= heap.distances()[j]);
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
        let indices: Vec<i32> = heap.indices()[0..10].to_vec();
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
        assert_eq!(heap.indices()[0..2], [1, 2]);
        assert_eq!(heap.indices()[2..4], [0, 2]);
        assert_eq!(heap.indices()[4..6], [0, 1]);
    }
}
