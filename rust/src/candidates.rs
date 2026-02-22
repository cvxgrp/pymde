/// Reservoir-sampled candidate lists for NN-Descent.
///
/// Each point has a max-heap of (priority, neighbor_index) pairs keyed by
/// random priority. When more than `max_candidates` arrive, only those with
/// the lowest priorities survive — ensuring uniform reservoir sampling.

pub struct CandidateHeap {
    max_candidates: usize,
    /// Flat arrays: priorities[n * max_candidates], indices[n * max_candidates]
    priorities: Vec<f32>,
    indices: Vec<i32>,
    /// Number of valid entries per point
    counts: Vec<usize>,
}

impl CandidateHeap {
    pub fn new(n: usize, max_candidates: usize) -> Self {
        Self {
            max_candidates,
            priorities: vec![f32::NEG_INFINITY; n * max_candidates],
            indices: vec![-1; n * max_candidates],
            counts: vec![0; n],
        }
    }

    pub fn clear(&mut self) {
        self.priorities.fill(f32::NEG_INFINITY);
        self.indices.fill(-1);
        self.counts.fill(0);
    }

    /// Push a candidate `neighbor` for `point` with `priority`.
    /// Returns true if the candidate was accepted.
    pub fn push(&mut self, point: usize, neighbor: i32, priority: f32) -> bool {
        let mc = self.max_candidates;
        let base = point * mc;
        let count = self.counts[point];

        // Duplicate check
        for j in 0..count {
            if self.indices[base + j] == neighbor {
                return false;
            }
        }

        if count < mc {
            // Still have room — just append and sift up
            let pos = count;
            self.priorities[base + pos] = priority;
            self.indices[base + pos] = neighbor;
            self.counts[point] = count + 1;
            self.sift_up(base, pos);
            true
        } else {
            // Full — check if this priority is lower than the max (root)
            if priority >= self.priorities[base] {
                return false;
            }
            // Replace root and sift down
            self.priorities[base] = priority;
            self.indices[base] = neighbor;
            self.sift_down(base, mc);
            true
        }
    }

    /// Get the candidate indices for a point. Returns the valid entries.
    pub fn get(&self, point: usize) -> &[i32] {
        let base = point * self.max_candidates;
        let count = self.counts[point];
        &self.indices[base..base + count]
    }

    /// Get the count of candidates for a point.
    #[cfg(test)]
    pub fn count(&self, point: usize) -> usize {
        self.counts[point]
    }

    fn sift_up(&mut self, base: usize, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if self.priorities[base + pos] <= self.priorities[base + parent] {
                break;
            }
            self.priorities.swap(base + pos, base + parent);
            self.indices.swap(base + pos, base + parent);
            pos = parent;
        }
    }

    fn sift_down(&mut self, base: usize, k: usize) {
        let mut pos = 0;
        loop {
            let left = 2 * pos + 1;
            if left >= k {
                break;
            }
            let right = left + 1;
            let mut largest = pos;
            if self.priorities[base + left] > self.priorities[base + largest] {
                largest = left;
            }
            if right < k && self.priorities[base + right] > self.priorities[base + largest] {
                largest = right;
            }
            if largest == pos {
                break;
            }
            self.priorities.swap(base + pos, base + largest);
            self.indices.swap(base + pos, base + largest);
            pos = largest;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_get() {
        let mut ch = CandidateHeap::new(2, 3);
        ch.push(0, 10, 0.5);
        ch.push(0, 11, 0.3);
        ch.push(0, 12, 0.8);
        assert_eq!(ch.count(0), 3);
        let candidates = ch.get(0);
        assert_eq!(candidates.len(), 3);
        assert!(candidates.contains(&10));
        assert!(candidates.contains(&11));
        assert!(candidates.contains(&12));
    }

    #[test]
    fn reservoir_limit() {
        let mut ch = CandidateHeap::new(1, 3);
        ch.push(0, 10, 0.9);
        ch.push(0, 11, 0.8);
        ch.push(0, 12, 0.7);
        assert_eq!(ch.count(0), 3);

        // Push with lower priority — should replace highest priority
        assert!(ch.push(0, 13, 0.1));
        assert_eq!(ch.count(0), 3);
        let candidates = ch.get(0);
        // The one with priority 0.9 should be gone, replaced by 0.1
        assert!(candidates.contains(&13));
    }

    #[test]
    fn reject_high_priority_when_full() {
        let mut ch = CandidateHeap::new(1, 2);
        ch.push(0, 10, 0.3);
        ch.push(0, 11, 0.5);
        // Full, push with higher priority than max — should be rejected
        assert!(!ch.push(0, 12, 0.6));
        assert_eq!(ch.count(0), 2);
    }

    #[test]
    fn duplicate_rejection() {
        let mut ch = CandidateHeap::new(1, 5);
        assert!(ch.push(0, 10, 0.5));
        assert!(!ch.push(0, 10, 0.3)); // duplicate
        assert_eq!(ch.count(0), 1);
    }

    #[test]
    fn clear_resets() {
        let mut ch = CandidateHeap::new(1, 3);
        ch.push(0, 10, 0.5);
        ch.push(0, 11, 0.3);
        ch.clear();
        assert_eq!(ch.count(0), 0);
        assert!(ch.get(0).is_empty());
    }

    #[test]
    fn multiple_points() {
        let mut ch = CandidateHeap::new(3, 2);
        ch.push(0, 10, 0.5);
        ch.push(1, 20, 0.3);
        ch.push(2, 30, 0.7);
        assert_eq!(ch.count(0), 1);
        assert_eq!(ch.count(1), 1);
        assert_eq!(ch.count(2), 1);
        assert!(ch.get(0).contains(&10));
        assert!(ch.get(1).contains(&20));
        assert!(ch.get(2).contains(&30));
    }
}
