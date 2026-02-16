use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

/// Deterministic seeded RNG wrapper.
pub struct NnRng {
    inner: ChaCha8Rng,
}

impl NnRng {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Random integer in [0, upper_exclusive).
    #[inline]
    pub fn rand_int(&mut self, upper_exclusive: usize) -> usize {
        self.inner.gen_range(0..upper_exclusive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let mut r1 = NnRng::new(42);
        let mut r2 = NnRng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.rand_int(1000), r2.rand_int(1000));
        }
    }

    #[test]
    fn test_range() {
        let mut r = NnRng::new(0);
        for _ in 0..1000 {
            let v = r.rand_int(10);
            assert!(v < 10);
        }
    }
}
