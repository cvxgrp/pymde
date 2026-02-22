/// Deterministic seeded RNG wrapper around ChaCha8.
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct Rng64 {
    inner: ChaCha8Rng,
}

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Random integer in [0, upper).
    #[inline]
    pub fn rand_int(&mut self, upper: usize) -> usize {
        self.inner.random_range(0..upper)
    }

    /// Random float in [0, 1).
    #[inline]
    pub fn rand_float(&mut self) -> f32 {
        self.inner.random::<f32>()
    }

    /// Raw u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.inner.random::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng64::new(42);
        let mut b = Rng64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds() {
        let mut a = Rng64::new(1);
        let mut b = Rng64::new(2);
        // Extremely unlikely to produce same sequence
        let same = (0..10).all(|_| a.next_u64() == b.next_u64());
        assert!(!same);
    }

    #[test]
    fn rand_int_in_range() {
        let mut rng = Rng64::new(0);
        for _ in 0..1000 {
            let v = rng.rand_int(10);
            assert!(v < 10);
        }
    }

    #[test]
    fn rand_float_in_range() {
        let mut rng = Rng64::new(0);
        for _ in 0..1000 {
            let v = rng.rand_float();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
