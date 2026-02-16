#[inline]
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean(a, b).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!((euclidean(&a, &b) - std::f32::consts::SQRT_2).abs() < 1e-6);
        assert!((squared_euclidean(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_distance() {
        let a = [3.0, 4.0, 5.0];
        assert_eq!(squared_euclidean(&a, &a), 0.0);
        assert_eq!(euclidean(&a, &a), 0.0);
    }

    #[test]
    fn test_high_dim() {
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();
        // Each dimension contributes 1.0 to squared distance
        assert!((squared_euclidean(&a, &b) - dim as f32).abs() < 1e-3);
        assert!((euclidean(&a, &b) - (dim as f32).sqrt()).abs() < 1e-3);
    }
}
