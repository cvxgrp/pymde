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

/// Squared Euclidean distance with early exit when partial sum exceeds `bound`.
/// Returns the exact distance if < bound, otherwise a value >= bound.
/// On aarch64, uses explicit NEON intrinsics with fused multiply-accumulate
/// and 4 independent accumulators for maximum pipeline utilization.
#[inline]
pub fn squared_euclidean_bounded(a: &[f32], b: &[f32], bound: f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        squared_euclidean_bounded_neon(a, b, bound)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        squared_euclidean_bounded_generic(a, b, bound)
    }
}

/// NEON-optimized bounded squared Euclidean distance.
/// Uses vfmaq_f32 (fused multiply-accumulate) with 4 persistent accumulators
/// that carry across chunks, avoiding reinitialization overhead.
#[cfg(target_arch = "aarch64")]
#[inline]
fn squared_euclidean_bounded_neon(a: &[f32], b: &[f32], bound: f32) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut i = 0;

    unsafe {
        // Persistent accumulators across all chunks
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        while i + 16 <= n {
            let d0 = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            acc0 = vfmaq_f32(acc0, d0, d0);

            let d1 = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
            acc1 = vfmaq_f32(acc1, d1, d1);

            let d2 = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 8)), vld1q_f32(b.as_ptr().add(i + 8)));
            acc2 = vfmaq_f32(acc2, d2, d2);

            let d3 = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 12)), vld1q_f32(b.as_ptr().add(i + 12)));
            acc3 = vfmaq_f32(acc3, d3, d3);

            // Horizontal reduction for bound check
            let combined = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            let sum = vaddvq_f32(combined);
            if sum >= bound {
                return sum;
            }
            i += 16;
        }

        // Final reduction
        let combined = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut sum = vaddvq_f32(combined);

        // Scalar remainder
        while i < n {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum += d * d;
            i += 1;
        }
        sum
    }
}

/// Generic (non-NEON) bounded squared Euclidean distance.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn squared_euclidean_bounded_generic(a: &[f32], b: &[f32], bound: f32) -> f32 {
    let n = a.len();
    let mut sum = 0.0f32;
    let mut i = 0;

    while i + 16 <= n {
        unsafe {
            let mut chunk = 0.0f32;
            for j in 0..16 {
                let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
                chunk += d * d;
            }
            sum += chunk;
        }
        if sum >= bound {
            return sum;
        }
        i += 16;
    }

    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// Dot product using NEON FMA on aarch64, generic autovectorization elsewhere.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        dot_product_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_product_generic(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut i = 0;

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(a.as_ptr().add(i + 8)), vld1q_f32(b.as_ptr().add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(a.as_ptr().add(i + 12)), vld1q_f32(b.as_ptr().add(i + 12)));
            i += 16;
        }

        let combined = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut sum = vaddvq_f32(combined);

        while i < n {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        sum
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_product_generic(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = 0.0f32;
    let mut i = 0;

    while i + 16 <= n {
        unsafe {
            let mut chunk = 0.0f32;
            for j in 0..16 {
                chunk += *a.get_unchecked(i + j) * *b.get_unchecked(i + j);
            }
            sum += chunk;
        }
        i += 16;
    }

    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
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

    #[test]
    fn test_bounded_vs_unbounded() {
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let exact = squared_euclidean(&a, &b);
        let bounded = squared_euclidean_bounded(&a, &b, f32::INFINITY);
        assert!((exact - bounded).abs() < 1e-3, "bounded={bounded} exact={exact}");
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
        assert!((dot_product(&a, &b) - expected).abs() < 1e-6);
    }
}
