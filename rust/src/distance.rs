/// Squared Euclidean distance and dot product with SIMD acceleration.

/// Squared Euclidean distance: sum((a_i - b_i)^2).
#[inline]
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { squared_euclidean_neon(a, b, n) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        squared_euclidean_scalar(a, b, n)
    }
}

/// Squared Euclidean distance with early exit when running sum >= bound.
/// Returns the partial or full sum. Caller checks result >= bound for rejection.
#[inline]
pub fn squared_euclidean_bounded(a: &[f32], b: &[f32], bound: f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { squared_euclidean_bounded_neon(a, b, n, bound) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        squared_euclidean_bounded_scalar(a, b, n, bound)
    }
}

/// Dot product: sum(a_i * b_i).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_neon(a, b, n) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_product_scalar(a, b, n)
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON implementations
// ---------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn squared_euclidean_neon(a: &[f32], b: &[f32], n: usize) -> f32 {
    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let chunks = n / 16;
        let mut i = 0;
        for _ in 0..chunks {
            let a0 = vld1q_f32(a.as_ptr().add(i));
            let b0 = vld1q_f32(b.as_ptr().add(i));
            let d0 = vsubq_f32(a0, b0);
            sum0 = vfmaq_f32(sum0, d0, d0);

            let a1 = vld1q_f32(a.as_ptr().add(i + 4));
            let b1 = vld1q_f32(b.as_ptr().add(i + 4));
            let d1 = vsubq_f32(a1, b1);
            sum1 = vfmaq_f32(sum1, d1, d1);

            let a2 = vld1q_f32(a.as_ptr().add(i + 8));
            let b2 = vld1q_f32(b.as_ptr().add(i + 8));
            let d2 = vsubq_f32(a2, b2);
            sum2 = vfmaq_f32(sum2, d2, d2);

            let a3 = vld1q_f32(a.as_ptr().add(i + 12));
            let b3 = vld1q_f32(b.as_ptr().add(i + 12));
            let d3 = vsubq_f32(a3, b3);
            sum3 = vfmaq_f32(sum3, d3, d3);

            i += 16;
        }

        let sum01 = vaddq_f32(sum0, sum1);
        let sum23 = vaddq_f32(sum2, sum3);
        let sum = vaddq_f32(sum01, sum23);
        let mut result = vaddvq_f32(sum);

        // 4-element tail
        while i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            let dv = vsubq_f32(av, bv);
            result += vaddvq_f32(vmulq_f32(dv, dv));
            i += 4;
        }

        // Scalar remainder
        while i < n {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            result += d * d;
            i += 1;
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn squared_euclidean_bounded_neon(
    a: &[f32],
    b: &[f32],
    n: usize,
    bound: f32,
) -> f32 {
    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let chunks = n / 16;
        let mut i = 0;
        for _ in 0..chunks {
            let a0 = vld1q_f32(a.as_ptr().add(i));
            let b0 = vld1q_f32(b.as_ptr().add(i));
            let d0 = vsubq_f32(a0, b0);
            sum0 = vfmaq_f32(sum0, d0, d0);

            let a1 = vld1q_f32(a.as_ptr().add(i + 4));
            let b1 = vld1q_f32(b.as_ptr().add(i + 4));
            let d1 = vsubq_f32(a1, b1);
            sum1 = vfmaq_f32(sum1, d1, d1);

            let a2 = vld1q_f32(a.as_ptr().add(i + 8));
            let b2 = vld1q_f32(b.as_ptr().add(i + 8));
            let d2 = vsubq_f32(a2, b2);
            sum2 = vfmaq_f32(sum2, d2, d2);

            let a3 = vld1q_f32(a.as_ptr().add(i + 12));
            let b3 = vld1q_f32(b.as_ptr().add(i + 12));
            let d3 = vsubq_f32(a3, b3);
            sum3 = vfmaq_f32(sum3, d3, d3);

            i += 16;

            // Early exit check every 16 elements
            let s01 = vaddq_f32(sum0, sum1);
            let s23 = vaddq_f32(sum2, sum3);
            let s = vaddq_f32(s01, s23);
            if vaddvq_f32(s) >= bound {
                return vaddvq_f32(s);
            }
        }

        let sum01 = vaddq_f32(sum0, sum1);
        let sum23 = vaddq_f32(sum2, sum3);
        let sum = vaddq_f32(sum01, sum23);
        let mut result = vaddvq_f32(sum);

        // 4-element tail
        while i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            let dv = vsubq_f32(av, bv);
            result += vaddvq_f32(vmulq_f32(dv, dv));
            i += 4;
        }

        // Scalar remainder
        while i < n {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            result += d * d;
            i += 1;
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_product_neon(a: &[f32], b: &[f32], n: usize) -> f32 {
    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let chunks = n / 16;
        let mut i = 0;
        for _ in 0..chunks {
            let a0 = vld1q_f32(a.as_ptr().add(i));
            let b0 = vld1q_f32(b.as_ptr().add(i));
            sum0 = vfmaq_f32(sum0, a0, b0);

            let a1 = vld1q_f32(a.as_ptr().add(i + 4));
            let b1 = vld1q_f32(b.as_ptr().add(i + 4));
            sum1 = vfmaq_f32(sum1, a1, b1);

            let a2 = vld1q_f32(a.as_ptr().add(i + 8));
            let b2 = vld1q_f32(b.as_ptr().add(i + 8));
            sum2 = vfmaq_f32(sum2, a2, b2);

            let a3 = vld1q_f32(a.as_ptr().add(i + 12));
            let b3 = vld1q_f32(b.as_ptr().add(i + 12));
            sum3 = vfmaq_f32(sum3, a3, b3);

            i += 16;
        }

        let sum01 = vaddq_f32(sum0, sum1);
        let sum23 = vaddq_f32(sum2, sum3);
        let sum = vaddq_f32(sum01, sum23);
        let mut result = vaddvq_f32(sum);

        // 4-element tail
        while i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            result += vaddvq_f32(vmulq_f32(av, bv));
            i += 4;
        }

        // Scalar remainder
        while i < n {
            result += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback implementations
// ---------------------------------------------------------------------------
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn squared_euclidean_scalar(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = n / 16;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            for j in 0..4 {
                let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
                sum0 += d * d;
            }
            for j in 4..8 {
                let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
                sum1 += d * d;
            }
            for j in 8..12 {
                let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
                sum2 += d * d;
            }
            for j in 12..16 {
                let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
                sum3 += d * d;
            }
        }
        i += 16;
    }

    let mut result = sum0 + sum1 + sum2 + sum3;
    while i < n {
        unsafe {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            result += d * d;
        }
        i += 1;
    }
    result
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn squared_euclidean_bounded_scalar(a: &[f32], b: &[f32], n: usize, bound: f32) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = n / 16;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            for j in 0..4 {
                let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
                sum0 += d * d;
            }
            for j in 4..8 {
                let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
                sum1 += d * d;
            }
            for j in 8..12 {
                let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
                sum2 += d * d;
            }
            for j in 12..16 {
                let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
                sum3 += d * d;
            }
        }
        i += 16;
        let partial = sum0 + sum1 + sum2 + sum3;
        if partial >= bound {
            return partial;
        }
    }

    let mut result = sum0 + sum1 + sum2 + sum3;
    while i < n {
        unsafe {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            result += d * d;
        }
        i += 1;
    }
    result
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = n / 16;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            for j in 0..4 {
                sum0 += *a.get_unchecked(i + j) * *b.get_unchecked(i + j);
            }
            for j in 4..8 {
                sum1 += *a.get_unchecked(i + j) * *b.get_unchecked(i + j);
            }
            for j in 8..12 {
                sum2 += *a.get_unchecked(i + j) * *b.get_unchecked(i + j);
            }
            for j in 12..16 {
                sum3 += *a.get_unchecked(i + j) * *b.get_unchecked(i + j);
            }
        }
        i += 16;
    }

    let mut result = sum0 + sum1 + sum2 + sum3;
    while i < n {
        unsafe {
            result += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_euclidean_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // (3^2 + 3^2 + 3^2) = 27
        let d = squared_euclidean(&a, &b);
        assert!((d - 27.0).abs() < 1e-5);
    }

    #[test]
    fn squared_euclidean_identical() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!(squared_euclidean(&a, &a) < 1e-10);
    }

    #[test]
    fn squared_euclidean_large_dim() {
        // Test with >16 dimensions to exercise SIMD path
        let n = 128;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
        let d = squared_euclidean(&a, &b);
        // Each diff is 1.0, sum of 128 ones = 128
        assert!((d - 128.0).abs() < 1e-3);
    }

    #[test]
    fn squared_euclidean_odd_dim() {
        // Non-multiple-of-16 to test remainder
        let n = 35;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let d = squared_euclidean(&a, &b);
        assert!((d - expected).abs() < 1e-3, "got {d}, expected {expected}");
    }

    #[test]
    fn squared_euclidean_dim_20() {
        // 20 = 16 + 4: exercises the 4-element SIMD tail exactly
        let n = 20;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let d = squared_euclidean(&a, &b);
        assert!((d - expected).abs() < 1e-3, "got {d}, expected {expected}");
    }

    #[test]
    fn bounded_agrees_with_unbounded() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let full = squared_euclidean(&a, &b);
        // With a bound larger than the full distance, should return the same value
        let bounded = squared_euclidean_bounded(&a, &b, full + 1.0);
        assert!(
            (full - bounded).abs() < 1e-3,
            "full={full}, bounded={bounded}"
        );
    }

    #[test]
    fn bounded_early_exit() {
        let n = 128;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![0.0; n];
        let full = squared_euclidean(&a, &b);
        // Set a very low bound — the bounded version should exit early
        // and return something >= bound
        let bound = 10.0;
        let result = squared_euclidean_bounded(&a, &b, bound);
        assert!(result >= bound, "result={result}, bound={bound}");
        // But the full result is much larger
        assert!(full > bound);
    }

    #[test]
    fn dot_product_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let d = dot_product(&a, &b);
        assert!((d - 32.0).abs() < 1e-5);
    }

    #[test]
    fn dot_product_large_dim() {
        let n = 128;
        let a: Vec<f32> = vec![1.0; n];
        let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let d = dot_product(&a, &b);
        let expected: f32 = (0..n).map(|i| i as f32).sum();
        assert!((d - expected).abs() < 1e-2, "got {d}, expected {expected}");
    }

    #[test]
    fn dot_product_odd_dim() {
        let n = 37;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let d = dot_product(&a, &b);
        assert!((d - expected).abs() < 1e-2, "got {d}, expected {expected}");
    }

    #[test]
    fn dot_product_dim_20() {
        // 20 = 16 + 4: exercises the 4-element SIMD tail
        let n = 20;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let d = dot_product(&a, &b);
        assert!((d - expected).abs() < 1e-2, "got {d}, expected {expected}");
    }
}
