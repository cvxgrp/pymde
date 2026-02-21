use std::os::raw::c_int;

const ROW_MAJOR: c_int = 101;
const NO_TRANS: c_int = 111;
const TRANS: c_int = 112;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

#[cfg(not(target_os = "macos"))]
#[link(name = "openblas")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

/// Call sgemm: C = alpha * A @ B^T + beta * C
/// A is (m, k) row-major, B is (n, k) row-major, C is (m, n) row-major.
#[inline]
pub(crate) unsafe fn sgemm_nn_t(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
) {
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a,
            k as c_int, // lda = number of columns of A
            b,
            k as c_int, // ldb = number of columns of B
            beta,
            c,
            n as c_int, // ldc = number of columns of C
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_nn_t_identity() {
        // A = [[1,0],[0,1]], B = [[1,0],[0,1]]
        // C = -2 * A @ B^T = -2 * I
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        unsafe { sgemm_nn_t(2, 2, 2, -2.0, a.as_ptr(), b.as_ptr(), 0.0, c.as_mut_ptr()) };
        assert_eq!(c, [-2.0, 0.0, 0.0, -2.0]);
    }

    #[test]
    fn sgemm_nn_t_simple() {
        // A = [[1,2,3]], B = [[4,5,6],[7,8,9]]
        // A @ B^T = [1*4+2*5+3*6, 1*7+2*8+3*9] = [32, 50]
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = [0.0f32; 2];
        unsafe { sgemm_nn_t(1, 2, 3, 1.0, a.as_ptr(), b.as_ptr(), 0.0, c.as_mut_ptr()) };
        assert_eq!(c, [32.0, 50.0]);
    }

    #[test]
    fn sgemm_nn_t_beta_accumulate() {
        // Check that beta * C is accumulated
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [10.0f32, 20.0, 30.0, 40.0];
        // C = 1.0 * A @ B^T + 0.5 * C = I + [5,10,15,20]
        unsafe { sgemm_nn_t(2, 2, 2, 1.0, a.as_ptr(), b.as_ptr(), 0.5, c.as_mut_ptr()) };
        assert_eq!(c, [6.0, 10.0, 15.0, 21.0]);
    }
}
