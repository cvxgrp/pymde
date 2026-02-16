mod distance;
mod heap;
mod nndescent;
mod rng;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Approximate k-nearest neighbor graph via NN-Descent.
///
/// Parameters
/// ----------
/// data : np.ndarray
///     Data matrix of shape (n, d), dtype float32, C-contiguous.
/// n_neighbors : int
///     Number of neighbors to find **including self** (so actual k = n_neighbors - 1).
/// max_candidates : int
///     Maximum number of candidate neighbors to consider per iteration.
/// seed : int
///     Random seed for reproducibility.
/// verbose : bool
///     If True, print iteration info to stderr.
///
/// Returns
/// -------
/// (indices, distances) : tuple of np.ndarray
///     indices: shape (n, n_neighbors), dtype int32
///     distances: shape (n, n_neighbors), dtype float32
///     Row i, column 0 is always (i, 0.0) — the point itself.
#[pyfunction]
#[pyo3(signature = (data, n_neighbors, max_candidates=60, seed=42, verbose=false))]
fn nn_descent<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    n_neighbors: usize,
    max_candidates: usize,
    seed: u64,
    verbose: bool,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f32>>)> {
    let array = data.as_array();
    let n = array.shape()[0];
    let dim = array.shape()[1];

    // Get a contiguous slice of the data
    let data_slice = array
        .as_slice()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "data must be C-contiguous; use np.ascontiguousarray(data, dtype=np.float32)",
            )
        })?;

    // Run NN-Descent with GIL released
    let (flat_indices, flat_distances) = py.allow_threads(|| {
        nndescent::nn_descent(data_slice, n, dim, n_neighbors, max_candidates, seed, verbose)
    });

    // Convert flat vecs to 2D numpy arrays
    let indices_array = Array2::from_shape_vec((n, n_neighbors), flat_indices)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let distances_array = Array2::from_shape_vec((n, n_neighbors), flat_distances)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok((
        indices_array.into_pyarray(py),
        distances_array.into_pyarray(py),
    ))
}

#[pymodule]
fn _nndescent(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nn_descent, m)?)?;
    Ok(())
}
