mod bfs;
mod blas;
mod knn;

#[pyo3::pymodule]
mod _native {
    #[pymodule_export]
    use super::knn::knn_l2;
    #[pymodule_export]
    use super::bfs::breadth_first_directed;
}
