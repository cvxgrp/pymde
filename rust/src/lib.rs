mod bfs;
mod blas;
mod knn;

#[pyo3::pymodule]
mod _native {
    #[pymodule_export]
    use crate::knn::knn_l2;
    #[pymodule_export]
    use crate::bfs::breadth_first_directed;
}
