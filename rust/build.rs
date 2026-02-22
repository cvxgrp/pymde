fn main() {
    // On Windows CI, OpenBLAS is manually downloaded to C:\openblas.
    // OPENBLAS_LIB_DIR tells the linker where to find openblas.lib.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
}
