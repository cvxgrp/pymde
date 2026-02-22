fn main() {
    // This crate links against OpenBLAS for BLAS routines (see blas.rs).
    // On Windows CI, OpenBLAS is manually downloaded to C:\openblas since
    // there's no system package manager. This tells the linker where to
    // find openblas.lib.
    //
    // Backslashes are replaced with forward slashes because Cargo's
    // `cargo:rustc-link-search` directive doesn't escape them, so a path like
    // `C:\openblas\lib` gets mangled to `C:openblaslib`. Forward slashes work
    // fine with the MSVC linker.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", dir.replace('\\', "/"));
        }
    }
}
