fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", dir);
        }
    }
}
