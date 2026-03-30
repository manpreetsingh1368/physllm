//! build.rs — compiles HIP kernels and generates Rust FFI bindings.
//!
//! Requires:
//!   export ROCM_PATH=/opt/rocm   (or set via env var)
//!   AMD ROCm ≥ 6.0 installed
//!
//! Run with: cargo build --features rocm

use std::{env, path::PathBuf, process::Command};

fn main() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let out_dir   = PathBuf::from(env::var("OUT_DIR").unwrap());

    // ── 1. Compile HIP kernels (gemm, attention, softmax, rope) ──────────────
    #[cfg(feature = "rocm")]
    compile_hip_kernels(&rocm_path, &out_dir);

    // ── 2. Generate FFI bindings from hip_runtime_api.h ──────────────────────
    #[cfg(feature = "rocm")]
    generate_hip_bindings(&rocm_path, &out_dir);

    // ── 3. Link flags ────────────────────────────────────────────────────────
    #[cfg(feature = "rocm")]
    {
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-lib=dylib=hipblas");
        println!("cargo:rustc-link-lib=dylib=rocsolver");
        println!("cargo:rustc-link-lib=dylib=miopen");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-changed=../../kernels/");
    }
}

#[cfg(feature = "rocm")]
fn compile_hip_kernels(rocm_path: &str, out_dir: &PathBuf) {
    let hipcc  = format!("{rocm_path}/bin/hipcc");
    let kernel_dir = PathBuf::from("../../kernels");
    let kernels = [
        "gemm_f16.hip",
        "flash_attention.hip",
        "softmax.hip",
        "rope_embedding.hip",
        "layer_norm.hip",
        "kv_cache.hip",
    ];

    for kernel in &kernels {
        let src = kernel_dir.join(kernel);
        let obj = out_dir.join(kernel.replace(".hip", ".o"));
        println!("cargo:rerun-if-changed={}", src.display());

        let status = Command::new(&hipcc)
            .args([
                "--amdgpu-target=gfx1100,gfx1030,gfx906,gfx90a",  // RX 7900 / 6000 / MI series
                "-O3",
                "-ffast-math",
                "--offload-arch=gfx1100",
                "-I", &format!("{rocm_path}/include"),
                "-c",
                src.to_str().unwrap(),
                "-o",
                obj.to_str().unwrap(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {},
            Ok(s) => eprintln!("hipcc failed ({s}) for {kernel}"),
            Err(e) => eprintln!("hipcc not found ({e}); ROCm GPU acceleration unavailable"),
        }
    }

    // Link compiled objects into a static lib
    let ar = Command::new("ar")
        .args(["rcs", out_dir.join("libphysllm_kernels.a").to_str().unwrap()])
        .args(kernels.iter().map(|k| out_dir.join(k.replace(".hip", ".o"))))
        .status();

    if ar.map(|s| s.success()).unwrap_or(false) {
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=physllm_kernels");
    }
}

#[cfg(feature = "rocm")]
fn generate_hip_bindings(rocm_path: &str, out_dir: &PathBuf) {
    let bindings = bindgen::Builder::default()
        .header(format!("{rocm_path}/include/hip/hip_runtime_api.h"))
        .clang_arg(format!("-I{rocm_path}/include"))
        .allowlist_function("hip.*")
        .allowlist_type("hip.*")
        .allowlist_var("hip.*")
        .derive_debug(true)
        .derive_default(true)
        .generate()
        .expect("Failed to generate HIP bindings");

    bindings
        .write_to_file(out_dir.join("hip_bindings.rs"))
        .expect("Failed to write bindings");
}
