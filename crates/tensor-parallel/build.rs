// build.rs
use std::{env, path::PathBuf, process::Command};

fn main() {
    // In build.rs, features are checked via CARGO_FEATURE_* env vars
    let rocm = env::var("CARGO_FEATURE_ROCM").is_ok();
    let cuda = env::var("CARGO_FEATURE_CUDA").is_ok();

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    if rocm {
        build_rocm(&out);
    } else if cuda {
        build_cuda(&out);
    } else {
        panic!("tensor-parallel requires 'rocm' or 'cuda' feature");
    }
}

fn build_rocm(out: &PathBuf) {
    let rocm = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let arch = env::var("HIP_ARCH").unwrap_or_else(|_| "gfx942".into());

    // Find hipcc — check common locations
    let hipcc = [
        format!("{rocm}/bin/hipcc"),
        "/opt/rocm/bin/hipcc".into(),
        "hipcc".into(),
    ]
    .into_iter()
    .find(|p| {
        if p == "hipcc" {
            Command::new("which").arg("hipcc").status().map(|s| s.success()).unwrap_or(false)
        } else {
            std::path::Path::new(p).exists()
        }
    })
    .expect("hipcc not found — install ROCm or set ROCM_PATH");

    println!("cargo:rerun-if-changed=kernels/gpu_shim.hip");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:warning=Building tensor-parallel with hipcc={hipcc} arch={arch}");

    let obj = out.join("gpu_shim.o");
    let lib = out.join("libtensor_parallel_gpu.a");

    // hipcc flags: --offload-arch must be a single combined arg on some versions
    let status = Command::new(&hipcc)
        .args([
            &format!("--offload-arch={arch}"),  // combined form, more compatible
            "-O3",
            "-fPIC",
            &format!("--rocm-path={rocm}"),
            &format!("-I{rocm}/include"),
            &format!("-I{rocm}/hipblas/include"),
            "-DUSE_ROCM",
            "-x", "hip",          // explicitly tell clang it's HIP source
            "-c", "kernels/gpu_shim.hip",
            "-o", obj.to_str().unwrap(),
        ])
        .status()
        .expect("hipcc failed to start");

    assert!(status.success(), "hipcc compilation failed for gpu_shim.hip");

    Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .expect("ar failed");

    // Find actual rocm lib path
    let lib_paths = [
        format!("{rocm}/lib"),
        format!("{rocm}/lib64"),
    ];
    for p in &lib_paths {
        if std::path::Path::new(p).exists() {
            println!("cargo:rustc-link-search=native={p}");
        }
    }

    // hipblas lib path varies by ROCm version
    let hipblas_paths = [
        format!("{rocm}/hipblas/lib"),
        format!("{rocm}/lib"),       // newer ROCm puts hipblas in main lib
    ];
    for p in &hipblas_paths {
        if std::path::Path::new(p).exists() {
            println!("cargo:rustc-link-search=native={p}");
        }
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=tensor_parallel_gpu");
    println!("cargo:rustc-link-lib=dylib=rccl");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}

fn build_cuda(out: &PathBuf) {
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = format!("{cuda}/bin/nvcc");
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".into());

    println!("cargo:rerun-if-changed=kernels/gpu_shim.cu");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let obj = out.join("gpu_shim_cuda.o");
    let lib = out.join("libtensor_parallel_gpu_cuda.a");

    let status = Command::new(&nvcc)
        .args([
            &format!("-arch={arch}"),
            "-O3", "-Xcompiler", "-fPIC",
            &format!("-I{cuda}/include"),
            "-DUSE_CUDA",
            "-c", "kernels/gpu_shim.cu",
            "-o", obj.to_str().unwrap(),
        ])
        .status()
        .expect("nvcc not found");

    assert!(status.success(), "nvcc compilation failed");

    Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .expect("ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=static=tensor_parallel_gpu_cuda");
    println!("cargo:rustc-link-lib=dylib=nccl");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
