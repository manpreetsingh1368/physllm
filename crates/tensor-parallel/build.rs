// build.rs
use std::{env, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    #[cfg(feature = "rocm")]
    build_rocm(&out);

    #[cfg(feature = "cuda")]
    build_cuda(&out);

    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    compile_error!("tensor-parallel requires either 'rocm' or 'cuda' feature. CPU-only is not supported.");
}

fn build_rocm(out: &PathBuf) {
    let rocm = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let hipcc = format!("{rocm}/bin/hipcc");
    let arch  = env::var("HIP_ARCH").unwrap_or_else(|_| "gfx942".into()); // MI300X default

    println!("cargo:rerun-if-changed=kernels/gpu_shim.hip");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    let obj = out.join("gpu_shim.o");
    let lib = out.join("libtensor_parallel_gpu.a");

    let status = Command::new(&hipcc)
        .args([
            "--offload-arch", &arch,
            "-O3", "-fPIC",
            "--rocm-path", &rocm,
            &format!("-I{rocm}/include"),         // RCCL + HIP headers
            &format!("-I{rocm}/hipblas/include"),  // hipBLAS
            "-DUSE_ROCM",
            "-c", "kernels/gpu_shim.hip",
            "-o", obj.to_str().unwrap(),
        ])
        .status()
        .expect("hipcc not found — set ROCM_PATH or install ROCm");

    assert!(status.success(), "hipcc compilation failed for gpu_shim.hip");

    Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status().expect("ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={rocm}/lib");
    println!("cargo:rustc-link-search=native={rocm}/hipblas/lib");

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
        .expect("nvcc not found — set CUDA_HOME or install CUDA");

    assert!(status.success(), "nvcc compilation failed for gpu_shim.cu");

    Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status().expect("ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={cuda}/lib64");

    println!("cargo:rustc-link-lib=static=tensor_parallel_gpu_cuda");
    println!("cargo:rustc-link-lib=dylib=nccl");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
