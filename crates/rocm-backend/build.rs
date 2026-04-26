//! build.rs — compiles HIP kernels for AMD (ROCm) AND NVIDIA (CUDA via HIP).
//!
//! Requires either:
//!   AMD:    export ROCM_PATH=/opt/rocm   (ROCm >= 6.0)
//!   NVIDIA: export HIP_PLATFORM=nvidia   (HIP for NVIDIA)

use std::{env, path::PathBuf, process::Command};

fn main() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let out_dir   = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Detect platform
    let hip_platform = env::var("HIP_PLATFORM").unwrap_or_else(|_| {
        if std::path::Path::new("/opt/rocm/bin/hipcc").exists() {
            "amd".to_string()
        } else if std::path::Path::new("/usr/local/cuda/bin/nvcc").exists() {
            "nvidia".to_string()
        } else {
            "amd".to_string()
        }
    });

    println!("cargo:warning=PhysLLM GPU platform: {}", hip_platform);

    #[cfg(any(feature = "rocm", feature = "cuda"))]
    {
        compile_hip_kernels(&rocm_path, &out_dir, &hip_platform);
        generate_hip_bindings(&rocm_path, &out_dir);
    }

    // Link flags
    #[cfg(feature = "rocm")]
    {
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-lib=dylib=hipblas");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-changed=../../kernels/");
    }

    #[cfg(feature = "cuda")]
    {
        let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".into());
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-changed=../../kernels/");
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
fn compile_hip_kernels(rocm_path: &str, out_dir: &PathBuf, hip_platform: &str) {
    let hipcc = format!("{rocm_path}/bin/hipcc");
    let kernel_dir = PathBuf::from("../../kernels");

    // ALL kernels (old + new GPU ops)
    let kernels = [
        "gemm_f16",
        "flash_attention",
        "flash_attention_v2",
        "softmax",
        "rope_embedding",
        "rope_gpu",
        "layer_norm",
        "rms_norm_gpu",
        "kv_cache",
        "kv_cache_update",
        "silu",
        "residual_add",
        "embedding",
        "lm_head",
        "softmax_sample",
        "adam_optimizer",
        "mxfp4_dequant",
        "moe_combine",
        "moe_expert_forward",
        "moe_router",
    ];

    // Architecture flags based on platform
    let arch_flags: Vec<String> = if hip_platform == "nvidia" {
        vec![
            "--offload-arch=sm_70".into(),  // V100
            "--offload-arch=sm_80".into(),  // A100
            "--offload-arch=sm_86".into(),  // RTX 3090
            "--offload-arch=sm_89".into(),  // RTX 4090
            "--offload-arch=sm_90".into(),  // H100
        ]
    } else {
        vec![
            "--offload-arch=gfx90a".into(),   // MI250X
            "--offload-arch=gfx942".into(),   // MI300X
            "--offload-arch=gfx1100".into(),  // RX 7900 XTX
        ]
    };

    let mut compiled = Vec::new();

    for kernel_name in &kernels {
        let src = kernel_dir.join(format!("{}.hip", kernel_name));
        let obj = out_dir.join(format!("{}.o", kernel_name));

        if !src.exists() {
            eprintln!("Warning: kernel {} not found, skipping", src.display());
            continue;
        }
        println!("cargo:rerun-if-changed={}", src.display());

        let mut cmd = Command::new(&hipcc);
        for flag in &arch_flags {
            cmd.arg(flag);
        }
        cmd.args([
            "-O3", "-ffast-math", "-fPIC",
            "-I", &format!("{rocm_path}/include"),
            "-c", src.to_str().unwrap(),
            "-o", obj.to_str().unwrap(),
        ]);

        match cmd.status() {
            Ok(s) if s.success() => compiled.push(obj),
            Ok(s) => eprintln!("hipcc failed ({s}) for {kernel_name}"),
            Err(e) => eprintln!("hipcc not found ({e}); GPU acceleration unavailable"),
        }
    }

    // Create static library from compiled objects
    if !compiled.is_empty() {
        let lib_path = out_dir.join("libphysllm_kernels.a");
        let mut ar_cmd = Command::new("ar");
        ar_cmd.args(["rcs", lib_path.to_str().unwrap()]);
        for obj in &compiled {
            ar_cmd.arg(obj.to_str().unwrap());
        }
        if ar_cmd.status().map(|s| s.success()).unwrap_or(false) {
            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=physllm_kernels");
            println!("cargo:warning=Compiled {} GPU kernels", compiled.len());
        }
    }
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
fn generate_hip_bindings(rocm_path: &str, out_dir: &PathBuf) {
    let hip_include = format!("{rocm_path}/include/hip/hip_runtime_api.h");
    if !std::path::Path::new(&hip_include).exists() {
        eprintln!("Warning: HIP headers not found, skipping bindgen");
        // Write empty bindings
        std::fs::write(out_dir.join("hip_bindings.rs"), "// No HIP bindings\n").ok();
        return;
    }

    let bindings = bindgen::Builder::default()
        .header(hip_include)
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
