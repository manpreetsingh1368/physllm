//! build.rs — Compiles HIP kernels and generates Rust FFI bindings for ROCm.
//!
//! Fixes applied vs. original:
//!   1. Kernel path now computed from CARGO_MANIFEST_DIR (absolute, reliable)
//!   2. --amdgpu-target removed; only --offload-arch used (ROCm 6+/7 compatible)
//!   3. Missing kernels (gemm_f16, softmax, kv_cache) created inline if absent
//!   4. bindgen clang args include GCC include path to fix limits.h not found
//!   5. Duplicate --offload-arch=gfx1100 removed from args list
//!   6. hipcc path uses ROCM_PATH env var with fallback, not hardcoded

use std::{env, path::PathBuf, process::Command};

fn main() {
    let rocm_path = env::var("ROCM_PATH")
        .or_else(|_| env::var("HIP_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".into());

    let out_dir  = PathBuf::from(env::var("OUT_DIR").unwrap());

    // ── IMPORTANT: kernel dir is relative to the *workspace* root, ──────────
    // not the crate root. CARGO_MANIFEST_DIR is the crate root
    // (physllm/crates/rocm-backend), so go up two levels to reach
    // physllm/kernels/.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir   = manifest_dir.join("../../kernels");

    // Canonicalise so error messages show the real path
    let kernel_dir = kernel_dir.canonicalize().unwrap_or_else(|_| {
        // If the kernels dir doesn't exist yet, create it
        std::fs::create_dir_all(&kernel_dir).ok();
        kernel_dir.clone()
    });

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    #[cfg(feature = "rocm")]
    {
        // Ensure all required kernel files exist (create stubs for missing ones)
        ensure_kernels_exist(&kernel_dir);

        compile_hip_kernels(&rocm_path, &kernel_dir, &out_dir);
        generate_hip_bindings(&rocm_path, &out_dir);

        // Link flags
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-lib=dylib=hipblas");

        // Only link these if they exist (not all ROCm installs include them)
        if std::path::Path::new(&format!("{rocm_path}/lib/libmiopen.so")).exists()
            || std::path::Path::new(&format!("{rocm_path}/lib/libMIOpen.so")).exists()
        {
            println!("cargo:rustc-link-lib=dylib=MIOpen");
        }
    }

    // Always emit rerun-if-changed for all kernel files so incremental builds work
    for kernel in KERNELS {
        println!("cargo:rerun-if-changed={}", kernel_dir.join(kernel).display());
    }
}

/// All kernel files the build expects.
const KERNELS: &[&str] = &[
    "gemm_f16.hip",
    "flash_attention.hip",
    "softmax.hip",
    "rope_embedding.hip",
    "layer_norm.hip",
    "kv_cache.hip",
];

/// GPU targets to compile for (ROCm 6+/7 uses --offload-arch).
const OFFLOAD_ARCHES: &[&str] = &[
    "gfx1100",  // RX 7900 XTX/XT (RDNA3)
    "gfx1030",  // RX 6900/6800 XT (RDNA2)
    "gfx906",   // Instinct MI50/60 (Vega 20)
    "gfx90a",   // Instinct MI200 series (CDNA2)
    "gfx942",   // Instinct MI300X (CDNA3)
];

#[cfg(feature = "rocm")]
fn compile_hip_kernels(rocm_path: &str, kernel_dir: &PathBuf, out_dir: &PathBuf) {
    // Try hipcc first, fall back to direct clang++ with HIP flags
    let hipcc = format!("{rocm_path}/bin/hipcc");
    let hipcc_exists = std::path::Path::new(&hipcc).exists();

    if !hipcc_exists {
        eprintln!(
            "cargo:warning=hipcc not found at {hipcc}. \
             Set ROCM_PATH to your ROCm installation directory."
        );
        return;
    }

    let mut obj_files: Vec<PathBuf> = Vec::new();

    for kernel in KERNELS {
        let src = kernel_dir.join(kernel);
        let obj = out_dir.join(kernel.replace(".hip", ".o"));

        if !src.exists() {
            eprintln!("cargo:warning=Kernel source not found: {}", src.display());
            continue;
        }

        // Build --offload-arch args (one per target, no duplicates)
        let mut args: Vec<String> = Vec::new();
        for arch in OFFLOAD_ARCHES {
            args.push(format!("--offload-arch={arch}"));
        }
        args.extend([
            "-O3".into(),
            "-ffast-math".into(),
            "-fPIC".into(),
            format!("-I{rocm_path}/include"),
            "-c".into(),
            "-x".into(),
            "hip".into(),
            src.to_string_lossy().to_string(),
            "-o".into(),
            obj.to_string_lossy().to_string(),
        ]);

        let status = Command::new(&hipcc)
            .args(&args)
            .status();

        match status {
            Ok(s) if s.success() => {
                obj_files.push(obj);
            }
            Ok(s) => {
                eprintln!("cargo:warning=hipcc failed (exit {s}) for {kernel}");
            }
            Err(e) => {
                eprintln!("cargo:warning=Could not run hipcc: {e}");
            }
        }
    }

    if obj_files.is_empty() {
        eprintln!("cargo:warning=No kernel objects compiled — GPU acceleration unavailable");
        return;
    }

    // Archive compiled objects into a static library
    let lib_path = out_dir.join("libphysllm_kernels.a");
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .args(&obj_files)
        .status();

    if ar_status.map(|s| s.success()).unwrap_or(false) {
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=physllm_kernels");
    } else {
        eprintln!("cargo:warning=ar failed — kernels not linked as static lib");
    }
}

#[cfg(feature = "rocm")]
fn generate_hip_bindings(rocm_path: &str, out_dir: &PathBuf) {
    let header = format!("{rocm_path}/include/hip/hip_runtime_api.h");
    if !std::path::Path::new(&header).exists() {
        eprintln!(
            "cargo:warning=HIP header not found at {header}. \
             Skipping binding generation — using pre-generated stubs."
        );
        write_stub_bindings(out_dir);
        return;
    }

    // Find GCC include path to fix "limits.h not found" with bindgen's
    // embedded clang, which doesn't know about the system GCC installation.
    let gcc_includes = find_gcc_include_path();

    let mut builder = bindgen::Builder::default()
        .header(&header)
        .clang_arg(format!("-I{rocm_path}/include"))
        // Use the ROCm clang rather than the system one (important for HIP types)
        .clang_arg(format!("--rocm-path={rocm_path}"))
        // Fix: add GCC resource dir so clang finds limits.h, stddef.h, etc.
        .clang_arg(format!(
            "--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/{}",
            find_gcc_version()
        ))
        // Alternative fix for limits.h on systems where above doesn't work:
        .clang_arg("-D__GNUC__=13")
        .allowlist_function("hip.*")
        .allowlist_function("hipblas.*")
        .allowlist_function("rocblas.*")
        .allowlist_type("hip.*")
        .allowlist_type("hipblas.*")
        .allowlist_var("HIP.*")
        .allowlist_var("HIPBLAS.*")
        .derive_debug(true)
        .derive_default(true)
        // Avoid generating bindings for system headers (avoids limits.h issue entirely)
        .allowlist_recursively(false);

    // Add GCC include dirs as -isystem (lower priority, suppresses warnings)
    for path in &gcc_includes {
        builder = builder.clang_arg(format!("-isystem{path}"));
    }

    match builder.generate() {
        Ok(bindings) => {
            bindings
                .write_to_file(out_dir.join("hip_bindings.rs"))
                .expect("Failed to write HIP bindings");
        }
        Err(e) => {
            eprintln!("cargo:warning=bindgen failed: {e}. Using stub bindings.");
            write_stub_bindings(out_dir);
        }
    }
}

/// Find GCC include directories for the installed GCC version.
fn find_gcc_include_path() -> Vec<String> {
    let version = find_gcc_version();
    let mut paths = Vec::new();
    let candidates = [
        format!("/usr/lib/gcc/x86_64-linux-gnu/{version}/include"),
        format!("/usr/lib/gcc/x86_64-linux-gnu/{version}/include-fixed"),
        "/usr/include/x86_64-linux-gnu".into(),
        "/usr/include".into(),
    ];
    for p in candidates {
        if std::path::Path::new(&p).exists() { paths.push(p); }
    }
    paths
}

/// Detect installed GCC version number (e.g. "13").
fn find_gcc_version() -> String {
    // Try reading from gcc --version
    let output = Command::new("gcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    // Parse "gcc (Ubuntu 13.x.x-...) 13.x.x"
    for token in output.split_whitespace() {
        if token.starts_with(|c: char| c.is_ascii_digit()) {
            let major = token.split('.').next().unwrap_or("13");
            return major.to_string();
        }
    }

    // Fallback: scan /usr/lib/gcc/x86_64-linux-gnu/ for directories
    if let Ok(entries) = std::fs::read_dir("/usr/lib/gcc/x86_64-linux-gnu/") {
        let mut versions: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                if n.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                    Some(n)
                } else { None }
            })
            .collect();
        versions.sort();
        if let Some(v) = versions.last() { return v.clone(); }
    }

    "13".to_string() // safe fallback for Ubuntu 22.04/24.04
}

/// Write a minimal stub bindings file so the crate compiles even when
/// bindgen fails (e.g. on machines without ROCm installed).
fn write_stub_bindings(out_dir: &PathBuf) {
    let stubs = r#"
// Auto-generated stub bindings — real bindings require ROCm installed.
// The cpu-only feature flag disables all GPU code paths.
pub type hipStream_t  = *mut ::std::os::raw::c_void;
pub type hipblasHandle_t = *mut ::std::os::raw::c_void;

#[repr(C)] #[derive(Debug, Default)] pub struct hipDeviceProp_tR0600 {
    pub name:          [::std::os::raw::c_char; 256],
    pub gcnArchName:   [::std::os::raw::c_char; 256],
    pub totalGlobalMem: usize,
    pub multiProcessorCount: ::std::os::raw::c_int,
    pub warpSize:       ::std::os::raw::c_int,
    pub maxThreadsPerBlock: ::std::os::raw::c_int,
}

#[allow(non_camel_case_types)]
#[repr(u32)] #[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum hipMemcpyKind {
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
}

#[allow(non_camel_case_types)]
#[repr(u32)] pub enum hipblasOperation_t {
    HIPBLAS_OP_N = 111,
    HIPBLAS_OP_T = 112,
}
pub use hipblasOperation_t::*;

extern "C" {
    pub fn hipSetDevice(device_id: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
    pub fn hipGetDeviceProperties_v2(props: *mut hipDeviceProp_tR0600, device: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> ::std::os::raw::c_int;
    pub fn hipStreamDestroy(stream: hipStream_t) -> ::std::os::raw::c_int;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> ::std::os::raw::c_int;
    pub fn hipMalloc(ptr: *mut *mut ::std::os::raw::c_void, size: usize) -> ::std::os::raw::c_int;
    pub fn hipFree(ptr: *mut ::std::os::raw::c_void) -> ::std::os::raw::c_int;
    pub fn hipMemcpy(dst: *mut ::std::os::raw::c_void, src: *const ::std::os::raw::c_void, size: usize, kind: hipMemcpyKind) -> ::std::os::raw::c_int;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> ::std::os::raw::c_int;
    pub fn hipblasCreate(handle: *mut hipblasHandle_t) -> ::std::os::raw::c_int;
    pub fn hipblasDestroy(handle: hipblasHandle_t) -> ::std::os::raw::c_int;
    pub fn hipblasHgemm(
        handle: hipblasHandle_t,
        transa: hipblasOperation_t, transb: hipblasOperation_t,
        m: ::std::os::raw::c_int, n: ::std::os::raw::c_int, k: ::std::os::raw::c_int,
        alpha: *const half::f16,
        A: *const half::f16, lda: ::std::os::raw::c_int,
        B: *const half::f16, ldb: ::std::os::raw::c_int,
        beta:  *const half::f16,
        C: *mut half::f16, ldc: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
"#;
    std::fs::write(out_dir.join("hip_bindings.rs"), stubs)
        .expect("Failed to write stub bindings");
}

/// Create any missing kernel source files with minimal valid HIP stubs.
/// This prevents the "no such file or directory" error while you add the
/// real implementations.
fn ensure_kernels_exist(kernel_dir: &PathBuf) {
    std::fs::create_dir_all(kernel_dir).ok();

    let missing_kernels: &[(&str, &str)] = &[
        ("gemm_f16.hip", GEMM_STUB),
        ("softmax.hip",  SOFTMAX_STUB),
        ("kv_cache.hip", KV_CACHE_STUB),
    ];

    for (filename, source) in missing_kernels {
        let path = kernel_dir.join(filename);
        if !path.exists() {
            eprintln!("cargo:warning=Creating stub kernel: {filename}");
            std::fs::write(&path, source)
                .unwrap_or_else(|e| eprintln!("cargo:warning=Could not write {filename}: {e}"));
        }
    }
}

// ── Stub kernel sources (compile cleanly, no-ops) ────────────────────────────

const GEMM_STUB: &str = r#"
// gemm_f16.hip — f16 GEMM via rocBLAS (stub).
// Real implementation delegates to hipblasHgemm in ops.rs.
// This file exists so the build system can compile the kernel list.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// No custom GEMM kernel needed — we use hipblasHgemm directly from Rust.
// This stub satisfies the build dependency.
extern "C" void gemm_f16_noop() {}
"#;

const SOFTMAX_STUB: &str = r#"
// softmax.hip — Fused online softmax kernel.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>

#define WARP_SIZE 64

__device__ float warp_max(float val) {
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor(val, mask));
    return val;
}
__device__ float warp_sum(float val) {
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
        val += __shfl_xor(val, mask);
    return val;
}

// In-place row-wise softmax on f16 tensor [rows, cols].
__global__ void softmax_f16_kernel(
    __half* __restrict__ x,
    int rows, int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __half* row_ptr = x + row * cols;

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        max_val = fmaxf(max_val, __half2float(row_ptr[i]));
    max_val = warp_max(max_val);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(__half2float(row_ptr[i]) - max_val);
        row_ptr[i] = __float2half(e);
        sum += e;
    }
    sum = warp_sum(sum);

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_ptr[i] = __float2half(__half2float(row_ptr[i]) / sum);
}

extern "C" void softmax_f16(hipStream_t stream, __half* x, int rows, int cols) {
    hipLaunchKernelGGL(softmax_f16_kernel,
        dim3(rows), dim3(WARP_SIZE), 0, stream, x, rows, cols);
}
"#;

const KV_CACHE_STUB: &str = r#"
// kv_cache.hip — KV cache append and gather operations.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Append new K/V slice to the cache at position `offset`.
// cache:   [batch, heads, max_seq, head_dim]
// new_kv:  [batch, heads, new_tokens, head_dim]
__global__ void kv_cache_append_kernel(
    __half* __restrict__       cache,
    const __half* __restrict__ new_kv,
    int batch, int heads, int max_seq, int head_dim,
    int offset, int new_tokens
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * heads * new_tokens * head_dim;
    if (idx >= total) return;

    const int d   =  idx % head_dim;
    const int tok = (idx / head_dim) % new_tokens;
    const int h   = (idx / head_dim / new_tokens) % heads;
    const int b   =  idx / head_dim / new_tokens / heads;

    const int cache_idx =
        ((b * heads + h) * max_seq + (offset + tok)) * head_dim + d;
    cache[cache_idx] = new_kv[idx];
}

extern "C" void kv_cache_append(
    hipStream_t stream,
    __half* cache, const __half* new_kv,
    int batch, int heads, int max_seq, int head_dim,
    int offset, int new_tokens
) {
    const int total   = batch * heads * new_tokens * head_dim;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    hipLaunchKernelGGL(
        kv_cache_append_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        cache, new_kv, batch, heads, max_seq, head_dim, offset, new_tokens
    );
}
"#;
