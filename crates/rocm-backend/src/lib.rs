//! rocm-backend — GPU acceleration for PhysLLM.
//! Supports AMD (ROCm/HIP) and NVIDIA (CUDA via HIP).

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

pub mod device;
pub mod kernels;
pub mod memory;
pub mod hipblas_ffi;
pub mod ops;
pub mod tensor;

#[cfg(any(feature = "rocm", feature = "cuda"))]
pub mod hip_ffi {
    include!(concat!(env!("OUT_DIR"), "/hip_bindings.rs"));
}

pub use device::GpuDevice;
pub use memory::MemoryPool;
pub use tensor::DeviceTensor;

// Export ALL operations (old + new GPU ops)
pub use ops::{
    matmul_f16, flash_attention, rope_embed, rms_norm,
    // New 100% GPU operations
    rms_norm_gpu, rope_gpu, silu_multiply_gpu, residual_add_gpu,
    embedding_gpu, lm_head_gpu, softmax_sample_gpu,
    kv_cache_update_gpu, flash_attention_v2, adam_update_gpu,
    // MoE operations
    moe_router_gpu, moe_expert_forward_gpu, moe_combine_gpu, mxfp4_dequant_gpu,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("HIP error {code}: {msg}")]
    Hip { code: i32, msg: String },
    #[error("GPU device not found: {0}")]
    DeviceNotFound(String),
    #[error("Out of GPU memory: requested {requested_mb}MB, available {available_mb}MB")]
    OutOfMemory { requested_mb: usize, available_mb: usize },
    #[error("Kernel launch failed: {0}")]
    KernelLaunch(String),
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}

pub type Result<T> = std::result::Result<T, BackendError>;
