//! rocm-backend — AMD GPU acceleration layer for PhysLLM.

pub mod device;
pub mod tensor;
pub mod ops;
pub mod memory;
pub mod kernels;
pub mod runtime;

pub use device::{GpuDevice, GpuProperties};
pub use tensor::DeviceTensor;
pub use memory::MemoryPool;
pub use ops::{matmul_f16, flash_attention, rope_embed, rms_norm};

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
    #[error("{0}")]
    Other(String),
    #[error("Runtime error: {0}")]
    Runtime(String),
}
pub type Result<T> = std::result::Result<T, BackendError>;
// HIP EFI hip_bindings
#[allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]
pub mod hip_ffi {
    include!(concat!(env!("OUT_DIR"), "/hip_bindings.rs"));
}
pub mod attention_ops;
