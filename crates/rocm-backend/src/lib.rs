//! rocm-backend — AMD GPU acceleration layer for PhysLLM.
//!
//! Architecture:
//!   GpuDevice (HIP device handle)
//!     ├── TensorBuffer  (device-side memory, typed)
//!     ├── KernelRunner  (dispatch GEMM / attention / etc.)
//!     └── MemoryPool    (pre-allocated pool, avoids hipMalloc overhead)

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

pub mod device;
pub mod kernels;
pub mod memory;
pub mod ops;
pub mod tensor;

#[cfg(feature = "rocm")]
pub mod hip_ffi {
    include!(concat!(env!("OUT_DIR"), "/hip_bindings.rs"));
}

pub use device::GpuDevice;
pub use memory::MemoryPool;
pub use ops::{matmul_f16, flash_attention, rope_embed, rms_norm};
pub use tensor::DeviceTensor;

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
