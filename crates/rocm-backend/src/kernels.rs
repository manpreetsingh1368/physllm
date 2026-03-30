//! kernels.rs — Kernel registry and launch helpers.

/// List of all compiled HIP kernel names.
pub const KERNEL_NAMES: &[&str] = &[
    "flash_attention_kernel",
    "rope_kernel",
    "layer_norm_kernel",
];
