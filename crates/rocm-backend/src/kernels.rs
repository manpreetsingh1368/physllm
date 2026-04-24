pub const KERNEL_NAMES: &[&str] = &[
    "gemm_f16_noop",
    "softmax_f16",
    "kv_cache_append",
    "flash_attention",
    "rope_embed",
    "rms_norm",
];

/// Extern declarations for the HIP kernels (used when rocm feature is active).
#[cfg(feature = "rocm")]
extern "C" {
    pub fn softmax_f16(
        stream: *mut std::ffi::c_void,
        x: *mut half::f16,
        rows: i32, cols: i32,
    );

    pub fn kv_cache_append(
        stream: *mut std::ffi::c_void,
        cache: *mut half::f16, new_kv: *const half::f16,
        batch: i32, heads: i32, max_seq: i32, head_dim: i32,
        offset: i32, new_tokens: i32,
    );

    pub fn flash_attention(
        stream: *mut std::ffi::c_void,
        q: *const half::f16, k: *const half::f16, v: *const half::f16,
        o: *mut half::f16,
        seq: i32, d: i32, scale: f32, causal: i32,
    );

    pub fn rope_embed(
        stream: *mut std::ffi::c_void,
        x: *mut half::f16,
        seq: i32, heads: i32, head_dim: i32, offset: i32, theta: f32,
    );

    pub fn rms_norm(
        stream: *mut std::ffi::c_void,
        x: *mut half::f16,
        weight: *const half::f16,
        rows: i32, cols: i32, eps: f32,
    );
}
