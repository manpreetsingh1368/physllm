//! GPU-accelerated attention operations

use crate::{DeviceTensor, GpuDevice, Result};
use half::f16;

#[cfg(feature = "rocm")]
extern "C" {
    fn launch_flash_attention(
        q: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut std::ffi::c_void,
        batch: i32,
        num_heads: i32,
        num_kv_heads: i32,
        seq_q: i32,
        seq_kv: i32,
        head_dim: i32,
        scale: f32,
        stream: *mut std::ffi::c_void,
    );
}

/// GPU flash attention: O = softmax(Q @ K^T / sqrt(d)) @ V
pub fn flash_attention_gpu(
    dev: &GpuDevice,
    q: &DeviceTensor<f16>,      // [batch, num_heads, seq_q, head_dim]
    k: &DeviceTensor<f16>,      // [batch, num_kv_heads, seq_kv, head_dim]
    v: &DeviceTensor<f16>,      // [batch, num_kv_heads, seq_kv, head_dim]
    output: &mut DeviceTensor<f16>, // [batch, num_heads, seq_q, head_dim]
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
) -> Result<()> {
    
    #[cfg(feature = "rocm")]
    unsafe {
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        launch_flash_attention(
            q.as_ptr() as *const std::ffi::c_void,
            k.as_ptr() as *const std::ffi::c_void,
            v.as_ptr() as *const std::ffi::c_void,
            output.as_mut_ptr() as *mut std::ffi::c_void,
            batch as i32,
            num_heads as i32,
            num_kv_heads as i32,
            seq_q as i32,
            seq_kv as i32,
            head_dim as i32,
            scale,
            std::ptr::null_mut(), // Default stream
        );
        
        // Synchronize to ensure completion
        use crate::hip_ffi::*;
        hipDeviceSynchronize();
    }
    
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (dev, q, k, v, output, batch, num_heads, num_kv_heads, seq_q, seq_kv, head_dim);
        return Err(crate::BackendError::Runtime("ROCm not enabled".into()));
    }
    
    Ok(())
}
