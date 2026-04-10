//! ops.rs — High-level GPU operation dispatch.
//!
//! These functions call our compiled HIP kernels (or rocBLAS) for each op.
//! CPU fallback paths use ndarray + rayon.

use crate::{DeviceTensor, GpuDevice, BackendError, Result};
use half::f16;
use tracing::trace;

// ── Matrix Multiply (GEMM) ────────────────────────────────────────────────────

/// C = A @ B  (f16, column-major, via rocBLAS hipblasGemmEx)
pub fn matmul_f16(
    dev: &GpuDevice,
    a:   &DeviceTensor<f16>,   // [M, K]
    b:   &DeviceTensor<f16>,   // [K, N]
    c:   &mut DeviceTensor<f16>, // [M, N]
) -> Result<()> {
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();

    trace!("matmul_f16 [{m}x{k}] @ [{k}x{n}]");

    if b.rows() != k {
        return Err(BackendError::ShapeMismatch(
            format!("matmul: a.cols ({k}) != b.rows ({})", b.rows())
        ));
    }
// GPU safety (not test properly yet)
    #[cfg(feature = "rocm")]

    unsafe {
        use crate::hip_ffi::*;
        // rocBLAS is column-major; transpose A and B so we get row-major result
        let alpha = f16::from_f32(1.0);
        let beta  = f16::from_f32(0.0);
        // Use hipblas for high-level BLAS
        let handle = get_or_create_hipblas_handle(dev)?;

        let err = hipblasHgemm(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            n as i32, m as i32, k as i32,
            &alpha as *const _ as *const _,
            b.raw_ptr() as *const _,  n as i32,
            a.raw_ptr() as *const _,  k as i32,
            &beta  as *const _ as *const _,
            c.raw_ptr() as *mut _,    n as i32,
        );
        if err != 0 {
            return Err(BackendError::Hip { code: err, msg: "hipblasHgemm".into() });
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        // CPU fallback via ndarray
        use ndarray::Array2;
        let a_host = a.copy_to_host()?
            .iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        let b_host = b.copy_to_host()?
            .iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        let a_arr = Array2::from_shape_vec((m, k), a_host).unwrap();
        let b_arr = Array2::from_shape_vec((k, n), b_host).unwrap();
        let c_arr = a_arr.dot(&b_arr);
        let c_f16: Vec<f16> = c_arr.iter().map(|&x| f16::from_f32(x)).collect();
        c.copy_from_host(&c_f16)?;
    }

    Ok(())
}

// ── Flash Attention ───────────────────────────────────────────────────────────

/// Flash Attention 2 forward pass (f16).
///
/// # Arguments
/// * `q`    — [batch, heads, seq_q, head_dim]
/// * `k`    — [batch, heads, seq_k, head_dim]
/// * `v`    — [batch, heads, seq_k, head_dim]
/// * `out`  — [batch, heads, seq_q, head_dim]
/// * `scale`— 1/sqrt(head_dim)
/// * `causal` — apply causal mask
pub fn flash_attention(
    dev:    &GpuDevice,
    q:      &DeviceTensor<f16>,
    k:      &DeviceTensor<f16>,
    v:      &DeviceTensor<f16>,
    out:    &mut DeviceTensor<f16>,
    scale:  f32,
    causal: bool,
) -> Result<()> {
  let (b,h,sq,sk,d) = validate_attention(q,k,v,out)?;
    //trace!("flash_attn batch={batch} heads={heads} seq_q={seq_q} seq_k={seq_k} d={head_dim}");
// just update in name here
        trace!(
            "flash_attention b={b} h={h} sq={sq} sk={sk} d={d} causal={causal}"
        );

    #[cfg(feature = "rocm")]
    unsafe {
        // no aliasing
        // all tensors validated 
        // Calling  compiled flash_attention.hip kernel
        flash_attention_hip(
            dev.raw_stream(),
            q.raw_ptr(), 
            k.raw_ptr(),
             v.raw_ptr(), 
             out.raw_ptr(),
          b as i32,
          h as i32,
          sq as i32,
          sk as i32,
          d as i32,
          scale,
          causal as i32,
        );
        hip_check("flash_attention")?;
        #[cfg(debug_assertions)]
        hip_sync(dev)?;
    }

    #[cfg(not(feature = "rocm"))]
    {
        // Naive O(seq²) CPU attention for fallback/testing
        cpu_naive_attention(q, k, v, out, scale, causal)?;
    }

    Ok(())
}

// ── RoPE Embeddings 
/// Rotary Position Embedding in-place on Q and K tensors.
pub fn rope_embed(
    dev: &GpuDevice,
    q: &mut DeviceTensor<f16>,
    k: &mut DeviceTensor<f16>,
    seq_offset:  usize,
    theta:       f32,
) -> Result<()> {
    let head_dim = *q.shape().last().unwrap_or(&0);
    let seq_len  = q.shape().get(q.shape().len().saturating_sub(2)).copied().unwrap_or(1);
    trace!("rope seq_len={seq_len} head_dim={head_dim} offset={seq_offset}");

    #[cfg(feature = "rocm")]
    unsafe {
        rope_embedding_hip(
            dev.raw_stream(),
            q.raw_ptr(), k.raw_ptr(),
            seq_len as i32, head_dim as i32,
            seq_offset as i32, theta,
            q.numel() as i32,
        );
    }

    #[cfg(not(feature = "rocm"))]
    cpu_rope(q, k, seq_offset, theta)?;

    Ok(())
}

// ── RMS Normalisation ─────────────────────────────────────────────────────────

/// RMSNorm(x, weight) in-place. Common in Llama-style models.
pub fn rms_norm(
    dev:     &GpuDevice,
    x:       &mut DeviceTensor<f16>,
    weight:  &DeviceTensor<f16>,
    eps:     f32,
) -> Result<()> {
    #[cfg(feature = "rocm")]
    unsafe {
        let rows = x.rows();
        let cols = x.cols();
        layer_norm_hip(
            dev.raw_stream(),
            x.raw_ptr(), weight.raw_ptr(), std::ptr::null(),
            rows as i32, cols as i32, eps, 1 /* rms_only */,
        );
    }

    #[cfg(not(feature = "rocm"))]
    cpu_rms_norm(x, weight, eps)?;

    Ok(())
}

// ── HIP kernel extern declarations ───────────────────────────────────────────

#[cfg(feature = "rocm")]
extern "C" {
    fn flash_attention_hip(
        stream: *mut std::ffi::c_void,
        q: *const f16, k: *const f16, v: *const f16, out: *mut f16,
        batch: i32, heads: i32, seq_q: i32, seq_k: i32, head_dim: i32,
        scale: f32, causal: i32,
    );

    fn rope_embedding_hip(
        stream: *mut std::ffi::c_void,
        q: *mut f16, k: *mut f16,
        seq_len: i32, head_dim: i32,
        offset: i32, theta: f32, numel: i32,
    );

    fn layer_norm_hip(
        stream: *mut std::ffi::c_void,
        x: *mut f16, gamma: *const f16, beta: *const f16,
        rows: i32, cols: i32, eps: f32, rms_only: i32,
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn extract_4d_shape(t: &DeviceTensor<f16>) -> Result<[usize; 4]> {
    if t.shape().len() != 4 {
        return Err(BackendError::ShapeMismatch(
            format!("expected 4D tensor, got {:?}", t.shape())
        ));
    }
    Ok([t.shape()[0], t.shape()[1], t.shape()[2], t.shape()[3]])
}

#[cfg(not(feature = "rocm"))]
fn cpu_naive_attention(
    q: &DeviceTensor<f16>, k: &DeviceTensor<f16>, v: &DeviceTensor<f16>,
    out: &mut DeviceTensor<f16>, scale: f32, causal: bool,
) -> Result<()> {
    // Simplified single-head implementation for testing
    let seq = q.shape()[q.shape().len() - 2];
    let d   = *q.shape().last().unwrap();
    let q_host: Vec<f32> = q.copy_to_host()?.iter().map(|x| x.to_f32()).collect();
    let k_host: Vec<f32> = k.copy_to_host()?.iter().map(|x| x.to_f32()).collect();
    let v_host: Vec<f32> = v.copy_to_host()?.iter().map(|x| x.to_f32()).collect();

    let mut o_host = vec![0f32; q.numel()];

    for i in 0..seq {
        let mut scores = vec![f32::NEG_INFINITY; seq];
        for j in 0..seq {
            if causal && j > i { continue; }
            let dot: f32 = (0..d).map(|dd| q_host[i*d+dd] * k_host[j*d+dd]).sum();
            scores[j] = dot * scale;
        }
        // softmax
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let weights: Vec<f32> = exp.iter().map(|&e| e / sum).collect();
        // weighted sum
        for dd in 0..d {
            o_host[i*d+dd] = (0..seq).map(|j| weights[j] * v_host[j*d+dd]).sum();
        }
    }

    let o_f16: Vec<f16> = o_host.iter().map(|&x| f16::from_f32(x)).collect();
    out.copy_from_host(&o_f16)?;
    Ok(())
}

#[cfg(not(feature = "rocm"))]
fn cpu_rope(
    q: &mut DeviceTensor<f16>, k: &mut DeviceTensor<f16>,
    offset: usize, theta: f32,
) -> Result<()> {
    let apply = |t: &mut DeviceTensor<f16>| -> Result<()> {
        let head_dim = *t.shape().last().unwrap();
        let seq = t.shape()[t.shape().len().saturating_sub(2)];
        let mut host: Vec<f32> = t.copy_to_host()?.iter().map(|x| x.to_f32()).collect();
        for s in 0..seq {
            let pos = (s + offset) as f32;
            for i in 0..head_dim / 2 {
                let freq = pos / theta.powf(2.0 * i as f32 / head_dim as f32);
                let (sin, cos) = freq.sin_cos();
                let idx = s * head_dim + i * 2;
                if idx + 1 < host.len() {
                    let (x0, x1) = (host[idx], host[idx + 1]);
                    host[idx]   = x0 * cos - x1 * sin;
                    host[idx+1] = x0 * sin + x1 * cos;
                }
            }
        }
        let f16s: Vec<f16> = host.iter().map(|&x| f16::from_f32(x)).collect();
        t.copy_from_host(&f16s)?;
        Ok(())
    };
    apply(q)?;
    apply(k)?;
    Ok(())
}

#[cfg(not(feature = "rocm"))]
fn cpu_rms_norm(
    x: &mut DeviceTensor<f16>, weight: &DeviceTensor<f16>, eps: f32,
) -> Result<()> {
    let d = x.cols();
    let mut host: Vec<f32> = x.copy_to_host()?.iter().map(|v| v.to_f32()).collect();
    let w: Vec<f32> = weight.copy_to_host()?.iter().map(|v| v.to_f32()).collect();
    for row in host.chunks_mut(d) {
        let rms = (row.iter().map(|&v| v*v).sum::<f32>() / d as f32 + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() { *v = *v / rms * w[i]; }
    }
    let f16s: Vec<f16> = host.iter().map(|&x| f16::from_f32(x)).collect();
    x.copy_from_host(&f16s)?;
    Ok(())
}

// Stub — in real code this would cache the handle in thread-local storage
#[cfg(feature = "rocm")]
unsafe fn get_or_create_hipblas_handle(
    _dev: &GpuDevice,
) -> Result<crate::hip_ffi::hipblasHandle_t> {
    let mut handle = std::ptr::null_mut();
    crate::hip_ffi::hipblasCreate(&mut handle);
    Ok(handle)
}
