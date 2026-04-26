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
    let [batch, heads, seq_q, head_dim] = extract_4d_shape(q)?;
    let [_, _, seq_k, _]                = extract_4d_shape(k)?;
    trace!("flash_attn batch={batch} heads={heads} seq_q={seq_q} seq_k={seq_k} d={head_dim}");

    #[cfg(feature = "rocm")]
    unsafe {
        // Call our compiled flash_attention.hip kernel
        flash_attention_hip(
            dev.raw_stream(),
            q.raw_ptr(), k.raw_ptr(), v.raw_ptr(), out.raw_ptr(),
            batch as i32, heads as i32,
            seq_q as i32, seq_k as i32, head_dim as i32,
            scale,
            causal as i32,
        );
    }

    #[cfg(not(feature = "rocm"))]
    {
        // Naive O(seq²) CPU attention for fallback/testing
        cpu_naive_attention(q, k, v, out, scale, causal)?;
    }

    Ok(())
}

// ── RoPE Embeddings ───────────────────────────────────────────────────────────

/// Rotary Position Embedding in-place on Q and K tensors.
pub fn rope_embed(
    dev:         &GpuDevice,
    q:           &mut DeviceTensor<f16>,
    k:           &mut DeviceTensor<f16>,
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

// ══════════════════════════════════════════════════════════════════════════════
// NEW GPU-ONLY OPERATIONS (added for 100% GPU compute)
// ══════════════════════════════════════════════════════════════════════════════

// ── GPU RMS Norm ──────────────────────────────────────────────────────────────

#[cfg(any(feature = "rocm", feature = "cuda"))]
extern "C" {
    fn launch_rms_norm(
        input: *const std::ffi::c_void, weight: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch_size: i32, hidden_dim: i32, eps: f32, stream: *mut std::ffi::c_void,
    );
    fn launch_rope_gpu(
        qk: *mut std::ffi::c_void,
        num_heads: i32, seq_len: i32, head_dim: i32,
        position_offset: i32, theta: f32, stream: *mut std::ffi::c_void,
    );
    fn launch_silu_multiply(
        gate: *const std::ffi::c_void, up: *const std::ffi::c_void,
        output: *mut std::ffi::c_void, size: i32, stream: *mut std::ffi::c_void,
    );
    fn launch_residual_add(
        output: *mut std::ffi::c_void, residual: *const std::ffi::c_void,
        size: i32, stream: *mut std::ffi::c_void,
    );
    fn launch_embedding_lookup(
        embed_table: *const std::ffi::c_void, token_ids: *const std::ffi::c_void,
        output: *mut std::ffi::c_void, hidden_dim: i32, num_tokens: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_lm_head(
        hidden: *const std::ffi::c_void, weight: *const std::ffi::c_void,
        logits: *mut std::ffi::c_void, hidden_dim: i32, vocab_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_softmax_sample(
        logits: *mut std::ffi::c_void, output_token: *mut std::ffi::c_void,
        vocab_size: i32, temperature: f32, seed: u64, stream: *mut std::ffi::c_void,
    );
    fn launch_kv_cache_update(
        cache: *mut std::ffi::c_void, new_kv: *const std::ffi::c_void,
        position: i32, num_heads: i32, head_dim: i32, stream: *mut std::ffi::c_void,
    );
    fn launch_flash_attention_v2(
        q: *const std::ffi::c_void, k: *const std::ffi::c_void,
        v: *const std::ffi::c_void, o: *mut std::ffi::c_void,
        batch: i32, num_heads: i32, num_kv_heads: i32,
        seq_q: i32, seq_kv: i32, head_dim: i32, scale: f32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_adam_update(
        param: *mut std::ffi::c_void, grad: *const std::ffi::c_void,
        m: *mut std::ffi::c_void, v: *mut std::ffi::c_void,
        lr: f32, beta1: f32, beta2: f32, epsilon: f32,
        weight_decay: f32, numel: i32, stream: *mut std::ffi::c_void,
    );
}

/// GPU RMS normalization
pub fn rms_norm_gpu(
    dev: &GpuDevice,
    input: &DeviceTensor<f16>,
    weight: &DeviceTensor<f16>,
    output: &mut DeviceTensor<f16>,
    eps: f32,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        let batch = input.rows() as i32;
        let hidden = input.cols() as i32;
        launch_rms_norm(
            input.raw_ptr() as *const _, weight.raw_ptr() as *const _,
            output.raw_ptr() as *mut _, batch, hidden, eps, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// GPU RoPE application
pub fn rope_gpu(
    dev: &GpuDevice,
    qk: &mut DeviceTensor<f16>,
    num_heads: usize, seq_len: usize, head_dim: usize,
    position_offset: usize, theta: f32,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_rope_gpu(
            qk.raw_ptr() as *mut _,
            num_heads as i32, seq_len as i32, head_dim as i32,
            position_offset as i32, theta, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// GPU SiLU activation with element-wise multiply
pub fn silu_multiply_gpu(
    dev: &GpuDevice,
    gate: &DeviceTensor<f16>,
    up: &DeviceTensor<f16>,
    output: &mut DeviceTensor<f16>,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_silu_multiply(
            gate.raw_ptr() as *const _, up.raw_ptr() as *const _,
            output.raw_ptr() as *mut _, gate.numel() as i32, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// GPU residual addition: output += residual
pub fn residual_add_gpu(
    dev: &GpuDevice,
    output: &mut DeviceTensor<f16>,
    residual: &DeviceTensor<f16>,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_residual_add(
            output.raw_ptr() as *mut _, residual.raw_ptr() as *const _,
            output.numel() as i32, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// GPU embedding lookup
pub fn embedding_gpu(
    dev: &GpuDevice,
    embed_table: &DeviceTensor<f16>,
    token_ids: &DeviceTensor<i32>,
    output: &mut DeviceTensor<f16>,
    hidden_dim: usize,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_embedding_lookup(
            embed_table.raw_ptr() as *const _, token_ids.raw_ptr() as *const _,
            output.raw_ptr() as *mut _, hidden_dim as i32,
            token_ids.numel() as i32, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// GPU LM head projection: logits = hidden @ lm_head_weight^T
pub fn lm_head_gpu(
    dev: &GpuDevice,
    hidden: &DeviceTensor<f16>,
    weight: &DeviceTensor<f16>,
    logits: &mut DeviceTensor<f32>,
    hidden_dim: usize, vocab_size: usize,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_lm_head(
            hidden.raw_ptr() as *const _, weight.raw_ptr() as *const _,
            logits.raw_ptr() as *mut _, hidden_dim as i32,
            vocab_size as i32, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// GPU softmax + sampling
pub fn softmax_sample_gpu(
    dev: &GpuDevice,
    logits: &mut DeviceTensor<f32>,
    output_token: &mut DeviceTensor<i32>,
    vocab_size: usize, temperature: f32, seed: u64,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_softmax_sample(
            logits.raw_ptr() as *mut _, output_token.raw_ptr() as *mut _,
            vocab_size as i32, temperature, seed, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// GPU KV cache update
pub fn kv_cache_update_gpu(
    dev: &GpuDevice,
    cache: &mut DeviceTensor<f16>,
    new_kv: &DeviceTensor<f16>,
    position: usize, num_heads: usize, head_dim: usize,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_kv_cache_update(
            cache.raw_ptr() as *mut _, new_kv.raw_ptr() as *const _,
            position as i32, num_heads as i32, head_dim as i32, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// GPU Flash Attention v2
pub fn flash_attention_v2(
    dev: &GpuDevice,
    q: &DeviceTensor<f16>, k: &DeviceTensor<f16>,
    v: &DeviceTensor<f16>, out: &mut DeviceTensor<f16>,
    num_heads: usize, num_kv_heads: usize,
    seq_q: usize, seq_kv: usize, head_dim: usize,
) -> Result<()> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_flash_attention_v2(
            q.raw_ptr() as *const _, k.raw_ptr() as *const _,
            v.raw_ptr() as *const _, out.raw_ptr() as *mut _,
            1, num_heads as i32, num_kv_heads as i32,
            seq_q as i32, seq_kv as i32, head_dim as i32,
            scale, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// GPU Adam optimizer step
pub fn adam_update_gpu(
    dev: &GpuDevice,
    param: &mut DeviceTensor<f16>, grad: &DeviceTensor<f16>,
    m: &mut DeviceTensor<f16>, v: &mut DeviceTensor<f16>,
    lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_adam_update(
            param.raw_ptr() as *mut _, grad.raw_ptr() as *const _,
            m.raw_ptr() as *mut _, v.raw_ptr() as *mut _,
            lr, beta1, beta2, epsilon, weight_decay,
            param.numel() as i32, dev.raw_stream(),
        );
    }
    dev.synchronise()
}

// ══════════════════════════════════════════════════════════════════════════════
// MIXTURE-OF-EXPERTS (MoE) OPERATIONS
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(any(feature = "rocm", feature = "cuda"))]
extern "C" {
    fn launch_moe_router(
        hidden: *const std::ffi::c_void,
        router_weight: *const std::ffi::c_void,
        expert_indices: *mut std::ffi::c_void,
        expert_weights: *mut std::ffi::c_void,
        batch_size: i32, hidden_dim: i32,
        num_experts: i32, top_k: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_moe_expert_silu(
        input: *const std::ffi::c_void,
        w_gate: *const std::ffi::c_void,
        w_up: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        hidden_dim: i32, intermediate_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_moe_expert_down(
        input: *const std::ffi::c_void,
        w_down: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        hidden_dim: i32, intermediate_dim: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_moe_combine(
        expert_outputs: *const std::ffi::c_void,
        expert_weights: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        hidden_dim: i32, top_k: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mxfp4_dequant(
        packed_data: *const std::ffi::c_void,
        scales: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        numel: i32, block_size: i32,
        stream: *mut std::ffi::c_void,
    );
}

/// MoE Router: select top-k experts for each token
pub fn moe_router_gpu(
    dev: &GpuDevice,
    hidden: &DeviceTensor<f16>,
    router_weight: &DeviceTensor<f16>,
    expert_indices: &mut DeviceTensor<i32>,
    expert_weights: &mut DeviceTensor<f32>,
    num_experts: usize,
    top_k: usize,
) -> Result<()> {
    let batch = hidden.rows() as i32;
    let hidden_dim = hidden.cols() as i32;
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_moe_router(
            hidden.raw_ptr() as *const _, router_weight.raw_ptr() as *const _,
            expert_indices.raw_ptr() as *mut _, expert_weights.raw_ptr() as *mut _,
            batch, hidden_dim, num_experts as i32, top_k as i32,
            dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// Run a single MoE expert's SwiGLU forward pass
pub fn moe_expert_forward_gpu(
    dev: &GpuDevice,
    input: &DeviceTensor<f16>,
    w_gate: &DeviceTensor<f16>,
    w_up: &DeviceTensor<f16>,
    w_down: &DeviceTensor<f16>,
    intermediate: &mut DeviceTensor<f16>,
    output: &mut DeviceTensor<f16>,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Result<()> {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        // SiLU(gate) * up → intermediate
        launch_moe_expert_silu(
            input.raw_ptr() as *const _,
            w_gate.raw_ptr() as *const _, w_up.raw_ptr() as *const _,
            intermediate.raw_ptr() as *mut _,
            hidden_dim as i32, intermediate_dim as i32,
            dev.raw_stream(),
        );
        // Down projection → output
        launch_moe_expert_down(
            intermediate.raw_ptr() as *const _,
            w_down.raw_ptr() as *const _,
            output.raw_ptr() as *mut _,
            hidden_dim as i32, intermediate_dim as i32,
            dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// Combine top-k expert outputs with routing weights
pub fn moe_combine_gpu(
    dev: &GpuDevice,
    expert_outputs: &DeviceTensor<f16>,  // [top_k, hidden_dim]
    expert_weights: &DeviceTensor<f32>,  // [top_k]
    output: &mut DeviceTensor<f16>,      // [hidden_dim]
    top_k: usize,
) -> Result<()> {
    let hidden_dim = output.numel() as i32;
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_moe_combine(
            expert_outputs.raw_ptr() as *const _,
            expert_weights.raw_ptr() as *const _,
            output.raw_ptr() as *mut _,
            hidden_dim, top_k as i32,
            dev.raw_stream(),
        );
    }
    dev.synchronise()
}

/// MXFP4 dequantization: convert packed 4-bit weights to FP16
pub fn mxfp4_dequant_gpu(
    dev: &GpuDevice,
    packed: &DeviceTensor<u8>,    // Packed FP4 data
    scales: &DeviceTensor<f16>,   // Per-block scales
    output: &mut DeviceTensor<f16>,
    block_size: usize,
) -> Result<()> {
    let numel = output.numel() as i32;
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    unsafe {
        launch_mxfp4_dequant(
            packed.raw_ptr() as *const _,
            scales.raw_ptr() as *const _,
            output.raw_ptr() as *mut _,
            numel, block_size as i32,
            dev.raw_stream(),
        );
    }
    dev.synchronise()
}
