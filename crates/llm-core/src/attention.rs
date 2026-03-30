//! attention.rs — Grouped Query Attention (GQA) module.
//! Full implementation is integrated into model.rs transformer_block().
//! This module exposes the standalone attention fn for unit testing.

use crate::{config::ModelConfig, Result};
use rocm_backend::{GpuDevice, DeviceTensor, flash_attention, rope_embed};
use half::f16;

/// Compute GQA attention for a single layer.
/// Handles head-count mismatch between Q (num_heads) and K/V (num_kv_heads)
/// by repeating K/V heads to match Q (KV-head broadcasting).
pub fn grouped_query_attention(
    device:     &GpuDevice,
    q:          &mut DeviceTensor<f16>,   // [batch, num_heads, seq, head_dim]
    k:          &mut DeviceTensor<f16>,   // [batch, num_kv_heads, seq, head_dim]
    v:          &mut DeviceTensor<f16>,   // [batch, num_kv_heads, seq, head_dim]
    config:     &ModelConfig,
    seq_offset: usize,
    causal:     bool,
) -> Result<DeviceTensor<f16>> {
    let [batch, num_heads, seq, head_dim] = [
        q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]
    ];

    // Apply RoPE to Q and K
    rope_embed(device, q, k, seq_offset, config.rope_theta)?;

    // If num_heads > num_kv_heads, broadcast K/V (GQA)
    // groups = num_heads / num_kv_heads
    let groups = config.num_heads / config.num_kv_heads;
    let k_bcast;
    let v_bcast;
    let (k_final, v_final) = if groups > 1 {
        // Repeat K/V heads: [batch, num_kv_heads, seq, head_dim] → [batch, num_heads, seq, head_dim]
        let kh = k.copy_to_host()?;
        let vh = v.copy_to_host()?;
        let mut k_rep = Vec::with_capacity(batch * num_heads * seq * head_dim);
        let mut v_rep = Vec::with_capacity(batch * num_heads * seq * head_dim);
        for b in 0..batch {
            for h in 0..num_heads {
                let kv_h = h / groups;
                let kv_offset = b * config.num_kv_heads * seq * head_dim + kv_h * seq * head_dim;
                k_rep.extend_from_slice(&kh[kv_offset..kv_offset + seq * head_dim]);
                v_rep.extend_from_slice(&vh[kv_offset..kv_offset + seq * head_dim]);
            }
        }
        k_bcast = DeviceTensor::from_slice(&k_rep, &[batch, num_heads, seq, head_dim])?;
        v_bcast = DeviceTensor::from_slice(&v_rep, &[batch, num_heads, seq, head_dim])?;
        (&k_bcast, &v_bcast)
    } else {
        (k as &DeviceTensor<f16>, v as &DeviceTensor<f16>)
    };

    let mut out = DeviceTensor::<f16>::alloc(&[batch, num_heads, seq, head_dim])?;
    flash_attention(device, q, k_final, v_final, &mut out, config.attention_scale(), causal)?;

    Ok(out)
}
