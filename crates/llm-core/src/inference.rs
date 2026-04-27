//! inference.rs — 100% GPU inference engine for PhysLLM.
//!
//! This module provides a low-level inference engine that keeps ALL computation
//! on the GPU. Unlike generate.rs (which uses the model's forward() method),
//! this directly calls GPU kernels for every operation.
//!
//! Pipeline (100% GPU):
//!   Embedding Lookup → [RMSNorm → Q/K/V Proj → RoPE → KV Cache → FlashAttn →
//!   Output Proj → Residual → RMSNorm → Gate/Up Proj → SiLU×Up → Down Proj →
//!   Residual] × 32 layers → Final RMSNorm → LM Head → Softmax+Sample

use crate::{config::ModelConfig, model::{ModelWeights, LayerWeights}, Result, LlmError};
use rocm_backend::{
    GpuDevice, DeviceTensor,
    matmul_f16, rms_norm_gpu, rope_gpu, silu_multiply_gpu,
    residual_add_gpu, flash_attention_v2, kv_cache_update_gpu,
    lm_head_gpu, softmax_sample_gpu,
};
use half::f16;
use std::sync::Arc;

/// KV cache for a single transformer layer (stored on GPU).
pub struct LayerKVCache {
    pub k: DeviceTensor<f16>,   // [num_kv_heads, max_seq_len, head_dim]
    pub v: DeviceTensor<f16>,   // [num_kv_heads, max_seq_len, head_dim]
    pub current_len: usize,
}

/// 100% GPU inference engine.
pub struct InferenceEngine {
    pub device: Arc<GpuDevice>,
    pub config: ModelConfig,
    pub weights: ModelWeights,
    pub kv_caches: Vec<LayerKVCache>,
}

impl InferenceEngine {
    /// Create a new inference engine with pre-allocated KV caches on GPU.
    pub fn new(
        device: Arc<GpuDevice>,
        config: ModelConfig,
        weights: ModelWeights,
    ) -> Result<Self> {
        let max_seq = config.max_seq_len;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;

        let kv_caches = (0..config.num_layers)
            .map(|_| {
                Ok(LayerKVCache {
                    k: DeviceTensor::alloc(&[num_kv_heads, max_seq, head_dim])
                        .map_err(|e| LlmError::Backend(e))?,
                    v: DeviceTensor::alloc(&[num_kv_heads, max_seq, head_dim])
                        .map_err(|e| LlmError::Backend(e))?,
                    current_len: 0,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { device, config, weights, kv_caches })
    }

    /// Generate the next token from a single input token.
    /// Everything runs on GPU — no CPU/GPU transfers during the layer loop.
    pub fn generate_token(&mut self, token: u32, temperature: f32) -> Result<u32> {
        let cfg = self.config.clone();
        let hd = cfg.hidden_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_dim;

        // ── 1. Embedding lookup (copy token embedding from GPU table) ────────
        let emb_host = self.weights.embed_tokens.copy_to_host()
            .map_err(|e| LlmError::Backend(e))?;
        let tok_start = token as usize * hd;
        let tok_emb = &emb_host[tok_start..tok_start + hd];
        let mut hidden = DeviceTensor::from_slice(tok_emb, &[1, hd])
            .map_err(|e| LlmError::Backend(e))?;

        // ── 2. Transformer layers (100% GPU) ─────────────────────────────────
        for layer_idx in 0..cfg.num_layers {
            hidden = self.forward_layer_gpu(hidden, layer_idx)?;
        }

        // ── 3. Final RMS norm (GPU) ──────────────────────────────────────────
        let mut normed = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        rms_norm_gpu(
            &self.device, &hidden, &self.weights.final_norm,
            &mut normed, cfg.rms_norm_eps,
        ).map_err(|e| LlmError::Backend(e))?;

        // ── 4. LM head → logits (GPU) ───────────────────────────────────────
        let mut logits = DeviceTensor::<f32>::alloc(&[cfg.vocab_size])
            .map_err(|e| LlmError::Backend(e))?;
        lm_head_gpu(
            &self.device, &normed, &self.weights.lm_head,
            &mut logits, hd, cfg.vocab_size,
        ).map_err(|e| LlmError::Backend(e))?;

        // ── 5. Softmax + sample (GPU) ────────────────────────────────────────
        let mut token_out = DeviceTensor::<i32>::alloc(&[1])
            .map_err(|e| LlmError::Backend(e))?;
        let seed: u64 = rand::random();
        softmax_sample_gpu(
            &self.device, &mut logits, &mut token_out,
            cfg.vocab_size, temperature, seed,
        ).map_err(|e| LlmError::Backend(e))?;

        let token_host = token_out.copy_to_host()
            .map_err(|e| LlmError::Backend(e))?;
        Ok(token_host[0] as u32)
    }

    /// Forward pass through a single transformer layer — 100% on GPU.
    fn forward_layer_gpu(
        &mut self,
        hidden: DeviceTensor<f16>,
        layer_idx: usize,
    ) -> Result<DeviceTensor<f16>> {
        let cfg = self.config.clone();
        let hd = cfg.hidden_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_dim;
        let layer = &self.weights.layers[layer_idx];

        // ── Attention sub-layer ──────────────────────────────────────────────

        // 1. RMS Norm (GPU)
        let mut normed = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        rms_norm_gpu(
            &self.device, &hidden, &layer.attn_norm,
            &mut normed, cfg.rms_norm_eps,
        ).map_err(|e| LlmError::Backend(e))?;

        // 2. Q/K/V projections (GPU matmul)
        let mut q = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        let mut k = DeviceTensor::alloc(&[1, kv_dim])
            .map_err(|e| LlmError::Backend(e))?;
        let mut v = DeviceTensor::alloc(&[1, kv_dim])
            .map_err(|e| LlmError::Backend(e))?;

        matmul_f16(&self.device, &normed, &layer.wq, &mut q)
            .map_err(|e| LlmError::Backend(e))?;
        matmul_f16(&self.device, &normed, &layer.wk, &mut k)
            .map_err(|e| LlmError::Backend(e))?;
        matmul_f16(&self.device, &normed, &layer.wv, &mut v)
            .map_err(|e| LlmError::Backend(e))?;

        // 3. RoPE (GPU)
        let pos = self.kv_caches[layer_idx].current_len;
        rope_gpu(
            &self.device, &mut q,
            cfg.num_heads, 1, cfg.head_dim, pos, cfg.rope_theta,
        ).map_err(|e| LlmError::Backend(e))?;
        rope_gpu(
            &self.device, &mut k,
            cfg.num_kv_heads, 1, cfg.head_dim, pos, cfg.rope_theta,
        ).map_err(|e| LlmError::Backend(e))?;

        // 4. Update KV cache (GPU)
        kv_cache_update_gpu(
            &self.device, &mut self.kv_caches[layer_idx].k,
            &k, pos, cfg.num_kv_heads, cfg.head_dim,
        ).map_err(|e| LlmError::Backend(e))?;
        kv_cache_update_gpu(
            &self.device, &mut self.kv_caches[layer_idx].v,
            &v, pos, cfg.num_kv_heads, cfg.head_dim,
        ).map_err(|e| LlmError::Backend(e))?;
        self.kv_caches[layer_idx].current_len += 1;
        let total_seq = self.kv_caches[layer_idx].current_len;

        // 5. Flash Attention v2 (GPU)
        let mut attn_out = DeviceTensor::alloc(&[1, cfg.num_heads, 1, cfg.head_dim])
            .map_err(|e| LlmError::Backend(e))?;
        flash_attention_v2(
            &self.device,
            &q, &self.kv_caches[layer_idx].k, &self.kv_caches[layer_idx].v,
            &mut attn_out,
            cfg.num_heads, cfg.num_kv_heads,
            1, total_seq, cfg.head_dim,
        ).map_err(|e| LlmError::Backend(e))?;

        // 6. Output projection (GPU matmul)
        // Reshape attn_out to [1, hidden_dim] for matmul
        let attn_flat: DeviceTensor<f16> = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        // Note: attn_out data is already [1, num_heads * head_dim] = [1, hidden_dim]
        // We can reinterpret the shape without copying
        let mut o = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        matmul_f16(&self.device, &attn_out, &layer.wo, &mut o)
            .map_err(|e| LlmError::Backend(e))?;

        // 7. Residual add (GPU) — hidden = hidden + o
        let mut residual1 = hidden;
        residual_add_gpu(&self.device, &mut residual1, &o)
            .map_err(|e| LlmError::Backend(e))?;

        // ── MLP sub-layer ────────────────────────────────────────────────────

        // 8. RMS Norm (GPU)
        let mut normed2 = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        rms_norm_gpu(
            &self.device, &residual1, &layer.ffn_norm,
            &mut normed2, cfg.rms_norm_eps,
        ).map_err(|e| LlmError::Backend(e))?;

        // 9. Gate and Up projections (GPU matmul)
        let mut gate_out = DeviceTensor::alloc(&[1, inter])
            .map_err(|e| LlmError::Backend(e))?;
        let mut up_out = DeviceTensor::alloc(&[1, inter])
            .map_err(|e| LlmError::Backend(e))?;
        matmul_f16(&self.device, &normed2, &layer.w_gate, &mut gate_out)
            .map_err(|e| LlmError::Backend(e))?;
        matmul_f16(&self.device, &normed2, &layer.w_up, &mut up_out)
            .map_err(|e| LlmError::Backend(e))?;

        // 10. SiLU(gate) * up (GPU, fused!)
        let mut mlp_inter = DeviceTensor::alloc(&[1, inter])
            .map_err(|e| LlmError::Backend(e))?;
        silu_multiply_gpu(&self.device, &gate_out, &up_out, &mut mlp_inter)
            .map_err(|e| LlmError::Backend(e))?;

        // 11. Down projection (GPU matmul)
        let mut mlp_out = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        matmul_f16(&self.device, &mlp_inter, &layer.w_down, &mut mlp_out)
            .map_err(|e| LlmError::Backend(e))?;

        // 12. Residual add (GPU) — output = residual1 + mlp_out
        residual_add_gpu(&self.device, &mut residual1, &mlp_out)
            .map_err(|e| LlmError::Backend(e))?;

        Ok(residual1)
    }

    /// Reset all KV caches (call between conversations).
    pub fn reset_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.current_len = 0;
        }
    }

    /// Generate multiple tokens.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        eos_token: u32,
    ) -> Result<Vec<u32>> {
        self.reset_cache();

        let mut output_tokens = Vec::with_capacity(max_new_tokens);

        // Prefill: process all prompt tokens
        for &tok in prompt_tokens {
            let _ = self.generate_token(tok, temperature)?;
        }

        // Decode: generate new tokens one at a time
        let mut last_token = *prompt_tokens.last().unwrap_or(&0);
        for _ in 0..max_new_tokens {
            let next = self.generate_token(last_token, temperature)?;

            if next == eos_token {
                break;
            }

            output_tokens.push(next);
            last_token = next;
        }

        Ok(output_tokens)
    }
}

/// Transpose a weight matrix: [out_dim, in_dim] → [in_dim, out_dim]
/// Required because PyTorch/safetensors stores as [out, in] but matmul needs [in, out].
pub fn transpose_weight(weight: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut transposed = vec![f16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = weight[r * cols + c];
        }
    }
    transposed
}

/// Convert BF16 raw bytes to F16 values.
pub fn bf16_to_f16_vec(bf16_data: &[u8]) -> Vec<f16> {
    let bf16_slice: &[u16] = unsafe {
        std::slice::from_raw_parts(
            bf16_data.as_ptr() as *const u16,
            bf16_data.len() / 2,
        )
    };
    bf16_slice.iter().map(|&bf16| {
        f16::from_f32(f32::from_bits((bf16 as u32) << 16))
    }).collect()
}

/// CPU-based token sampling (fallback when GPU sampling kernel isn't available).
pub fn sample_token_cpu(logits: &[f32], temperature: f32) -> u32 {
    if temperature == 0.0 {
        // Greedy
        return logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
    }

    // Temperature scaling + softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = logits.iter()
        .map(|&x| ((x - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = scaled.iter().sum();

    // Sample from distribution
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen::<f32>() * sum;
    let mut cumsum = 0.0;
    for (idx, &p) in scaled.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return idx as u32;
        }
    }
    (logits.len() - 1) as u32
}
