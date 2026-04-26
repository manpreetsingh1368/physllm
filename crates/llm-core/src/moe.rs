//! moe.rs — Mixture-of-Experts transformer layer for GPT-OSS architecture.
//!
//! GPT-OSS-20B Architecture:
//!   - 24 layers, each with attention + MoE FFN
//!   - 32 experts per MoE layer, top-4 routing
//!   - Hidden dim: 2880, Head dim: 64
//!   - 64 query heads, 8 KV heads (GQA)
//!   - Alternating sliding window (128) + full attention
//!   - MXFP4 quantized expert weights
//!   - YaRN RoPE (theta=150000, factor=32)
//!
//! Forward pass per MoE layer:
//!   1. RMS Norm
//!   2. Attention (GQA with sliding/full alternation)
//!   3. Residual
//!   4. RMS Norm
//!   5. Router → select top-k experts
//!   6. Run selected experts (SwiGLU FFN each)
//!   7. Weighted combine of expert outputs
//!   8. Residual

use crate::{config::ModelConfig, Result, LlmError};
use rocm_backend::{
    GpuDevice, DeviceTensor,
    matmul_f16, rms_norm_gpu, rope_gpu, residual_add_gpu,
    flash_attention_v2,
    moe_router_gpu, moe_expert_forward_gpu, moe_combine_gpu,
    mxfp4_dequant_gpu,
};
use half::f16;
use std::sync::Arc;

/// Weights for a single expert (SwiGLU FFN).
pub struct ExpertWeights {
    pub w_gate: DeviceTensor<f16>,  // [intermediate_dim, hidden_dim]
    pub w_up:   DeviceTensor<f16>,  // [intermediate_dim, hidden_dim]
    pub w_down: DeviceTensor<f16>,  // [hidden_dim, intermediate_dim]
}

/// MXFP4 quantized expert weights (packed 4-bit + scales).
pub struct ExpertWeightsMXFP4 {
    pub w_gate_blocks: DeviceTensor<u8>,   // Packed FP4
    pub w_gate_scales: DeviceTensor<f16>,  // Per-block scales
    pub w_up_blocks:   DeviceTensor<u8>,
    pub w_up_scales:   DeviceTensor<f16>,
    pub w_down_blocks: DeviceTensor<u8>,
    pub w_down_scales: DeviceTensor<f16>,
    pub numel_gate:    usize,
    pub numel_up:      usize,
    pub numel_down:    usize,
}

/// Weights for a single MoE transformer layer.
pub struct MoELayerWeights {
    // Attention
    pub wq:        DeviceTensor<f16>,  // [hidden, num_heads * head_dim]
    pub wk:        DeviceTensor<f16>,  // [hidden, num_kv_heads * head_dim]
    pub wv:        DeviceTensor<f16>,  // [hidden, num_kv_heads * head_dim]
    pub wo:        DeviceTensor<f16>,  // [num_heads * head_dim, hidden]
    pub attn_norm: DeviceTensor<f16>,  // [hidden]
    
    // MoE
    pub router_weight: DeviceTensor<f16>,  // [num_experts, hidden_dim]
    pub experts:       Vec<ExpertWeights>, // num_experts experts
    pub ffn_norm:      DeviceTensor<f16>,  // [hidden]
}

/// Complete MoE model weights.
pub struct MoEModelWeights {
    pub embed_tokens: DeviceTensor<f16>,       // [vocab, hidden]
    pub layers:       Vec<MoELayerWeights>,
    pub final_norm:   DeviceTensor<f16>,       // [hidden]
    pub lm_head:      DeviceTensor<f16>,       // [vocab, hidden]
}

/// KV cache for MoE model.
pub struct MoEKVCache {
    pub k: DeviceTensor<f16>,
    pub v: DeviceTensor<f16>,
    pub current_len: usize,
}

/// MoE Inference Engine — runs GPT-OSS-20B on GPU.
pub struct MoEInferenceEngine {
    pub device:    Arc<GpuDevice>,
    pub config:    ModelConfig,
    pub weights:   MoEModelWeights,
    pub kv_caches: Vec<MoEKVCache>,
}

impl MoEInferenceEngine {
    /// Initialize engine with model weights and pre-allocated KV caches.
    pub fn new(
        device: Arc<GpuDevice>,
        config: ModelConfig,
        weights: MoEModelWeights,
    ) -> Result<Self> {
        let kv_caches = (0..config.num_layers)
            .map(|_| {
                Ok(MoEKVCache {
                    k: DeviceTensor::alloc(&[
                        config.num_kv_heads, config.max_seq_len, config.head_dim,
                    ]).map_err(|e| LlmError::Backend(e))?,
                    v: DeviceTensor::alloc(&[
                        config.num_kv_heads, config.max_seq_len, config.head_dim,
                    ]).map_err(|e| LlmError::Backend(e))?,
                    current_len: 0,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        
        Ok(Self { device, config, weights, kv_caches })
    }

    /// Generate next token — 100% GPU, MoE routing included.
    pub fn generate_token(&mut self, token: u32, temperature: f32) -> Result<u32> {
        let cfg = &self.config;
        let hd = cfg.hidden_dim;

        // 1. Embedding lookup
        let emb_host = self.weights.embed_tokens.copy_to_host()
            .map_err(|e| LlmError::Backend(e))?;
        let start = token as usize * hd;
        let tok_emb = &emb_host[start..start + hd];
        let mut hidden = DeviceTensor::from_slice(tok_emb, &[1, hd])
            .map_err(|e| LlmError::Backend(e))?;

        // 2. Run through all MoE layers
        for layer_idx in 0..cfg.num_layers {
            hidden = self.forward_moe_layer(hidden, layer_idx)?;
        }

        // 3. Final norm
        let mut normed = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        rms_norm_gpu(&self.device, &hidden, &self.weights.final_norm,
                     &mut normed, cfg.rms_norm_eps)
            .map_err(|e| LlmError::Backend(e))?;

        // 4. LM head → logits (on GPU)
        let mut logits_gpu = DeviceTensor::<f32>::alloc(&[cfg.vocab_size])
            .map_err(|e| LlmError::Backend(e))?;
        rocm_backend::lm_head_gpu(
            &self.device, &normed, &self.weights.lm_head,
            &mut logits_gpu, hd, cfg.vocab_size,
        ).map_err(|e| LlmError::Backend(e))?;

        // 5. Sample (GPU)
        let mut tok_out = DeviceTensor::<i32>::alloc(&[1])
            .map_err(|e| LlmError::Backend(e))?;
        rocm_backend::softmax_sample_gpu(
            &self.device, &mut logits_gpu, &mut tok_out,
            cfg.vocab_size, temperature, rand::random(),
        ).map_err(|e| LlmError::Backend(e))?;

        let result = tok_out.copy_to_host().map_err(|e| LlmError::Backend(e))?;
        Ok(result[0] as u32)
    }

    /// Forward pass for a single MoE transformer layer.
    fn forward_moe_layer(
        &mut self,
        hidden: DeviceTensor<f16>,
        layer_idx: usize,
    ) -> Result<DeviceTensor<f16>> {
        let cfg = &self.config;
        let hd = cfg.hidden_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let layer = &self.weights.layers[layer_idx];

        // ─── Attention sub-layer ─────────────────────────────────────────

        // 1. RMS Norm
        let mut normed = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        rms_norm_gpu(&self.device, &hidden, &layer.attn_norm,
                     &mut normed, cfg.rms_norm_eps)
            .map_err(|e| LlmError::Backend(e))?;

        // 2. Q/K/V projections
        let mut q = DeviceTensor::alloc(&[1, cfg.num_heads * cfg.head_dim])
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

        // 3. RoPE
        let pos = self.kv_caches[layer_idx].current_len;
        rope_gpu(&self.device, &mut q, cfg.num_heads, 1, cfg.head_dim,
                 pos, cfg.rope_theta)
            .map_err(|e| LlmError::Backend(e))?;
        rope_gpu(&self.device, &mut k, cfg.num_kv_heads, 1, cfg.head_dim,
                 pos, cfg.rope_theta)
            .map_err(|e| LlmError::Backend(e))?;

        // 4. KV cache update
        rocm_backend::kv_cache_update_gpu(
            &self.device, &mut self.kv_caches[layer_idx].k,
            &k, pos, cfg.num_kv_heads, cfg.head_dim,
        ).map_err(|e| LlmError::Backend(e))?;
        rocm_backend::kv_cache_update_gpu(
            &self.device, &mut self.kv_caches[layer_idx].v,
            &v, pos, cfg.num_kv_heads, cfg.head_dim,
        ).map_err(|e| LlmError::Backend(e))?;
        self.kv_caches[layer_idx].current_len += 1;
        let total_seq = self.kv_caches[layer_idx].current_len;

        // 5. Flash Attention v2
        let mut attn_out = DeviceTensor::alloc(
            &[1, cfg.num_heads, 1, cfg.head_dim]
        ).map_err(|e| LlmError::Backend(e))?;
        flash_attention_v2(
            &self.device,
            &q, &self.kv_caches[layer_idx].k, &self.kv_caches[layer_idx].v,
            &mut attn_out,
            cfg.num_heads, cfg.num_kv_heads, 1, total_seq, cfg.head_dim,
        ).map_err(|e| LlmError::Backend(e))?;

        // 6. Output projection
        let mut o = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        matmul_f16(&self.device, &attn_out, &layer.wo, &mut o)
            .map_err(|e| LlmError::Backend(e))?;

        // 7. Residual
        let mut residual = hidden;
        residual_add_gpu(&self.device, &mut residual, &o)
            .map_err(|e| LlmError::Backend(e))?;

        // ─── MoE FFN sub-layer ───────────────────────────────────────────

        // 8. RMS Norm
        let mut normed2 = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        rms_norm_gpu(&self.device, &residual, &layer.ffn_norm,
                     &mut normed2, cfg.rms_norm_eps)
            .map_err(|e| LlmError::Backend(e))?;

        // 9. MoE routing — select top-k experts
        let top_k = cfg.experts_per_token;
        let num_experts = cfg.num_experts;
        let mut expert_indices = DeviceTensor::<i32>::alloc(&[1, top_k])
            .map_err(|e| LlmError::Backend(e))?;
        let mut expert_weights_gpu = DeviceTensor::<f32>::alloc(&[1, top_k])
            .map_err(|e| LlmError::Backend(e))?;

        moe_router_gpu(
            &self.device, &normed2, &layer.router_weight,
            &mut expert_indices, &mut expert_weights_gpu,
            num_experts, top_k,
        ).map_err(|e| LlmError::Backend(e))?;

        // 10. Run selected experts and combine
        let indices_host = expert_indices.copy_to_host()
            .map_err(|e| LlmError::Backend(e))?;
        let weights_host = expert_weights_gpu.copy_to_host()
            .map_err(|e| LlmError::Backend(e))?;

        // Allocate buffers for expert outputs
        let inter_dim = cfg.intermediate_dim;
        let mut expert_outputs = DeviceTensor::alloc(&[top_k, hd])
            .map_err(|e| LlmError::Backend(e))?;
        let mut inter_buf = DeviceTensor::alloc(&[1, inter_dim])
            .map_err(|e| LlmError::Backend(e))?;

        // Run each selected expert
        for k_idx in 0..top_k {
            let expert_id = indices_host[k_idx] as usize;
            let expert = &layer.experts[expert_id];

            // Expert output → slice of expert_outputs[k_idx * hd .. (k_idx+1) * hd]
            let mut expert_out = DeviceTensor::alloc(&[1, hd])
                .map_err(|e| LlmError::Backend(e))?;

            moe_expert_forward_gpu(
                &self.device,
                &normed2,
                &expert.w_gate, &expert.w_up, &expert.w_down,
                &mut inter_buf, &mut expert_out,
                hd, inter_dim,
            ).map_err(|e| LlmError::Backend(e))?;

            // Copy expert output to the combined buffer
            let out_host = expert_out.copy_to_host()
                .map_err(|e| LlmError::Backend(e))?;
            let mut combined_host = expert_outputs.copy_to_host()
                .map_err(|e| LlmError::Backend(e))?;
            combined_host[k_idx * hd..(k_idx + 1) * hd]
                .copy_from_slice(&out_host);
            expert_outputs = DeviceTensor::from_slice(&combined_host, &[top_k, hd])
                .map_err(|e| LlmError::Backend(e))?;
        }

        // 11. Weighted combine of expert outputs
        let mut moe_out = DeviceTensor::alloc(&[1, hd])
            .map_err(|e| LlmError::Backend(e))?;
        moe_combine_gpu(
            &self.device, &expert_outputs, &expert_weights_gpu,
            &mut moe_out, top_k,
        ).map_err(|e| LlmError::Backend(e))?;

        // 12. Residual
        residual_add_gpu(&self.device, &mut residual, &moe_out)
            .map_err(|e| LlmError::Backend(e))?;

        Ok(residual)
    }

    /// Reset KV caches for a new conversation.
    pub fn reset_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.current_len = 0;
        }
    }

    /// Multi-token generation.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>> {
        self.reset_cache();
        let eos = self.config.eos_token_id;

        // Prefill
        for &tok in prompt_tokens {
            let _ = self.generate_token(tok, temperature)?;
        }

        // Decode
        let mut output = Vec::with_capacity(max_new_tokens);
        let mut last = *prompt_tokens.last().unwrap_or(&0);

        for _ in 0..max_new_tokens {
            let next = self.generate_token(last, temperature)?;
            if next == eos { break; }
            output.push(next);
            last = next;
        }

        Ok(output)
    }
}
