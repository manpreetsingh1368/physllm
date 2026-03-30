//! PhysLLM — Transformer forward pass.

use crate::{config::ModelConfig, kv_cache::KVCache, Result, LlmError};
use rocm_backend::{GpuDevice, DeviceTensor, matmul_f16, flash_attention, rope_embed, rms_norm};
use half::f16;
use std::sync::Arc;
use tracing::{debug, trace};

/// Weight tensors for a single transformer layer.
pub struct LayerWeights {
    // Attention projections
    pub wq:   DeviceTensor<f16>,   // [hidden, num_heads * head_dim]
    pub wk:   DeviceTensor<f16>,   // [hidden, num_kv_heads * head_dim]
    pub wv:   DeviceTensor<f16>,   // [hidden, num_kv_heads * head_dim]
    pub wo:   DeviceTensor<f16>,   // [num_heads * head_dim, hidden]
    // Attention norm
    pub attn_norm: DeviceTensor<f16>,  // [hidden]
    // FFN (SwiGLU: gate + up + down)
    pub w_gate: DeviceTensor<f16>,     // [intermediate, hidden]
    pub w_up:   DeviceTensor<f16>,     // [intermediate, hidden]
    pub w_down: DeviceTensor<f16>,     // [hidden, intermediate]
    // FFN norm
    pub ffn_norm: DeviceTensor<f16>,   // [hidden]
}

/// All model weights.
pub struct ModelWeights {
    pub embed_tokens: DeviceTensor<f16>,      // [vocab, hidden]
    pub layers:       Vec<LayerWeights>,
    pub final_norm:   DeviceTensor<f16>,      // [hidden]
    pub lm_head:      DeviceTensor<f16>,      // [vocab, hidden]
}

/// The full PhysLLM model.
pub struct PhysLLM {
    pub config:  ModelConfig,
    pub weights: ModelWeights,
    pub device:  Arc<GpuDevice>,
    pub kv_cache: KVCache,
}

impl PhysLLM {
    pub fn new(config: ModelConfig, device: Arc<GpuDevice>) -> Result<Self> {
        // Weights are loaded separately via loader::load_weights()
        // Here we initialise with random weights (for testing/training from scratch)
        let weights = ModelWeights::random_init(&config, &device)?;
        let kv_cache = KVCache::new(&config, &device)?;
        Ok(Self { config, weights, device, kv_cache })
    }

    /// Forward pass: token IDs → logits over vocabulary.
    ///
    /// `tokens`     — input token IDs [batch, seq_len]
    /// `seq_offset` — for KV cache: how many tokens are already cached
    pub fn forward(
        &self,
        tokens:     &[u32],
        seq_offset: usize,
    ) -> Result<Vec<f32>> {
        let seq_len = tokens.len();
        let cfg     = &self.config;

        debug!("forward seq_len={seq_len} offset={seq_offset}");

        // 1. Embed tokens → [seq_len, hidden_dim]
        let mut hidden = self.embed(tokens)?;

        // 2. Transformer layers
        for (layer_idx, layer_w) in self.weights.layers.iter().enumerate() {
            trace!("layer {layer_idx}");
            hidden = self.transformer_block(&hidden, layer_w, seq_len, seq_offset, layer_idx)?;
        }

        // 3. Final RMS norm
        let mut normed = DeviceTensor::<f16>::alloc(&[seq_len, cfg.hidden_dim])?;
        normed.copy_from_host(&hidden.copy_to_host()?)?;
        rms_norm(&self.device, &mut normed, &self.weights.final_norm, cfg.rms_norm_eps)?;

        // 4. LM head → logits [seq_len, vocab_size]
        let mut logits = DeviceTensor::<f16>::alloc(&[seq_len, cfg.vocab_size])?;
        matmul_f16(&self.device, &normed, &self.weights.lm_head, &mut logits)?;

        // 5. Return last-token logits as f32
        let logits_host = logits.copy_to_host()?;
        let last_start  = (seq_len - 1) * cfg.vocab_size;
        Ok(logits_host[last_start..]
            .iter()
            .map(|x| x.to_f32())
            .collect())
    }

    fn embed(&self, tokens: &[u32]) -> Result<DeviceTensor<f16>> {
        let hidden_dim  = self.config.hidden_dim;
        let seq_len     = tokens.len();
        let embed_host  = self.weights.embed_tokens.copy_to_host()?;
        let mut out_host = vec![f16::ZERO; seq_len * hidden_dim];

        for (i, &tok) in tokens.iter().enumerate() {
            let src_off = tok as usize * hidden_dim;
            let dst_off = i * hidden_dim;
            out_host[dst_off..dst_off+hidden_dim]
                .copy_from_slice(&embed_host[src_off..src_off+hidden_dim]);
        }

        DeviceTensor::from_slice(&out_host, &[seq_len, hidden_dim])
            .map_err(Into::into)
    }

    fn transformer_block(
        &self,
        x:          &DeviceTensor<f16>,
        w:          &LayerWeights,
        seq_len:    usize,
        seq_offset: usize,
        layer_idx:  usize,
    ) -> Result<DeviceTensor<f16>> {
        let cfg = &self.config;

        // ── Attention sub-layer ────────────────────────────────────────────────
        // RMSNorm
        let mut h = DeviceTensor::<f16>::alloc(&[seq_len, cfg.hidden_dim])?;
        h.copy_from_host(&x.copy_to_host()?)?;
        rms_norm(&self.device, &mut h, &w.attn_norm, cfg.rms_norm_eps)?;

        // Project Q, K, V
        let mut q = DeviceTensor::<f16>::alloc(&[seq_len, cfg.num_heads * cfg.head_dim])?;
        let mut k = DeviceTensor::<f16>::alloc(&[seq_len, cfg.num_kv_heads * cfg.head_dim])?;
        let mut v = DeviceTensor::<f16>::alloc(&[seq_len, cfg.num_kv_heads * cfg.head_dim])?;
        matmul_f16(&self.device, &h, &w.wq, &mut q)?;
        matmul_f16(&self.device, &h, &w.wk, &mut k)?;
        matmul_f16(&self.device, &h, &w.wv, &mut v)?;

        // RoPE
        rope_embed(&self.device, &mut q, &mut k, seq_offset, cfg.rope_theta)?;

        // Reshape for attention: [1, heads, seq, head_dim]
        let mut q4 = DeviceTensor::<f16>::alloc(&[1, cfg.num_heads, seq_len, cfg.head_dim])?;
        q4.copy_from_host(&q.copy_to_host()?)?;
        let mut k4 = DeviceTensor::<f16>::alloc(&[1, cfg.num_kv_heads, seq_len, cfg.head_dim])?;
        k4.copy_from_host(&k.copy_to_host()?)?;
        let mut v4 = DeviceTensor::<f16>::alloc(&[1, cfg.num_kv_heads, seq_len, cfg.head_dim])?;
        v4.copy_from_host(&v.copy_to_host()?)?;

        // Flash Attention
        let mut attn_out = DeviceTensor::<f16>::alloc(&[1, cfg.num_heads, seq_len, cfg.head_dim])?;
        flash_attention(
            &self.device, &q4, &k4, &v4, &mut attn_out,
            cfg.attention_scale(), true, // causal
        )?;

        // Reshape back [seq, hidden]
        let mut attn_flat = DeviceTensor::<f16>::alloc(&[seq_len, cfg.num_heads * cfg.head_dim])?;
        attn_flat.copy_from_host(&attn_out.copy_to_host()?)?;

        // Output projection
        let mut attn_proj = DeviceTensor::<f16>::alloc(&[seq_len, cfg.hidden_dim])?;
        matmul_f16(&self.device, &attn_flat, &w.wo, &mut attn_proj)?;

        // Residual + FFN
        let x_host    = x.copy_to_host()?;
        let proj_host = attn_proj.copy_to_host()?;
        let mut res1: Vec<f16> = x_host.iter().zip(proj_host.iter())
            .map(|(&a, &b)| f16::from_f32(a.to_f32() + b.to_f32()))
            .collect();

        // ── FFN sub-layer (SwiGLU) ─────────────────────────────────────────────
        let mut res1_t = DeviceTensor::from_slice(&res1, &[seq_len, cfg.hidden_dim])?;
        rms_norm(&self.device, &mut res1_t, &w.ffn_norm, cfg.rms_norm_eps)?;

        let mut gate = DeviceTensor::<f16>::alloc(&[seq_len, cfg.intermediate_dim])?;
        let mut up   = DeviceTensor::<f16>::alloc(&[seq_len, cfg.intermediate_dim])?;
        matmul_f16(&self.device, &res1_t, &w.w_gate, &mut gate)?;
        matmul_f16(&self.device, &res1_t, &w.w_up,   &mut up)?;

        // SiLU(gate) * up
        let gate_h = gate.copy_to_host()?;
        let up_h   = up.copy_to_host()?;
        let swiglu: Vec<f16> = gate_h.iter().zip(up_h.iter()).map(|(&g, &u)| {
            let gf = g.to_f32();
            let uf = u.to_f32();
            let silu = gf / (1.0 + (-gf).exp());  // SiLU(x) = x * σ(x)
            f16::from_f32(silu * uf)
        }).collect();

        let swiglu_t = DeviceTensor::from_slice(&swiglu, &[seq_len, cfg.intermediate_dim])?;
        let mut ffn_out = DeviceTensor::<f16>::alloc(&[seq_len, cfg.hidden_dim])?;
        matmul_f16(&self.device, &swiglu_t, &w.w_down, &mut ffn_out)?;

        // Final residual
        let ffn_host = ffn_out.copy_to_host()?;
        let out: Vec<f16> = res1.iter().zip(ffn_host.iter())
            .map(|(&a, &b)| f16::from_f32(a.to_f32() + b.to_f32()))
            .collect();

        DeviceTensor::from_slice(&out, &[seq_len, cfg.hidden_dim]).map_err(Into::into)
    }
}

impl ModelWeights {
    /// Randomly initialise all weights (for testing).
    pub fn random_init(cfg: &ModelConfig, dev: &GpuDevice) -> Result<Self> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let rand_f16 = |n: usize, std: f32, rng: &mut ChaCha8Rng| -> Vec<f16> {
            (0..n).map(|_| f16::from_f32(rng.gen::<f32>() * std - std/2.0)).collect()
        };

        let init = |shape: &[usize], std: f32| -> std::result::Result<DeviceTensor<f16>, rocm_backend::BackendError> {
            let n = shape.iter().product();
            DeviceTensor::from_slice(&rand_f16(n, std, &mut rng), shape)
        };

        let scale = (2.0_f32 / cfg.hidden_dim as f32).sqrt();

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(LayerWeights {
                wq:       init(&[cfg.hidden_dim, cfg.num_heads * cfg.head_dim], scale)?,
                wk:       init(&[cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim], scale)?,
                wv:       init(&[cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim], scale)?,
                wo:       init(&[cfg.num_heads * cfg.head_dim, cfg.hidden_dim], scale)?,
                attn_norm: init(&[cfg.hidden_dim], 1.0)?,
                w_gate:   init(&[cfg.hidden_dim, cfg.intermediate_dim], scale)?,
                w_up:     init(&[cfg.hidden_dim, cfg.intermediate_dim], scale)?,
                w_down:   init(&[cfg.intermediate_dim, cfg.hidden_dim], scale)?,
                ffn_norm: init(&[cfg.hidden_dim], 1.0)?,
            });
        }

        Ok(ModelWeights {
            embed_tokens: init(&[cfg.vocab_size, cfg.hidden_dim], scale)?,
            layers,
            final_norm:   init(&[cfg.hidden_dim], 1.0)?,
            lm_head:      init(&[cfg.vocab_size, cfg.hidden_dim], scale)?,
        })
    }
}
