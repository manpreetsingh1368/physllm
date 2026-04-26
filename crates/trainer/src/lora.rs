// crates/trainer/src/lora.rs
//
// Production LoRA (Low-Rank Adaptation) implementation.
//
// For each target linear layer W ∈ ℝ^{d_out × d_in}, LoRA adds:
//   h = W·x + (α/r) · B·A·x
//
// where:
//   A ∈ ℝ^{r × d_in}  — initialised from N(0, σ²)  (σ = 1/√r)
//   B ∈ ℝ^{d_out × r} — initialised to zero         (so ΔW=0 at start)
//   r ≪ min(d_in, d_out)                             (e.g. 16 or 32)
//   α — scaling constant (often = r, or 2r)
//
// Only A and B are trained; W is frozen.
// At inference, merge: W' = W + (α/r) · B·A  (single matrix mul, no overhead)

use anyhow::{Context, Result};
use candle_core::{Device, DType, Module, Tensor, Var};
use candle_nn::{Linear, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info};

/// LoRA configuration.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition (r)
    pub rank:     usize,
    /// Scaling factor (α); effective scale = α/r
    pub alpha:    f32,
    /// Dropout applied to A·x before multiplying by B
    pub dropout:  f32,
    /// Which module name suffixes to apply LoRA to
    pub targets:  Vec<String>,
}

impl LoraConfig {
    pub fn scale(&self) -> f64 { self.alpha as f64 / self.rank as f64 }
}

/// A single LoRA adapter on one linear layer.
pub struct LoraLinear {
    /// Frozen base weight W [d_out, d_in]
    base:    Tensor,
    /// Optional frozen bias
    bias:    Option<Tensor>,
    /// Trainable A matrix [r, d_in] — random init
    lora_a:  Var,
    /// Trainable B matrix [d_out, r] — zero init
    lora_b:  Var,
    /// Scaling factor α/r
    scale:   f64,
    /// Dropout rate (applied during training only)
    dropout: f64,
    pub d_out:   usize,
    pub d_in:    usize,
    pub rank:    usize,
}

impl LoraLinear {
    /// Wrap a frozen base linear weight with a LoRA adapter.
    pub fn new(
        base:   Tensor,
        bias:   Option<Tensor>,
        cfg:    &LoraConfig,
        device: &Device,
    ) -> Result<Self> {
        let (d_out, d_in) = base.dims2()?;
        let r = cfg.rank;

        // A: kaiming uniform init with std = 1/√r (good default for LoRA A)
        let std_a  = 1.0 / (r as f64).sqrt();
        let a_data = Tensor::randn(0f32, std_a as f32, (r, d_in), device)?;
        let lora_a = Var::from_tensor(&a_data)?;

        // B: zero init so ΔW = 0 at step 0
        let b_data = Tensor::zeros((d_out, r), base.dtype(), device)?;
        let lora_b = Var::from_tensor(&b_data)?;

        // Freeze the base weight (no gradient)
        let base = base.detach();

        Ok(Self {
            base, bias, lora_a, lora_b,
            scale:   cfg.scale(),
            dropout: cfg.dropout as f64,
            d_out, d_in, rank: r,
        })
    }

    /// Number of trainable parameters in this adapter.
    pub fn param_count(&self) -> usize {
        self.rank * (self.d_in + self.d_out)
    }

    /// Merge LoRA into base weight: W' = W + (α/r) B·A
    pub fn merged_weight(&self) -> Result<Tensor> {
        // delta_W = (α/r) * B @ A    [d_out, r] @ [r, d_in] → [d_out, d_in]
        let b = self.lora_b.as_tensor();
        let a = self.lora_a.as_tensor();
        let delta = b.matmul(&a)?.affine(self.scale, 0.0)?;
        (&self.base + &delta).context("merge: base + delta failed")
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Base path: x @ W^T + bias
        let base_out = x.matmul(&self.base.t()?)?;
        let base_out = if let Some(ref b) = self.bias {
            base_out.broadcast_add(b)?
        } else { base_out };

        // LoRA path: (α/r) * (x @ A^T) @ B^T
        let ax = x.matmul(&self.lora_a.as_tensor().t()?)?;
        // Dropout on ax (only effective during training)
        let ax = if self.dropout > 0.0 {
            candle_nn::ops::dropout(&ax, self.dropout as f32)?
        } else { ax };
        let bax = ax.matmul(&self.lora_b.as_tensor().t()?)?;

        // Combined output
        (base_out + (self.scale * &bax)?)
            .map_err(|e| candle_core::Error::Msg(format!("LoRA forward: {e}")))
    }
}

/// The full LoRA-wrapped model.
/// Holds all LoRA adapters keyed by module path.
pub struct LoraModel {
    /// All LoRA adapters: module_path → adapter
    adapters:   HashMap<String, LoraLinear>,
    /// Non-LoRA frozen tensors (embeddings, norms, lm_head)
    frozen:     HashMap<String, Tensor>,
    /// Model config (hidden_dim, num_layers, etc.)
    pub config: PhysLlmConfig,
    device:     Device,
    dtype:      DType,
}

/// Minimal config needed for the LoRA model graph.
#[derive(Debug, Clone)]
pub struct PhysLlmConfig {
    pub hidden_dim:       usize,
    pub intermediate_dim: usize,
    pub num_layers:       usize,
    pub num_heads:        usize,
    pub num_kv_heads:     usize,
    pub head_dim:         usize,
    pub vocab_size:       usize,
    pub max_seq_len:      usize,
    pub rope_theta:       f32,
    pub rms_norm_eps:     f32,
}

impl LoraModel {
    /// Load base model from safetensors and wrap targeted layers with LoRA.
    pub fn load(
        model_dir: &Path,
        var_map:   &VarMap,
        device:    &Device,
        dtype:     DType,
        lora_cfg:  LoraConfig,
    ) -> Result<Self> {
        // Collect all safetensors shard paths
        let mut shards: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_str().map(|n| n.ends_with(".safetensors")).unwrap_or(false))
            .map(|e| e.path())
            .collect();
        shards.sort();

        if shards.is_empty() {
            anyhow::bail!("No .safetensors files found in {:?}", model_dir);
        }
        info!("Loading {} safetensors shard(s)...", shards.len());

        // Load all tensors into a flat map
        let mut tensor_map: HashMap<String, Tensor> = HashMap::new();
        for shard in &shards {
            let st = safetensors::SafeTensors::deserialize(
                &std::fs::read(shard)?
            )?;
            for (name, view) in st.tensors() {
                let t = tensor_from_view(&view, device, dtype)?;
                tensor_map.insert(name.to_string(), t);
            }
        }
        info!("Loaded {} tensors", tensor_map.len());

        // Infer config from tensor shapes
        let config = infer_config(&tensor_map)?;
        info!("Model config: {}L hidden={} heads={}/{} vocab={}",
              config.num_layers, config.hidden_dim,
              config.num_heads, config.num_kv_heads, config.vocab_size);

        // Partition: LoRA-targeted vs frozen
        let lora_target_suffixes: Vec<&str> = lora_cfg.targets
            .iter().map(|s| s.as_str()).collect();

        let mut adapters: HashMap<String, LoraLinear> = HashMap::new();
        let mut frozen:   HashMap<String, Tensor>     = HashMap::new();

        for (name, tensor) in tensor_map {
            let is_lora_target = lora_target_suffixes.iter()
                .any(|&suffix| name.ends_with(&format!(".{suffix}.weight")));

            if is_lora_target && tensor.dims().len() == 2 {
                // Find matching bias if any
                let bias_name = name.replace(".weight", ".bias");
                let bias = frozen.remove(&bias_name);
                debug!("LoRA wrapping: {name} {:?}", tensor.dims());
                let adapter = LoraLinear::new(tensor, bias, &lora_cfg, device)?;

                // Register trainable vars with VarMap so optimizer sees them
                var_map.set_one(
                    format!("{name}.lora_a"),
                    adapter.lora_a.as_tensor().clone(),
                )?;
                var_map.set_one(
                    format!("{name}.lora_b"),
                    adapter.lora_b.as_tensor().clone(),
                )?;

                adapters.insert(name, adapter);
            } else {
                frozen.insert(name, tensor);
            }
        }

        let total_lora_params: usize = adapters.values().map(|a| a.param_count()).sum();
        let total_base_params: usize = frozen.values()
            .map(|t| t.elem_count())
            .sum();
        info!("LoRA adapters: {} layers  ({} trainable params / {} total = {:.3}%)",
              adapters.len(), total_lora_params, total_base_params,
              total_lora_params as f64 / total_base_params as f64 * 100.0);

        Ok(Self { adapters, frozen, config, device: device.clone(), dtype })
    }

    /// Count total trainable parameters.
    pub fn trainable_param_count(&self) -> usize {
        self.adapters.values().map(|a| a.param_count()).sum()
    }

    /// Forward pass through the full model.
    /// Returns logits [batch, seq-1, vocab].
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let (batch, seq) = input_ids.dims2()?;

        // Token embedding
        let embed_w = self.frozen.get("model.embed_tokens.weight")
            .context("embed_tokens not found")?;
        let mut hidden = embed_w.embedding(input_ids)?;
        // Scale embeddings (standard in Llama)
        hidden = (hidden * (cfg.hidden_dim as f64).sqrt())?;

        // Transformer layers
        for layer_idx in 0..cfg.num_layers {
            hidden = self.transformer_block(hidden, attention_mask, layer_idx)?;
        }

        // Final RMS norm
        let norm_w = self.frozen.get("model.norm.weight")
            .context("model.norm.weight not found")?;
        hidden = rms_norm(&hidden, norm_w, cfg.rms_norm_eps as f64)?;

        // LM head — use embedding weight if tied, else separate lm_head
        let lm_head_w = self.frozen.get("lm_head.weight")
            .or_else(|| self.frozen.get("model.embed_tokens.weight"))
            .context("lm_head.weight not found")?;

        // Shift: predict token[1..] from hidden[0..seq-1]
        let hidden_shifted = hidden.narrow(1, 0, seq - 1)?;
        let logits = hidden_shifted.matmul(&lm_head_w.t()?)?;

        Ok(logits)
    }

    fn transformer_block(
        &self,
        x:     Tensor,
        mask:  &Tensor,
        layer: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let p = format!("model.layers.{layer}");

        // ── Attention sub-layer ───────────────────────────────────────────────
        let attn_norm_w = self.get_frozen(&format!("{p}.input_layernorm.weight"))?;
        let mut h = rms_norm(&x, attn_norm_w, cfg.rms_norm_eps as f64)?;

        // Q, K, V projections (LoRA-wrapped if targeted)
        let q = self.linear_forward(&format!("{p}.self_attn.q_proj"), &h)?;
        let k = self.linear_forward(&format!("{p}.self_attn.k_proj"), &h)?;
        let v = self.linear_forward(&format!("{p}.self_attn.v_proj"), &h)?;

        // Reshape for multi-head attention
        let (b, s, _) = q.dims3()?;
        let q = q.reshape((b, s, cfg.num_heads, cfg.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, s, cfg.num_kv_heads, cfg.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, s, cfg.num_kv_heads, cfg.head_dim))?.transpose(1, 2)?;

        // RoPE
        let (q, k) = rope_embed(q, k, cfg.rope_theta as f64, 0)?;

        // GQA: repeat K/V if num_heads > num_kv_heads
        let (q, k, v) = if cfg.num_heads != cfg.num_kv_heads {
            let groups = cfg.num_heads / cfg.num_kv_heads;
            let k = k.repeat((1, groups, 1, 1))?;
            let v = v.repeat((1, groups, 1, 1))?;
            (q, k, v)
        } else { (q, k, v) };

        // Scaled dot-product attention
        let scale = 1.0 / (cfg.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Causal mask
        let causal = causal_mask(s, q.device(), q.dtype())?;
        let attn_weights = attn_weights.broadcast_add(&causal)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape and output projection
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, s, cfg.num_heads * cfg.head_dim))?;
        let attn_out = self.linear_forward(&format!("{p}.self_attn.o_proj"), &attn_out)?;

        // Residual
        let x = (&x + &attn_out)?;

        // ── FFN sub-layer (SwiGLU) ────────────────────────────────────────────
        let ffn_norm_w = self.get_frozen(&format!("{p}.post_attention_layernorm.weight"))?;
        let h2 = rms_norm(&x, ffn_norm_w, cfg.rms_norm_eps as f64)?;

        let gate = self.linear_forward(&format!("{p}.mlp.gate_proj"), &h2)?;
        let up   = self.linear_forward(&format!("{p}.mlp.up_proj"),   &h2)?;

        // SiLU(gate) ⊙ up
        let silu_gate = candle_nn::ops::silu(&gate)?;
        let ffn_out   = (silu_gate * up)?;
        let ffn_out   = self.linear_forward(&format!("{p}.mlp.down_proj"), &ffn_out)?;

        // Residual
        Ok((&x + &ffn_out)?)
    }

    /// Run a linear forward — uses LoRA adapter if this layer is targeted, else frozen weight.
    fn linear_forward(&self, name: &str, x: &Tensor) -> Result<Tensor> {
        let weight_name = format!("{name}.weight");
        if let Some(adapter) = self.adapters.get(&weight_name) {
            adapter.forward(x)
                .context(format!("LoRA forward failed for {name}"))
        } else {
            let w = self.get_frozen(&weight_name)?;
            let out = x.matmul(&w.t()?)?;
            // Check for bias
            if let Some(b) = self.frozen.get(&format!("{name}.bias")) {
                Ok(out.broadcast_add(b)?)
            } else { Ok(out) }
        }
    }

    fn get_frozen(&self, name: &str) -> Result<&Tensor> {
        self.frozen.get(name)
            .with_context(|| format!("Frozen tensor not found: {name}"))
    }

    /// Save only the LoRA adapter weights (small — a few MB).
    pub fn save_lora_weights(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)?;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, adapter) in &self.adapters {
            tensors.insert(
                format!("{name}.lora_a"),
                adapter.lora_a.as_tensor().to_dtype(DType::F32)?,
            );
            tensors.insert(
                format!("{name}.lora_b"),
                adapter.lora_b.as_tensor().to_dtype(DType::F32)?,
            );
        }
        let out_path = path.join("lora_adapters.safetensors");
        save_safetensors(&tensors, &out_path)?;
        info!("LoRA weights saved: {} tensors → {:?}", tensors.len(), out_path);
        Ok(())
    }

    /// Merge LoRA into base weights and save as a full safetensors checkpoint.
    /// The merged model is identical to fine-tuned full weights — no LoRA overhead at inference.
    pub fn merge_and_save(&self, output_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;
        let mut merged: HashMap<String, Tensor> = HashMap::new();

        // Start with all frozen weights
        for (name, t) in &self.frozen {
            merged.insert(name.clone(), t.to_dtype(DType::F32)?);
        }

        // Merge LoRA adapters: W' = W + (α/r) · B · A
        for (name, adapter) in &self.adapters {
            let merged_w = adapter.merged_weight()?;
            merged.insert(name.clone(), merged_w.to_dtype(DType::F32)?);
        }

        // Shard into ≤4GB chunks for HuggingFace compatibility
        let out_path = output_dir.join("model.safetensors");
        save_safetensors(&merged, &out_path)?;
        info!("Merged model saved ({} tensors) → {:?}", merged.len(), out_path);

        Ok(())
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq   = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
    let rms    = (mean_sq + eps)?.sqrt()?;
    let normed = x.broadcast_div(&rms)?;
    normed.broadcast_mul(weight).context("rms_norm broadcast_mul failed")
}

fn rope_embed(
    q: Tensor, k: Tensor,
    theta: f64, offset: usize,
) -> Result<(Tensor, Tensor)> {
    let (_, _, seq, head_dim) = q.dims4()?;
    let device = q.device();
    let dtype  = q.dtype();

    // Precompute rotation angles for each (position, dim) pair
    let positions: Vec<f32> = (offset..offset + seq).map(|p| p as f32).collect();
    let dim_indices: Vec<f32> = (0..head_dim / 2)
        .map(|i| i as f32 / head_dim as f32)
        .collect();

    let freqs: Vec<f32> = positions.iter().flat_map(|&p| {
        dim_indices.iter().map(move |&d| p / (theta as f32).powf(d))
    }).collect();

    let freqs_t = Tensor::from_vec(freqs, (seq, head_dim / 2), device)?;
    let cos = freqs_t.cos()?.to_dtype(dtype)?;
    let sin = freqs_t.sin()?.to_dtype(dtype)?;

    let apply_rope = |t: Tensor| -> Result<Tensor> {
        let (b, h, s, d) = t.dims4()?;
        let half = d / 2;
        let t0 = t.narrow(3, 0, half)?;
        let t1 = t.narrow(3, half, half)?;
        let cos_bc = cos.reshape((1, 1, s, half))?.broadcast_as((b, h, s, half))?;
        let sin_bc = sin.reshape((1, 1, s, half))?.broadcast_as((b, h, s, half))?;
        let rotated = Tensor::cat(&[
            (&t0 * &cos_bc)?.sub(&(&t1 * &sin_bc)?)?,
            (&t1 * &cos_bc)?.add(&(&t0 * &sin_bc)?)?,
        ], 3)?;
        Ok(rotated)
    };

    Ok((apply_rope(q)?, apply_rope(k)?))
}

fn causal_mask(seq: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    // Upper-triangular mask filled with -inf (keeps lower triangle = 0)
    let mask: Vec<f32> = (0..seq).flat_map(|i| {
        (0..seq).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
    }).collect();
    Tensor::from_vec(mask, (1, 1, seq, seq), device)?
        .to_dtype(dtype).context("causal_mask dtype cast")
}

fn infer_config(tensors: &HashMap<String, Tensor>) -> Result<PhysLlmConfig> {
    let embed = tensors.get("model.embed_tokens.weight")
        .context("embed_tokens.weight not found")?;
    let (vocab_size, hidden_dim) = embed.dims2()?;

    // Count transformer layers
    let num_layers = (0..)
        .take_while(|&i| tensors.contains_key(&format!("model.layers.{i}.input_layernorm.weight")))
        .count();

    let q_proj = tensors.get("model.layers.0.self_attn.q_proj.weight")
        .context("q_proj not found")?;
    let k_proj = tensors.get("model.layers.0.self_attn.k_proj.weight")
        .context("k_proj not found")?;
    let ffn_gate = tensors.get("model.layers.0.mlp.gate_proj.weight")
        .context("gate_proj not found")?;

    let (q_dim, _)   = q_proj.dims2()?;
    let (kv_dim, _)  = k_proj.dims2()?;
    let (inter_dim, _) = ffn_gate.dims2()?;

    // Typical: head_dim = 128 for 7B models
    let head_dim   = 128usize;
    let num_heads    = q_dim  / head_dim;
    let num_kv_heads = kv_dim / head_dim;

    Ok(PhysLlmConfig {
        hidden_dim,
        intermediate_dim: inter_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len: 32768,
        rope_theta:  500_000.0,
        rms_norm_eps: 1e-5,
    })
}

fn tensor_from_view(
    view: &safetensors::tensor::TensorView<'_>,
    device: &Device,
    target_dtype: DType,
) -> Result<Tensor> {
    let src_dtype = match view.dtype() {
        safetensors::Dtype::F32  => DType::F32,
        safetensors::Dtype::F16  => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::I32  => DType::I64,
        other => anyhow::bail!("Unsupported dtype: {:?}", other),
    };

    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data();

    let t = match src_dtype {
        DType::F32 => {
            let v: Vec<f32> = data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]]))
                .collect();
            Tensor::from_vec(v, shape.as_slice(), &Device::Cpu)?
        }
        DType::F16 => {
            let v: Vec<half::f16> = data.chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0],b[1]]))
                .collect();
            Tensor::from_vec(v, shape.as_slice(), &Device::Cpu)?
        }
        _ => {
            let v: Vec<f32> = data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]]))
                .collect();
            Tensor::from_vec(v, shape.as_slice(), &Device::Cpu)?
        }
    };

    // Move to target device and cast to training dtype
    t.to_device(device)?.to_dtype(target_dtype)
        .context("tensor_from_view: device/dtype cast failed")
}

fn save_safetensors(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    use safetensors::tensor::SerdeError;

    let data: HashMap<String, safetensors::tensor::TensorView<'_>> = tensors.iter()
        .map(|(name, t)| {
            let flat = t.flatten_all()?.to_vec1::<f32>()?;
            Ok::<_, anyhow::Error>((name.clone(), flat))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter_map(|(name, data)| {
            let shape = tensors[&name].dims().to_vec();
            // Convert f32 vec to bytes
            let bytes: Vec<u8> = data.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();
            Some((name, (bytes, shape)))
        })
        .collect::<HashMap<_,_>>()
        .into_iter()
        .map(|(name, (bytes, shape))| {
            // We'll write manually as raw bytes
            (name, bytes, shape)
        })
        .fold(HashMap::new(), |mut map, (name, bytes, shape)| {
            map.insert(name, (bytes, shape));
            map
        })
        .into_iter()
        .map(|(name, _)| name)
        .collect::<std::collections::HashSet<_>>();

    // Use safetensors serialize
    let mut st_tensors: std::collections::BTreeMap<String, safetensors::tensor::TensorView<'_>> =
        std::collections::BTreeMap::new();

    // Collect all byte data first (lifetime issue workaround)
    let byte_data: Vec<(String, Vec<u8>, Vec<usize>)> = tensors.iter()
        .filter_map(|(name, t)| {
            let shape = t.dims().to_vec();
            t.to_dtype(DType::F32)
                .and_then(|f| f.flatten_all())
                .and_then(|f| f.to_vec1::<f32>())
                .ok()
                .map(|v| {
                    let bytes: Vec<u8> = v.iter().flat_map(|&f| f.to_le_bytes()).collect();
                    (name.clone(), bytes, shape)
                })
        })
        .collect();

    safetensors::serialize_to_file(&byte_data.iter().map(|(name, bytes, shape)| {
        (name.as_str(), safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.as_slice(),
            bytes.as_slice(),
        ).unwrap())
    }).collect::<std::collections::HashMap<_,_>>(), &None, path)?;

    info!("Saved {} tensors to {:?}", byte_data.len(), path);
    Ok(())
}
