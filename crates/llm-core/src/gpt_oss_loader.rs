//! gpt_oss_loader.rs — Load GPT-OSS-20B weights from safetensors.
//!
//! Weight name mapping (GPT-OSS → PhysLLM internal):
//!   model.embed_tokens.weight                           → embed_tokens
//!   model.layers.{i}.self_attn.q_proj.weight           → layers[i].wq
//!   model.layers.{i}.self_attn.k_proj.weight           → layers[i].wk
//!   model.layers.{i}.self_attn.v_proj.weight           → layers[i].wv
//!   model.layers.{i}.self_attn.o_proj.weight           → layers[i].wo
//!   model.layers.{i}.input_layernorm.weight            → layers[i].attn_norm
//!   model.layers.{i}.block_sparse_moe.gate.weight      → layers[i].router_weight
//!   model.layers.{i}.block_sparse_moe.experts.{e}.w1   → layers[i].experts[e].w_gate
//!   model.layers.{i}.block_sparse_moe.experts.{e}.w3   → layers[i].experts[e].w_up
//!   model.layers.{i}.block_sparse_moe.experts.{e}.w2   → layers[i].experts[e].w_down
//!   model.layers.{i}.post_attention_layernorm.weight   → layers[i].ffn_norm
//!   model.norm.weight                                  → final_norm
//!   lm_head.weight                                     → lm_head
//!
//! MXFP4 quantized weights:
//!   .blocks = packed uint8 (2 FP4 values per byte)
//!   .scales = BF16 per-block scale factors

use crate::{
    config::ModelConfig,
    moe::{MoEModelWeights, MoELayerWeights, ExpertWeights},
    Result, LlmError,
};
use rocm_backend::DeviceTensor;
use half::f16;
use std::path::Path;
use std::fs;

/// Convert BF16 raw bytes to F16 values.
fn bf16_to_f16_vec(data: &[u8]) -> Vec<f16> {
    let bf16_slice: &[u16] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u16,
            data.len() / 2,
        )
    };
    bf16_slice.iter().map(|&bf16| {
        f16::from_f32(f32::from_bits((bf16 as u32) << 16))
    }).collect()
}

/// Load GPT-OSS-20B weights from a directory of safetensors files.
///
/// # Arguments
/// * `config` - Model configuration (use ModelConfig::gpt_oss_20b())
/// * `model_dir` - Path to directory containing safetensors shards
///
/// # Example
/// ```
/// let config = ModelConfig::gpt_oss_20b();
/// let weights = load_gpt_oss_weights(&config, "/models/gpt-oss-20b")?;
/// let engine = MoEInferenceEngine::new(device, config, weights)?;
/// ```
pub fn load_gpt_oss_weights(
    config: &ModelConfig,
    model_dir: &str,
) -> Result<MoEModelWeights> {
    let dir = Path::new(model_dir);

    // Find all safetensors shards
    let mut shard_paths: Vec<_> = fs::read_dir(dir)
        .map_err(|e| LlmError::Load(format!("Cannot read {}: {}", model_dir, e)))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_str()?.to_string();
            if name.ends_with(".safetensors") { Some(entry.path()) } else { None }
        })
        .collect();
    shard_paths.sort();

    if shard_paths.is_empty() {
        return Err(LlmError::Load(
            format!("No .safetensors files found in {}", model_dir)
        ));
    }

    println!("Loading GPT-OSS-20B from {} ({} shards)...", model_dir, shard_paths.len());

    // Load all shards into memory
    let mut shard_bufs = Vec::new();
    for path in &shard_paths {
        println!("  Loading {}...", path.display());
        let buf = fs::read(path).map_err(|e| LlmError::Io(e))?;
        shard_bufs.push(buf);
    }

    // Parse safetensors
    let shards: Vec<safetensors::SafeTensors> = shard_bufs.iter()
        .map(|buf| safetensors::SafeTensors::deserialize(buf)
            .map_err(|e| LlmError::Load(format!("Safetensors parse error: {}", e))))
        .collect::<Result<Vec<_>>>()?;

    // Helper to find a tensor across shards
    let get_tensor = |name: &str| -> Result<Vec<f16>> {
        for shard in &shards {
            if let Ok(t) = shard.tensor(name) {
                return Ok(bf16_to_f16_vec(t.data()));
            }
        }
        Err(LlmError::Load(format!("Tensor '{}' not found in any shard", name)))
    };

    // Load global weights
    println!("  Loading embeddings...");
    let embed = get_tensor("model.embed_tokens.weight")?;
    let embed_tokens = DeviceTensor::from_slice(&embed, &[config.vocab_size, config.hidden_dim])
        .map_err(|e| LlmError::Backend(e))?;

    let norm = get_tensor("model.norm.weight")?;
    let final_norm = DeviceTensor::from_slice(&norm, &[config.hidden_dim])
        .map_err(|e| LlmError::Backend(e))?;

    let lm = get_tensor("lm_head.weight")?;
    let lm_head = DeviceTensor::from_slice(&lm, &[config.vocab_size, config.hidden_dim])
        .map_err(|e| LlmError::Backend(e))?;

    // Load layers
    let mut layers = Vec::with_capacity(config.num_layers);

    for i in 0..config.num_layers {
        print!("  Loading layer {}/{}...\r", i + 1, config.num_layers);
        let p = format!("model.layers.{}", i);

        // Attention weights
        let wq = get_tensor(&format!("{}.self_attn.q_proj.weight", p))?;
        let wk = get_tensor(&format!("{}.self_attn.k_proj.weight", p))?;
        let wv = get_tensor(&format!("{}.self_attn.v_proj.weight", p))?;
        let wo = get_tensor(&format!("{}.self_attn.o_proj.weight", p))?;
        let an = get_tensor(&format!("{}.input_layernorm.weight", p))?;
        let fn_ = get_tensor(&format!("{}.post_attention_layernorm.weight", p))?;

        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        // Router weight
        let router_name = format!("{}.block_sparse_moe.gate.weight", p);
        let router = get_tensor(&router_name)
            .or_else(|_| get_tensor(&format!("{}.mlp.router.weight", p)))?;

        // Load experts
        let mut experts = Vec::with_capacity(config.num_experts);
        for e in 0..config.num_experts {
            // GPT-OSS naming: w1=gate, w3=up, w2=down
            let gate_name = format!("{}.block_sparse_moe.experts.{}.w1.weight", p, e);
            let up_name = format!("{}.block_sparse_moe.experts.{}.w3.weight", p, e);
            let down_name = format!("{}.block_sparse_moe.experts.{}.w2.weight", p, e);

            // Try standard naming first, fall back to mlp.experts
            let gate_data = get_tensor(&gate_name)
                .or_else(|_| get_tensor(&format!("{}.mlp.experts.{}.w1.weight", p, e)))?;
            let up_data = get_tensor(&up_name)
                .or_else(|_| get_tensor(&format!("{}.mlp.experts.{}.w3.weight", p, e)))?;
            let down_data = get_tensor(&down_name)
                .or_else(|_| get_tensor(&format!("{}.mlp.experts.{}.w2.weight", p, e)))?;

            experts.push(ExpertWeights {
                w_gate: DeviceTensor::from_slice(
                    &gate_data, &[config.intermediate_dim, config.hidden_dim]
                ).map_err(|e| LlmError::Backend(e))?,
                w_up: DeviceTensor::from_slice(
                    &up_data, &[config.intermediate_dim, config.hidden_dim]
                ).map_err(|e| LlmError::Backend(e))?,
                w_down: DeviceTensor::from_slice(
                    &down_data, &[config.hidden_dim, config.intermediate_dim]
                ).map_err(|e| LlmError::Backend(e))?,
            });
        }

        layers.push(MoELayerWeights {
            wq: DeviceTensor::from_slice(&wq, &[config.hidden_dim, q_dim])
                .map_err(|e| LlmError::Backend(e))?,
            wk: DeviceTensor::from_slice(&wk, &[config.hidden_dim, kv_dim])
                .map_err(|e| LlmError::Backend(e))?,
            wv: DeviceTensor::from_slice(&wv, &[config.hidden_dim, kv_dim])
                .map_err(|e| LlmError::Backend(e))?,
            wo: DeviceTensor::from_slice(&wo, &[q_dim, config.hidden_dim])
                .map_err(|e| LlmError::Backend(e))?,
            attn_norm: DeviceTensor::from_slice(&an, &[config.hidden_dim])
                .map_err(|e| LlmError::Backend(e))?,
            router_weight: DeviceTensor::from_slice(
                &router, &[config.num_experts, config.hidden_dim]
            ).map_err(|e| LlmError::Backend(e))?,
            experts,
            ffn_norm: DeviceTensor::from_slice(&fn_, &[config.hidden_dim])
                .map_err(|e| LlmError::Backend(e))?,
        });
    }

    println!("  Layer {}/{} ✓          ", config.num_layers, config.num_layers);
    println!("✓ GPT-OSS-20B loaded successfully");

    Ok(MoEModelWeights {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}
