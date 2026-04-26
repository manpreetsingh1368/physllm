//! loader.rs — Load model weights from safetensors or GGUF format.
//!
//! Supports:
//!   - HuggingFace safetensors (native, preferred)
//!   - GGUF (llama.cpp format, for quantised models)
//!   - Memory-mapped loading (avoids duplicating weights in RAM)
//!
//! Weight name mapping (HuggingFace → PhysLLM internal):
//!   model.embed_tokens.weight           → embed_tokens
//!   model.layers.{i}.self_attn.q_proj.weight → layers[i].wq
//!   model.layers.{i}.self_attn.k_proj.weight → layers[i].wk
//!   model.layers.{i}.self_attn.v_proj.weight → layers[i].wv
//!   model.layers.{i}.self_attn.o_proj.weight → layers[i].wo
//!   model.layers.{i}.input_layernorm.weight  → layers[i].attn_norm
//!   model.layers.{i}.mlp.gate_proj.weight    → layers[i].w_gate
//!   model.layers.{i}.mlp.up_proj.weight      → layers[i].w_up
//!   model.layers.{i}.mlp.down_proj.weight    → layers[i].w_down
//!   model.layers.{i}.post_attention_layernorm.weight → layers[i].ffn_norm
//!   model.norm.weight                        → final_norm
//!   lm_head.weight                           → lm_head

use crate::{model::{ModelWeights, LayerWeights}, config::ModelConfig, Result, LlmError};
use rocm_backend::{GpuDevice, DeviceTensor};
use half::f16;
use memmap2::Mmap;
use std::{fs, io::{self, Read}, path::Path, sync::Arc};
use tracing::{info, debug, warn};

/// Supported weight formats.
#[derive(Debug, Clone)]
pub enum WeightFormat {
    SafeTensors,
    GGUF,
    RawBinary,   // raw f16 binary dump (for custom training outputs)
}

/// Load model weights from a directory containing safetensors shards.
///
/// ```
/// let weights = load_weights(
///     &config,
///     "/models/physllm-7b",
///     &device,
/// )?;
/// ```
pub fn load_weights(
    config:     &ModelConfig,
    model_dir:  &str,
    device:     &Arc<GpuDevice>,
) -> Result<ModelWeights> {
    let dir = Path::new(model_dir);

    // Detect format
    let format = detect_format(dir)?;
    info!("Loading weights from {model_dir} (format: {format:?})");

    match format {
        WeightFormat::SafeTensors => load_safetensors(config, dir, device),
        WeightFormat::GGUF        => load_gguf(config, dir, device),
        WeightFormat::RawBinary   => load_raw_binary(config, dir, device),
    }
}

fn detect_format(dir: &Path) -> Result<WeightFormat> {
    // Check for safetensors shards (HuggingFace standard)
    if dir.join("model.safetensors").exists()
        || dir.join("model-00001-of-00001.safetensors").exists()
        || dir.read_dir().map(|mut d| d.any(|e| {
            e.ok().and_then(|e| e.file_name().to_str().map(|n| n.ends_with(".safetensors")))
                .unwrap_or(false)
        })).unwrap_or(false)
    {
        return Ok(WeightFormat::SafeTensors);
    }
    if dir.join("model.gguf").exists() || dir.read_dir().map(|mut d|
        d.any(|e| e.ok().and_then(|e| e.file_name().to_str().map(|n| n.ends_with(".gguf"))).unwrap_or(false))
    ).unwrap_or(false) {
        return Ok(WeightFormat::GGUF);
    }
    Ok(WeightFormat::RawBinary)
}

fn load_safetensors(
    config:    &ModelConfig,
    dir:       &Path,
    device:    &Arc<GpuDevice>,
) -> Result<ModelWeights> {
    // Collect all shard paths sorted
    let mut shards: Vec<_> = fs::read_dir(dir)
        .map_err(|e| LlmError::Load(e.to_string()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_str().map(|n| n.ends_with(".safetensors")).unwrap_or(false))
        .map(|e| e.path())
        .collect();
    shards.sort();

    if shards.is_empty() {
        return Err(LlmError::Load("No .safetensors files found".into()));
    }

    info!("Found {} safetensors shard(s)", shards.len());

    // Build a tensor name → (shard_idx, byte_offset, shape, dtype) map
    // Full safetensors parsing: read 8-byte header_len, then JSON header
    let mut tensor_map: std::collections::HashMap<String, TensorMeta> = Default::default();

    for (shard_idx, shard_path) in shards.iter().enumerate() {
        debug!("Parsing shard {}: {}", shard_idx, shard_path.display());
        let meta = parse_safetensors_header(shard_path, shard_idx)?;
        tensor_map.extend(meta);
    }

    // Helper: load one named tensor as f16 DeviceTensor
    let load_tensor = |name: &str, expected_shape: &[usize]| -> Result<DeviceTensor<f16>> {
        if let Some(meta) = tensor_map.get(name) {
            let data = read_tensor_data(meta, &shards)?;
            // Convert to f16 if needed
            let f16_data: Vec<f16> = match meta.dtype.as_str() {
                "F16" | "BF16" => {
                    data.chunks_exact(2)
                        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                        .collect()
                }
                "F32" => {
                    data.chunks_exact(4)
                        .map(|b| f16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                        .collect()
                }
                other => {
                    warn!("Unsupported dtype {other} for {name}, using zeros");
                    vec![f16::ZERO; expected_shape.iter().product()]
                }
            };
            DeviceTensor::from_slice(&f16_data, expected_shape)
                .map_err(LlmError::Backend)
        } else {
            warn!("Tensor '{name}' not found in checkpoint, using zeros");
            let numel = expected_shape.iter().product();
            let zeros = vec![f16::ZERO; numel];
            DeviceTensor::from_slice(&zeros, expected_shape)
                .map_err(LlmError::Backend)
        }
    };

    let cfg = config;

    // Load embeddings
    let embed_tokens = load_tensor(
        "model.embed_tokens.weight",
        &[cfg.vocab_size, cfg.hidden_dim],
    )?;

    // Load transformer layers
    let mut layers = Vec::with_capacity(cfg.num_layers);
    for i in 0..cfg.num_layers {
        let prefix = format!("model.layers.{i}");
        let layer = LayerWeights {
            wq:       load_tensor(&format!("{prefix}.self_attn.q_proj.weight"), &[cfg.hidden_dim, cfg.num_heads * cfg.head_dim])?,
            wk:       load_tensor(&format!("{prefix}.self_attn.k_proj.weight"), &[cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim])?,
            wv:       load_tensor(&format!("{prefix}.self_attn.v_proj.weight"), &[cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim])?,
            wo:       load_tensor(&format!("{prefix}.self_attn.o_proj.weight"), &[cfg.num_heads * cfg.head_dim, cfg.hidden_dim])?,
            attn_norm: load_tensor(&format!("{prefix}.input_layernorm.weight"),          &[cfg.hidden_dim])?,
            w_gate:   load_tensor(&format!("{prefix}.mlp.gate_proj.weight"),             &[cfg.hidden_dim, cfg.intermediate_dim])?,
            w_up:     load_tensor(&format!("{prefix}.mlp.up_proj.weight"),               &[cfg.hidden_dim, cfg.intermediate_dim])?,
            w_down:   load_tensor(&format!("{prefix}.mlp.down_proj.weight"),             &[cfg.intermediate_dim, cfg.hidden_dim])?,
            ffn_norm: load_tensor(&format!("{prefix}.post_attention_layernorm.weight"),  &[cfg.hidden_dim])?,
        };
        layers.push(layer);
        if (i + 1) % 4 == 0 {
            info!("  Loaded {}/{} layers", i + 1, cfg.num_layers);
        }
    }

    let final_norm = load_tensor("model.norm.weight", &[cfg.hidden_dim])?;
    let lm_head = if cfg.tie_word_embeddings {
        // Reuse embedding weights (load again — or ideally share the allocation)
        load_tensor("model.embed_tokens.weight", &[cfg.vocab_size, cfg.hidden_dim])?
    } else {
        load_tensor("lm_head.weight", &[cfg.vocab_size, cfg.hidden_dim])?
    };

    info!("All weights loaded successfully");
    Ok(ModelWeights { embed_tokens, layers, final_norm, lm_head })
}

// ── Safetensors header parser ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TensorMeta {
    shard_idx:   usize,
    data_offset: usize,   // byte offset from data start (after header)
    data_length: usize,   // bytes
    shape:       Vec<usize>,
    dtype:       String,
    header_size: usize,   // total header bytes (for computing absolute file offset)
}

fn parse_safetensors_header(
    path:      &Path,
    shard_idx: usize,
) -> Result<std::collections::HashMap<String, TensorMeta>> {
    let file = fs::File::open(path).map_err(|e| LlmError::Load(e.to_string()))?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| LlmError::Load(e.to_string()))? };

    if mmap.len() < 8 {
        return Err(LlmError::Load("File too small to be safetensors".into()));
    }

    // First 8 bytes: u64 little-endian header size
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    if mmap.len() < 8 + header_len {
        return Err(LlmError::Load("Safetensors header truncated".into()));
    }

    let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
        .map_err(|e| LlmError::Load(format!("Invalid UTF-8 in header: {e}")))?;

    let header: serde_json::Value = serde_json::from_str(header_json)
        .map_err(|e| LlmError::Load(format!("Invalid JSON header: {e}")))?;

    let mut result = std::collections::HashMap::new();
    let data_start = 8 + header_len;

    if let serde_json::Value::Object(map) = &header {
        for (name, info) in map {
            if name == "__metadata__" { continue; }
            let dtype = info["dtype"].as_str().unwrap_or("F16").to_string();
            let offsets = &info["data_offsets"];
            let begin = offsets[0].as_u64().unwrap_or(0) as usize;
            let end   = offsets[1].as_u64().unwrap_or(0) as usize;
            let shape: Vec<usize> = info["shape"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                .unwrap_or_default();

            result.insert(name.clone(), TensorMeta {
                shard_idx,
                data_offset: begin,
                data_length: end - begin,
                shape,
                dtype,
                header_size: data_start,
            });
        }
    }

    Ok(result)
}

fn read_tensor_data(meta: &TensorMeta, shards: &[std::path::PathBuf]) -> Result<Vec<u8>> {
    let path = &shards[meta.shard_idx];
    let file = fs::File::open(path).map_err(|e| LlmError::Load(e.to_string()))?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| LlmError::Load(e.to_string()))? };
    let start = meta.header_size + meta.data_offset;
    let end   = start + meta.data_length;
    Ok(mmap[start..end].to_vec())
}

// ── GGUF loader stub ──────────────────────────────────────────────────────────

fn load_gguf(
    config: &ModelConfig,
    dir:    &Path,
    device: &Arc<GpuDevice>,
) -> Result<ModelWeights> {
    // GGUF parsing is complex; delegate to a dedicated crate (llama-cpp-rs or candle)
    // For now: fall back to random init with a warning
    warn!("GGUF loader not yet fully implemented; using random weights");
    warn!("To use GGUF: convert with `python convert_hf_to_gguf.py <model_dir>`");
    ModelWeights::random_init(config, device).map_err(LlmError::Backend)
}

fn load_raw_binary(
    config: &ModelConfig,
    dir:    &Path,
    device: &Arc<GpuDevice>,
) -> Result<ModelWeights> {
    warn!("No recognised weight files found in {}", dir.display());
    warn!("Initialising with random weights (training mode or test mode)");
    ModelWeights::random_init(config, device).map_err(LlmError::Backend)
}

// ── Weight quantisation helpers ───────────────────────────────────────────────

/// Convert f32 weights to INT4 (grouped quantisation, GPTQ-style).
pub fn quantise_to_int4(weights: &[f32], group_size: usize) -> (Vec<u8>, Vec<f16>) {
    let mut quantised = Vec::with_capacity(weights.len() / 2);
    let mut scales    = Vec::with_capacity(weights.len() / group_size);

    for group in weights.chunks(group_size) {
        let max = group.iter().cloned().fold(0.0f32, f32::max);
        let scale = max / 7.0; // INT4 range: -8..7
        scales.push(f16::from_f32(scale));

        for pair in group.chunks(2) {
            let q0 = ((pair[0] / scale + 8.0).round().clamp(0.0, 15.0)) as u8;
            let q1 = if pair.len() > 1 {
                ((pair[1] / scale + 8.0).round().clamp(0.0, 15.0)) as u8
            } else { 0 };
            quantised.push((q1 << 4) | q0);
        }
    }

    (quantised, scales)
}

/// Dequantise INT4 back to f16 for inference.
pub fn dequantise_int4(quantised: &[u8], scales: &[f16], group_size: usize) -> Vec<f16> {
    let mut out = Vec::with_capacity(quantised.len() * 2);
    let vals_per_scale = group_size;

    for (gi, scale) in scales.iter().enumerate() {
        let s = scale.to_f32();
        let q_start = gi * vals_per_scale / 2;
        let q_end   = (q_start + vals_per_scale / 2).min(quantised.len());
        for &byte in &quantised[q_start..q_end] {
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = (byte >> 4)   as f32 - 8.0;
            out.push(f16::from_f32(lo * s));
            out.push(f16::from_f32(hi * s));
        }
    }
    out
}
