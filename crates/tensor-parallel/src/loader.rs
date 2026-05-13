// src/loader.rs
use std::{collections::HashMap, path::{Path, PathBuf}, sync::Arc};
use memmap2::Mmap;
use safetensors::SafeTensors;
use half::f16;
use tracing::{debug, info};

use crate::{config::TpConfig, error::{TpError, TpResult}};

/// One rank's slice of a weight tensor.
pub struct ShardedWeight {
    pub name: String,
    pub full_shape:  Vec<usize>,
    pub shard_shape: Vec<usize>,
    pub shard_dim:   usize,
    /// Raw fp16 bytes — ready to upload to device via hipMemcpy / cudaMemcpy.
    pub data: Vec<u8>,
    pub numel: usize,
}

impl ShardedWeight {
    pub fn as_fp16(&self) -> &[f16] { bytemuck::cast_slice(&self.data) }
}

pub struct WeightLoader {
    config: Arc<TpConfig>,
    files:  Vec<PathBuf>,
    maps:   Vec<Mmap>,
}

impl WeightLoader {
    /// Scan a directory for *.safetensors files and mmap them all.
    pub fn from_dir(dir: &Path, config: Arc<TpConfig>) -> TpResult<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
            .collect();
        files.sort();

        if files.is_empty() {
            return Err(TpError::Other(anyhow::anyhow!(
                "No .safetensors files in {}", dir.display()
            )));
        }

        info!(rank = config.rank, n_files = files.len(), "WeightLoader: mmap'd safetensor files");

        let maps = files.iter().map(|f| {
            let file = std::fs::File::open(f).map_err(TpError::Io)?;
            // SAFETY: read-only mmap, file outlives map (held in vec alongside it).
            unsafe { Mmap::map(&file).map_err(TpError::Io) }
        }).collect::<TpResult<Vec<_>>>()?;

        Ok(Self { config, files, maps })
    }

    /// Load one tensor shard for this rank.
    ///
    /// `shard_dim = None`  → full copy (biases, norms, router weights)
    /// `shard_dim = Some(d)` → slice dim d into world_size equal parts
    pub fn load(&self, name: &str, shard_dim: Option<usize>) -> TpResult<ShardedWeight> {
        for map in &self.maps {
            let st = SafeTensors::deserialize(map).map_err(|e| TpError::WeightLoad {
                name: name.to_string(), source: anyhow::anyhow!("{e}"),
            })?;

            let Ok(view) = st.tensor(name) else { continue };

            let full_shape: Vec<usize> = view.shape().to_vec();
            let raw_bytes = view.data(); // fp16 bytes from file

            debug!(rank = self.config.rank, tensor = name, shape = ?full_shape, shard_dim, "Loading shard");

            let (data, shard_shape, dim) = match shard_dim {
                None => {
                    (raw_bytes.to_vec(), full_shape.clone(), 0)
                }
                Some(dim) => {
                    let start = self.config.shard_start(full_shape[dim]);
                    let end   = self.config.shard_end(full_shape[dim]);
                    let sliced = self.slice_dim_bytes(raw_bytes, &full_shape, dim, start, end);
                    let mut shape = full_shape.clone();
                    shape[dim] = end - start;
                    (sliced, shape, dim)
                }
            };

            let numel: usize = shard_shape.iter().product();
            return Ok(ShardedWeight { name: name.to_string(), full_shape, shard_shape, shard_dim: dim, data, numel });
        }

        Err(TpError::WeightLoad {
            name: name.to_string(),
            source: anyhow::anyhow!("tensor not found in any safetensors file"),
        })
    }

    /// Load all weights for one transformer layer, pre-sharded for this rank.
    /// Follows LLaMA / GPT-OSS naming. Returns name → ShardedWeight.
    pub fn load_layer(&self, layer_idx: usize) -> TpResult<HashMap<String, ShardedWeight>> {
        let p = format!("model.layers.{layer_idx}");
        let mut w = HashMap::new();

        // Q/K/V — column parallel (split output dim 0)
        for proj in ["q_proj", "k_proj", "v_proj"] {
            w.insert(proj.to_string(), self.load(&format!("{p}.self_attn.{proj}.weight"), Some(0))?);
        }
        // O proj — row parallel (split input dim 1)
        w.insert("o_proj".into(), self.load(&format!("{p}.self_attn.o_proj.weight"), Some(1))?);

        // MLP gate/up — column parallel; down — row parallel
        for proj in ["gate_proj", "up_proj"] {
            if let Ok(weight) = self.load(&format!("{p}.mlp.{proj}.weight"), Some(0)) {
                w.insert(proj.to_string(), weight);
            }
        }
        if let Ok(weight) = self.load(&format!("{p}.mlp.down_proj.weight"), Some(1)) {
            w.insert("down_proj".into(), weight);
        }

        // Norms — full copy on every rank
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            if let Ok(weight) = self.load(&format!("{p}.{norm}.weight"), None) {
                w.insert(norm.to_string(), weight);
            }
        }

        Ok(w)
    }

    /// Load MoE expert weights for this rank's local experts.
    /// Experts are distributed whole across ranks (expert-parallel).
    pub fn load_moe_layer(
        &self,
        layer_idx: usize,
        n_experts: usize,
    ) -> TpResult<Vec<HashMap<String, ShardedWeight>>> {
        let world = self.config.world_size;
        let experts_per_rank = (n_experts + world - 1) / world;
        let offset = self.config.rank * experts_per_rank;
        let end    = (offset + experts_per_rank).min(n_experts);

        let p = format!("model.layers.{layer_idx}.mlp");
        let mut result = Vec::new();

        for expert_id in offset..end {
            let mut w = HashMap::new();
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                // Expert weights are NOT sharded further — each rank owns whole experts.
                let key = format!("{p}.experts.{expert_id}.{proj}.weight");
                w.insert(proj.to_string(), self.load(&key, None)?);
            }
            result.push(w);
        }
        Ok(result)
    }

    // Slice `bytes` (fp16, row-major) along `dim` from `start..end` (elements, not bytes).
    fn slice_dim_bytes(
        &self, bytes: &[u8], shape: &[usize], dim: usize, start: usize, end: usize,
    ) -> Vec<u8> {
        let elem_size = 2usize; // fp16
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();
        let old_stride = shape[dim] * inner * elem_size;
        let new_stride = (end - start) * inner * elem_size;
        let src_offset = start * inner * elem_size;

        let mut out = Vec::with_capacity(outer * new_stride);
        for o in 0..outer {
            let base = o * old_stride + src_offset;
            out.extend_from_slice(&bytes[base..base + new_stride]);
        }
        out
    }
}
