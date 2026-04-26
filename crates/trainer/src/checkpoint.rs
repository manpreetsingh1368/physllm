// crates/trainer/src/checkpoint.rs
//
// Production checkpoint manager.
//
// Saves:
//   {output_dir}/step_{N}/
//     lora_adapters.safetensors   — LoRA A and B weights
//     optimizer_state.safetensors — AdamW moment estimates
//     meta.json                  — step, loss, val_ppl, timestamp, sha256
//
//   {output_dir}/best/           — copy of the best val_ppl checkpoint
//   {output_dir}/final/          — merged model (written at end of training)
//
// Rotation: keeps the latest `max_keep` checkpoints, deletes older ones.

use crate::{lora::LoraModel, optimizer::AdamW};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use chrono::Utc;
use sha2::{Sha256, Digest};

/// Metadata stored alongside each checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub step:       usize,
    pub train_loss: f64,
    pub val_ppl:    f64,
    pub timestamp:  String,
    pub sha256:     String,
    pub elapsed_s:  u64,
    pub lr:         f64,
}

pub struct CheckpointManager {
    output_dir:  PathBuf,
    max_keep:    usize,
    saved_steps: Vec<usize>,
    train_start: std::time::Instant,
}

impl CheckpointManager {
    pub fn new(output_dir: PathBuf, max_keep: usize) -> Self {
        std::fs::create_dir_all(&output_dir).ok();
        Self {
            output_dir,
            max_keep,
            saved_steps: Vec::new(),
            train_start: std::time::Instant::now(),
        }
    }

    /// Save a training checkpoint.
    pub fn save(
        &mut self,
        model:      &LoraModel,
        optimizer:  &AdamW,
        step:       usize,
        train_loss: f64,
    ) -> Result<PathBuf> {
        let ckpt_dir = self.output_dir.join(format!("step_{step:07}"));
        std::fs::create_dir_all(&ckpt_dir)?;

        // 1. Save LoRA weights
        model.save_lora_weights(&ckpt_dir)?;

        // 2. Save optimizer state
        let opt_state = optimizer.state_dict()?;
        let opt_path  = ckpt_dir.join("optimizer_state.safetensors");
        save_tensors_to_file(&opt_state, &opt_path)?;

        // 3. Compute SHA256 of LoRA weights for integrity verification
        let lora_path = ckpt_dir.join("lora_adapters.safetensors");
        let sha256 = compute_sha256(&lora_path)?;

        // 4. Write metadata
        let meta = CheckpointMeta {
            step,
            train_loss,
            val_ppl: 0.0,  // filled in by save_best
            timestamp: Utc::now().to_rfc3339(),
            sha256,
            elapsed_s: self.train_start.elapsed().as_secs(),
            lr: optimizer.current_lr(),
        };
        write_meta(&ckpt_dir, &meta)?;

        info!("Checkpoint: {:?}  loss={train_loss:.4}  sha256={}",
              ckpt_dir, &meta.sha256[..8]);

        // 5. Rotate old checkpoints
        self.saved_steps.push(step);
        self.rotate_checkpoints()?;

        Ok(ckpt_dir)
    }

    /// Save the best-validation checkpoint (no rotation — always kept).
    pub fn save_best(
        &self,
        model:      &LoraModel,
        optimizer:  &AdamW,
        step:       usize,
        train_loss: f64,
        val_ppl:    f64,
    ) -> Result<()> {
        let best_dir = self.output_dir.join("best");
        // Overwrite previous best
        if best_dir.exists() {
            std::fs::remove_dir_all(&best_dir)?;
        }
        std::fs::create_dir_all(&best_dir)?;

        model.save_lora_weights(&best_dir)?;
        let opt_state = optimizer.state_dict()?;
        save_tensors_to_file(&opt_state, &best_dir.join("optimizer_state.safetensors"))?;

        let sha256 = compute_sha256(&best_dir.join("lora_adapters.safetensors"))?;
        let meta = CheckpointMeta {
            step, train_loss, val_ppl,
            timestamp: Utc::now().to_rfc3339(),
            sha256, elapsed_s: 0, lr: optimizer.current_lr(),
        };
        write_meta(&best_dir, &meta)?;
        info!("Best checkpoint saved: val_ppl={val_ppl:.2}");
        Ok(())
    }

    /// Load a checkpoint from a directory. Returns the metadata.
    pub fn load_checkpoint(
        &self,
        ckpt_dir:  &Path,
        model:     &mut LoraModel,
        optimizer: &mut AdamW,
    ) -> Result<CheckpointMeta> {
        // Read and verify metadata
        let meta_path = ckpt_dir.join("meta.json");
        let meta: CheckpointMeta = serde_json::from_str(
            &std::fs::read_to_string(&meta_path)
                .with_context(|| format!("Cannot read {:?}", meta_path))?
        )?;

        // Verify SHA256 integrity
        let lora_path = ckpt_dir.join("lora_adapters.safetensors");
        let actual_sha = compute_sha256(&lora_path)?;
        if actual_sha != meta.sha256 {
            anyhow::bail!(
                "Checkpoint integrity check failed!\n\
                 Expected SHA256: {}\n\
                 Actual SHA256:   {}\n\
                 File may be corrupt.",
                meta.sha256, actual_sha
            );
        }
        info!("Checkpoint integrity OK (sha256={})", &meta.sha256[..8]);

        // Load LoRA weights
        load_lora_weights(model, &lora_path)?;

        // Load optimizer state
        let opt_path = ckpt_dir.join("optimizer_state.safetensors");
        if opt_path.exists() {
            let opt_state = load_tensors_from_file(&opt_path)?;
            optimizer.load_state_dict(&opt_state)?;
        } else {
            warn!("Optimizer state not found — starting fresh optimizer");
        }

        info!("Resumed from step={} loss={:.4}", meta.step, meta.train_loss);
        Ok(meta)
    }

    /// List all checkpoint directories sorted by step.
    pub fn list_checkpoints(&self) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(&self.output_dir) else { return vec![]; };
        let mut dirs: Vec<(usize, PathBuf)> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with("step_") {
                    name[5..].parse::<usize>().ok().map(|n| (n, e.path()))
                } else { None }
            })
            .collect();
        dirs.sort_by_key(|(n, _)| *n);
        dirs.into_iter().map(|(_, p)| p).collect()
    }

    fn rotate_checkpoints(&mut self) -> Result<()> {
        if self.saved_steps.len() <= self.max_keep { return Ok(()); }

        // Keep the most recent `max_keep`; delete the rest
        let to_delete = self.saved_steps.len() - self.max_keep;
        let steps_to_delete: Vec<usize> = self.saved_steps.drain(..to_delete).collect();

        for step in steps_to_delete {
            let dir = self.output_dir.join(format!("step_{step:07}"));
            if dir.exists() {
                std::fs::remove_dir_all(&dir)
                    .with_context(|| format!("Failed to delete checkpoint {:?}", dir))?;
                info!("Rotated old checkpoint: step_{step:07}");
            }
        }
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn write_meta(dir: &Path, meta: &CheckpointMeta) -> Result<()> {
    let json = serde_json::to_string_pretty(meta)?;
    std::fs::write(dir.join("meta.json"), json)?;
    Ok(())
}

fn compute_sha256(path: &Path) -> Result<String> {
    let data = std::fs::read(path)
        .with_context(|| format!("Cannot read {:?} for SHA256", path))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(format!("{:x}", hasher.finalize()))
}

fn save_tensors_to_file(
    tensors: &std::collections::HashMap<String, candle_core::Tensor>,
    path:    &Path,
) -> Result<()> {
    // Serialise to safetensors format
    let data: Vec<(String, Vec<u8>, Vec<usize>)> = tensors.iter()
        .filter_map(|(name, t)| {
            t.to_dtype(candle_core::DType::F32)
                .and_then(|tf| tf.flatten_all())
                .and_then(|tf| tf.to_vec1::<f32>())
                .ok()
                .map(|v| {
                    let bytes: Vec<u8> = v.iter().flat_map(|&f| f.to_le_bytes()).collect();
                    let shape = t.dims().to_vec();
                    (name.clone(), bytes, shape)
                })
        })
        .collect();

    let tensors_map: std::collections::HashMap<&str, safetensors::tensor::TensorView<'_>> =
        data.iter().filter_map(|(name, bytes, shape)| {
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                shape.as_slice(),
                bytes.as_slice(),
            ).ok().map(|v| (name.as_str(), v))
        }).collect();

    safetensors::serialize_to_file(&tensors_map, &None, path)
        .context("safetensors serialize_to_file failed")?;
    Ok(())
}

fn load_tensors_from_file(
    path: &Path,
) -> Result<std::collections::HashMap<String, candle_core::Tensor>> {
    let bytes = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut map = std::collections::HashMap::new();
    for (name, view) in st.tensors() {
        let data: Vec<f32> = view.data().chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]]))
            .collect();
        let shape = view.shape().to_vec();
        let t = candle_core::Tensor::from_vec(data, shape.as_slice(), &candle_core::Device::Cpu)?;
        map.insert(name.to_string(), t);
    }
    Ok(map)
}

fn load_lora_weights(model: &mut LoraModel, path: &Path) -> Result<()> {
    let state = load_tensors_from_file(path)?;
    for (name, adapter) in model.adapters.iter_mut() {
        let a_key = format!("{name}.lora_a");
        let b_key = format!("{name}.lora_b");
        if let Some(a) = state.get(&a_key) {
            adapter.lora_a.set(&a.to_dtype(adapter.lora_a.dtype())?)?;
        }
        if let Some(b) = state.get(&b_key) {
            adapter.lora_b.set(&b.to_dtype(adapter.lora_b.dtype())?)?;
        }
    }
    info!("LoRA weights loaded from {:?}", path);
    Ok(())
}
