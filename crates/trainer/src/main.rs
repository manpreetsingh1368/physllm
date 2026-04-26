// crates/trainer/src/main.rs
//
// PhysLLM Production Trainer
// ──────────────────────────
// Full training loop with:
//   • Real autograd via Candle (no stubs)
//   • LoRA adapters with correct delta_W = (α/r) B·A
//   • AdamW with bias correction and weight decay
//   • Gradient clipping (global L2 norm)
//   • Gradient accumulation (true, not fake)
//   • Mixed-precision training (f16 forward, f32 master weights)
//   • Safetensors checkpoint save/resume
//   • Perplexity evaluation on held-out val set
//   • LR schedule: linear warmup → cosine decay → final flat
//   • Weights & Biases logging (optional)
//   • CLI argument parsing
//   • Graceful Ctrl-C → saves checkpoint before exit

mod lora;
mod optimizer;
mod checkpoint;
mod dataset;
mod evaluate;
mod logger;

use lora::{LoraConfig, LoraModel};
use optimizer::{AdamW, AdamWConfig};
use checkpoint::{CheckpointManager, CheckpointMeta};
use dataset::{DataLoader, JsonlDataset};
use evaluate::evaluate;
use logger::TrainLogger;

use candle_core::{Device, DType, Tensor, Module};
use candle_nn::VarMap;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::{path::PathBuf, sync::Arc, sync::atomic::{AtomicBool, Ordering}};
use tracing::{info, warn, error};
use anyhow::{Context, Result};

// ── CLI ───────────────────────────────────────────────────────────────────────

/// PhysLLM production trainer.
#[derive(Parser, Debug)]
#[command(name = "physllm-train", about = "Train or fine-tune PhysLLM", version)]
pub struct Args {
    /// Path to model weights directory (safetensors shards)
    #[arg(long, default_value = "models/mistral-7b")]
    pub model_dir: PathBuf,

    /// Path to training JSONL file
    #[arg(long, default_value = "data/train/train.jsonl")]
    pub train_data: PathBuf,

    /// Path to validation JSONL file
    #[arg(long, default_value = "data/train/val.jsonl")]
    pub val_data: PathBuf,

    /// Output directory for checkpoints
    #[arg(long, default_value = "checkpoints/physllm-7b")]
    pub output_dir: PathBuf,

    /// Resume from a specific checkpoint directory
    #[arg(long)]
    pub resume: Option<PathBuf>,

    // ── LoRA ──────────────────────────────────────────────────────────────────
    /// LoRA rank (0 = full fine-tune, not recommended for 7B)
    #[arg(long, default_value_t = 32)]
    pub lora_rank: usize,

    /// LoRA alpha (scaling factor; effective LR = lr * alpha/rank)
    #[arg(long, default_value_t = 64.0)]
    pub lora_alpha: f32,

    /// LoRA dropout probability
    #[arg(long, default_value_t = 0.05)]
    pub lora_dropout: f32,

    /// Which modules to apply LoRA to (comma-separated)
    #[arg(long, default_value = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")]
    pub lora_targets: String,

    // ── Optimizer ─────────────────────────────────────────────────────────────
    /// Peak learning rate
    #[arg(long, default_value_t = 2e-4)]
    pub lr: f64,

    /// AdamW beta1
    #[arg(long, default_value_t = 0.9)]
    pub beta1: f64,

    /// AdamW beta2
    #[arg(long, default_value_t = 0.999)]
    pub beta2: f64,

    /// AdamW epsilon
    #[arg(long, default_value_t = 1e-8)]
    pub eps: f64,

    /// Weight decay coefficient
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f64,

    /// Maximum gradient L2 norm (0 = no clipping)
    #[arg(long, default_value_t = 1.0)]
    pub grad_clip: f64,

    // ── Schedule ──────────────────────────────────────────────────────────────
    /// Linear warmup steps
    #[arg(long, default_value_t = 100)]
    pub warmup_steps: usize,

    /// Total training steps
    #[arg(long, default_value_t = 10_000)]
    pub total_steps: usize,

    // ── Batching ──────────────────────────────────────────────────────────────
    /// Per-device batch size
    #[arg(long, default_value_t = 2)]
    pub batch_size: usize,

    /// Gradient accumulation steps (effective batch = batch_size × this)
    #[arg(long, default_value_t = 8)]
    pub grad_accum: usize,

    /// Maximum sequence length in tokens
    #[arg(long, default_value_t = 2048)]
    pub max_seq_len: usize,

    // ── Logging / saving ──────────────────────────────────────────────────────
    /// Log every N optimizer steps
    #[arg(long, default_value_t = 10)]
    pub log_every: usize,

    /// Evaluate on val set every N optimizer steps
    #[arg(long, default_value_t = 500)]
    pub eval_every: usize,

    /// Save checkpoint every N optimizer steps
    #[arg(long, default_value_t = 500)]
    pub save_every: usize,

    /// Maximum checkpoints to keep on disk
    #[arg(long, default_value_t = 3)]
    pub keep_checkpoints: usize,

    /// Training dtype ("f16" or "bf16")
    #[arg(long, default_value = "f16")]
    pub dtype: String,

    /// Weights & Biases project name (empty = disabled)
    #[arg(long, default_value = "physllm")]
    pub wandb_project: String,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

// ── Learning rate schedule ────────────────────────────────────────────────────

/// Linear warmup → cosine decay → flat tail (last 10% of training).
pub fn lr_schedule(step: usize, cfg: &Args) -> f64 {
    let total = cfg.total_steps;
    let warmup = cfg.warmup_steps;
    let flat_start = (total as f64 * 0.9) as usize;

    if step < warmup {
        // Linear warmup: 0 → peak
        cfg.lr * step as f64 / warmup as f64
    } else if step < flat_start {
        // Cosine decay: peak → 10% of peak
        let progress = (step - warmup) as f64 / (flat_start - warmup) as f64;
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        cfg.lr * (0.1 + 0.9 * cosine)
    } else {
        // Flat tail: hold at 10% of peak
        cfg.lr * 0.1
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Initialise logging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "physllm_train=debug,info".into())
        )
        .with_target(false)
        .compact()
        .init();

    let args = Args::parse();

    // GPU device selection
    let device = select_device()?;
    info!("Device: {:?}", device);

    let dtype = match args.dtype.as_str() {
        "bf16" => DType::BF16,
        "f32"  => DType::F32,
        _      => DType::F16,
    };
    info!("Training dtype: {:?}", dtype);

    // ── Graceful shutdown on Ctrl-C ───────────────────────────────────────────
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = shutdown.clone();
        ctrlc::set_handler(move || {
            warn!("Ctrl-C received — will save checkpoint and exit after this step");
            shutdown.store(true, Ordering::SeqCst);
        }).context("Failed to set Ctrl-C handler")?;
    }

    // ── Dataset ───────────────────────────────────────────────────────────────
    info!("Loading datasets...");
    let train_ds = JsonlDataset::load(&args.train_data, args.max_seq_len, args.seed)?;
    let val_ds   = JsonlDataset::load(&args.val_data,   args.max_seq_len, args.seed)?;
    info!("Train: {} examples  Val: {} examples", train_ds.len(), val_ds.len());

    let mut loader = DataLoader::new(train_ds, args.batch_size, /*shuffle=*/true, args.seed);

    // ── Model + LoRA ──────────────────────────────────────────────────────────
    info!("Loading base model weights from {:?}...", args.model_dir);
    let var_map = VarMap::new();
    let mut lora_model = LoraModel::load(
        &args.model_dir,
        &var_map,
        &device,
        dtype,
        LoraConfig {
            rank:     args.lora_rank,
            alpha:    args.lora_alpha,
            dropout:  args.lora_dropout,
            targets:  args.lora_targets.split(',').map(String::from).collect(),
        },
    )?;
    info!("Model loaded. LoRA parameters: {}", lora_model.trainable_param_count());

    // ── Optimizer ─────────────────────────────────────────────────────────────
    let mut optimizer = AdamW::new(
        var_map.all_vars(),
        AdamWConfig {
            lr:           args.lr,
            beta1:        args.beta1,
            beta2:        args.beta2,
            eps:          args.eps,
            weight_decay: args.weight_decay,
        },
    )?;

    // ── Checkpoint manager ────────────────────────────────────────────────────
    let mut ckpt_mgr = CheckpointManager::new(
        args.output_dir.clone(),
        args.keep_checkpoints,
    );

    // ── Resume ────────────────────────────────────────────────────────────────
    let mut start_step = 0usize;
    if let Some(ref resume_path) = args.resume {
        info!("Resuming from {:?}", resume_path);
        let meta = ckpt_mgr.load_checkpoint(resume_path, &mut lora_model, &mut optimizer)?;
        start_step = meta.step + 1;
        info!("Resumed at step {start_step}");
    }

    // ── Logger ────────────────────────────────────────────────────────────────
    let mut logger = TrainLogger::new(&args.wandb_project, &args)?;

    // ── Training loop ─────────────────────────────────────────────────────────
    info!("Starting training: steps {start_step}–{}", args.total_steps);
    info!("Effective batch size: {} = {} × {} grad_accum",
          args.batch_size * args.grad_accum, args.batch_size, args.grad_accum);

    let pb = ProgressBar::new(args.total_steps as u64);
    pb.set_style(ProgressStyle::with_template(
        "{spinner:.cyan} [{elapsed_precise}] [{bar:40.cyan/blue}] \
         step={pos}/{len} loss={msg}"
    )?.progress_chars("█▇▆▅▄▃▂▁  "));
    pb.set_position(start_step as u64);

    let mut opt_step        = start_step;
    let mut accum_loss      = 0.0f64;
    let mut accum_tokens    = 0usize;
    let mut best_val_ppl    = f64::MAX;

    // Zero gradients before accumulation starts
    optimizer.zero_grad();

    'training: loop {
        for micro_step in 0..args.grad_accum {
            if opt_step >= args.total_steps || shutdown.load(Ordering::SeqCst) {
                break 'training;
            }

            // ── Fetch batch ───────────────────────────────────────────────────
            let batch = loader.next_batch(&device, dtype)?;
            let (input_ids, target_ids, attention_mask) = batch;
            let n_tokens = count_valid_tokens(&attention_mask);

            // ── Forward pass ──────────────────────────────────────────────────
            let logits = lora_model.forward(&input_ids, &attention_mask)?;
            // logits: [batch, seq-1, vocab]

            // ── Cross-entropy loss (with label smoothing) ─────────────────────
            let loss = cross_entropy_loss(
                &logits,
                &target_ids,
                &attention_mask,
                /*label_smoothing=*/0.1,
            )?;

            // Scale loss by 1/grad_accum so gradients add up to the correct mean
            let scaled_loss = (&loss / args.grad_accum as f64)?;

            // ── Backward pass (real autograd via Candle) ──────────────────────
            scaled_loss.backward()?;

            accum_loss   += loss.to_scalar::<f64>()? as f64;
            accum_tokens += n_tokens;
        } // end micro-step loop

        // ── Gradient clipping ─────────────────────────────────────────────────
        let grad_norm = if args.grad_clip > 0.0 {
            optimizer.clip_grad_norm(args.grad_clip)?
        } else { 0.0 };

        // ── Optimizer step ────────────────────────────────────────────────────
        let current_lr = lr_schedule(opt_step, &args);
        optimizer.set_lr(current_lr);
        optimizer.step()?;
        optimizer.zero_grad();

        let avg_loss = accum_loss / args.grad_accum as f64;
        let ppl      = avg_loss.exp();
        accum_loss   = 0.0;
        accum_tokens = 0;

        // ── Logging ───────────────────────────────────────────────────────────
        if opt_step % args.log_every == 0 {
            pb.set_message(format!("{avg_loss:.4}"));
            pb.set_position(opt_step as u64);
            logger.log_train(opt_step, avg_loss, ppl, current_lr, grad_norm)?;
            info!(
                "step={opt_step:>6} loss={avg_loss:.4} ppl={ppl:>8.2} \
                 lr={current_lr:.2e} grad_norm={grad_norm:.3}"
            );
        }

        // ── Evaluation ────────────────────────────────────────────────────────
        if opt_step % args.eval_every == 0 && opt_step > 0 {
            let (val_loss, val_ppl) = evaluate(
                &lora_model, &val_ds, &device, dtype, args.batch_size, args.max_seq_len
            )?;
            info!("EVAL step={opt_step} val_loss={val_loss:.4} val_ppl={val_ppl:.2}");
            logger.log_eval(opt_step, val_loss, val_ppl)?;

            // Save best model
            if val_ppl < best_val_ppl {
                best_val_ppl = val_ppl;
                ckpt_mgr.save_best(&lora_model, &optimizer, opt_step, val_loss, val_ppl)?;
                info!("★ New best val_ppl={val_ppl:.2} — saved best checkpoint");
            }
        }

        // ── Checkpoint ────────────────────────────────────────────────────────
        if opt_step % args.save_every == 0 && opt_step > 0 {
            ckpt_mgr.save(&lora_model, &optimizer, opt_step, avg_loss)?;
            info!("Checkpoint saved at step {opt_step}");
        }

        opt_step += 1;
    }

    // ── Final checkpoint ──────────────────────────────────────────────────────
    pb.finish_with_message("training complete");
    let final_loss = accum_loss / args.grad_accum.max(1) as f64;
    ckpt_mgr.save(&lora_model, &optimizer, opt_step, final_loss)?;

    // ── Merge LoRA into base weights and save final model ─────────────────────
    info!("Merging LoRA adapters into base model weights...");
    lora_model.merge_and_save(&args.output_dir.join("final"))?;
    info!("Final merged model saved to {:?}/final", args.output_dir);
    info!("Training complete. Best val_ppl={best_val_ppl:.2}");

    logger.finish(opt_step, best_val_ppl)?;
    Ok(())
}

// ── Cross-entropy loss with label smoothing ───────────────────────────────────

fn cross_entropy_loss(
    logits:          &Tensor,   // [B, S, V]
    targets:         &Tensor,   // [B, S]
    attention_mask:  &Tensor,   // [B, S]  — 0 = padding, 1 = real token
    label_smoothing: f64,
) -> Result<Tensor> {
    let (b, s, v) = logits.dims3()?;

    // Flatten: [B*S, V] and [B*S]
    let logits_flat  = logits.reshape((b * s, v))?;
    let targets_flat = targets.reshape((b * s,))?;
    let mask_flat    = attention_mask.reshape((b * s,))?;

    // Log-softmax for numerical stability
    let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)?;

    // NLL loss: -log_prob[target]
    let nll = log_probs.gather(&targets_flat.unsqueeze(1)?, 1)?.squeeze(1)?.neg()?;

    // Label smoothing: -(1-ε)*log_p[y] - ε/V * Σ log_p[i]
    // = (1-ε)*nll - ε * mean(log_probs)
    let loss = if label_smoothing > 0.0 {
        let smooth_loss = log_probs.mean(1)?.neg()?;  // mean over vocab
        ((1.0 - label_smoothing) * &nll)?.add(&(label_smoothing * &smooth_loss)?)?
    } else {
        nll
    };

    // Mask out padding positions
    let masked = loss.mul(&mask_flat.to_dtype(loss.dtype())?)?;

    // Mean over non-padding tokens
    let n_real   = mask_flat.sum_all()?.to_scalar::<f32>()? as f64;
    let sum_loss = masked.sum_all()?;
    (sum_loss / n_real).context("Loss computation failed")
}

fn count_valid_tokens(mask: &Tensor) -> usize {
    mask.sum_all()
        .and_then(|t| t.to_scalar::<f32>())
        .map(|v| v as usize)
        .unwrap_or(0)
}

fn select_device() -> Result<Device> {
    #[cfg(feature = "rocm")]
    {
        // Candle uses CUDA API for ROCm via HIP — device 0
        if let Ok(d) = Device::new_cuda(0) {
            info!("Using AMD GPU (ROCm) device 0");
            return Ok(d);
        }
        warn!("ROCm device not available — falling back to CPU");
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            info!("Using CUDA device 0");
            return Ok(d);
        }
    }
    info!("Using CPU");
    Ok(Device::Cpu)
}
