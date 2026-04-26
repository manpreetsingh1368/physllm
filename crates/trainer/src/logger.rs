// crates/trainer/src/logger.rs
//
// Training metrics logger.
// Writes JSONL metrics to disk and optionally to Weights & Biases.

use crate::Args;
use anyhow::Result;
use std::{fs::{File, OpenOptions}, io::Write, path::PathBuf};
use chrono::Utc;
use tracing::debug;

/// Logged metrics for one training step.
#[derive(serde::Serialize)]
struct TrainMetrics {
    step:      usize,
    loss:      f64,
    ppl:       f64,
    lr:        f64,
    grad_norm: f64,
    timestamp: String,
    kind:      &'static str,
}

pub struct TrainLogger {
    log_file:   Option<File>,
    log_path:   PathBuf,
    project:    String,
    wandb_run:  Option<String>,
}

impl TrainLogger {
    pub fn new(project: &str, args: &Args) -> Result<Self> {
        let log_path = args.output_dir.join("train_log.jsonl");
        std::fs::create_dir_all(&args.output_dir)?;

        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        // Initialise W&B if project name is provided and wandb crate is compiled in
        let wandb_run = None;
        #[cfg(feature = "wandb-log")]
        let wandb_run = if !project.is_empty() {
            // wandb::init(project, None) returns a run ID
            match wandb::init(project, None) {
                Ok(run_id) => {
                    tracing::info!("W&B run: {run_id}");
                    Some(run_id)
                }
                Err(e) => {
                    tracing::warn!("W&B init failed: {e}");
                    None
                }
            }
        } else { None };

        Ok(Self {
            log_file:  Some(log_file),
            log_path,
            project:   project.to_string(),
            wandb_run,
        })
    }

    pub fn log_train(
        &mut self,
        step:      usize,
        loss:      f64,
        ppl:       f64,
        lr:        f64,
        grad_norm: f64,
    ) -> Result<()> {
        let metrics = TrainMetrics {
            step, loss, ppl, lr, grad_norm,
            timestamp: Utc::now().to_rfc3339(),
            kind: "train",
        };

        // Write to JSONL
        if let Some(ref mut f) = self.log_file {
            writeln!(f, "{}", serde_json::to_string(&metrics)?)?;
        }

        // W&B
        #[cfg(feature = "wandb-log")]
        if self.wandb_run.is_some() {
            let _ = wandb::log(&[
                ("train/loss",      loss),
                ("train/ppl",       ppl),
                ("train/lr",        lr),
                ("train/grad_norm", grad_norm),
            ], step);
        }

        Ok(())
    }

    pub fn log_eval(&mut self, step: usize, loss: f64, ppl: f64) -> Result<()> {
        let metrics = serde_json::json!({
            "step": step, "loss": loss, "ppl": ppl,
            "timestamp": Utc::now().to_rfc3339(), "kind": "eval",
        });
        if let Some(ref mut f) = self.log_file {
            writeln!(f, "{}", metrics)?;
        }

        #[cfg(feature = "wandb-log")]
        if self.wandb_run.is_some() {
            let _ = wandb::log(&[("eval/loss", loss), ("eval/ppl", ppl)], step);
        }

        Ok(())
    }

    pub fn finish(&mut self, final_step: usize, best_val_ppl: f64) -> Result<()> {
        let summary = serde_json::json!({
            "final_step": final_step,
            "best_val_ppl": best_val_ppl,
            "completed_at": Utc::now().to_rfc3339(),
            "kind": "summary",
        });
        if let Some(ref mut f) = self.log_file {
            writeln!(f, "{}", summary)?;
        }
        tracing::info!("Training log saved to {:?}", self.log_path);
        Ok(())
    }
}

/// Parse and pretty-print a train_log.jsonl file.
pub fn print_log_summary(log_path: &PathBuf) -> Result<()> {
    use std::io::{BufRead, BufReader};
    let file = File::open(log_path)?;
    let reader = BufReader::new(file);

    let mut train_steps = Vec::new();
    let mut eval_steps  = Vec::new();

    for line in reader.lines().filter_map(|l| l.ok()) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            match v["kind"].as_str() {
                Some("train") => train_steps.push(v),
                Some("eval")  => eval_steps.push(v),
                _ => {}
            }
        }
    }

    println!("Training summary:");
    println!("  Total steps: {}", train_steps.len());
    if let Some(last) = train_steps.last() {
        println!("  Final train loss: {:.4}", last["loss"].as_f64().unwrap_or(0.0));
        println!("  Final train ppl:  {:.2}", last["ppl"].as_f64().unwrap_or(0.0));
    }
    if let Some(best_eval) = eval_steps.iter()
        .min_by(|a, b| a["ppl"].as_f64().unwrap_or(f64::MAX)
                        .partial_cmp(&b["ppl"].as_f64().unwrap_or(f64::MAX))
                        .unwrap_or(std::cmp::Ordering::Equal))
    {
        println!("  Best val ppl:     {:.2} (step {})",
                 best_eval["ppl"].as_f64().unwrap_or(0.0),
                 best_eval["step"].as_u64().unwrap_or(0));
    }

    Ok(())
}
