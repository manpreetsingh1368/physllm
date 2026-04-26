// crates/trainer/src/evaluate.rs
//
// Validation evaluation: compute loss and perplexity over the full val set.
// Uses the same cross-entropy logic as training but without backward passes.

use crate::{
    lora::LoraModel,
    dataset::{JsonlDataset, DataLoader},
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops;
use tracing::debug;

/// Evaluate the model on the validation set.
/// Returns (val_loss, val_perplexity).
pub fn evaluate(
    model:       &LoraModel,
    val_ds:      &JsonlDataset,
    device:      &Device,
    dtype:       DType,
    batch_size:  usize,
    max_seq_len: usize,
) -> Result<(f64, f64)> {
    let n_batches = (val_ds.len() + batch_size - 1) / batch_size;
    // Cap at 200 batches to keep eval fast
    let n_batches = n_batches.min(200);

    // Create a non-shuffled loader for reproducible eval
    let val_copy = JsonlDataset {
        examples:    val_ds.examples.clone(),
        max_seq_len,
    };
    let mut loader = DataLoader::new(val_copy, batch_size, /*shuffle=*/false, 0);

    let mut total_loss   = 0.0f64;
    let mut total_tokens = 0usize;

    for batch_idx in 0..n_batches {
        let (input_ids, target_ids, attention_mask) = loader.next_batch(device, dtype)?;
        let (b, s) = input_ids.dims2()?;
        let vocab = model.config.vocab_size;

        // Forward (no gradient tracking needed)
        let logits = model.forward(&input_ids, &attention_mask)?;
        // logits: [B, S-1, V]  (model already shifts internally)

        let (_, s_out, _) = logits.dims3()?;

        // Flatten
        let logits_flat  = logits.reshape((b * s_out, vocab))?;
        let targets_flat = target_ids.reshape((b * s_out,))?;
        let mask_flat    = attention_mask.reshape((b * s_out,))?;

        // Only evaluate on non-padding positions with valid targets (≥ 0)
        let log_probs = ops::log_softmax(&logits_flat, 1)?;
        let nll = log_probs
            .gather(&targets_flat.clamp(0i64, vocab as i64 - 1)?.unsqueeze(1)?, 1)?
            .squeeze(1)?
            .neg()?;

        // Mask: ignore padding (mask=0) and ignored positions (target=-100)
        let valid_mask = mask_flat
            .mul(&targets_flat.ge(0i64)?.to_dtype(DType::F32)?)?;

        let batch_loss   = nll.mul(&valid_mask)?.sum_all()?.to_scalar::<f32>()? as f64;
        let batch_tokens = valid_mask.sum_all()?.to_scalar::<f32>()? as usize;

        total_loss   += batch_loss;
        total_tokens += batch_tokens;

        debug!("Eval batch {}/{n_batches}: tokens={batch_tokens}", batch_idx + 1);
    }

    if total_tokens == 0 {
        return Ok((f64::INFINITY, f64::INFINITY));
    }

    let avg_loss = total_loss / total_tokens as f64;
    let ppl      = avg_loss.exp();
    Ok((avg_loss, ppl))
}
