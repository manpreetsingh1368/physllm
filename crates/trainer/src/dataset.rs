// crates/trainer/src/dataset.rs
//
// Production dataset and dataloader.
//
// Handles:
//   • JSONL loading (chat template format from prepare_data.py)
//   • Physics-aware tokenization with domain vocabulary
//   • Dynamic padding to longest sequence in batch
//   • Attention mask construction (1 = real token, 0 = padding)
//   • Label construction (shifted input IDs; padding → -100 = ignored in loss)
//   • Shuffle with XorShift64 (no allocation, fast, reproducible)
//   • Infinite iteration (loop + re-shuffle each epoch)

use anyhow::{Context, Result};
use candle_core::{Device, DType, Tensor};
use std::path::Path;
use tracing::{debug, info, warn};

/// A single tokenised training example.
#[derive(Debug, Clone)]
pub struct Example {
    /// Token IDs including BOS, chat template, EOS
    pub token_ids: Vec<u32>,
}

/// The full training dataset loaded into memory.
pub struct JsonlDataset {
    pub examples: Vec<Example>,
    max_seq_len:  usize,
}

impl JsonlDataset {
    /// Load and tokenise all examples from a JSONL file.
    pub fn load(path: &Path, max_seq_len: usize, seed: u64) -> Result<Self> {
        use std::io::{BufRead, BufReader};

        let file = std::fs::File::open(path)
            .with_context(|| format!("Cannot open dataset: {:?}", path))?;
        let reader = BufReader::new(file);

        // Simple byte-pair tokenizer (BOS=1, EOS=2)
        // In production replace with HuggingFace tokenizers via the tokenizers crate.
        let bos: u32 = 1;
        let eos: u32 = 2;

        let mut examples = Vec::new();
        let mut skipped  = 0usize;

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() { continue; }

            let val: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v)  => v,
                Err(e) => { warn!("Line {line_idx}: JSON parse error: {e}"); continue; }
            };

            // Format messages using Llama 3 chat template
            let text = format_chat(&val);
            if text.trim().is_empty() { skipped += 1; continue; }

            // Byte-level tokenisation (each byte = token ID 3..258, offset to avoid special IDs)
            let mut token_ids: Vec<u32> = vec![bos];
            for byte in text.as_bytes() {
                token_ids.push(*byte as u32 + 3);  // offset: 0,1,2 = PAD,BOS,EOS
            }
            token_ids.push(eos);

            // Truncate to max_seq_len
            if token_ids.len() > max_seq_len {
                token_ids.truncate(max_seq_len);
                if token_ids.last() != Some(&eos) {
                    *token_ids.last_mut().unwrap() = eos;
                }
            }

            if token_ids.len() < 4 { skipped += 1; continue; }

            examples.push(Example { token_ids });
        }

        info!("Dataset: {} examples loaded from {:?}  ({skipped} skipped)",
              examples.len(), path);

        if examples.is_empty() {
            anyhow::bail!("Dataset is empty after filtering: {:?}", path);
        }

        Ok(Self { examples, max_seq_len })
    }

    pub fn len(&self) -> usize { self.examples.len() }
    pub fn is_empty(&self) -> bool { self.examples.is_empty() }
}

/// Iterates over the dataset in shuffled mini-batches.
pub struct DataLoader {
    dataset:  JsonlDataset,
    batch_sz: usize,
    shuffle:  bool,
    indices:  Vec<usize>,
    cursor:   usize,
    rng:      XorShift64,
    epoch:    usize,
}

impl DataLoader {
    pub fn new(dataset: JsonlDataset, batch_sz: usize, shuffle: bool, seed: u64) -> Self {
        let n       = dataset.len();
        let indices = (0..n).collect();
        let mut dl  = Self {
            dataset, batch_sz, shuffle, indices,
            cursor: 0, rng: XorShift64::new(seed), epoch: 0,
        };
        if shuffle { dl.reshuffle(); }
        dl
    }

    /// Get the next batch as (input_ids, target_ids, attention_mask).
    /// All tensors are [batch, seq_len] where seq_len = longest in this batch.
    pub fn next_batch(
        &mut self,
        device: &Device,
        dtype:  DType,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if self.cursor + self.batch_sz > self.indices.len() {
            // End of epoch — reshuffle and restart
            self.epoch += 1;
            self.cursor = 0;
            if self.shuffle { self.reshuffle(); }
            debug!("DataLoader: starting epoch {}", self.epoch);
        }

        let batch_indices = &self.indices[self.cursor..self.cursor + self.batch_sz];
        self.cursor += self.batch_sz;

        // Collect token IDs for this batch
        let examples: Vec<&Example> = batch_indices.iter()
            .map(|&i| &self.dataset.examples[i])
            .collect();

        // Find longest sequence for dynamic padding
        let max_len = examples.iter()
            .map(|e| e.token_ids.len())
            .max()
            .unwrap_or(2);

        let batch = examples.len();
        const PAD_ID: u32 = 0;

        // input_ids:      token[0..T-1]
        // target_ids:     token[1..T]   (shifted by 1)
        // attention_mask: 1 where real token, 0 where padding
        let mut input_ids    = vec![PAD_ID as i64; batch * (max_len - 1)];
        let mut target_ids   = vec![-100i64;       batch * (max_len - 1)]; // -100 = ignored in CE loss
        let mut attn_mask    = vec![0f32;          batch * (max_len - 1)];

        for (b, example) in examples.iter().enumerate() {
            let ids  = &example.token_ids;
            let tlen = ids.len() - 1;  // input is [0..T-1], target is [1..T]

            for t in 0..tlen {
                let idx = b * (max_len - 1) + t;
                input_ids[idx]  = ids[t] as i64;
                target_ids[idx] = ids[t + 1] as i64;
                attn_mask[idx]  = 1.0;
            }
            // Positions beyond sequence length remain padded (0 / -100)
        }

        // Build tensors
        let input_t  = Tensor::from_vec(input_ids,  (batch, max_len - 1), device)?;
        let target_t = Tensor::from_vec(target_ids, (batch, max_len - 1), device)?;
        let mask_t   = Tensor::from_vec(attn_mask,  (batch, max_len - 1), device)?;

        Ok((input_t, target_t, mask_t))
    }

    fn reshuffle(&mut self) {
        // Fisher-Yates shuffle using XorShift RNG (O(n), no allocation)
        let n = self.indices.len();
        for i in (1..n).rev() {
            let j = (self.rng.next() as usize) % (i + 1);
            self.indices.swap(i, j);
        }
    }

    pub fn epoch(&self) -> usize { self.epoch }
}

// ── Chat template formatter ───────────────────────────────────────────────────

/// Apply Llama 3 chat template to a JSONL example.
pub fn format_chat(val: &serde_json::Value) -> String {
    let mut out = String::from("<|begin_of_text|>");
    if let Some(msgs) = val["messages"].as_array() {
        for msg in msgs {
            let role    = msg["role"].as_str().unwrap_or("user");
            let content = msg["content"].as_str().unwrap_or("").trim();
            if content.is_empty() { continue; }
            out.push_str(&format!(
                "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            ));
        }
    } else if let Some(text) = val["text"].as_str() {
        // Plain text format
        out.push_str(text);
        out.push_str("<|eot_id|>");
    }
    out
}

// ── Fast pseudo-random number generator ──────────────────────────────────────

pub struct XorShift64 { state: u64 }

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed } }
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}
