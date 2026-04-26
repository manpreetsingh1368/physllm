//! generate.rs — Autoregressive token generation with sampling strategies.

use crate::{model::PhysLLM, tokenizer::PhysTokenizer, Result, LlmError};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Sampling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Maximum tokens to generate
    pub max_new_tokens:  usize,
    /// Temperature (1.0 = no change, <1 = sharper, >1 = more random)
    pub temperature:     f32,
    /// Top-p nucleus sampling (1.0 = disabled)
    pub top_p:           f32,
    /// Top-k sampling (0 = disabled)
    pub top_k:           usize,
    /// Repetition penalty (1.0 = none)
    pub repetition_penalty: f32,
    /// Stop sequences (text)
    pub stop_sequences:  Vec<String>,
    /// Random seed (None = random)
    pub seed:            Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens:    512,
            temperature:       0.7,
            top_p:             0.9,
            top_k:             50,
            repetition_penalty: 1.1,
            stop_sequences:    vec!["</s>".into(), "<|eot_id|>".into()],
            seed:              None,
        }
    }
}

/// Input to the generation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt:   String,
    pub system:   Option<String>,
    pub sampling: SamplingParams,
}

/// Output from the generation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text:         String,
    pub tokens_in:    usize,
    pub tokens_out:   usize,
    pub finish_reason: FinishReason,
    pub time_ms:      u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason { Length, StopSequence, Eos }

/// Full generation pipeline (prefill + decode).
pub fn generate(
    model:     &PhysLLM,
    tokenizer: &PhysTokenizer,
    request:   &GenerateRequest,
) -> Result<GenerateResponse> {
    let t0 = std::time::Instant::now();

    // ── Format prompt ─────────────────────────────────────────────────────────
    let system = request.system.as_deref()
        .unwrap_or(&model.config.domain.system_prompt);
    let formatted = format_chat_prompt(system, &request.prompt);

    // ── Tokenise ──────────────────────────────────────────────────────────────
    let mut tokens = tokenizer.encode(&formatted)?;
    let tokens_in = tokens.len();

    if tokens_in >= model.config.max_seq_len {
        return Err(LlmError::ContextOverflow {
            tokens: tokens_in,
            max:    model.config.max_seq_len,
        });
    }

    info!("generate: {tokens_in} input tokens, max_new={}", request.sampling.max_new_tokens);

    // ── Prefill ───────────────────────────────────────────────────────────────
    let mut offset = 0;
    let prefill_logits = model.forward(&tokens, offset)?;
    offset = tokens.len();

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut rng = match request.sampling.seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None    => ChaCha8Rng::from_entropy(),
    };

    let mut output_tokens: Vec<u32> = Vec::with_capacity(request.sampling.max_new_tokens);
    let mut logits = prefill_logits;
    let mut finish = FinishReason::Length;

    for _ in 0..request.sampling.max_new_tokens {
        // Apply repetition penalty
        apply_repetition_penalty(&mut logits, &tokens, request.sampling.repetition_penalty);

        // Sample
        let next_token = sample(
            &logits,
            request.sampling.temperature,
            request.sampling.top_p,
            request.sampling.top_k,
            &mut rng,
        );

        debug!("sampled token {next_token}");

        // Check EOS
        if next_token == model.config.eos_token_id {
            finish = FinishReason::Eos;
            break;
        }

        output_tokens.push(next_token);
        tokens.push(next_token);

        // Check stop sequences
        let decoded_so_far = tokenizer.decode(&output_tokens)?;
        if request.sampling.stop_sequences.iter().any(|s| decoded_so_far.contains(s.as_str())) {
            finish = FinishReason::StopSequence;
            break;
        }

        if offset >= model.config.max_seq_len - 1 {
            break;
        }

        // Next forward pass (single token, KV cache used)
        logits = model.forward(&[next_token], offset)?;
        offset += 1;
    }

    let text = tokenizer.decode(&output_tokens)?;
    let tokens_out = output_tokens.len();
    let time_ms = t0.elapsed().as_millis() as u64;

    info!("generated {tokens_out} tokens in {time_ms}ms ({:.1} tok/s)",
          tokens_out as f64 / time_ms as f64 * 1000.0);

    Ok(GenerateResponse { text, tokens_in, tokens_out, finish_reason: finish, time_ms })
}

fn sample(logits: &[f32], temp: f32, top_p: f32, top_k: usize, rng: &mut ChaCha8Rng) -> u32 {
    // Temperature scaling
    let mut scaled: Vec<f32> = if temp == 0.0 {
        // Greedy
        let max_idx = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        return max_idx as u32;
    } else {
        logits.iter().map(|&x| x / temp).collect()
    };

    // Softmax
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    // Top-K filter
    let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
    sorted_indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    if top_k > 0 && top_k < probs.len() {
        for &i in &sorted_indices[top_k..] { probs[i] = 0.0; }
    }

    // Top-P filter
    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (rank, &i) in sorted_indices.iter().enumerate() {
        cumsum += probs[i];
        if cumsum >= top_p {
            cutoff = rank + 1;
            break;
        }
    }
    for &i in &sorted_indices[cutoff..] { probs[i] = 0.0; }

    // Renormalise
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    // Sample
    let r: f32 = rng.gen();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum { return i as u32; }
    }
    *sorted_indices.first().unwrap_or(&0) as u32
}

fn apply_repetition_penalty(logits: &mut [f32], seen: &[u32], penalty: f32) {
    if penalty == 1.0 { return; }
    for &tok in seen {
        let i = tok as usize;
        if i < logits.len() {
            logits[i] = if logits[i] > 0.0 {
                logits[i] / penalty
            } else {
                logits[i] * penalty
            };
        }
    }
}

fn format_chat_prompt(system: &str, user: &str) -> String {
    // Llama 3 chat template
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
         {system}<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\n\
         {user}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}
