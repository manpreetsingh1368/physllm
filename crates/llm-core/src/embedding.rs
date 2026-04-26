//! embedding.rs — Token and positional embedding utilities.

use half::f16;

/// Look up token embeddings for a batch of token IDs.
///
/// # Arguments
/// * `table`  — embedding matrix [vocab_size, hidden_dim] (on host)
/// * `tokens` — token IDs [seq_len]
///
/// Returns a [seq_len, hidden_dim] slice.
pub fn embed_tokens(table: &[f16], tokens: &[u32], hidden_dim: usize) -> Vec<f16> {
    let mut out = Vec::with_capacity(tokens.len() * hidden_dim);
    for &tok in tokens {
        let off = tok as usize * hidden_dim;
        out.extend_from_slice(&table[off..off + hidden_dim]);
    }
    out
}

/// Scale embeddings by sqrt(hidden_dim) as in the original Transformer paper.
pub fn scale_embeddings(embeddings: &mut [f16], hidden_dim: usize) {
    let scale = f16::from_f32((hidden_dim as f32).sqrt());
    for v in embeddings.iter_mut() {
        *v = f16::from_f32(v.to_f32() * scale.to_f32());
    }
}

/// Sinusoidal positional encoding (original "Attention Is All You Need").
/// Returns a [seq_len, hidden_dim] table.
pub fn sinusoidal_pe(seq_len: usize, hidden_dim: usize) -> Vec<f32> {
    let mut pe = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        for i in (0..hidden_dim).step_by(2) {
            let theta = pos as f32 / 10_000_f32.powf(i as f32 / hidden_dim as f32);
            pe[pos * hidden_dim + i]     = theta.sin();
            if i + 1 < hidden_dim {
                pe[pos * hidden_dim + i + 1] = theta.cos();
            }
        }
    }
    pe
}
