//! RMS Normalization

use half::f16;

/// Apply RMS normalization
pub fn rms_norm(
    x: &[f16],              // Input: [seq_len, hidden_dim]
    weight: &[f16],         // Weight: [hidden_dim]
    seq_len: usize,
    hidden_dim: usize,
    eps: f32,
) -> Vec<f16> {
    let mut output = vec![f16::ZERO; seq_len * hidden_dim];
    
    for s in 0..seq_len {
        let start = s * hidden_dim;
        let end = start + hidden_dim;
        let row = &x[start..end];
        
        // Compute RMS
        let sum_sq: f32 = row.iter().map(|&v| {
            let f = v.to_f32();
            f * f
        }).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
        
        // Normalize and scale
        for i in 0..hidden_dim {
            let normalized = row[i].to_f32() / rms;
            output[start + i] = f16::from_f32(normalized * weight[i].to_f32());
        }
    }
    
    output
}
