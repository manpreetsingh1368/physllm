//! Rotary Position Embeddings (RoPE)

use half::f16;

/// Apply RoPE to query or key tensors
pub fn apply_rope(
    x: &mut [f16],      // [seq_len, num_heads, head_dim]
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    position_offset: usize, // For KV cache
) {
    assert_eq!(x.len(), seq_len * num_heads * head_dim);
    
    for pos in 0..seq_len {
        let actual_pos = (pos + position_offset) as f32;
        
        for head in 0..num_heads {
            for i in 0..(head_dim / 2) {
                let freq = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
                let angle = actual_pos * freq;
                let (sin, cos) = angle.sin_cos();
                
                let idx_base = pos * num_heads * head_dim + head * head_dim;
                let idx_0 = idx_base + 2 * i;
                let idx_1 = idx_base + 2 * i + 1;
                
                let x0 = x[idx_0].to_f32();
                let x1 = x[idx_1].to_f32();
                
                // Rotation matrix
                x[idx_0] = f16::from_f32(x0 * cos - x1 * sin);
                x[idx_1] = f16::from_f32(x0 * sin + x1 * cos);
            }
        }
    }
}
