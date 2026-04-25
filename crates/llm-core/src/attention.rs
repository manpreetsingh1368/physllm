//! Multi-head attention with GQA and KV cache

use half::f16;
use crate::rope::apply_rope;

pub struct KVCache {
    pub k: Vec<f16>,  // [max_seq_len, num_kv_heads, head_dim]
    pub v: Vec<f16>,  // [max_seq_len, num_kv_heads, head_dim]
    pub current_len: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            k: vec![f16::ZERO; max_seq_len * num_kv_heads * head_dim],
            v: vec![f16::ZERO; max_seq_len * num_kv_heads * head_dim],
            current_len: 0,
        }
    }
}

/// Compute attention: Q @ K^T @ V with RoPE and KV cache
pub fn attention(
    q: &[f16],          // Query: [seq_len, num_heads, head_dim]
    k_new: &[f16],      // Key: [seq_len, num_kv_heads, head_dim]
    v_new: &[f16],      // Value: [seq_len, num_kv_heads, head_dim]
    kv_cache: &mut KVCache,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) -> Vec<f16> {
    
    // Apply RoPE to Q and K
    let mut q_rope = q.to_vec();
    let mut k_rope = k_new.to_vec();
    
    apply_rope(&mut q_rope, seq_len, num_heads, head_dim, rope_theta, kv_cache.current_len);
    apply_rope(&mut k_rope, seq_len, num_kv_heads, head_dim, rope_theta, kv_cache.current_len);
    
    // Append new K, V to cache
    let cache_start = kv_cache.current_len * num_kv_heads * head_dim;
    for i in 0..(seq_len * num_kv_heads * head_dim) {
        kv_cache.k[cache_start + i] = k_rope[i];
        kv_cache.v[cache_start + i] = v_new[i];
    }
    kv_cache.current_len += seq_len;
    
    let total_seq = kv_cache.current_len;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Output: [seq_len, num_heads, head_dim]
    let mut output = vec![f16::ZERO; seq_len * num_heads * head_dim];
    
    // For each query position
    for q_pos in 0..seq_len {
        // For each query head
        for h in 0..num_heads {
            let kv_head = h * num_kv_heads / num_heads; // GQA
            
            let q_offset = (q_pos * num_heads + h) * head_dim;
            let q_vec = &q_rope[q_offset..q_offset + head_dim];
            
            // Compute attention scores over all KV positions
            let mut scores = vec![0.0f32; total_seq];
            for kv_pos in 0..total_seq {
                let k_offset = (kv_pos * num_kv_heads + kv_head) * head_dim;
                let k_vec = &kv_cache.k[k_offset..k_offset + head_dim];
                
                let dot: f32 = q_vec.iter().zip(k_vec.iter())
                    .map(|(q, k)| q.to_f32() * k.to_f32())
                    .sum();
                scores[kv_pos] = dot * scale;
            }
            
            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let attn_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();
            
            // Weighted sum of values
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for kv_pos in 0..total_seq {
                    let v_offset = (kv_pos * num_kv_heads + kv_head) * head_dim;
                    sum += attn_weights[kv_pos] * kv_cache.v[v_offset + d].to_f32();
                }
                let out_idx = (q_pos * num_heads + h) * head_dim + d;
                output[out_idx] = f16::from_f32(sum);
            }
        }
    }
    
    output
}
