//! Simplified inference for Mistral 7B

use crate::Result;
use rocm_backend::{GpuDevice, DeviceTensor, ops::matmul_f16};
use half::f16;
use std::sync::Arc;

/// Transpose a weight matrix: [out_dim, in_dim] → [in_dim, out_dim]
fn transpose_weight(weight: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut transposed = vec![f16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = weight[r * cols + c];
        }
    }
    transposed
}

pub fn layer_forward(
    dev: &GpuDevice,
    x: &[f16],
    seq_len: usize,
    hidden_dim: usize,
    wq: &DeviceTensor<f16>,
    wk: &DeviceTensor<f16>,
    wv: &DeviceTensor<f16>,
    wo: &DeviceTensor<f16>,
    w_gate: &DeviceTensor<f16>,  // [14336, 4096]
    w_up: &DeviceTensor<f16>,    // [14336, 4096]
    w_down: &DeviceTensor<f16>,  // [4096, 14336]
) -> Result<Vec<f16>> {
    
    // Upload input
    let gpu_x = DeviceTensor::from_slice(x, &[seq_len, hidden_dim])?;
    
    // Transpose gate and up weights for matmul: [14336, 4096] → [4096, 14336]
    let gate_weights = w_gate.copy_to_host()?;
    let up_weights = w_up.copy_to_host()?;
    
    let gate_transposed = transpose_weight(&gate_weights, 14336, 4096);
    let up_transposed = transpose_weight(&up_weights, 14336, 4096);
    
    let gpu_gate_t = DeviceTensor::from_slice(&gate_transposed, &[4096, 14336])?;
    let gpu_up_t = DeviceTensor::from_slice(&up_transposed, &[4096, 14336])?;
    
    // Gate projection on GPU: [seq_len, 4096] @ [4096, 14336] = [seq_len, 14336]
    let mut gpu_gate_out = DeviceTensor::alloc(&[seq_len, 14336])?;
    matmul_f16(dev, &gpu_x, &gpu_gate_t, &mut gpu_gate_out)?;
    
    // Up projection on GPU
    let mut gpu_up_out = DeviceTensor::alloc(&[seq_len, 14336])?;
    matmul_f16(dev, &gpu_x, &gpu_up_t, &mut gpu_up_out)?;
    
    // Copy back for activation (SiLU is CPU for now)
    let gate_out = gpu_gate_out.copy_to_host()?;
    let up_out = gpu_up_out.copy_to_host()?;
    
    // SiLU(gate) * up
    let intermediate: Vec<f16> = gate_out.iter().zip(up_out.iter())
        .map(|(&g, &u)| {
            let g_f32 = g.to_f32();
            let silu = g_f32 / (1.0 + (-g_f32).exp());
            f16::from_f32(silu * u.to_f32())
        })
        .collect();
    
    // Down projection: [seq_len, 14336] @ [14336, 4096] = [seq_len, 4096]
    // w_down is already [4096, 14336], need to transpose
    let down_weights = w_down.copy_to_host()?;
    let down_transposed = transpose_weight(&down_weights, 4096, 14336);
    let gpu_down_t = DeviceTensor::from_slice(&down_transposed, &[14336, 4096])?;
    
    let gpu_intermediate = DeviceTensor::from_slice(&intermediate, &[seq_len, 14336])?;
    let mut gpu_output = DeviceTensor::alloc(&[seq_len, 4096])?;
    matmul_f16(dev, &gpu_intermediate, &gpu_down_t, &mut gpu_output)?;
    
    // Residual connection
    let mut output = gpu_output.copy_to_host()?;
    for (out, &inp) in output.iter_mut().zip(x.iter()) {
        *out = f16::from_f32(out.to_f32() + inp.to_f32());
    }
    
    Ok(output)
}

pub fn sample_token(logits: &[f32], temperature: f32) -> u32 {
    if temperature == 0.0 {
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap()
    } else {
        let scaled: Vec<f32> = logits.iter()
            .map(|&x| (x / temperature).exp())
            .collect();
        let sum: f32 = scaled.iter().sum();
        let probs: Vec<f32> = scaled.iter().map(|&x| x / sum).collect();
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let rand_val: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (idx, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand_val < cumsum {
                return idx as u32;
            }
        }
        (logits.len() - 1) as u32
    }
}
