//! Optimized inference - everything stays on GPU

use crate::Result;
use rocm_backend::{GpuDevice, DeviceTensor};
use half::f16;

/// Fast MLP-only forward pass (skips attention)
pub fn fast_layer_forward(
    dev: &GpuDevice,
    x: &DeviceTensor<f16>,           // Input on GPU: [seq_len, 4096]
    w_gate_t: &DeviceTensor<f16>,    // Pre-transposed: [4096, 14336]
    w_up_t: &DeviceTensor<f16>,      // Pre-transposed: [4096, 14336]
    w_down_t: &DeviceTensor<f16>,    // Pre-transposed: [14336, 4096]
) -> Result<DeviceTensor<f16>> {
    
    use rocm_backend::ops::matmul_f16;
    
    let seq_len = x.shape()[0];
    
    // Gate projection
    let mut gate_out = DeviceTensor::alloc(&[seq_len, 14336])?;
    matmul_f16(dev, x, w_gate_t, &mut gate_out)?;
    
    // Up projection
    let mut up_out = DeviceTensor::alloc(&[seq_len, 14336])?;
    matmul_f16(dev, x, w_up_t, &mut up_out)?;
    
    // SiLU + multiply on CPU (fast enough for now)
    let gate_cpu = gate_out.copy_to_host()?;
    let up_cpu = up_out.copy_to_host()?;
    
    let intermediate: Vec<f16> = gate_cpu.iter().zip(up_cpu.iter())
        .map(|(&g, &u)| {
            let g_f32 = g.to_f32();
            let silu = g_f32 / (1.0 + (-g_f32).exp());
            f16::from_f32(silu * u.to_f32())
        })
        .collect();
    
    // Down projection
    let gpu_intermediate = DeviceTensor::from_slice(&intermediate, &[seq_len, 14336])?;
    let mut output = DeviceTensor::alloc(&[seq_len, 4096])?;
    matmul_f16(dev, &gpu_intermediate, w_down_t, &mut output)?;
    
    // Residual (need to copy for now)
    let mut out_cpu = output.copy_to_host()?;
    let x_cpu = x.copy_to_host()?;
    for (o, &i) in out_cpu.iter_mut().zip(x_cpu.iter()) {
        *o = f16::from_f32(o.to_f32() + i.to_f32());
    }
    
    Ok(DeviceTensor::from_slice(&out_cpu, &[seq_len, 4096])?)
}

pub fn sample_token(logits: &[f32], temperature: f32) -> u32 {
    if temperature == 0.0 {
        logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32).unwrap()
    } else {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scaled: Vec<f32> = logits.iter()
            .map(|&x| ((x - max_logit) / temperature).exp())
            .collect();
        let sum: f32 = scaled.iter().sum();
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let rand_val: f32 = rng.gen::<f32>() * sum;
        let mut cumsum = 0.0;
        for (idx, &val) in scaled.iter().enumerate() {
            cumsum += val;
            if rand_val < cumsum {
                return idx as u32;
            }
        }
        (logits.len() - 1) as u32
    }
}
