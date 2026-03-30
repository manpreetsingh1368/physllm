//! ffn.rs — SwiGLU Feed-Forward Network (used in Llama / Mistral).
//!
//! FFN(x) = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )
//!
//! SiLU(x) = x * σ(x)  where σ is the sigmoid function.

use crate::Result;
use rocm_backend::{GpuDevice, DeviceTensor, matmul_f16};
use half::f16;

/// SwiGLU FFN forward pass (in-place on existing allocations for efficiency).
pub fn swiglu_ffn(
    device:    &GpuDevice,
    x:         &DeviceTensor<f16>,     // [seq, hidden]
    w_gate:    &DeviceTensor<f16>,     // [hidden, intermediate]
    w_up:      &DeviceTensor<f16>,     // [hidden, intermediate]
    w_down:    &DeviceTensor<f16>,     // [intermediate, hidden]
    seq_len:   usize,
    hidden:    usize,
    inter:     usize,
) -> Result<DeviceTensor<f16>> {
    let mut gate_out = DeviceTensor::<f16>::alloc(&[seq_len, inter])?;
    let mut up_out   = DeviceTensor::<f16>::alloc(&[seq_len, inter])?;

    matmul_f16(device, x, w_gate, &mut gate_out)?;
    matmul_f16(device, x, w_up,   &mut up_out)?;

    // SiLU(gate) ⊙ up — applied on host for non-ROCm builds, on GPU via kernel otherwise
    let gate_h = gate_out.copy_to_host()?;
    let up_h   = up_out.copy_to_host()?;
    let swiglu: Vec<f16> = gate_h.iter().zip(up_h.iter()).map(|(&g, &u)| {
        let gf = g.to_f32();
        let uf = u.to_f32();
        let silu = gf / (1.0 + (-gf).exp());
        f16::from_f32(silu * uf)
    }).collect();

    let swiglu_t = DeviceTensor::from_slice(&swiglu, &[seq_len, inter])?;
    let mut out  = DeviceTensor::<f16>::alloc(&[seq_len, hidden])?;
    matmul_f16(device, &swiglu_t, w_down, &mut out)?;

    Ok(out)
}

/// Apply SiLU activation element-wise (CPU path).
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SiLU to a slice in-place.
pub fn silu_inplace(v: &mut [f32]) {
    for x in v.iter_mut() { *x = silu(*x); }
}
