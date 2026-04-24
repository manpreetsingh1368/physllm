//! ops.rs — Tensor operations (CPU fallback + ROCm acceleration).

use crate::{DeviceTensor, GpuDevice, BackendError, Result};
use half::f16;

/// f16 matrix multiply: C = A · B  where A is [m, k], B is [k, n], C is [m, n].
pub fn matmul_f16(
    dev: &GpuDevice,
    a: &DeviceTensor<f16>,
    b: &DeviceTensor<f16>,
    c: &mut DeviceTensor<f16>,
) -> Result<()> {
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();
    if b.rows() != k {
        return Err(BackendError::ShapeMismatch(
            format!("matmul: a.cols({k}) != b.rows({})", b.rows())
        ));
    }

    #[cfg(feature = "rocm")]
    unsafe {
        use crate::hip_ffi::*;
        let mut handle: hipblasHandle_t = std::ptr::null_mut();
        if hipblasCreate(&mut handle) != 0 {
            return Err(BackendError::Runtime("hipblasCreate failed".into()));
        }
        hipblasSetStream(handle, dev.raw_stream());

        let alpha = f16::from_f32(1.0);
        let beta  = f16::from_f32(0.0);

        // Note: hipBLAS uses column-major like Fortran; we transpose both
        // operands to get row-major behaviour.
        let err = hipblasHgemm(
            handle,
            HIPBLAS_OP_T, HIPBLAS_OP_T,
            m as i32, n as i32, k as i32,
            &alpha,
            a.raw_ptr() as *const _, k as i32,
            b.raw_ptr() as *const _, n as i32,
            &beta,
            c.raw_ptr() as *mut _, m as i32,
        );
        hipblasDestroy(handle);
        if err != 0 {
            return Err(BackendError::Runtime(format!("hipblasHgemm returned {err}")));
        }
        return Ok(());
    }

    // CPU fallback: naive O(m·k·n) matmul
    #[cfg(not(feature = "rocm"))]
    {
        let a_h = a.copy_to_host()?;
        let b_h = b.copy_to_host()?;
        let mut c_h = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_h[i*k + l].to_f32() * b_h[l*n + j].to_f32();
                }
                c_h[i*n + j] = sum;
            }
        }
        let c_f16: Vec<f16> = c_h.iter().map(|&x| f16::from_f32(x)).collect();
        c.copy_from_host(&c_f16)?;
        Ok(())
    }
}

/// Scaled dot-product attention: O = softmax(Q·K^T · scale) · V
pub fn flash_attention(
    _dev: &GpuDevice,
    q: &DeviceTensor<f16>,
    k: &DeviceTensor<f16>,
    v: &DeviceTensor<f16>,
    out: &mut DeviceTensor<f16>,
    scale: f32,
    causal: bool,
) -> Result<()> {
    let seq = *q.shape().get(q.shape().len().saturating_sub(2)).unwrap_or(&1);
    let d   = *q.shape().last().unwrap_or(&0);

    let q_h: Vec<f32> = q.copy_to_host()?.iter().map(|x| x.to_f32()).collect();
    let k_h: Vec<f32> = k.copy_to_host()?.iter().map(|x| x.to_f32()).collect();
    let v_h: Vec<f32> = v.copy_to_host()?.iter().map(|x| x.to_f32()).collect();
    let mut o_h = vec![0f32; q.numel()];

    for i in 0..seq {
        let mut scores = vec![f32::NEG_INFINITY; seq];
        for j in 0..seq {
            if causal && j > i { continue; }
            scores[j] = (0..d).map(|dd| q_h[i*d+dd] * k_h[j*d+dd]).sum::<f32>() * scale;
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        for dd in 0..d {
            o_h[i*d + dd] = (0..seq).map(|j| exp[j] / sum * v_h[j*d+dd]).sum();
        }
    }

    let o_f16: Vec<f16> = o_h.iter().map(|&x| f16::from_f32(x)).collect();
    out.copy_from_host(&o_f16)
}

/// Apply rotary positional embedding to Q and K tensors in-place.
pub fn rope_embed(
    _dev: &GpuDevice,
    q: &mut DeviceTensor<f16>,
    k: &mut DeviceTensor<f16>,
    offset: usize,
    theta: f32,
) -> Result<()> {
    let apply = |t: &mut DeviceTensor<f16>| -> Result<()> {
        let hd  = *t.shape().last().unwrap_or(&0);
        let seq = *t.shape().get(t.shape().len().saturating_sub(2)).unwrap_or(&1);
        let mut h: Vec<f32> = t.copy_to_host()?.iter().map(|x| x.to_f32()).collect();
        for s in 0..seq {
            let pos = (s + offset) as f32;
            for i in 0..hd/2 {
                let freq = pos / theta.powf(2.0 * i as f32 / hd as f32);
                let (sin, cos) = freq.sin_cos();
                let idx = s * hd + i * 2;
                if idx + 1 < h.len() {
                    let (x0, x1) = (h[idx], h[idx+1]);
                    h[idx]     = x0 * cos - x1 * sin;
                    h[idx + 1] = x0 * sin + x1 * cos;
                }
            }
        }
        let f16s: Vec<f16> = h.iter().map(|&x| f16::from_f32(x)).collect();
        t.copy_from_host(&f16s)
    };
    apply(q)?;
    apply(k)
}

/// RMS normalisation: x = x / sqrt(mean(x²) + eps) * weight
pub fn rms_norm(
    _dev: &GpuDevice,
    x: &mut DeviceTensor<f16>,
    weight: &DeviceTensor<f16>,
    eps: f32,
) -> Result<()> {
    let d = x.cols();
    if weight.len() != d {
        return Err(BackendError::ShapeMismatch(
            format!("rms_norm: weight.len({}) != x.cols({d})", weight.len())
        ));
    }
    let mut h: Vec<f32> = x.copy_to_host()?.iter().map(|v| v.to_f32()).collect();
    let w: Vec<f32> = weight.copy_to_host()?.iter().map(|v| v.to_f32()).collect();
    for row in h.chunks_mut(d) {
        let rms = (row.iter().map(|&v| v*v).sum::<f32>() / d as f32 + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = *v / rms * w.get(i).copied().unwrap_or(1.0);
        }
    }
    let f16s: Vec<f16> = h.iter().map(|&x| f16::from_f32(x)).collect();
    x.copy_from_host(&f16s)
}

/// Fused softmax along the last dimension (in-place).
pub fn softmax_f16(_dev: &GpuDevice, x: &mut DeviceTensor<f16>) -> Result<()> {
    let cols = x.cols();
    if cols == 0 { return Ok(()); }
    let mut h: Vec<f32> = x.copy_to_host()?.iter().map(|v| v.to_f32()).collect();
    for row in h.chunks_mut(cols) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() { *v /= sum; }
        }
    }
    let f16s: Vec<f16> = h.iter().map(|&x| f16::from_f32(x)).collect();
    x.copy_from_host(&f16s)
}
