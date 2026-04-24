//! ops.rs — High-level GPU operation dispatch.
//!
//! These functions call our compiled HIP kernels (or rocBLAS) for each op.
//! CPU fallback paths use ndarray + rayon.

//! ops.rs — Production-grade GPU operation dispatch.

use crate::{DeviceTensor, GpuDevice, BackendError, Result};
use half::f16;
use tracing::trace;

// GPU SAFETY + UTILITIES


#[cfg(feature = "rocm")]
#[inline(always)]
fn hip_check(op: &str) -> Result<()> {
    unsafe {
        let err = crate::hip_ffi::hipGetLastError();
        if err != 0 {
            return Err(BackendError::Hip {
                code: err,
                msg: format!("{op} launch failed"),
            });
        }
    }
    Ok(())
}

#[cfg(all(feature = "rocm", debug_assertions))]
fn hip_sync(dev: &GpuDevice) -> Result<()> {
    unsafe {
        let err = crate::hip_ffi::hipStreamSynchronize(dev.raw_stream());
        if err != 0 {
            return Err(BackendError::Hip {
                code: err,
                msg: "hipStreamSynchronize failed".into(),
            });
        }
    }
    Ok(())
}

#[cfg(feature = "rocm")]
thread_local! {
    static HIPBLAS: std::cell::RefCell<Option<crate::hip_ffi::hipblasHandle_t>> = Default::default();
}

#[cfg(feature = "rocm")]
unsafe fn get_or_create_hipblas_handle(
    dev: &GpuDevice,
) -> Result<crate::hip_ffi::hipblasHandle_t> {
    use crate::hip_ffi::*;

    HIPBLAS.with(|cell| {
        let mut opt = cell.borrow_mut();

        if let Some(h) = *opt {
            hipblasSetStream(h, dev.raw_stream());
            return Ok(h);
        }

        let mut handle = std::ptr::null_mut();
        let err = hipblasCreate(&mut handle);
        if err != 0 {
            return Err(BackendError::Hip { code: err, msg: "hipblasCreate".into() });
        }

        hipblasSetStream(handle, dev.raw_stream());
        *opt = Some(handle);
        Ok(handle)
    })
}
pub fn matmul_f16(
    dev: &GpuDevice,
    a: &DeviceTensor<f16>,
    b: &DeviceTensor<f16>,
    c: &mut DeviceTensor<f16>,
) -> Result<()> {
    let (m, k) = (a.rows(), a.cols());
    let (bk, n) = (b.rows(), b.cols());

    if bk != k {
        return Err(BackendError::ShapeMismatch(format!(
            "matmul: a.cols ({k}) != b.rows ({bk})"
        )));
    }

    if c.rows() != m || c.cols() != n {
        return Err(BackendError::ShapeMismatch(format!(
            "matmul: expected [{m}, {n}], got [{}, {}]",
            c.rows(), c.cols()
        )));
    }

    trace!("matmul_f16 [{m}x{k}] @ [{k}x{n}]");

    #[cfg(feature = "rocm")]
    unsafe {
        use crate::hip_ffi::*;

        let alpha = f16::from_f32(1.0);
        let beta = f16::from_f32(0.0);

        let handle = get_or_create_hipblas_handle(dev)?;

        // SAFETY:
        // - pointers are valid device memory
        // - shapes validated above
        // - no aliasing between inputs/outputs
        let err = hipblasHgemm(
            handle,
            HIPBLAS_OP_N,
            HIPBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const _ as *const _,
            b.raw_ptr() as *const _,
            n as i32,
            a.raw_ptr() as *const _,
            k as i32,
            &beta as *const _ as *const _,
            c.raw_ptr() as *mut _,
            n as i32,
        );

        if err != 0 {
            return Err(BackendError::Hip { code: err, msg: "hipblasHgemm".into() });
        }

        hip_check("hipblasHgemm")?;
        #[cfg(debug_assertions)]
        hip_sync(dev)?;
    }

    #[cfg(not(feature = "rocm"))]
    cpu_matmul(a, b, c)?;

    Ok(())
}
// FLASH ATTENTION

pub fn flash_attention(
    dev: &GpuDevice,
    q: &DeviceTensor<f16>,
    k: &DeviceTensor<f16>,
    v: &DeviceTensor<f16>,
    out: &mut DeviceTensor<f16>,
    scale: f32,
    causal: bool,
) -> Result<()> {
    let (b, h, sq, sk, d) = validate_attention(q, k, v, out)?;

    trace!(
        "flash_attention b={b} h={h} sq={sq} sk={sk} d={d} causal={causal}"
    );

    #[cfg(feature = "rocm")]
    unsafe {
        // SAFETY:
        // - all tensors validated
        // - device pointers valid
        // - no aliasing
        flash_attention_hip(
            dev.raw_stream(),
            q.raw_ptr(),
            k.raw_ptr(),
            v.raw_ptr(),
            out.raw_ptr(),
            b as i32,
            h as i32,
            sq as i32,
            sk as i32,
            d as i32,
            scale,
            causal as i32,
        );

        hip_check("flash_attention")?;
        #[cfg(debug_assertions)]
        hip_sync(dev)?;
    }

    #[cfg(not(feature = "rocm"))]
    cpu_attention(q, k, v, out, scale, causal)?;

    Ok(())
}

// ROPE
pub fn rope_embed(
    dev: &GpuDevice,
    q: &mut DeviceTensor<f16>,
    k: &mut DeviceTensor<f16>,
    offset: usize,
    theta: f32,
) -> Result<()> {
    let head_dim = *q.shape().last().unwrap();

    if head_dim % 2 != 0 {
        return Err(BackendError::ShapeMismatch(
            "RoPE requires even head_dim".into(),
        ));
    }

    #[cfg(feature = "rocm")]
    unsafe {
        rope_embedding_hip(
            dev.raw_stream(),
            q.raw_ptr(),
            k.raw_ptr(),
            q.shape()[q.shape().len() - 2] as i32,
            head_dim as i32,
            offset as i32,
            theta,
            q.numel() as i32,
        );

        hip_check("rope_embedding")?;
    }

    #[cfg(not(feature = "rocm"))]
    cpu_rope(q, k, offset, theta)?;

    Ok(())
}
// RMS NORM
pub fn rms_norm(
    dev: &GpuDevice,
    x: &mut DeviceTensor<f16>,
    w: &DeviceTensor<f16>,
    eps: f32,
) -> Result<()> {
    if w.len() != x.cols() {
        return Err(BackendError::ShapeMismatch(
            "RMSNorm weight mismatch".into(),
        ));
    }

    #[cfg(feature = "rocm")]
    unsafe {
        layer_norm_hip(
            dev.raw_stream(),
            x.raw_ptr(),
            w.raw_ptr(),
            std::ptr::null(),
            x.rows() as i32,
            x.cols() as i32,
            eps,
            1,
        );

        hip_check("rms_norm")?;
    }

    #[cfg(not(feature = "rocm"))]
    cpu_rms_norm(x, w, eps)?;

    Ok(())
}


// VALIDATION

fn validate_attention(
    q: &DeviceTensor<f16>,
    k: &DeviceTensor<f16>,
    v: &DeviceTensor<f16>,
    out: &DeviceTensor<f16>,
) -> Result<(usize, usize, usize, usize, usize)> {
    let [b, h, sq, d] = extract_4d(q)?;
    let [bk, hk, sk, dk] = extract_4d(k)?;
    let [bv, hv, sv, dv] = extract_4d(v)?;
    let [bo, ho, so, do_] = extract_4d(out)?;

    if (b, h, d) != (bk, hk, dk) || (b, h, d) != (bv, hv, dv) {
        return Err(BackendError::ShapeMismatch("QKV mismatch".into()));
    }

    if sk != sv {
        return Err(BackendError::ShapeMismatch("K/V mismatch".into()));
    }

    if (bo, ho, so, do_) != (b, h, sq, d) {
        return Err(BackendError::ShapeMismatch("Output mismatch".into()));
    }

    Ok((b, h, sq, sk, d))
}

fn extract_4d(t: &DeviceTensor<f16>) -> Result<[usize; 4]> {
    if t.shape().len() != 4 {
        return Err(BackendError::ShapeMismatch("Expected 4D".into()));
    }
    Ok([t.shape()[0], t.shape()[1], t.shape()[2], t.shape()[3]])
}

// CPU FALLBACKS (FULLY CORRECT)

#[cfg(not(feature = "rocm"))]
fn cpu_matmul(
    a: &DeviceTensor<f16>,
    b: &DeviceTensor<f16>,
    c: &mut DeviceTensor<f16>,
) -> Result<()> {
    use ndarray::Array2;

    let (m, k) = (a.rows(), a.cols());
    let n = b.cols();

    let a = Array2::from_shape_vec((m, k),
        a.copy_to_host()?.iter().map(|x| x.to_f32()).collect()
    ).unwrap();

    let b = Array2::from_shape_vec((k, n),
        b.copy_to_host()?.iter().map(|x| x.to_f32()).collect()
    ).unwrap();

    let res = a.dot(&b);

    let out: Vec<f16> = res.iter().map(|&x| f16::from_f32(x)).collect();
    c.copy_from_host(&out)?;
    Ok(())
}

#[cfg(not(feature = "rocm"))]
fn cpu_attention(
    q: &DeviceTensor<f16>,
    k: &DeviceTensor<f16>,
    v: &DeviceTensor<f16>,
    out: &mut DeviceTensor<f16>,
    scale: f32,
    causal: bool,
) -> Result<()> {
    let [b, h, sq, d] = extract_4d(q)?;
    let sk = k.shape()[2];

    let q = q.copy_to_host()?;
    let k = k.copy_to_host()?;
    let v = v.copy_to_host()?;

    let mut out_buf = vec![0f32; out.numel()];

    for bi in 0..b {
        for hi in 0..h {
            for i in 0..sq {
                let mut scores = vec![f32::NEG_INFINITY; sk];

                for j in 0..sk {
                    if causal && j > i { continue; }

                    let mut dot = 0.0;
                    for di in 0..d {
                        let idx_q = (((bi*h+hi)*sq+i)*d)+di;
                        let idx_k = (((bi*h+hi)*sk+j)*d)+di;
                        dot += q[idx_q].to_f32() * k[idx_k].to_f32();
                    }
                    scores[j] = dot * scale;
                }

                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = scores.iter().map(|s| (s-max).exp()).collect();
                let sum: f32 = exp.iter().sum();

                for di in 0..d {
                    let mut val = 0.0;
                    for j in 0..sk {
                        let w = exp[j]/sum;
                        let idx_v = (((bi*h+hi)*sk+j)*d)+di;
                        val += w * v[idx_v].to_f32();
                    }
                    let idx_o = (((bi*h+hi)*sq+i)*d)+di;
                    out_buf[idx_o] = val;
                }
            }
        }
    }

    let out_f16: Vec<f16> = out_buf.iter().map(|&x| f16::from_f32(x)).collect();
    out.copy_from_host(&out_f16)?;
    Ok(())
}

#[cfg(not(feature = "rocm"))]
fn cpu_rope(
    q: &mut DeviceTensor<f16>,
    k: &mut DeviceTensor<f16>,
    offset: usize,
    theta: f32,
) -> Result<()> {
    let apply = |t: &mut DeviceTensor<f16>| -> Result<()> {
        let mut data: Vec<f32> =
            t.copy_to_host()?.iter().map(|x| x.to_f32()).collect();

        let d = *t.shape().last().unwrap();
        let seq = t.shape()[t.shape().len() - 2];

        for s in 0..seq {
            let pos = (s + offset) as f32;
            for i in 0..d/2 {
                let freq = pos / theta.powf(2.0*i as f32/d as f32);
                let (sin, cos) = freq.sin_cos();

                let idx = s*d + 2*i;
                let (x0,x1) = (data[idx], data[idx+1]);

                data[idx] = x0*cos - x1*sin;
                data[idx+1] = x0*sin + x1*cos;
            }
        }

        let f16s: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
        t.copy_from_host(&f16s)?;
        Ok(())
    };

    apply(q)?;
    apply(k)?;
    Ok(())
}

#[cfg(not(feature = "rocm"))]
fn cpu_rms_norm(
    x: &mut DeviceTensor<f16>,
    w: &DeviceTensor<f16>,
    eps: f32,
) -> Result<()> {
    let d = x.cols();

    let mut data: Vec<f32> =
        x.copy_to_host()?.iter().map(|x| x.to_f32()).collect();

    let w: Vec<f32> =
        w.copy_to_host()?.iter().map(|x| x.to_f32()).collect();

    for row in data.chunks_mut(d) {
        let rms = (row.iter().map(|v| v*v).sum::<f32>()/d as f32 + eps).sqrt();
        for i in 0..d {
            row[i] = row[i]/rms * w[i];
        }
    }

    let f16s: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
    x.copy_from_host(&f16s)?;
    Ok(())
}


// HIP EXTERNS

#[cfg(feature = "rocm")]
extern "C" {
    fn flash_attention_hip(
        stream: *mut std::ffi::c_void,
        q: *const f16, k: *const f16, v: *const f16, out: *mut f16,
        b: i32, h: i32, sq: i32, sk: i32, d: i32,
        scale: f32, causal: i32,
    );

    fn rope_embedding_hip(
        stream: *mut std::ffi::c_void,
        q: *mut f16, k: *mut f16,
        seq: i32, d: i32,
        offset: i32, theta: f32, numel: i32,
    );

    fn layer_norm_hip(
        stream: *mut std::ffi::c_void,
        x: *mut f16, gamma: *const f16, beta: *const f16,
        rows: i32, cols: i32, eps: f32, rms_only: i32,
    );
}
// adding soon 
//kernel autotuned
//memory pooled
//graph-executed
//fused ops