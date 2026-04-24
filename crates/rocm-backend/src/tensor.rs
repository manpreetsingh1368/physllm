//! tensor.rs — GPU tensor with host-to-device transfer.
//!
//! In cpu-only mode the "device" memory is just host memory, so the tensor
//! behaves identically but without any GPU copies.

use crate::{BackendError, Result};
use half::f16;
use std::marker::PhantomData;

pub trait Element: Copy + bytemuck::Pod + Send + Sync + 'static {
    fn zero() -> Self;
}
impl Element for f16 { fn zero() -> Self { f16::from_f32(0.0) } }
impl Element for f32 { fn zero() -> Self { 0.0 } }
impl Element for f64 { fn zero() -> Self { 0.0 } }
impl Element for u8  { fn zero() -> Self { 0 } }
impl Element for i32 { fn zero() -> Self { 0 } }
impl Element for u32 { fn zero() -> Self { 0 } }
impl Element for i64 { fn zero() -> Self { 0 } }

pub struct DeviceTensor<T: Element> {
    ptr:     *mut T,
    shape:   Vec<usize>,
    numel:   usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Element> Send for DeviceTensor<T> {}
unsafe impl<T: Element> Sync for DeviceTensor<T> {}

impl<T: Element> DeviceTensor<T> {
    pub fn alloc(shape: &[usize]) -> Result<Self> {
        let numel = shape.iter().product::<usize>();
        let ptr = if numel == 0 {
            std::ptr::null_mut()
        } else {
            #[cfg(feature = "rocm")]
            unsafe {
                use crate::hip_ffi::*;
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                let size = numel * std::mem::size_of::<T>();
                let err = hipMalloc(&mut p, size);
                if err != 0 {
                    return Err(BackendError::OutOfMemory {
                        requested_mb: size / 1_048_576,
                        available_mb: 0,
                    });
                }
                p as *mut T
            }
            #[cfg(not(feature = "rocm"))]
            {
                let mut v: Vec<T> = Vec::with_capacity(numel);
                // Zero-initialise via bytemuck
                for _ in 0..numel { v.push(T::zero()); }
                let p = v.as_mut_ptr();
                std::mem::forget(v);
                p
            }
        };
        Ok(Self { ptr, shape: shape.to_vec(), numel, _marker: PhantomData })
    }

    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Self> {
        assert_eq!(data.len(), shape.iter().product::<usize>(),
                   "from_slice: data/shape size mismatch");
        let mut t = Self::alloc(shape)?;
        t.copy_from_host(data)?;
        Ok(t)
    }

    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        assert_eq!(data.len(), self.numel);
        if self.ptr.is_null() { return Ok(()); }
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let err = hipMemcpy(
                self.ptr as *mut _,
                data.as_ptr() as *const _,
                self.numel * std::mem::size_of::<T>(),
                hipMemcpyKind::hipMemcpyHostToDevice,
            );
            if err != 0 {
                return Err(BackendError::Hip { code: err, msg: "hipMemcpy H2D".into() });
            }
        }
        #[cfg(not(feature = "rocm"))]
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
        Ok(())
    }

    pub fn copy_to_host(&self) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(self.numel);
        for _ in 0..self.numel { out.push(T::zero()); }
        if self.ptr.is_null() { return Ok(out); }
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let err = hipMemcpy(
                out.as_mut_ptr() as *mut _,
                self.ptr as *const _,
                self.numel * std::mem::size_of::<T>(),
                hipMemcpyKind::hipMemcpyDeviceToHost,
            );
            if err != 0 {
                return Err(BackendError::Hip { code: err, msg: "hipMemcpy D2H".into() });
            }
        }
        #[cfg(not(feature = "rocm"))]
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, out.as_mut_ptr(), self.numel);
        }
        Ok(out)
    }

    pub fn shape(&self)   -> &[usize] { &self.shape }
    pub fn numel(&self)   -> usize    { self.numel }
    pub fn len(&self)     -> usize    { self.numel }
    pub fn is_empty(&self) -> bool    { self.numel == 0 }
    pub fn raw_ptr(&self) -> *mut T   { self.ptr }
    pub fn rows(&self)    -> usize {
        if self.shape.len() >= 2 { self.shape[self.shape.len()-2] } else { 1 }
    }
    pub fn cols(&self)    -> usize { *self.shape.last().unwrap_or(&0) }

    /// Number of batches (shape[0] if 3+ dims, else 1).
    pub fn batch(&self) -> usize {
        if self.shape.len() >= 3 { self.shape[0] } else { 1 }
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel * std::mem::size_of::<T>()
    }
}

impl<T: Element> Drop for DeviceTensor<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() { return; }
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            hipFree(self.ptr as *mut _);
        }
        #[cfg(not(feature = "rocm"))]
        unsafe {
            // Reconstruct the Vec so Rust's allocator frees it properly
            let _ = Vec::from_raw_parts(self.ptr, self.numel, self.numel);
        }
    }
}
