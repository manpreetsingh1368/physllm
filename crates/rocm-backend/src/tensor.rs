//! DeviceTensor — typed GPU buffer with shape tracking.

use crate::{BackendError, Result};
use half::f16;
use std::marker::PhantomData;

/// Supported element types for GPU tensors.
pub trait Element: Copy + bytemuck::Pod + Send + Sync + 'static {
    fn dtype_name() -> &'static str;
    fn bytes() -> usize { std::mem::size_of::<Self>() }
}

impl Element for f16  { fn dtype_name() -> &'static str { "f16" } }
impl Element for f32  { fn dtype_name() -> &'static str { "f32" } }
impl Element for f64  { fn dtype_name() -> &'static str { "f64" } }
impl Element for u8   { fn dtype_name() -> &'static str { "u8"  } }
impl Element for i32  { fn dtype_name() -> &'static str { "i32" } }
impl Element for u32  { fn dtype_name() -> &'static str { "u32" } }

/// A tensor living on the GPU (HIP device memory).
pub struct DeviceTensor<T: Element> {
    ptr:     *mut T,
    shape:   Vec<usize>,
    strides: Vec<usize>,
    numel:   usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Element> Send for DeviceTensor<T> {}
unsafe impl<T: Element> Sync for DeviceTensor<T> {}

impl<T: Element> DeviceTensor<T> {
    /// Allocate uninitialised GPU memory with the given shape.
    pub fn alloc(shape: &[usize]) -> Result<Self> {
        let numel      = shape.iter().product::<usize>();
        let byte_count = numel * T::bytes();

        if numel == 0 {
            return Ok(Self { ptr: std::ptr::null_mut(), shape: shape.to_vec(), strides: vec![], numel: 0, _marker: PhantomData });
        }

        let ptr = Self::hip_malloc(byte_count)?;
        let strides = Self::compute_strides(shape);
        Ok(Self { ptr, shape: shape.to_vec(), strides, numel, _marker: PhantomData })
    }

    /// Allocate and copy data from a host slice.
    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Self> {
        let mut t = Self::alloc(shape)?;
        t.copy_from_host(data)?;
        Ok(t)
    }

    /// Copy from host (CPU) memory to device.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        assert_eq!(data.len(), self.numel, "host slice length != tensor numel");
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let byte_count = self.numel * T::bytes();
            let err = hipMemcpy(
                self.ptr as *mut _,
                data.as_ptr() as *const _,
                byte_count,
                hipMemcpyKind_hipMemcpyHostToDevice,
            );
            if err != 0 {
                return Err(BackendError::Hip { code: err as i32, msg: "hipMemcpy H2D".into() });
            }
        }
        Ok(())
    }

    /// Copy from device to host (CPU) slice.
    pub fn copy_to_host(&self) -> Result<Vec<T>> {
        let mut out = vec![T::zeroed(); self.numel];
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let byte_count = self.numel * T::bytes();
            let err = hipMemcpy(
                out.as_mut_ptr() as *mut _,
                self.ptr as *const _,
                byte_count,
                hipMemcpyKind_hipMemcpyDeviceToHost,
            );
            if err != 0 {
                return Err(BackendError::Hip { code: err as i32, msg: "hipMemcpy D2H".into() });
            }
        }
        Ok(out)
    }

    pub fn shape(&self)   -> &[usize] { &self.shape   }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub fn numel(&self)   -> usize    { self.numel    }
    pub fn raw_ptr(&self) -> *mut T   { self.ptr      }

    pub fn rows(&self)    -> usize    { if self.shape.len() >= 2 { self.shape[self.shape.len()-2] } else { 1 } }
    pub fn cols(&self)    -> usize    { *self.shape.last().unwrap_or(&0) }

    fn hip_malloc(bytes: usize) -> Result<*mut T> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = hipMalloc(&mut ptr, bytes);
            if err != 0 {
                return Err(BackendError::Hip { code: err as i32, msg: format!("hipMalloc({bytes} bytes)") });
            }
            return Ok(ptr as *mut T);
        }
        #[cfg(not(feature = "rocm"))]
        {
            // CPU fallback — use aligned vec
            let mut v: Vec<T> = vec![T::zeroed(); bytes / T::bytes()];
            let ptr = v.as_mut_ptr();
            std::mem::forget(v);
            Ok(ptr)
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
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
            // reconstruct Vec and drop it
            let _ = Vec::from_raw_parts(self.ptr, self.numel, self.numel);
        }
    }
}

// Helper: zeroed value for any Pod type
trait Zeroed: Sized { fn zeroed() -> Self; }
impl<T: bytemuck::Zeroable> Zeroed for T { fn zeroed() -> Self { unsafe { std::mem::zeroed() } } }
