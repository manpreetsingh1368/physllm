use std::collections::BTreeMap;
use crate::{BackendError, Result};

pub struct GpuAllocator {
    free: BTreeMap<usize, Vec<*mut std::ffi::c_void>>,
}

impl GpuAllocator {
    pub fn new() -> Self {
        Self {
            free: BTreeMap::new(),
        }
    }

    pub fn alloc(&mut self, size: usize) -> Result<*mut std::ffi::c_void> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;

            if let Some((&_size, list)) = self.free.range_mut(size..).next() {
                if let Some(ptr) = list.pop() {
                    return Ok(ptr);
                }
            }

            let mut ptr = std::ptr::null_mut();
            let err = hipMalloc(&mut ptr, size);

            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipMalloc failed".into(),
                });
            }

            Ok(ptr)
        }

        #[cfg(not(feature = "rocm"))]
        Ok(std::ptr::null_mut())
    }

    pub fn free(&mut self, ptr: *mut std::ffi::c_void, size: usize) {
        self.free.entry(size).or_default().push(ptr);
    }
}