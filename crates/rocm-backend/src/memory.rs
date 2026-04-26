//! memory.rs — GPU memory pool to reduce hipMalloc overhead.

use crate::{BackendError, Result};

/// A simple slab-based GPU memory pool.
/// Allocates a large block upfront and suballocates from it.
pub struct MemoryPool {
    total_bytes:  usize,
    used_bytes:   usize,
    base_ptr:     *mut u8,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    /// Allocate `size_mb` megabytes of GPU memory.
    pub fn new(size_mb: usize) -> Result<Self> {
        let total = size_mb * 1024 * 1024;
        let base_ptr = Self::hip_malloc(total)?;
        Ok(Self { total_bytes: total, used_bytes: 0, base_ptr })
    }

    /// Suballocate a block (bump allocator — no free/reuse).
    pub fn alloc(&mut self, bytes: usize) -> Result<*mut u8> {
        let aligned = (bytes + 255) & !255;  // 256-byte alignment
        if self.used_bytes + aligned > self.total_bytes {
            return Err(BackendError::OutOfMemory {
                requested_mb: aligned / (1024 * 1024),
                available_mb: (self.total_bytes - self.used_bytes) / (1024 * 1024),
            });
        }
        let ptr = unsafe { self.base_ptr.add(self.used_bytes) };
        self.used_bytes += aligned;
        Ok(ptr)
    }

    /// Reset the pool (free all suballocations — does NOT call hipFree).
    pub fn reset(&mut self) {
        self.used_bytes = 0;
    }

    pub fn used_mb(&self)  -> usize { self.used_bytes  / (1024 * 1024) }
    pub fn total_mb(&self) -> usize { self.total_bytes / (1024 * 1024) }
    pub fn free_mb(&self)  -> usize { self.total_mb() - self.used_mb() }

    fn hip_malloc(bytes: usize) -> Result<*mut u8> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = hipMalloc(&mut ptr, bytes);
            if err != 0 {
                return Err(BackendError::Hip { code: err, msg: format!("Pool hipMalloc({} MB)", bytes/1024/1024) });
            }
            return Ok(ptr as *mut u8);
        }
        #[cfg(not(feature = "rocm"))]
        {
            let layout = std::alloc::Layout::from_size_align(bytes, 256).unwrap();
            Ok(unsafe { std::alloc::alloc(layout) })
        }
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        if self.base_ptr.is_null() { return; }
        #[cfg(feature = "rocm")]
        unsafe { crate::hip_ffi::hipFree(self.base_ptr as *mut _); }
        #[cfg(not(feature = "rocm"))]
        unsafe {
            let layout = std::alloc::Layout::from_size_align(self.total_bytes, 256).unwrap();
            std::alloc::dealloc(self.base_ptr, layout);
        }
    }
}

/// Query free GPU VRAM.
#[cfg(feature = "rocm")]
pub fn query_free_vram() -> Result<usize> {
    unsafe {
        use crate::hip_ffi::*;
        let mut free: usize = 0;
        let mut total: usize = 0;
        let err = hipMemGetInfo(&mut free, &mut total);
        if err != 0 {
            return Err(BackendError::Hip { code: err, msg: "hipMemGetInfo".into() });
        }
        Ok(free)
    }
}

#[cfg(not(feature = "rocm"))]
pub fn query_free_vram() -> Result<usize> { Ok(usize::MAX) }
