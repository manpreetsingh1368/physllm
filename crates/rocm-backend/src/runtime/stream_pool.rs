//! stream_pool.rs — Pool of HIP streams for concurrent kernel execution.
//!
//! Streams allow independent kernels to run in parallel on the GPU.
//! The pool uses a lock-free SegQueue so stream acquisition doesn't serialise.

use crossbeam_queue::SegQueue;
use crate::{BackendError, Result, device::RawStream};
use tracing::debug;

pub struct StreamPool {
    /// Lock-free queue of available streams
    global: SegQueue<RawStream>,
    /// Number of streams created (for logging)
    n_streams: usize,
}

impl StreamPool {
    pub fn new(n: usize) -> Result<Self> {
        let global = SegQueue::new();
        for i in 0..n {
            let raw = create_stream()?;
            global.push(raw);
            debug!("StreamPool: created stream {i}");
        }
        Ok(Self { global, n_streams: n })
    }

    /// Get an available stream. Blocks if none available by returning the default
    /// (stream 0 aka nullptr, which is the default HIP stream).
    pub fn get(&self) -> RawStream {
        self.global.pop().unwrap_or(RawStream(std::ptr::null_mut()))
    }

    /// Return a stream to the pool.
    pub fn release(&self, s: RawStream) {
        self.global.push(s);
    }

    /// Wait for all queued work on every stream to finish.
    pub fn synchronize_all(&self) {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            // Drain, sync each, return
            let mut streams = Vec::new();
            while let Some(s) = self.global.pop() {
                hipStreamSynchronize(s.0 as hipStream_t);
                streams.push(s);
            }
            for s in streams {
                self.global.push(s);
            }
        }
    }

    pub fn len(&self) -> usize { self.n_streams }
}

impl Drop for StreamPool {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            while let Some(s) = self.global.pop() {
                hipStreamDestroy(s.0 as hipStream_t);
            }
        }
    }
}

fn create_stream() -> Result<RawStream> {
    #[cfg(feature = "rocm")]
    unsafe {
        use crate::hip_ffi::*;
        let mut s: hipStream_t = std::ptr::null_mut();
        let err = hipStreamCreate(&mut s);
        if err != 0 {
            return Err(BackendError::Hip { code: err, msg: "hipStreamCreate".into() });
        }
        return Ok(RawStream(s as *mut _));
    }
    #[cfg(not(feature = "rocm"))]
    Ok(RawStream(std::ptr::null_mut()))
}
