use std::{cell::RefCell, sync::Arc};
use crossbeam_queue::SegQueue;

#[derive(Clone, Copy, Debug)]
pub struct RawStream(pub *mut std::ffi::c_void);

unsafe impl Send for RawStream {}
unsafe impl Sync for RawStream {}

pub struct StreamPool {
    global: Arc<SegQueue<RawStream>>,
}

thread_local! {
    static LOCAL_STREAM: RefCell<Option<RawStream>> = RefCell::new(None);
}

impl StreamPool {
    pub fn new() -> Self {
        Self {
            global: Arc::new(SegQueue::new()),
        }
    }

    pub fn acquire(&self) -> RawStream {
        if let Some(s) = LOCAL_STREAM.with(|tls| tls.borrow_mut().take()) {
            return s;
        }

        if let Some(s) = self.global.pop() {
            return s;
        }

        Self::create_stream()
    }

    pub fn release(&self, stream: RawStream) {
        LOCAL_STREAM.with(|tls| {
            *tls.borrow_mut() = Some(stream);
        });
    }

    fn create_stream() -> RawStream {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let mut s = std::ptr::null_mut();
            hipStreamCreate(&mut s);
            RawStream(s as _)
        }

        #[cfg(not(feature = "rocm"))]
        RawStream(std::ptr::null_mut())
    }
}