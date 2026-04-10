use crate::{BackendError, Result};
use super::{RawStream, StreamPool};
use std::sync::Arc;

pub struct GpuFuture {
    pub(crate) stream: RawStream,
    pub(crate) pool: Arc<StreamPool>,
}

impl GpuFuture {
    pub fn wait(self) -> Result<()> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let err = hipStreamSynchronize(self.stream.0 as _);

            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "Stream sync failed".into(),
                });
            }
        }

        self.pool.release(self.stream);
        Ok(())
    }
}