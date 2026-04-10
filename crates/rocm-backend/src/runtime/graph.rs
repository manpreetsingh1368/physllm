use crate::{BackendError, Result};
use super::RawStream;

pub struct GpuGraph {
    pub(crate) graph: *mut std::ffi::c_void,
    pub(crate) exec:  *mut std::ffi::c_void,
}

impl GpuGraph {
    pub fn launch(&self, stream: RawStream) -> Result<()> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;

            let err = hipGraphLaunch(self.exec as _, stream.0 as _);

            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipGraphLaunch failed".into(),
                });
            }
        }

        Ok(())
    }
}