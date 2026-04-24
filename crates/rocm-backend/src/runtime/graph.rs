// graph.rs — HIP/CUDA graph capture.


use crate::{BackendError, Result, device::RawStream};

pub struct GraphCapture {
    stream: RawStream,
}

pub struct CapturedGraph {
    pub exec: *mut std::ffi::c_void,
}

unsafe impl Send for CapturedGraph {}
unsafe impl Sync for CapturedGraph {}

impl GraphCapture {
    /// Start capturing kernels launched on the given stream.
    pub fn begin(stream: RawStream) -> Result<Self> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let err = hipStreamBeginCapture(stream.0 as hipStream_t, 0);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipStreamBeginCapture".into(),
                });
            }
        }
        Ok(Self { stream })
    }

    /// Finish capture and produce an executable graph.
    pub fn end(self) -> Result<CapturedGraph> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let mut graph: hipGraph_t = std::ptr::null_mut();
            let err = hipStreamEndCapture(self.stream.0 as hipStream_t, &mut graph);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipStreamEndCapture".into(),
                });
            }
            let mut exec: hipGraphExec_t = std::ptr::null_mut();
            let err = hipGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            hipGraphDestroy(graph);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipGraphInstantiate".into(),
                });
            }
            return Ok(CapturedGraph { exec: exec as *mut _ });
        }
        #[cfg(not(feature = "rocm"))]
        Ok(CapturedGraph { exec: std::ptr::null_mut() })
    }
}

impl CapturedGraph {
    /// Replay the captured sequence of kernels.
    pub fn launch(&self, stream: RawStream) -> Result<()> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let err = hipGraphLaunch(self.exec as _, stream.0 as hipStream_t);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipGraphLaunch".into(),
                });
            }
        }
        Ok(())
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            if !self.exec.is_null() {
                hipGraphExecDestroy(self.exec as _);
            }
        }
    }
}
