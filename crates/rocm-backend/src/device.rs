//! GpuDevice — AMD GPU device lifecycle and info.

use crate::{BackendError, Result};
use std::sync::Arc;
use parking_lot::Mutex;
use tracing::{info, warn};

/// Properties of a detected AMD GPU.
#[derive(Debug, Clone)]
pub struct GpuProperties {
    pub device_id:        i32,
    pub name:             String,
    pub gfx_arch:         String,   // e.g. "gfx1100"
    pub vram_total_mb:    usize,
    pub vram_free_mb:     usize,
    pub compute_units:    u32,
    pub wavefront_size:   u32,
    pub max_workgroup:    u32,
    pub rocm_version:     String,
}

/// A handle to a single AMD GPU device.
#[derive(Debug)]
pub struct GpuDevice {
    pub props:   GpuProperties,
    stream:      Arc<Mutex<RawStream>>,
}

// Raw pointer wrapper (HIP stream); Send/Sync are safe because we serialise via Mutex.
#[derive(Debug)]
struct RawStream(*mut std::ffi::c_void);
unsafe impl Send for RawStream {}
unsafe impl Sync for RawStream {}

impl GpuDevice {
    /// Enumerate and open the best available AMD GPU.
    pub fn open_best() -> Result<Self> {
        Self::open_device(0)
    }

    /// Open a specific device by index.
    pub fn open_device(idx: i32) -> Result<Self> {
        #[cfg(feature = "rocm")]
        {
            use crate::hip_ffi::*;
            unsafe {
                // hipSetDevice
                let err = hipSetDevice(idx);
                if err != 0 {
                    return Err(BackendError::Hip {
                        code: err as i32,
                        msg: format!("hipSetDevice({idx}) failed"),
                    });
                }

                let mut props: hipDeviceProp_tR0600 = std::mem::zeroed();
                hipGetDevicePropertiesR0600(&mut props, idx);

                let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                    .to_string_lossy()
                    .to_string();
                let arch = std::ffi::CStr::from_ptr(props.gcnArchName.as_ptr())
                    .to_string_lossy()
                    .to_string();

                info!("Opened AMD GPU [{idx}]: {name} ({arch})");
                info!(
                    "  VRAM: {:.1} GB  CU: {}  Wavefront: {}",
                    props.totalGlobalMem as f64 / 1e9,
                    props.multiProcessorCount,
                    props.warpSize,
                );

                // Create a HIP stream
                let mut stream: hipStream_t = std::ptr::null_mut();
                hipStreamCreate(&mut stream);

                Ok(GpuDevice {
                    props: GpuProperties {
                        device_id:     idx,
                        name,
                        gfx_arch:      arch,
                        vram_total_mb: props.totalGlobalMem / (1024 * 1024),
                        vram_free_mb:  0, // updated via memory::query_free()
                        compute_units: props.multiProcessorCount as u32,
                        wavefront_size: props.warpSize as u32,
                        max_workgroup: props.maxThreadsPerBlock as u32,
                        rocm_version: detect_rocm_version(),
                    },
                    stream: Arc::new(Mutex::new(RawStream(stream as *mut _))),
                })
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            warn!("ROCm not enabled; using CPU fallback device.");
            Ok(GpuDevice {
                props: GpuProperties {
                    device_id: -1,
                    name: "CPU Fallback".into(),
                    gfx_arch: "none".into(),
                    vram_total_mb: 0,
                    vram_free_mb: 0,
                    compute_units: rayon::current_num_threads() as u32,
                    wavefront_size: 0,
                    max_workgroup: 0,
                    rocm_version: "N/A".into(),
                },
                stream: Arc::new(Mutex::new(RawStream(std::ptr::null_mut()))),
            })
        }
    }

    /// Synchronise the HIP stream (wait for all GPU ops to complete).
    pub fn synchronise(&self) -> Result<()> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let stream = self.stream.lock();
            let err = hipStreamSynchronize(stream.0 as hipStream_t);
            if err != 0 {
                return Err(BackendError::Hip { code: err as i32, msg: "hipStreamSynchronize".into() });
            }
        }
        Ok(())
    }

    /// Raw HIP stream pointer (for kernel launch helpers).
    pub(crate) fn raw_stream(&self) -> *mut std::ffi::c_void {
        self.stream.lock().0
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let s = self.stream.lock();
            if !s.0.is_null() {
                hipStreamDestroy(s.0 as _);
            }
        }
    }
}

fn detect_rocm_version() -> String {
    std::fs::read_to_string("/opt/rocm/.info/version")
        .or_else(|_| std::fs::read_to_string("/opt/rocm/version.txt"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}
