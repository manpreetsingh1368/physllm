//! device.rs — GPU device management.

use crate::{BackendError, Result, runtime::stream_pool::StreamPool};
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct GpuProperties {
    pub device_id:      i32,
    pub name:           String,
    pub gfx_arch:       String,
    pub vram_total_mb:  usize,
    pub vram_free_mb:   usize,
    pub compute_units:  u32,
    pub wavefront_size: u32,
    pub clock_mhz:      u32,
    pub memory_bw_gbps: f32,
}

/// Opaque wrapper around a HIP stream pointer, making it Send+Sync.
#[derive(Copy, Clone, Debug)]
pub struct RawStream(pub *mut std::ffi::c_void);
unsafe impl Send for RawStream {}
unsafe impl Sync for RawStream {}

pub struct GpuDevice {
    pub props:        GpuProperties,
    pub stream_pool:  Arc<StreamPool>,
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuDevice({})", self.props.name)
    }
}

impl GpuDevice {
    /// Open the first available GPU (device 0).
    pub fn open_best() -> Result<Self> {
        #[cfg(feature = "rocm")]
        {
            let mut count = 0i32;
            unsafe {
                use crate::hip_ffi::*;
                let err = hipGetDeviceCount(&mut count);
                if err != 0 {
                    return Err(BackendError::DeviceNotFound(
                        format!("hipGetDeviceCount returned {err}")
                    ));
                }
            }
            if count <= 0 {
                return Err(BackendError::DeviceNotFound("No GPU devices found".into()));
            }
            Self::open(0)
        }
        #[cfg(not(feature = "rocm"))]
        {
            Self::open(-1)
        }
    }

    /// Open a specific device by ID.
    pub fn open(idx: i32) -> Result<Self> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;

            let err = hipSetDevice(idx);
            if err != 0 {
                return Err(BackendError::Hip { code: err, msg: format!("hipSetDevice({idx})") });
            }

            let mut props: hipDeviceProp_tR0600 = std::mem::zeroed();
            let err = hipGetDeviceProperties_v2(&mut props, idx);
            if err != 0 {
                return Err(BackendError::Hip { code: err, msg: "hipGetDeviceProperties_v2".into() });
            }

            let name = std::ffi::CStr::from_ptr(props.name.as_ptr()).to_string_lossy().to_string();
            let arch = std::ffi::CStr::from_ptr(props.gcnArchName.as_ptr()).to_string_lossy().to_string();

            let mut free: usize = 0;
            let mut total: usize = 0;
            hipMemGetInfo(&mut free, &mut total);

            info!("GPU[{idx}]: {} ({}) {}MB total, {}MB free",
                  name, arch, total / 1_048_576, free / 1_048_576);

            let stream_pool = Arc::new(StreamPool::new(4)?);

            Ok(GpuDevice {
                props: GpuProperties {
                    device_id:      idx,
                    name,
                    gfx_arch:       arch,
                    vram_total_mb:  total / 1_048_576,
                    vram_free_mb:   free / 1_048_576,
                    compute_units:  props.multiProcessorCount as u32,
                    wavefront_size: props.warpSize as u32,
                    clock_mhz:      (props.clockRate / 1000) as u32,
                    memory_bw_gbps: (props.memoryClockRate as f32 / 1e6)
                                    * (props.memoryBusWidth as f32 / 8.0) * 2.0,
                },
                stream_pool,
            })
        }

        #[cfg(not(feature = "rocm"))]
        {
            warn!("cpu-only mode — GPU operations are emulated on CPU");
            let stream_pool = Arc::new(StreamPool::new(1)?);
            Ok(GpuDevice {
                props: GpuProperties {
                    device_id:      idx,
                    name:           "CPU (cpu-only mode)".into(),
                    gfx_arch:       "none".into(),
                    vram_total_mb:  0,
                    vram_free_mb:   0,
                    compute_units:  rayon::current_num_threads() as u32,
                    wavefront_size: 0,
                    clock_mhz:      0,
                    memory_bw_gbps: 0.0,
                },
                stream_pool,
            })
        }
    }

    /// Get a stream from the pool for GPU work submission.
    pub fn raw_stream(&self) -> *mut std::ffi::c_void {
        self.stream_pool.get().0
    }

    /// Wait for all queued work to finish across all streams.
    pub fn synchronise(&self) -> Result<()> {
        self.stream_pool.synchronize_all();
        Ok(())
    }

    /// Query current free VRAM in MB.
    pub fn free_vram_mb(&self) -> usize {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;
            let mut free = 0usize;
            let mut total = 0usize;
            let err = hipMemGetInfo(&mut free, &mut total);
            if err == 0 { return free / 1_048_576; }
        }
        0
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // Ensure all pending GPU work finishes before tearing down the device.
        let _ = self.synchronise();
    }
}
