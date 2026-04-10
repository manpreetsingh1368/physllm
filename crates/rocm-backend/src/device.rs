//! GpuDevice — AMD GPU device lifecycle and info 

use crate::{BackendError, Result};
use std::sync::Arc;
use tracing::{info, warn};

/// Properties of a detected AMD GPU.
#[derive(Debug, Clone)]
pub struct GpuProperties {
    pub device_id:        i32,
    pub name:             String,
    pub gfx_arch:         String,
    pub vram_total_mb:    usize,
    pub vram_free_mb:     usize,
    pub compute_units:    u32,
    pub wavefront_size:   u32,
    pub max_workgroup:    u32,
    pub rocm_version:     String,
}

/// Raw pointer wrapper (HIP stream)
#[derive(Debug)]
struct RawStream(*mut std::ffi::c_void);

unsafe impl Send for RawStream {}
unsafe impl Sync for RawStream {}

/// A handle to a single AMD GPU device.
#[derive(Debug)]
pub struct GpuDevice {
    pub props:  GpuProperties,
    stream:     Arc<RawStream>, // 🚀 no mutex
}

impl GpuDevice {
    /// Enumerate all available devices.
    pub fn enumerate() -> Result<Vec<GpuProperties>> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;

            let mut count = 0;
            let err = hipGetDeviceCount(&mut count);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipGetDeviceCount failed".into(),
                });
            }

            let mut devices = Vec::new();

            for idx in 0..count {
                let mut props: hipDeviceProp_tR0600 = std::mem::zeroed();

                let err = hipGetDeviceProperties_v2(&mut props, idx);
                if err != 0 {
                    warn!("Skipping device {idx}: hipGetDeviceProperties failed");
                    continue;
                }

                let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                    .to_string_lossy()
                    .to_string();

                let arch = std::ffi::CStr::from_ptr(props.gcnArchName.as_ptr())
                    .to_string_lossy()
                    .to_string();

                let mut free = 0usize;
                let mut total = 0usize;
                hipMemGetInfo(&mut free, &mut total);

                devices.push(GpuProperties {
                    device_id: idx,
                    name,
                    gfx_arch: arch,
                    vram_total_mb: props.totalGlobalMem / (1024 * 1024),
                    vram_free_mb: free / (1024 * 1024),
                    compute_units: props.multiProcessorCount as u32,
                    wavefront_size: props.warpSize as u32,
                    max_workgroup: props.maxThreadsPerBlock as u32,
                    rocm_version: detect_rocm_version(),
                });
            }

            Ok(devices)
        }

        #[cfg(not(feature = "rocm"))]
        {
            Ok(vec![GpuProperties {
                device_id: -1,
                name: "CPU Fallback".into(),
                gfx_arch: "none".into(),
                vram_total_mb: 0,
                vram_free_mb: 0,
                compute_units: rayon::current_num_threads() as u32,
                wavefront_size: 0,
                max_workgroup: 0,
                rocm_version: "N/A".into(),
            }])
        }
    }

    /// Open the best available GPU (by VRAM, then compute units).
    pub fn open_best() -> Result<Self> {
        let devices = Self::enumerate()?;

        let best = devices
            .into_iter()
            .max_by_key(|d| (d.vram_total_mb, d.compute_units))
            .ok_or_else(|| BackendError::Other("No GPU devices found".into()))?;

        Self::open_device(best.device_id)
    }

    /// Open a specific device by index.
    pub fn open_device(idx: i32) -> Result<Self> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;

            let err = hipSetDevice(idx);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: format!("hipSetDevice({idx}) failed"),
                });
            }

            let mut props: hipDeviceProp_tR0600 = std::mem::zeroed();

            let err = hipGetDeviceProperties_v2(&mut props, idx);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipGetDeviceProperties_v2 failed".into(),
                });
            }

            let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .to_string();

            let arch = std::ffi::CStr::from_ptr(props.gcnArchName.as_ptr())
                .to_string_lossy()
                .to_string();

            let mut free = 0usize;
            let mut total = 0usize;
            hipMemGetInfo(&mut free, &mut total);

            info!("Opened AMD GPU [{idx}]: {name} ({arch})");
            info!(
                "  VRAM: {:.1} GB (free {:.1} GB)  CU: {}",
                total as f64 / 1e9,
                free as f64 / 1e9,
                props.multiProcessorCount,
            );

            let mut stream: hipStream_t = std::ptr::null_mut();

            let err = hipStreamCreate(&mut stream);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipStreamCreate failed".into(),
                });
            }

            Ok(GpuDevice {
                props: GpuProperties {
                    device_id: idx,
                    name,
                    gfx_arch: arch,
                    vram_total_mb: total / (1024 * 1024),
                    vram_free_mb: free / (1024 * 1024),
                    compute_units: props.multiProcessorCount as u32,
                    wavefront_size: props.warpSize as u32,
                    max_workgroup: props.maxThreadsPerBlock as u32,
                    rocm_version: detect_rocm_version(),
                },
                stream: Arc::new(RawStream(stream as *mut _)),
            })
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
                stream: Arc::new(RawStream(std::ptr::null_mut())),
            })
        }
    }

    /// Refresh VRAM stats.
    pub fn refresh_memory(&mut self) -> Result<()> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;

            let mut free = 0usize;
            let mut total = 0usize;

            let err = hipMemGetInfo(&mut free, &mut total);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipMemGetInfo failed".into(),
                });
            }

            self.props.vram_free_mb = free / (1024 * 1024);
            self.props.vram_total_mb = total / (1024 * 1024);
        }

        Ok(())
    }

    /// Synchronise the stream.
    pub fn synchronise(&self) -> Result<()> {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;

            let err = hipStreamSynchronize(self.stream.0 as _);
            if err != 0 {
                return Err(BackendError::Hip {
                    code: err,
                    msg: "hipStreamSynchronize failed".into(),
                });
            }
        }

        Ok(())
    }

    /// Raw HIP stream pointer.
    #[inline(always)]
    pub(crate) fn raw_stream(&self) -> *mut std::ffi::c_void {
        self.stream.0
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        unsafe {
            use crate::hip_ffi::*;

            if !self.stream.0.is_null() {
                let err = hipStreamDestroy(self.stream.0 as _);
                if err != 0 {
                    warn!("hipStreamDestroy failed during drop");
                }
            }
        }
    }
}

/// Detect ROCm version more robustly.
fn detect_rocm_version() -> String {
    std::env::var("ROCM_VERSION")
        .ok()
        .or_else(|| std::fs::read_to_string("/opt/rocm/.info/version").ok())
        .or_else(|| std::fs::read_to_string("/opt/rocm/version.txt").ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}