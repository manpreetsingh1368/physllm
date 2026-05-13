// src/comm/mod.rs
mod ffi;

use std::sync::Arc;
use parking_lot::Mutex;
use tracing::{debug, info, warn};

use crate::{config::TpConfig, error::{TpError, TpResult}};

// TpHandle 

/// One GPU's communicator + hipBLAS handle.
/// Wraps a raw C TpHandle pointer. All methods call into gpu_shim.hip.
pub struct TpHandle {
    ptr: *mut ffi::RawHandle,
    pub rank: usize,
    pub world_size: usize,
    pub device_id: i32,
}

// SAFETY: TpHandle is only accessed through Arc<Mutex<TpHandle>>.
unsafe impl Send for TpHandle {}
unsafe impl Sync for TpHandle {}

impl TpHandle {
    fn from_raw(ptr: *mut ffi::RawHandle) -> Self {
        // SAFETY: ptr is valid immediately after tp_init.
        let rank        = unsafe { ffi::tp_rank(ptr) }       as usize;
        let world_size  = unsafe { ffi::tp_world_size(ptr) } as usize;
        let device_id   = unsafe { ffi::tp_device_id(ptr) };
        Self { ptr, rank, world_size, device_id }
    }

    //  Collectives 

    /// In-place AllReduce (sum) fp16. Blocks until complete.
    /// `buf` — raw device pointer, `count` — number of fp16 elements.
    pub unsafe fn allreduce_fp16(&self, buf: *mut u8, count: usize) -> TpResult<()> {
        debug!(rank = self.rank, count, "AllReduce fp16");
        let rc = unsafe { ffi::tp_allreduce_fp16(self.ptr, buf as *mut _, count) };
        if rc != 0 { return Err(TpError::Collective { rank: self.rank, msg: format!("allreduce_fp16 rc={rc}") }); }
        Ok(())
    }

    /// In-place AllReduce (sum) fp32.
    pub unsafe fn allreduce_fp32(&self, buf: *mut u8, count: usize) -> TpResult<()> {
        let rc = unsafe { ffi::tp_allreduce_fp32(self.ptr, buf as *mut _, count) };
        if rc != 0 { return Err(TpError::Collective { rank: self.rank, msg: format!("allreduce_fp32 rc={rc}") }); }
        Ok(())
    }

    /// Non-blocking AllReduce fp16. Call `sync()` before reading result.
    /// Use this to overlap the reduce with the next layer's QKV projection.
    pub unsafe fn allreduce_fp16_async(&self, buf: *mut u8, count: usize) -> TpResult<()> {
        let rc = unsafe { ffi::tp_allreduce_fp16_async(self.ptr, buf as *mut _, count) };
        if rc != 0 { return Err(TpError::Collective { rank: self.rank, msg: format!("allreduce_async rc={rc}") }); }
        Ok(())
    }

    /// AllGather fp16. `recv_buf` must hold world_size × send_count fp16 elements.
    pub unsafe fn allgather_fp16(
        &self, send: *mut u8, recv: *mut u8, send_count: usize,
    ) -> TpResult<()> {
        let rc = unsafe { ffi::tp_allgather_fp16(self.ptr, send as *mut _, recv as *mut _, send_count) };
        if rc != 0 { return Err(TpError::Collective { rank: self.rank, msg: format!("allgather rc={rc}") }); }
        Ok(())
    }

    /// Broadcast fp16 from rank 0.
    pub unsafe fn broadcast_fp16(&self, buf: *mut u8, count: usize) -> TpResult<()> {
        let rc = unsafe { ffi::tp_broadcast_fp16(self.ptr, buf as *mut _, count) };
        if rc != 0 { return Err(TpError::Collective { rank: self.rank, msg: format!("broadcast rc={rc}") }); }
        Ok(())
    }

    /// Synchronise this rank's GPU stream (waits for all async ops).
    pub fn sync(&self) -> TpResult<()> {
        let rc = unsafe { ffi::tp_sync(self.ptr) };
        if rc != 0 { return Err(TpError::Collective { rank: self.rank, msg: "sync failed".into() }); }
        Ok(())
    }

    //  GEMMs 

    /// C = alpha * A @ B + beta * C   (all fp16 device ptrs, row-major)
    /// A: [m, k]  B: [k, n]  C: [m, n]
    pub unsafe fn gemm_fp16(
        &self,
        m: i32, n: i32, k: i32,
        a: *const u8, b: *const u8, c: *mut u8,
        alpha: f32, beta: f32,
    ) -> TpResult<()> {
        let rc = unsafe {
            ffi::tp_gemm_fp16(self.ptr, m, n, k, a as *const _, b as *const _, c as *mut _, alpha, beta)
        };
        if rc != 0 { return Err(TpError::Gemm { rank: self.rank, code: rc }); }
        Ok(())
    }

    /// C = alpha * A @ B.T + beta * C
    /// A: [m, k]  B stored as [n, k]  C: [m, n]
    pub unsafe fn gemm_fp16_bt(
        &self,
        m: i32, n: i32, k: i32,
        a: *const u8, b: *const u8, c: *mut u8,
        alpha: f32, beta: f32,
    ) -> TpResult<()> {
        let rc = unsafe {
            ffi::tp_gemm_fp16_bt(self.ptr, m, n, k, a as *const _, b as *const _, c as *mut _, alpha, beta)
        };
        if rc != 0 { return Err(TpError::Gemm { rank: self.rank, code: rc }); }
        Ok(())
    }
}

impl Drop for TpHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let rc = unsafe { ffi::tp_destroy(self.ptr) };
            if rc != 0 {
                warn!(rank = self.rank, "tp_destroy returned {rc}");
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

//  TpHandleGroup 

/// Owns all GPU handles for a tensor-parallel job.
pub struct TpHandleGroup {
    handles: Vec<Arc<Mutex<TpHandle>>>,
    pub config: TpConfig,
}

impl TpHandleGroup {
    /// Initialise. Calls `tp_init` which internally calls `ncclCommInitAll`.
    /// All `config.devices` must be visible in this process.
    pub fn init(config: TpConfig) -> TpResult<Self> {
        info!(
            world_size = config.world_size,
            strategy = ?config.strategy,
            devices = ?config.devices,
            "TpHandleGroup: initialising"
        );

        let mut devs = config.devices.clone();
        let raw = unsafe { ffi::tp_init(config.world_size as i32, devs.as_mut_ptr()) };
        if raw.is_null() {
            return Err(TpError::NotInitialised);
        }

        let handles = (0..config.world_size)
            .map(|i| {
                let ptr = unsafe { *raw.add(i) };
                Arc::new(Mutex::new(TpHandle::from_raw(ptr)))
            })
            .collect();

        // Free the outer C array (handles own the inner ptrs now)
        unsafe { libc_free(raw as *mut _) };

        info!("TpHandleGroup ready — {} GPUs", config.world_size);
        Ok(Self { handles, config })
    }

    pub fn handle(&self, rank: usize) -> Arc<Mutex<TpHandle>> {
        self.handles[rank].clone()
    }

    pub fn world_size(&self) -> usize { self.config.world_size }
}

extern "C" { fn free(ptr: *mut std::ffi::c_void); }
unsafe fn libc_free(ptr: *mut std::ffi::c_void) { unsafe { free(ptr) } }
