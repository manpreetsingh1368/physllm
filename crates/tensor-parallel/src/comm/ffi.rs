// src/comm/ffi.rs — raw FFI matching gpu_shim.hip
use std::ffi::c_void;

/// Opaque C TpHandle pointer
pub type RawHandle = c_void;

extern "C" {
    pub fn tp_init(world_size: i32, devices: *mut i32) -> *mut *mut RawHandle;
    pub fn tp_destroy(h: *mut RawHandle) -> i32;

    pub fn tp_allreduce_fp16(h: *mut RawHandle, buf: *mut c_void, count: usize) -> i32;
    pub fn tp_allreduce_fp32(h: *mut RawHandle, buf: *mut c_void, count: usize) -> i32;
    pub fn tp_allreduce_fp16_async(h: *mut RawHandle, buf: *mut c_void, count: usize) -> i32;
    pub fn tp_allgather_fp16(h: *mut RawHandle, send: *mut c_void, recv: *mut c_void, send_count: usize) -> i32;
    pub fn tp_broadcast_fp16(h: *mut RawHandle, buf: *mut c_void, count: usize) -> i32;
    pub fn tp_sync(h: *mut RawHandle) -> i32;

    pub fn tp_gemm_fp16(
        h: *mut RawHandle,
        m: i32, n: i32, k: i32,
        a: *const c_void, b: *const c_void, c: *mut c_void,
        alpha: f32, beta: f32,
    ) -> i32;

    pub fn tp_gemm_fp16_bt(
        h: *mut RawHandle,
        m: i32, n: i32, k: i32,
        a: *const c_void, b: *const c_void, c: *mut c_void,
        alpha: f32, beta: f32,
    ) -> i32;

    pub fn tp_rank(h: *mut RawHandle) -> i32;
    pub fn tp_world_size(h: *mut RawHandle) -> i32;
    pub fn tp_device_id(h: *mut RawHandle) -> i32;
}
