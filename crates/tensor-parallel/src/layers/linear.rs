// src/layers/linear.rs
//
// ColumnParallelLinear  y_local = x @ W_shard.T     no comm needed
// RowParallelLinear     y = AllReduce(x_shard @ W_shard.T)
//
// All tensors live on GPU device memory.
// `weight_ptr` and `bias_ptr` are device pointers returned by the caller
// after uploading ShardedWeight.data via hipMemcpy.

use std::sync::Arc;
use parking_lot::Mutex;
use tracing::trace;

use crate::{
    comm::TpHandle,
    config::TpConfig,
    error::TpResult,
    loader::ShardedWeight,
};

//  ColumnParallelLinear 

/// Q/K/V projections and MLP gate/up projections.
/// Each rank computes its column shard; no AllReduce needed.
/// Caller AllGathers if the full output is needed downstream.
pub struct ColumnParallelLinear {
    /// Device pointer to this rank's weight shard [out_shard, in_features] fp16
    pub weight_ptr: *mut u8,
    /// Device pointer to bias shard [out_shard] fp16, or null
    pub bias_ptr: *mut u8,
    pub in_features:  i32,
    pub out_features: i32,   // full output dim
    pub out_shard:    i32,   // out_features / world_size (this rank's slice)
    pub config: Arc<TpConfig>,
    pub handle: Arc<Mutex<TpHandle>>,
}

// SAFETY: device pointers are only accessed while holding the Mutex<TpHandle>.
unsafe impl Send for ColumnParallelLinear {}
unsafe impl Sync for ColumnParallelLinear {}

impl ColumnParallelLinear {
    pub fn from_shard(
        shard: &ShardedWeight,
        weight_ptr: *mut u8,  // caller uploads shard.data → GPU and passes ptr
        bias_ptr: *mut u8,    // null if no bias
        config: Arc<TpConfig>,
        handle: Arc<Mutex<TpHandle>>,
    ) -> Self {
        let out_shard    = shard.shard_shape[0] as i32;
        let in_features  = shard.shard_shape[1] as i32;
        let out_features = shard.full_shape[0] as i32;
        Self { weight_ptr, bias_ptr, in_features, out_features, out_shard, config, handle }
    }

    /// y_shard = x @ W_shard.T   (async, on the rank's stream)
    ///
    /// `x_ptr`  — device ptr [tokens, in_features] fp16
    /// `y_ptr`  — device ptr [tokens, out_shard] fp16 (pre-allocated by caller)
    /// `tokens` — batch × seq_len
    pub unsafe fn forward(
        &self, x_ptr: *const u8, y_ptr: *mut u8, tokens: i32,
    ) -> TpResult<()> {
        trace!(rank = self.config.rank, tokens, out_shard = self.out_shard, "ColParallel fwd");
        let h = self.handle.lock();
        // y = x @ W.T  →  gemm_fp16_bt(m=tokens, n=out_shard, k=in_features, A=x, B=W)
        unsafe {
            h.gemm_fp16_bt(
                tokens, self.out_shard, self.in_features,
                x_ptr, self.weight_ptr as *const u8, y_ptr,
                1.0, 0.0,
            )?;
        }
        // Bias add (fused into separate kernel in production — hipBLAS axpy or custom kernel)
        // TODO: call rms_norm / bias kernel from rocm-backend here if needed
        Ok(())
    }
}

//  RowParallelLinear 

/// O projection and MLP down projection.
/// Each rank computes a partial sum; AllReduce gives the full output.
pub struct RowParallelLinear {
    /// Device ptr to this rank's weight shard [out_features, in_shard] fp16
    pub weight_ptr: *mut u8,
    pub bias_ptr:   *mut u8,
    pub in_features:  i32,
    pub in_shard:     i32,  // in_features / world_size
    pub out_features: i32,
    pub config: Arc<TpConfig>,
    pub handle: Arc<Mutex<TpHandle>>,
}

unsafe impl Send for RowParallelLinear {}
unsafe impl Sync for RowParallelLinear {}

impl RowParallelLinear {
    pub fn from_shard(
        shard: &ShardedWeight,
        weight_ptr: *mut u8,
        bias_ptr: *mut u8,
        config: Arc<TpConfig>,
        handle: Arc<Mutex<TpHandle>>,
    ) -> Self {
        let in_shard     = shard.shard_shape[1] as i32;
        let out_features = shard.shard_shape[0] as i32;
        let in_features  = shard.full_shape[1] as i32;
        Self { weight_ptr, bias_ptr, in_features, in_shard, out_features, config, handle }
    }

    /// y = AllReduce(x_shard @ W_shard.T)
    ///
    /// `x_shard_ptr` — device ptr [tokens, in_shard] fp16 (this rank's input slice)
    /// `y_ptr`       — device ptr [tokens, out_features] fp16 (pre-allocated)
    /// After return: y holds the full reduced output on all ranks.
    pub unsafe fn forward(
        &self, x_shard_ptr: *const u8, y_ptr: *mut u8, tokens: i32,
    ) -> TpResult<()> {
        trace!(rank = self.config.rank, tokens, in_shard = self.in_shard, "RowParallel fwd");
        let numel = (tokens as usize) * (self.out_features as usize);

        {
            let h = self.handle.lock();
            // Partial sum: y_partial = x_shard @ W_shard.T
            unsafe {
                h.gemm_fp16_bt(
                    tokens, self.out_features, self.in_shard,
                    x_shard_ptr, self.weight_ptr as *const u8, y_ptr,
                    1.0, 0.0,
                )?;
            }
            // Non-blocking AllReduce — overlaps with caller's next operation
            unsafe { h.allreduce_fp16_async(y_ptr, numel)?; }
        }
        // Caller calls handle.sync() or chains the next async op.
        // Bias is applied by rank 0 only after sync (see ParallelAttention).
        Ok(())
    }
}
