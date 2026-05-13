// src/layers/attention.rs
//
// ParallelAttention — Megatron tensor-parallel MHA / GQA.
//
// Head distribution across world_size GPUs:
//   Q heads:    [rank * local_q .. (rank+1) * local_q)
//   KV heads:   same split, clamped to ≥1 for GQA where n_kv_heads < world_size
//
// Forward data flow (GPU only):
//   x -> QKV col-parallel GEMMs  (no comm, each rank gets head shard)
//       -> Flash Attention v2     (calls flash_attention_v2.hip from rocm-backend)
//       -> O  row-parallel GEMM   (partial sum)
//       -> async AllReduce        (sum partials → full residual)

// KV Cache:
//   kv_cache_ptr points to a pre-allocated device buffer managed by llm-core.
//   This layer writes K/V for the current position and reads the full context.

use std::sync::Arc;
use parking_lot::Mutex;
use tracing::trace;

use crate::{
    comm::TpHandle,
    config::TpConfig,
    error::TpResult,
    layers::linear::{ColumnParallelLinear, RowParallelLinear},
    loader::ShardedWeight,
};

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub n_heads:     usize,
    pub n_kv_heads:  usize,   // < n_heads for GQA; == n_heads for MHA
    pub head_dim:    usize,
    pub hidden_dim:  usize,
    pub max_seq_len: usize,
}

impl AttentionConfig {
    pub fn local_q_heads(&self, world: usize) -> usize  { (self.n_heads / world).max(1) }
    pub fn local_kv_heads(&self, world: usize) -> usize { (self.n_kv_heads / world).max(1) }
}

/// Tensor-parallel multi-head attention.
pub struct ParallelAttention {
    pub q_proj: ColumnParallelLinear,
    pub k_proj: ColumnParallelLinear,
    pub v_proj: ColumnParallelLinear,
    pub o_proj: RowParallelLinear,
    pub attn_cfg: AttentionConfig,
    pub config:   Arc<TpConfig>,
    pub handle:   Arc<Mutex<TpHandle>>,
}

impl ParallelAttention {
    pub fn from_shards(
        q_shard: &ShardedWeight, q_ptr: *mut u8,
        k_shard: &ShardedWeight, k_ptr: *mut u8,
        v_shard: &ShardedWeight, v_ptr: *mut u8,
        o_shard: &ShardedWeight, o_ptr: *mut u8,
        attn_cfg: AttentionConfig,
        config: Arc<TpConfig>,
        handle: Arc<Mutex<TpHandle>>,
    ) -> Self {
        let null = std::ptr::null_mut();
        Self {
            q_proj: ColumnParallelLinear::from_shard(q_shard, q_ptr, null, config.clone(), handle.clone()),
            k_proj: ColumnParallelLinear::from_shard(k_shard, k_ptr, null, config.clone(), handle.clone()),
            v_proj: ColumnParallelLinear::from_shard(v_shard, v_ptr, null, config.clone(), handle.clone()),
            o_proj: RowParallelLinear::from_shard(o_shard, o_ptr, null, config.clone(), handle.clone()),
            attn_cfg,
            config,
            handle,
        }
    }

    /// GPU forward pass.
    
    /// All pointers are device pointers (fp16).
    /// Buffers q_buf / k_buf / v_buf / attn_buf must be pre-allocated by llm-core
    /// with the correct sizes for this rank.
    
    /// # Parameters
    /// - `x_ptr`    [tokens, hidden_dim]
    /// - `q_buf`   [tokens, local_q_heads * head_dim]   scratch
    /// - `k_buf`  [tokens, local_kv_heads * head_dim]  scratch
    /// - `v_buf`   [tokens, local_kv_heads * head_dim]  scratch
    /// - `attn_buf`   [tokens, local_q_heads * head_dim]   flash-attn output
    /// - `out_ptr` [tokens, hidden_dim]                  final output
    /// - `kv_cache_ptr` [2, max_seq, n_kv_heads, head_dim]   persistent KV cache
    /// - `seq_offset`  position of first token (for RoPE + KV cache indexing)
    /// - `tokens`  number of tokens in this forward call
    pub unsafe fn forward(
        &self,
        x_ptr:        *const u8,
        q_buf:        *mut u8,
        k_buf:        *mut u8,
        v_buf:        *mut u8,
        attn_buf:     *mut u8,
        out_ptr:      *mut u8,
        kv_cache_ptr: *mut u8,
        seq_offset:   i32,
        tokens:       i32,
        // function pointers to rocm-backend kernels (injected by llm-core)
        rope_fn:         unsafe extern "C" fn(*mut u8, i32, i32, i32, i32),
        kv_cache_fn:     unsafe extern "C" fn(*mut u8, *const u8, *const u8, i32, i32),
        flash_attn_fn:   unsafe extern "C" fn(*const u8, *const u8, *const u8, *mut u8, i32, i32, i32),
    ) -> TpResult<()> {
        let world       = self.config.world_size;
        let local_q     = self.attn_cfg.local_q_heads(world) as i32;
        let local_kv    = self.attn_cfg.local_kv_heads(world) as i32;
        let head_dim    = self.attn_cfg.head_dim as i32;

        trace!(
            rank = self.config.rank, tokens, local_q, local_kv, seq_offset,
            "ParallelAttention::forward"
        );

        //  QKV projections (column-parallel, no comm) 
        unsafe {
            self.q_proj.forward(x_ptr, q_buf, tokens)?;
            self.k_proj.forward(x_ptr, k_buf, tokens)?;
            self.v_proj.forward(x_ptr, v_buf, tokens)?;
        }

        //  RoPE (in-place on Q and K) 
        // rope_fn: (buf, tokens, n_heads, head_dim, seq_offset)
        unsafe {
            rope_fn(q_buf, tokens, local_q, head_dim, seq_offset);
            rope_fn(k_buf, tokens, local_kv, head_dim, seq_offset);
        }

        //  KV cache update 
        // kv_cache_fn: (cache, k, v, tokens, seq_offset)
        unsafe { kv_cache_fn(kv_cache_ptr, k_buf, v_buf, tokens, seq_offset); }

        //  Flash Attention v2 
        // flash_attn_fn: (q, k_cache, v_cache, out, tokens, seq_len, head_dim)
        let seq_len = seq_offset + tokens;
        unsafe {
            flash_attn_fn(
                q_buf as *const u8,
                kv_cache_ptr as *const u8,
                kv_cache_ptr.add(                           // V starts after K in cache
                    (self.attn_cfg.max_seq_len * local_kv as usize * head_dim as usize * 2)
                ) as *const u8,
                attn_buf,
                tokens, seq_len, head_dim,
            );
        }

        //  Output projection (row-parallel → async AllReduce) 
        // attn_buf [tokens, local_q * head_dim] → out_ptr [tokens, hidden_dim]
        unsafe { self.o_proj.forward(attn_buf as *const u8, out_ptr, tokens)?; }

        // AllReduce was launched async in o_proj.forward().
        // llm-core calls handle.sync() after this returns, overlapping with
        // the residual add + RMSNorm kernels from rocm-backend.
        Ok(())
    }
}
