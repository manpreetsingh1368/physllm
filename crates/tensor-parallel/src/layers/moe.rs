// src/layers/moe.rs
//
// ParallelMoE — expert-parallel MoE layer.
//
// GPT-OSS-20B: 64 experts, top-2 routing, hidden=2880, intermediate=1536.
// Each GPU owns n_experts/world experts (whole, not sharded further).
//
// Forward protocol (avoids AllToAll):
//   1. Router GEMM runs on ALL ranks (replicated weights) → same routing decisions
//   2. Each rank runs its local experts on ALL tokens
//      (tokens not routed to this rank's experts contribute 0 after masking)
//   3. async AllReduce sums partial expert outputs across all ranks
//
// This is "zero-communication expert parallel" — trades extra compute
// (each rank processes all tokens through its experts) for zero AllToAll latency.
// On MI300X with 155GB free VRAM and Infinity Fabric, this beats AllToAll
// for batch sizes up to ~256 tokens.

use std::sync::Arc;
use parking_lot::Mutex;
use tracing::trace;

use crate::{
    comm::TpHandle,
    config::TpConfig,
    error::TpResult,
};

#[derive(Debug, Clone)]
pub struct MoEConfig {
    pub n_experts:        usize,
    pub top_k:            usize,
    pub hidden_dim:       usize,
    pub intermediate_dim: usize,
}

/// GPU state for one local expert (gate, up, down weight device ptrs).
pub struct GpuExpert {
    pub gate_ptr: *mut u8,  // device ptr [intermediate, hidden] fp16
    pub up_ptr:   *mut u8,  // device ptr [intermediate, hidden] fp16
    pub down_ptr: *mut u8,  // device ptr [hidden, intermediate] fp16
    pub intermediate: i32,
    pub hidden:       i32,
}

unsafe impl Send for GpuExpert {}
unsafe impl Sync for GpuExpert {}

/// Tensor-parallel (expert-parallel) MoE layer.
pub struct ParallelMoE {
    /// Local experts only (world_size GPUs share all n_experts).
    pub local_experts: Vec<GpuExpert>,
    /// Router weight device ptr [n_experts, hidden_dim] fp16 — replicated all ranks.
    pub router_ptr: *mut u8,
    pub moe_cfg:  MoEConfig,
    pub config:   Arc<TpConfig>,
    pub handle:   Arc<Mutex<TpHandle>>,
    /// Index of this rank's first expert in the global list.
    pub expert_offset: usize,
}

unsafe impl Send for ParallelMoE {}
unsafe impl Sync for ParallelMoE {}

impl ParallelMoE {
    pub fn new(
        local_experts: Vec<GpuExpert>,
        router_ptr: *mut u8,
        moe_cfg: MoEConfig,
        config: Arc<TpConfig>,
        handle: Arc<Mutex<TpHandle>>,
    ) -> Self {
        let experts_per_rank = (moe_cfg.n_experts + config.world_size - 1) / config.world_size;
        let expert_offset    = config.rank * experts_per_rank;
        Self { local_experts, router_ptr, moe_cfg, config, handle, expert_offset }
    }

    /// GPU forward pass.
    ///
    /// All pointers are device pointers (fp16).
    ///
    /// # Parameters
    /// - `x_ptr`         [tokens, hidden_dim] input
    /// - `out_ptr`       [tokens, hidden_dim] output (zeroed by caller before call)
    /// - `router_scores` [tokens, n_experts] scratch (pre-allocated)
    /// - `expert_buf`    [tokens, intermediate_dim] scratch per expert
    /// - `tokens`
    /// # Kernel function pointers (injected from rocm-backend)
    /// - `router_gemm_fn`  runs x @ router.T → router_scores, then softmax
    /// - `moe_router_fn`   top-k selection → returns routing indices + weights
    /// - `silu_fn`         SiLU(gate) ⊙ up in-place
    pub unsafe fn forward(
        &self,
        x_ptr:         *const u8,
        out_ptr:       *mut u8,
        router_scores: *mut u8,
        expert_buf_gate: *mut u8,
        expert_buf_up:   *mut u8,
        expert_buf_down: *mut u8,
        tokens: i32,
        // kernel fn pointers from rocm-backend
        router_gemm_fn: unsafe extern "C" fn(
            *const u8, *const u8, *mut u8, i32, i32, i32,  // x, router, scores, tokens, n_experts, hidden
        ),
        moe_route_fn: unsafe extern "C" fn(
            *const u8, i32, i32, i32,  // scores, tokens, n_experts, top_k
            *mut i32, *mut f32,        // out: expert_ids [tokens, top_k], weights [tokens, top_k]
        ),
        silu_fn: unsafe extern "C" fn(*mut u8, *const u8, usize),  // out_gate, up, numel
        gemm_fn: unsafe extern "C" fn(
            *const u8, *const u8, *mut u8, i32, i32, i32, f32, f32,
        ),
    ) -> TpResult<()> {
        let n_experts    = self.moe_cfg.n_experts as i32;
        let top_k        = self.moe_cfg.top_k as i32;
        let hidden       = self.moe_cfg.hidden_dim as i32;
        let inter        = self.moe_cfg.intermediate_dim as i32;
        let local_count  = self.local_experts.len();
        let offset       = self.expert_offset;

        trace!(
            rank = self.config.rank,
            expert_offset = offset,
            local_count,
            tokens,
            "ParallelMoE::forward"
        );

        //  Step 1: Router (same on all ranks — replicated weights) 
        // router_scores = softmax(x @ router.T)  [tokens, n_experts]
        unsafe {
            router_gemm_fn(x_ptr, self.router_ptr as *const u8, router_scores,
                           tokens, n_experts, hidden);
        }

        //  Step 2: Top-k routing 
        // Allocate on stack for small top_k; production uses pre-alloc device buf
        let route_elems = (tokens as usize) * (top_k as usize);
        let mut expert_ids: Vec<i32>  = vec![0i32;  route_elems];
        let mut gate_weights: Vec<f32> = vec![0f32; route_elems];

        unsafe {
            moe_route_fn(
                router_scores as *const u8,
                tokens, n_experts, top_k,
                expert_ids.as_mut_ptr(),
                gate_weights.as_mut_ptr(),
            );
        }

        //  Step 3: Local expert forward 
        // For each local expert: process ALL tokens, mask non-routed ones by
        // multiplying with gate weight (0 for tokens not routed here).
        for (local_idx, expert) in self.local_experts.iter().enumerate() {
            let global_id = (offset + local_idx) as i32;

            // gate = x @ W_gate.T   [tokens, inter]
            unsafe {
                gemm_fn(x_ptr, expert.gate_ptr as *const u8, expert_buf_gate,
                        tokens, inter, hidden, 1.0, 0.0);
            }

            // up = x @ W_up.T   [tokens, inter]
            unsafe {
                gemm_fn(x_ptr, expert.up_ptr as *const u8, expert_buf_up,
                        tokens, inter, hidden, 1.0, 0.0);
            }

            // SwiGLU: gate_buf = SiLU(gate) ⊙ up  (in-place on gate_buf)
            unsafe {
                silu_fn(
                    expert_buf_gate,
                    expert_buf_up as *const u8,
                    (tokens as usize) * (inter as usize),
                );
            }

            // down = gate_swiglu @ W_down.T   [tokens, hidden]
            // Write into expert_buf_down, then scale by routing weights + accumulate into out_ptr
            unsafe {
                gemm_fn(expert_buf_gate as *const u8, expert.down_ptr as *const u8,
                        expert_buf_down, tokens, hidden, inter, 1.0, 0.0);
            }

            // Accumulate: for each token, if routed to global_id, add gate_weight * expert_out
            // In production: replace this with moe_combine.hip kernel (already in kernels/).
            // That kernel does the masked scale+add in a single GPU pass.
            // Pseudo-code of what moe_combine.hip does:
            //   for t in 0..tokens:
            //     for k in 0..top_k:
            //       if expert_ids[t * top_k + k] == global_id:
            //         out[t, :] += gate_weights[t * top_k + k] * expert_buf_down[t, :]
            // TODO: call moe_combine.hip here once moe_combine_fn is injected
        }

        // Step 4: Async AllReduce — sum partial expert outputs 
        let numel = (tokens as usize) * (self.moe_cfg.hidden_dim);
        {
            let h = self.handle.lock();
            unsafe { h.allreduce_fp16_async(out_ptr, numel)?; }
        }
        // llm-core calls handle.sync() after this returns.

        Ok(())
    }
}
