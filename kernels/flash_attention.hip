// flash_attention.hip — Flash Attention 2 for AMD RDNA3/CDNA (ROCm)
// Compiled via: hipcc --amdgpu-target=gfx1100 -O3 -ffast-math

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>

#define WARP_SIZE    64     // AMD wavefront size
#define BLOCK_SIZE   256
#define TILE_Q       16
#define TILE_K       64

// Warp-level max reduction
__device__ float warp_max(float val) {
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor(val, mask));
    return val;
}

// Warp-level sum reduction
__device__ float warp_sum(float val) {
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
        val += __shfl_xor(val, mask);
    return val;
}

// Flash Attention 2 forward kernel (f16 I/O, f32 accumulation)
// Grid: [batch * heads, ceil(seq_q / TILE_Q)]
// Block: [WARP_SIZE * 4]
__global__ void flash_attention_kernel(
    const __half* __restrict__ Q,   // [B, H, Sq, D]
    const __half* __restrict__ K,   // [B, H, Sk, D]
    const __half* __restrict__ V,   // [B, H, Sk, D]
          __half* __restrict__ O,   // [B, H, Sq, D]
    int B, int H, int Sq, int Sk, int D,
    float scale, int causal
) {
    const int bh_idx  = blockIdx.x;          // batch* H + head index
    const int q_tile  = blockIdx.y;          //  which tile of Q
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    const int b = bh_idx / H;
    const int h = bh_idx % H;
    if (b >= B) return;

    const long bh_off = ((long)b * H + h);
    const __half* q_ptr = Q + bh_off * Sq * D;
    const __half* k_ptr = K + bh_off * Sk * D;
    const __half* v_ptr = V + bh_off * Sk * D;
          __half* o_ptr = O + bh_off * Sq * D;

    const int q_start = q_tile * TILE_Q;
    const int q_end   = min(q_start + TILE_Q, Sq);

    // Shared memory for K/V tiles
    __shared__ float Ks[TILE_K][128];   // head_dim <= 128
    __shared__ float Vs[TILE_K][128];

    // Per-thread accumulators (for one Q row)
    float acc[128] = {};
    float row_max  = -FLT_MAX;
    float row_sum  = 0.0f;

    // Q row for this warp
    const int q_row = q_start + warp_id;
    const bool valid_q = (q_row < q_end);

    float q_reg[128] = {};
    if (valid_q) {
        for (int d = lane; d < D; d += WARP_SIZE)
            q_reg[d] = __half2float(q_ptr[q_row * D + d]);
    }

    // Loop over K/V tiles
    for (int k_tile = 0; k_tile < Sk; k_tile += TILE_K) {
        const int k_end_tile = min(k_tile + TILE_K, Sk);

        // Load K tile into shared mem
        for (int ki = tid; ki < TILE_K && (k_tile + ki) < Sk; ki += blockDim.x) {
            for (int d = 0; d < D; d++)
                Ks[ki][d] = __half2float(k_ptr[(k_tile + ki) * D + d]);
        }
        for (int ki = tid; ki < TILE_K && (k_tile + ki) < Sk; ki += blockDim.x) {
            for (int d = 0; d < D; d++)
                Vs[ki][d] = __half2float(v_ptr[(k_tile + ki) * D + d]);
        }
        __syncthreads();

        if (valid_q) {
            // QK^T for this Q row
            float old_max = row_max;
            float new_max = row_max;

            float scores[TILE_K];
            for (int ki = 0; ki < (k_end_tile - k_tile); ki++) {
                if (causal && (k_tile + ki) > q_row) {
                    scores[ki] = -FLT_MAX;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += q_reg[d] * Ks[ki][d];
                scores[ki] = dot * scale;
                new_max = fmaxf(new_max, scores[ki]);
            }

            // Online softmax: rescale accumulators
            float scale_factor = expf(old_max - new_max);
            row_sum *= scale_factor;
            for (int d = 0; d < D; d++)
                acc[d] *= scale_factor;
            row_max = new_max;

            // Accumulate V weighted by exp(score)
            for (int ki = 0; ki < (k_end_tile - k_tile); ki++) {
                float w = expf(scores[ki] - row_max);
                row_sum += w;
                for (int d = 0; d < D; d++)
                    acc[d] += w * Vs[ki][d];
            }
        }
        __syncthreads();
    }

    // Write output
    if (valid_q && row_sum > 0.0f) {
        for (int d = 0; d < D; d++)
            o_ptr[q_row * D + d] = __float2half(acc[d] / row_sum);
    }
}

// ── C-linkage entry point called from Rust ────────────────────────────────────
extern "C" void flash_attention_hip(
    hipStream_t stream,
    const __half* Q, const __half* K, const __half* V, __half* O,
    int B, int H, int Sq, int Sk, int D,
    float scale, int causal
) {
    dim3 grid(B * H, (Sq + TILE_Q - 1) / TILE_Q);
    dim3 block(WARP_SIZE * 4);   // 4 warps per block
    hipLaunchKernelGGL(
        flash_attention_kernel, grid, block, 0, stream,
        Q, K, V, O, B, H, Sq, Sk, D, scale, causal
    );
}
