// kernels/softmax.hip
//
// Fused online softmax kernel (f16 I/O, f32 accumulation).
//
// Runs one threadblock per row. Each row is the attention score vector
// for one query token across all key positions. Fused max-find + exp + sum
// avoids two extra global memory passes compared to naive softmax.
//
// Grid:  dim3(rows)        — one block per row
// Block: dim3(WARP_SIZE)   — one warp per block (sufficient for head_dim ≤ 128)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>

#define WARP_SIZE 64   // AMD wavefront

//  Warp reductions 

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor(val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val += __shfl_xor(val, mask);
    return val;
}

//  Kernel 

// In-place row-wise softmax.
// x: [rows, cols] f16, modified in place.
__global__ void softmax_f16_kernel(
    __half* __restrict__ x,
    const int rows,
    const int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    __half* row_ptr = x + (long)row * cols;

    // Pass 1: find row maximum (numerically stable softmax)
    float row_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_max = fmaxf(row_max, __half2float(row_ptr[i]));
    row_max = warp_reduce_max(row_max);

    // Pass 2: exp(x - max) and accumulate sum
    float row_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(__half2float(row_ptr[i]) - row_max);
        row_ptr[i] = __float2half(e);
        row_sum += e;
    }
    row_sum = warp_reduce_sum(row_sum);

    // Pass 3: normalise
    const float inv_sum = (row_sum > 0.f) ? (1.0f / row_sum) : 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_ptr[i] = __float2half(__half2float(row_ptr[i]) * inv_sum);
}

// C-linkage entry point called from Rust 

extern "C" void softmax_f16(
    hipStream_t  stream,
    __half*      x,       // [rows, cols]
    int          rows,
    int          cols
) {
    dim3 grid(rows);
    dim3 block(WARP_SIZE);
    hipLaunchKernelGGL(
        softmax_f16_kernel,
        grid, block, 0, stream,
        x, rows, cols
    );
}

// Batched softmax: [batch * heads, seq, seq] attention matrices
extern "C" void softmax_f16_batched(
    hipStream_t stream,
    __half*     x,
    int         total_rows,   // batch * heads * seq
    int         cols          // seq (key length)
) {
    softmax_f16(stream, x, total_rows, cols);
}
