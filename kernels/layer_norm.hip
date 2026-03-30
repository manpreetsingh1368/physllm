// layer_norm.hip — RMSNorm and LayerNorm for AMD ROCm

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WARP_SIZE 64

__device__ float warp_reduce_sum(float val) {
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
        val += __shfl_xor(val, mask);
    return val;
}

__global__ void layer_norm_kernel(
    __half* __restrict__       X,       // [rows, cols] — modified in-place
    const __half* __restrict__ gamma,   // [cols]
    const __half* __restrict__ beta,    // [cols] — NULL for RMSNorm
    int rows, int cols,
    float eps, int rms_only
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    __half* row_ptr = X + row * cols;

    // Compute mean (only if not rms_only)
    float mean = 0.0f;
    if (!rms_only) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            sum += __half2float(row_ptr[i]);
        sum = warp_reduce_sum(sum);
        mean = sum / cols;
    }

    // Compute variance / RMS
    float var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __half2float(row_ptr[i]) - mean;
        var += v * v;
    }
    var = warp_reduce_sum(var);
    float inv_std = rsqrtf(var / cols + eps);

    // Normalise + scale + shift
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v    = (__half2float(row_ptr[i]) - mean) * inv_std;
        float g    = __half2float(gamma[i]);
        float b    = (beta && !rms_only) ? __half2float(beta[i]) : 0.0f;
        row_ptr[i] = __float2half(v * g + b);
    }
}

extern "C" void layer_norm_hip(
    hipStream_t stream,
    __half* X, const __half* gamma, const __half* beta,
    int rows, int cols, float eps, int rms_only
) {
    dim3 grid(rows);
    dim3 block(WARP_SIZE);
    hipLaunchKernelGGL(
        layer_norm_kernel, grid, block, 0, stream,
        X, gamma, beta, rows, cols, eps, rms_only
    );
}
