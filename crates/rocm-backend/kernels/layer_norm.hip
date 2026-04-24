
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
__global__ void rms_norm_k(__half* x, const __half* weight, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    __half* r = x + (long)row * cols;
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __half2float(r[i]);
        sum_sq += v * v;
    }
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.f;
    __syncthreads();
    atomicAdd(&shared_sum, sum_sq);
    __syncthreads();
    float rms = sqrtf(shared_sum / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __half2float(r[i]);
        float w = __half2float(weight[i]);
        r[i] = __float2half(v / rms * w);
    }
}
extern "C" void rms_norm(hipStream_t s, __half* x, const __half* weight,
                          int rows, int cols, float eps) {
    hipLaunchKernelGGL(rms_norm_k, dim3(rows), dim3(128), 0, s,
                       x, weight, rows, cols, eps);
}
