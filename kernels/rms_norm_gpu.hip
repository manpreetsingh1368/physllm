#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void rms_norm_kernel(
    const __half* input, const __half* weight, __half* output,
    int batch_size, int hidden_dim, float eps
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    float tsum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = __half2float(input[batch_idx * hidden_dim + i]);
        tsum += v * v;
    }
    smem[tid] = tsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    __shared__ float rms;
    if (tid == 0) rms = sqrtf(smem[0] / hidden_dim + eps);
    __syncthreads();

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        int idx = batch_idx * hidden_dim + i;
        float v = __half2float(input[idx]);
        output[idx] = __float2half((v / rms) * __half2float(weight[i]));
    }
}

extern "C" void launch_rms_norm(
    const void* input, const void* weight, void* output,
    int batch_size, int hidden_dim, float eps, void* stream
) {
    hipLaunchKernelGGL(rms_norm_kernel,
        dim3(batch_size), dim3(256), 256 * sizeof(float), (hipStream_t)stream,
        (const __half*)input, (const __half*)weight, (__half*)output,
        batch_size, hidden_dim, eps);
}
