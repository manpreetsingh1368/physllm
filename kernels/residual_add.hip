#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void residual_add_kernel(
    __half* output, const __half* residual, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = __float2half(__half2float(output[idx]) + __half2float(residual[idx]));
}

extern "C" void launch_residual_add(
    void* output, const void* residual, int size, void* stream
) {
    hipLaunchKernelGGL(residual_add_kernel,
        dim3((size + 255) / 256), dim3(256), 0, (hipStream_t)stream,
        (__half*)output, (const __half*)residual, size);
}
