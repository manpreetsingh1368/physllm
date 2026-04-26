#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void silu_multiply_kernel(
    const __half* gate, const __half* up, __half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu = g / (1.0f + expf(-g));
    output[idx] = __float2half(silu * u);
}

extern "C" void launch_silu_multiply(
    const void* gate, const void* up, void* output, int size, void* stream
) {
    hipLaunchKernelGGL(silu_multiply_kernel,
        dim3((size + 255) / 256), dim3(256), 0, (hipStream_t)stream,
        (const __half*)gate, (const __half*)up, (__half*)output, size);
}
