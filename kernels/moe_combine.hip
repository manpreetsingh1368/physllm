#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Combine expert outputs with routing weights
// output[d] = sum_k(expert_weights[k] * expert_outputs[k][d])
__global__ void moe_combine_kernel(
    const __half* expert_outputs,  // [top_k, hidden_dim]
    const float* expert_weights,   // [top_k]
    __half* output,                // [hidden_dim]
    int hidden_dim,
    int top_k
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden_dim) return;
    
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        sum += expert_weights[k] * __half2float(expert_outputs[k * hidden_dim + d]);
    }
    output[d] = __float2half(sum);
}

extern "C" void launch_moe_combine(
    const void* expert_outputs,
    const void* expert_weights,
    void* output,
    int hidden_dim,
    int top_k,
    void* stream
) {
    hipLaunchKernelGGL(moe_combine_kernel,
        dim3((hidden_dim + 255) / 256), dim3(256), 0, (hipStream_t)stream,
        (const __half*)expert_outputs,
        (const float*)expert_weights,
        (__half*)output,
        hidden_dim, top_k);
}
