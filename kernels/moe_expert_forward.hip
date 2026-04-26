#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MoE Expert: SwiGLU FFN for a single expert
// gate = SiLU(x @ W_gate^T) * (x @ W_up^T)
// output = gate @ W_down^T
// Each expert has its own W_gate, W_up, W_down
__global__ void moe_expert_silu_kernel(
    const __half* input,      // [hidden_dim]
    const __half* w_gate,     // [intermediate_dim, hidden_dim]
    const __half* w_up,       // [intermediate_dim, hidden_dim]
    __half* output,           // [intermediate_dim]
    int hidden_dim,
    int intermediate_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= intermediate_dim) return;
    
    // Gate projection
    float gate_val = 0.0f;
    float up_val = 0.0f;
    
    int w_off = idx * hidden_dim;
    for (int d = 0; d < hidden_dim; d++) {
        float x = __half2float(input[d]);
        gate_val += x * __half2float(w_gate[w_off + d]);
        up_val   += x * __half2float(w_up[w_off + d]);
    }
    
    // SiLU(gate) * up
    float silu = gate_val / (1.0f + expf(-gate_val));
    output[idx] = __float2half(silu * up_val);
}

// Down projection for expert output
__global__ void moe_expert_down_kernel(
    const __half* input,      // [intermediate_dim]
    const __half* w_down,     // [hidden_dim, intermediate_dim]
    __half* output,           // [hidden_dim]
    int hidden_dim,
    int intermediate_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;
    
    float sum = 0.0f;
    int w_off = idx * intermediate_dim;
    for (int d = 0; d < intermediate_dim; d++) {
        sum += __half2float(input[d]) * __half2float(w_down[w_off + d]);
    }
    output[idx] = __float2half(sum);
}

extern "C" void launch_moe_expert_silu(
    const void* input,
    const void* w_gate,
    const void* w_up,
    void* output,
    int hidden_dim,
    int intermediate_dim,
    void* stream
) {
    hipLaunchKernelGGL(moe_expert_silu_kernel,
        dim3((intermediate_dim + 255) / 256), dim3(256), 0, (hipStream_t)stream,
        (const __half*)input, (const __half*)w_gate,
        (const __half*)w_up, (__half*)output,
        hidden_dim, intermediate_dim);
}

extern "C" void launch_moe_expert_down(
    const void* input,
    const void* w_down,
    void* output,
    int hidden_dim,
    int intermediate_dim,
    void* stream
) {
    hipLaunchKernelGGL(moe_expert_down_kernel,
        dim3((hidden_dim + 255) / 256), dim3(256), 0, (hipStream_t)stream,
        (const __half*)input, (const __half*)w_down, (__half*)output,
        hidden_dim, intermediate_dim);
}
