#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void lm_head_kernel(
    const __half* hidden, const __half* weight,
    float* logits, int hidden_dim, int vocab_size
) {
    int vocab_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vocab_idx >= vocab_size) return;

    float sum = 0.0f;
    int w_off = vocab_idx * hidden_dim;
    for (int i = 0; i < hidden_dim; i++)
        sum += __half2float(hidden[i]) * __half2float(weight[w_off + i]);
    logits[vocab_idx] = sum;
}

extern "C" void launch_lm_head(
    const void* hidden, const void* weight,
    void* logits, int hidden_dim, int vocab_size, void* stream
) {
    hipLaunchKernelGGL(lm_head_kernel,
        dim3((vocab_size + 255) / 256), dim3(256), 0, (hipStream_t)stream,
        (const __half*)hidden, (const __half*)weight,
        (float*)logits, hidden_dim, vocab_size);
}
