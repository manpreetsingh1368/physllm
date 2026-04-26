#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void rope_gpu_kernel(
    __half* qk, int num_heads, int seq_len, int head_dim,
    int position_offset, float theta
) {
    int h = blockIdx.y;
    int pos = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim / 2) return;

    int position = position_offset + pos;
    int offset = (h * seq_len + pos) * head_dim;

    float freq = 1.0f / powf(theta, (2.0f * d) / (float)head_dim);
    float angle = position * freq;
    float cos_v = cosf(angle), sin_v = sinf(angle);

    float q0 = __half2float(qk[offset + 2 * d]);
    float q1 = __half2float(qk[offset + 2 * d + 1]);
    qk[offset + 2 * d]     = __float2half(q0 * cos_v - q1 * sin_v);
    qk[offset + 2 * d + 1] = __float2half(q0 * sin_v + q1 * cos_v);
}

extern "C" void launch_rope_gpu(
    void* qk, int num_heads, int seq_len, int head_dim,
    int position_offset, float theta, void* stream
) {
    dim3 grid(seq_len, num_heads);
    dim3 block(head_dim / 2);
    hipLaunchKernelGGL(rope_gpu_kernel,
        grid, block, 0, (hipStream_t)stream,
        (__half*)qk, num_heads, seq_len, head_dim, position_offset, theta);
}
