// rope_embedding.hip — Rotary Position Embedding for AMD ROCm

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

__global__ void rope_kernel(
    __half* __restrict__ Q,
    __half* __restrict__ K,
    int seq_len, int head_dim,
    int offset, float theta,
    int numel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 >= numel) return;

    // Decode position and dimension
    const int pair     = idx;
    const int dim_idx  = pair % (head_dim / 2);
    const int seq_pos  = (pair / (head_dim / 2)) % seq_len;
    const int head_idx = pair / (head_dim / 2 * seq_len);
    (void)head_idx;

    const int pos  = seq_pos + offset;
    const float freq = (float)pos / powf(theta, 2.0f * dim_idx / head_dim);
    float sin_f, cos_f;
    sincosf(freq, &sin_f, &cos_f);

    // Apply to Q
    {
        const int i0 = pair * 2;
        const int i1 = i0 + 1;
        if (i0 < numel && i1 < numel) {
            float x0 = __half2float(Q[i0]);
            float x1 = __half2float(Q[i1]);
            Q[i0] = __float2half(x0 * cos_f - x1 * sin_f);
            Q[i1] = __float2half(x0 * sin_f + x1 * cos_f);
        }
    }

    // Apply to K
    {
        const int i0 = pair * 2;
        const int i1 = i0 + 1;
        if (i0 < numel && i1 < numel) {
            float x0 = __half2float(K[i0]);
            float x1 = __half2float(K[i1]);
            K[i0] = __float2half(x0 * cos_f - x1 * sin_f);
            K[i1] = __float2half(x0 * sin_f + x1 * cos_f);
        }
    }
}

extern "C" void rope_embedding_hip(
    hipStream_t stream,
    __half* Q, __half* K,
    int seq_len, int head_dim,
    int offset, float theta,
    int numel
) {
    const int threads = 256;
    const int blocks  = (numel / 2 + threads - 1) / threads;
    hipLaunchKernelGGL(
        rope_kernel, dim3(blocks), dim3(threads), 0, stream,
        Q, K, seq_len, head_dim, offset, theta, numel
    );
}
