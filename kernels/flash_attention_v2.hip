#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void flash_attention_v2_kernel(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int batch, int num_heads, int num_kv_heads,
    int seq_q, int seq_kv, int head_dim, float scale
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_pos >= seq_q) return;

    int kv_h = h * num_kv_heads / num_heads;
    extern __shared__ float scores[];

    float max_score = -1e38f;
    for (int kv_pos = 0; kv_pos < seq_kv; kv_pos++) {
        float dot = 0.0f;
        int q_off = ((b * num_heads + h) * seq_q + q_pos) * head_dim;
        int k_off = ((b * num_kv_heads + kv_h) * seq_kv + kv_pos) * head_dim;
        for (int d = 0; d < head_dim; d++)
            dot += __half2float(Q[q_off + d]) * __half2float(K[k_off + d]);
        scores[kv_pos] = dot * scale;
        max_score = fmaxf(max_score, scores[kv_pos]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < seq_kv; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }
    for (int i = 0; i < seq_kv; i++) scores[i] /= sum_exp;

    int o_off = ((b * num_heads + h) * seq_q + q_pos) * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int kv_pos = 0; kv_pos < seq_kv; kv_pos++) {
            int v_off = ((b * num_kv_heads + kv_h) * seq_kv + kv_pos) * head_dim;
            sum += scores[kv_pos] * __half2float(V[v_off + d]);
        }
        O[o_off + d] = __float2half(sum);
    }
}

extern "C" void launch_flash_attention_v2(
    const void* Q, const void* K, const void* V, void* O,
    int batch, int num_heads, int num_kv_heads,
    int seq_q, int seq_kv, int head_dim, float scale, void* stream
) {
    dim3 block(256);
    dim3 grid((seq_q + 255) / 256, num_heads, batch);
    size_t smem = seq_kv * sizeof(float);
    hipLaunchKernelGGL(flash_attention_v2_kernel,
        grid, block, smem, (hipStream_t)stream,
        (const __half*)Q, (const __half*)K, (const __half*)V, (__half*)O,
        batch, num_heads, num_kv_heads, seq_q, seq_kv, head_dim, scale);
}
