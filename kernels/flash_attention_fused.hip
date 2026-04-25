#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Flash Attention kernel - fused QK^T, softmax, @V
__global__ void flash_attention_kernel(
    const __half* Q,           // [batch, num_heads, seq_q, head_dim]
    const __half* K,           // [batch, num_kv_heads, seq_kv, head_dim]
    const __half* V,           // [batch, num_kv_heads, seq_kv, head_dim]
    __half* O,                 // [batch, num_heads, seq_q, head_dim]
    int batch,
    int num_heads,
    int num_kv_heads,
    int seq_q,
    int seq_kv,
    int head_dim,
    float scale
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_pos >= seq_q) return;
    
    // GQA: map query head to KV head
    int kv_h = h * num_kv_heads / num_heads;
    
    // Compute attention for this query position
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;  // [seq_kv]
    
    // Q·K^T for all KV positions
    float max_score = -1e38f;
    for (int kv_pos = 0; kv_pos < seq_kv; kv_pos++) {
        float dot = 0.0f;
        
        int q_offset = ((b * num_heads + h) * seq_q + q_pos) * head_dim;
        int k_offset = ((b * num_kv_heads + kv_h) * seq_kv + kv_pos) * head_dim;
        
        for (int d = 0; d < head_dim; d++) {
            float q_val = __half2float(Q[q_offset + d]);
            float k_val = __half2float(K[k_offset + d]);
            dot += q_val * k_val;
        }
        
        scores[kv_pos] = dot * scale;
        max_score = fmaxf(max_score, scores[kv_pos]);
    }
    
    // Softmax
    float sum_exp = 0.0f;
    for (int kv_pos = 0; kv_pos < seq_kv; kv_pos++) {
        scores[kv_pos] = expf(scores[kv_pos] - max_score);
        sum_exp += scores[kv_pos];
    }
    
    for (int kv_pos = 0; kv_pos < seq_kv; kv_pos++) {
        scores[kv_pos] /= sum_exp;
    }
    
    // Weighted sum of values
    int o_offset = ((b * num_heads + h) * seq_q + q_pos) * head_dim;
    
    for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int kv_pos = 0; kv_pos < seq_kv; kv_pos++) {
            int v_offset = ((b * num_kv_heads + kv_h) * seq_kv + kv_pos) * head_dim;
            sum += scores[kv_pos] * __half2float(V[v_offset + d]);
        }
        O[o_offset + d] = __float2half(sum);
    }
}

extern "C" {
    void launch_flash_attention(
        const void* Q,
        const void* K,
        const void* V,
        void* O,
        int batch,
        int num_heads,
        int num_kv_heads,
        int seq_q,
        int seq_kv,
        int head_dim,
        float scale,
        void* stream
    ) {
        dim3 block(256);
        dim3 grid((seq_q + 255) / 256, num_heads, batch);
        
        size_t shared_mem = seq_kv * sizeof(float);
        
        hipLaunchKernelGGL(
            flash_attention_kernel,
            grid, block, shared_mem, (hipStream_t)stream,
            (const __half*)Q, (const __half*)K, (const __half*)V,
            (__half*)O,
            batch, num_heads, num_kv_heads,
            seq_q, seq_kv, head_dim, scale
        );
    }
}
