
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

//  QKV Projection 
// Projects input to Q, K, V all at once (fused)
__global__ void vision_qkv_proj_kernel(
    const __half* input,  // [batch, seq_len, embed_dim]
    const __half* qkv_weight, // [3 * embed_dim, embed_dim]
    const __half* qkv_bias,  // [3 * embed_dim]
    __half* qkv,   // [batch, seq_len, 3 * embed_dim]
    int batch_size,
    int seq_len,
    int embed_dim
) {
    int b   = blockIdx.z;
    int s   = blockIdx.y;
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = 3 * embed_dim;

    if (b >= batch_size || s >= seq_len || out >= total_out) return;

    float sum = 0.0f;
    int in_off = b * seq_len * embed_dim + s * embed_dim;
    int w_off = out * embed_dim;

    for (int i = 0; i < embed_dim; i++) {
        sum += __half2float(input[in_off + i]) * __half2float(qkv_weight[w_off + i]);
    }
    if (qkv_bias) sum += __half2float(qkv_bias[out]);

    qkv[b * seq_len * total_out + s * total_out + out] = __float2half(sum);
}

//  Vision Self-Attention 
// Full (non-causal) attention — all patches attend to all patches
__global__ void vision_attention_kernel(
    const __half* Q, // [batch, num_heads, seq_len, head_dim]
    const __half* K,  // [batch, num_heads, seq_len, head_dim]
    const __half* V, // [batch, num_heads, seq_len, head_dim]
    __half* output,   // [batch, num_heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || h >= num_heads || q >= seq_len) return;

    extern __shared__ float attn_scores[];

    // Compute attention scores for query position q
    float max_score = -1e38f;
    for (int k = 0; k < seq_len; k++) {
        float dot = 0.0f;
        int q_off = ((b * num_heads + h) * seq_len + q) * head_dim;
        int k_off = ((b * num_heads + h) * seq_len + k) * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(Q[q_off + d]) * __half2float(K[k_off + d]);
        }
        attn_scores[k] = dot * scale;
        max_score = fmaxf(max_score, attn_scores[k]);
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        attn_scores[k] = expf(attn_scores[k] - max_score);
        sum_exp += attn_scores[k];
    }
    for (int k = 0; k < seq_len; k++) attn_scores[k] /= sum_exp;

    // Weighted sum of values
    int out_off = ((b * num_heads + h) * seq_len + q) * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float weighted = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            int v_off = ((b * num_heads + h) * seq_len + k) * head_dim;
            weighted += attn_scores[k] * __half2float(V[v_off + d]);
        }
        output[out_off + d] = __float2half(weighted);
    }
}

//  Reshape QKV for Multi-Head 
// Splits fused QKV [batch, seq, 3*embed] into
// separate Q, K, V each [batch, heads, seq, head_dim]
__global__ void reshape_qkv_kernel(
    const __half* qkv,  // [batch, seq_len, 3 * embed_dim]
    __half* Q,    // [batch, num_heads, seq_len, head_dim]
    __half* K,
    __half* V,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int hd = blockIdx.x * blockDim.x + threadIdx.x;  // head * head_dim index

    int embed_dim = num_heads * head_dim;
    if (b >= batch_size || s >= seq_len || hd >= embed_dim) return;

    int h = hd / head_dim;
    int d = hd % head_dim;

    int qkv_off = b * seq_len * 3 * embed_dim + s * 3 * embed_dim;
    int out_off  = (b * num_heads + h) * seq_len * head_dim + s * head_dim + d;

    Q[out_off] = qkv[qkv_off + hd];
    K[out_off] = qkv[qkv_off + embed_dim + hd];
    V[out_off] = qkv[qkv_off + 2 * embed_dim + hd];
}

//  Output Projection 
__global__ void vision_out_proj_kernel(
    const __half* attn_out, // [batch, num_heads, seq_len, head_dim]
    const __half* o_weight, // [embed_dim, embed_dim]
    const __half* o_bias, // [embed_dim]
    __half* output, // [batch, seq_len, embed_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    int embed_dim = num_heads * head_dim;
    if (b >= batch_size || s >= seq_len || e >= embed_dim) return;

    float sum = 0.0f;
    for (int hd = 0; hd < embed_dim; hd++) {
        int h = hd / head_dim;
        int d = hd % head_dim;
        float attn_val = __half2float(
            attn_out[(b * num_heads + h) * seq_len * head_dim + s * head_dim + d]
        );
        sum += attn_val * __half2float(o_weight[e * embed_dim + hd]);
    }
    if (o_bias) sum += __half2float(o_bias[e]);

    output[b * seq_len * embed_dim + s * embed_dim + e] = __float2half(sum);
}

//  Vision FFN (MLP inside ViT) 
// Standard MLP with GELU activation (not SwiGLU — ViT uses plain GELU)
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void vision_ffn_kernel(
    const __half* input,// [batch, seq_len, embed_dim]
    const __half* w1,  // [ffn_dim, embed_dim]
    const __half* b1,// [ffn_dim]
    const __half* w2,    // [embed_dim, ffn_dim]
    const __half* b2,  // [embed_dim]
    __half* output,  // [batch, seq_len, embed_dim]
    int batch_size,
    int seq_len,
    int embed_dim,
    int ffn_dim
) {
    extern __shared__ float hidden[];

    int b = blockIdx.z;
    int s = blockIdx.y;
    int tid = threadIdx.x;

    if (b >= batch_size || s >= seq_len) return;

    int in_off = b * seq_len * embed_dim + s * embed_dim;

    // First layer: [embed_dim] → [ffn_dim] with GELU
    for (int f = tid; f < ffn_dim; f += blockDim.x) {
        float sum = 0.0f;
        for (int e = 0; e < embed_dim; e++) {
            sum += __half2float(input[in_off + e]) * __half2float(w1[f * embed_dim + e]);
        }
        if (b1) sum += __half2float(b1[f]);
        hidden[f] = gelu(sum);
    }
    __syncthreads();

    // Second layer: [ffn_dim] → [embed_dim]
    int out_off = b * seq_len * embed_dim + s * embed_dim;
    for (int e = tid; e < embed_dim; e += blockDim.x) {
        float sum = 0.0f;
        for (int f = 0; f < ffn_dim; f++) {
            sum += hidden[f] * __half2float(w2[e * ffn_dim + f]);
        }
        if (b2) sum += __half2float(b2[e]);
        output[out_off + e] = __float2half(sum);
    }
}

// Layer Norm for Vision 
__global__ void vision_layernorm_kernel(
    const __half* input,  // [batch, seq_len, embed_dim]
    const __half* weight,// [embed_dim]
    const __half* bias, // [embed_dim]
    __half* output,   // [batch, seq_len, embed_dim]
    int batch_size,
    int seq_len,
    int embed_dim,
    float eps
) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int tid = threadIdx.x;

    if (b >= batch_size || s >= seq_len) return;

    extern __shared__ float smem[];

    int off = b * seq_len * embed_dim + s * embed_dim;

    // Compute mean
    float thread_sum = 0.0f;
    for (int e = tid; e < embed_dim; e += blockDim.x)
        thread_sum += __half2float(input[off + e]);
    smem[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float mean = smem[0] / embed_dim;

    // Compute variance
    float thread_var = 0.0f;
    for (int e = tid; e < embed_dim; e += blockDim.x) {
        float diff = __half2float(input[off + e]) - mean;
        thread_var += diff * diff;
    }
    smem[tid] = thread_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float var = smem[0] / embed_dim;
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply scale/bias
    for (int e = tid; e < embed_dim; e += blockDim.x) {
        float normed = (__half2float(input[off + e]) - mean) * inv_std;
        float scale  = weight ? __half2float(weight[e]) : 1.0f;
        float shift  = bias   ? __half2float(bias[e])   : 0.0f;
        output[off + e] = __float2half(normed * scale + shift);
    }
}

//  C Exports 

extern "C" {

void launch_vision_qkv_proj(
    const void* input, const void* qkv_weight, const void* qkv_bias,
    void* qkv, int batch, int seq, int embed, void* stream
) {
    dim3 block(64);
    dim3 grid((3 * embed + 63) / 64, seq, batch);
    hipLaunchKernelGGL(vision_qkv_proj_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const __half*)input, (const __half*)qkv_weight, (const __half*)qkv_bias,
        (__half*)qkv, batch, seq, embed);
}

void launch_reshape_qkv(
    const void* qkv, void* Q, void* K, void* V,
    int batch, int heads, int seq, int head_dim, void* stream
) {
    int embed = heads * head_dim;
    dim3 block(64);
    dim3 grid((embed + 63) / 64, seq, batch);
    hipLaunchKernelGGL(reshape_qkv_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const __half*)qkv, (__half*)Q, (__half*)K, (__half*)V,
        batch, heads, seq, head_dim);
}

void launch_vision_attention(
    const void* Q, const void* K, const void* V, void* output,
    int batch, int heads, int seq, int head_dim, float scale, void* stream
) {
    dim3 block(32);
    dim3 grid((seq + 31) / 32, heads, batch);
    size_t smem = seq * sizeof(float);
    hipLaunchKernelGGL(vision_attention_kernel,
        grid, block, smem, (hipStream_t)stream,
        (const __half*)Q, (const __half*)K, (const __half*)V, (__half*)output,
        batch, heads, seq, head_dim, scale);
}

void launch_vision_out_proj(
    const void* attn_out, const void* o_weight, const void* o_bias, void* output,
    int batch, int heads, int seq, int head_dim, void* stream
) {
    int embed = heads * head_dim;
    dim3 block(64);
    dim3 grid((embed + 63) / 64, seq, batch);
    hipLaunchKernelGGL(vision_out_proj_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const __half*)attn_out, (const __half*)o_weight, (const __half*)o_bias,
        (__half*)output, batch, heads, seq, head_dim);
}

void launch_vision_ffn(
    const void* input, const void* w1, const void* b1,
    const void* w2, const void* b2, void* output,
    int batch, int seq, int embed, int ffn_dim, void* stream
) {
    dim3 block(256);
    dim3 grid(1, seq, batch);
    size_t smem = ffn_dim * sizeof(float);
    hipLaunchKernelGGL(vision_ffn_kernel,
        grid, block, smem, (hipStream_t)stream,
        (const __half*)input, (const __half*)w1, (const __half*)b1,
        (const __half*)w2, (const __half*)b2, (__half*)output,
        batch, seq, embed, ffn_dim);
}

void launch_vision_layernorm(
    const void* input, const void* weight, const void* bias, void* output,
    int batch, int seq, int embed, float eps, void* stream
) {
    dim3 grid(1, seq, batch);
    dim3 block(256);
    size_t smem = 256 * sizeof(float);
    hipLaunchKernelGGL(vision_layernorm_kernel,
        grid, block, smem, (hipStream_t)stream,
        (const __half*)input, (const __half*)weight, (const __half*)bias,
        (__half*)output, batch, seq, embed, eps);
}

} // extern "C"
