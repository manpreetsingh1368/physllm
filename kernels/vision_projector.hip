/**
 * vision_projector.hip — Vision-Language Projector
 *
 * Bridges the gap between the Vision Encoder (ViT) output space
 * and the LLM input space. This is the critical module that lets
 * the LLM "understand" image features.
 *
 * Two common architectures:
 *
 * 1. LLaVA-style MLP Projector:
 *    vision_features [batch, num_patches, vision_dim]
 *        → Linear(vision_dim, llm_dim)
 *        → GELU
 *        → Linear(llm_dim, llm_dim)
 *    = image_tokens [batch, num_patches, llm_dim]
 *
 * 2. Q-Former (BLIP-2 style) — more powerful but complex:
 *    Learned query vectors attend to vision features → compact tokens
 *
 * We implement the LLaVA MLP projector (simpler, still state-of-the-art).
 *
 * The output image_tokens are then PREPENDED to the text tokens:
 *   LLM input = [image_tokens] + [text_tokens]
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// GELU activation 
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

//  MLP Projector Layer 1: vision_dim → llm_dim with GELU 
__global__ void projector_layer1_kernel(
    const __half* input,   // [batch, num_tokens, vision_dim]
    const __half* weight,  // [llm_dim, vision_dim]
    const __half* bias,    // [llm_dim]
    __half* output,        // [batch, num_tokens, llm_dim]
    int batch_size,
    int num_tokens,
    int vision_dim,
    int llm_dim,
    bool apply_gelu
) {
    int b   = blockIdx.z;
    int t   = blockIdx.y;
    int out = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || t >= num_tokens || out >= llm_dim) return;

    float sum = 0.0f;
    int in_off  = b * num_tokens * vision_dim + t * vision_dim;
    int w_off   = out * vision_dim;

    for (int i = 0; i < vision_dim; i++) {
        sum += __half2float(input[in_off + i]) * __half2float(weight[w_off + i]);
    }
    if (bias) sum += __half2float(bias[out]);
    if (apply_gelu) sum = gelu_approx(sum);

    output[b * num_tokens * llm_dim + t * llm_dim + out] = __float2half(sum);
}

//  Concatenate Image + Text Tokens 
// Places image tokens BEFORE text tokens in the sequence
// Final: [CLS_omitted][img_token_0..N][text_token_0..M]
__global__ void concat_image_text_kernel(
    const __half* image_tokens,  // [batch, num_img_tokens, embed_dim]
    const __half* text_tokens,   // [batch, num_txt_tokens, embed_dim]
    __half* output,              // [batch, num_img_tokens + num_txt_tokens, embed_dim]
    int batch_size,
    int num_img,
    int num_txt,
    int embed_dim
) {
    int b   = blockIdx.z;
    int t   = blockIdx.y;   // combined token index
    int e   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_img + num_txt;

    if (b >= batch_size || t >= total || e >= embed_dim) return;

    int out_off = b * total * embed_dim + t * embed_dim + e;

    if (t < num_img) {
        // Copy image token
        output[out_off] = image_tokens[b * num_img * embed_dim + t * embed_dim + e];
    } else {
        // Copy text token
        int txt_t = t - num_img;
        output[out_off] = text_tokens[b * num_txt * embed_dim + txt_t * embed_dim + e];
    }
}

//  Special Image Token Marker 
// Marks image token positions in the attention mask
// (image tokens attend to each other AND text, text attends to all)
__global__ void create_vision_attention_mask_kernel(
    int* mask,          // [total_seq_len] — 1 = image token, 0 = text token
    int num_img,
    int num_txt
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_img + num_txt;
    if (t >= total) return;
    mask[t] = (t < num_img) ? 1 : 0;
}

//  Vision Feature Pooling 
// For compact representation: pool all patch tokens into fewer tokens
// Method: average pooling over spatial groups of patches
__global__ void spatial_pool_kernel(
    const __half* patch_tokens, // [batch, num_patches, embed_dim] (no CLS)
    __half* pooled, // [batch, pool_size, embed_dim]
    int batch_size,
    int num_patches,
    int pool_size,  // Target number of visual tokens
    int embed_dim
) {
    int b = blockIdx.z;
    int p = blockIdx.y;   // pooled patch index
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || p >= pool_size || e >= embed_dim) return;

    // How many patches map to each pooled token
    int patches_per_pool = (num_patches + pool_size - 1) / pool_size;
    int start = p * patches_per_pool;
    int end   = min(start + patches_per_pool, num_patches);

    float sum = 0.0f;
    int count = 0;
    for (int i = start; i < end; i++) {
        sum += __half2float(patch_tokens[b * num_patches * embed_dim + i * embed_dim + e]);
        count++;
    }
    pooled[b * pool_size * embed_dim + p * embed_dim + e] = __float2half(sum / count);
}

// C Exports 

extern "C" {

// Main projector forward (call twice: once with GELU, once without for 2-layer MLP)
void launch_projector_layer(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    int batch_size,
    int num_tokens,
    int in_dim,
    int out_dim,
    int apply_gelu,
    void* stream
) {
    dim3 block(64);
    dim3 grid((out_dim + 63) / 64, num_tokens, batch_size);
    hipLaunchKernelGGL(projector_layer1_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const __half*)input, (const __half*)weight, (const __half*)bias,
        (__half*)output,
        batch_size, num_tokens, in_dim, out_dim, (bool)apply_gelu);
}

void launch_concat_image_text(
    const void* image_tokens,
    const void* text_tokens,
    void* output,
    int batch_size,
    int num_img,
    int num_txt,
    int embed_dim,
    void* stream
) {
    dim3 block(64);
    dim3 grid((embed_dim + 63) / 64, num_img + num_txt, batch_size);
    hipLaunchKernelGGL(concat_image_text_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const __half*)image_tokens, (const __half*)text_tokens,
        (__half*)output, batch_size, num_img, num_txt, embed_dim);
}

void launch_spatial_pool(
    const void* patch_tokens,
    void* pooled,
    int batch_size,
    int num_patches,
    int pool_size,
    int embed_dim,
    void* stream
) {
    dim3 block(64);
    dim3 grid((embed_dim + 63) / 64, pool_size, batch_size);
    hipLaunchKernelGGL(spatial_pool_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const __half*)patch_tokens, (__half*)pooled,
        batch_size, num_patches, pool_size, embed_dim);
}

} // extern "C"