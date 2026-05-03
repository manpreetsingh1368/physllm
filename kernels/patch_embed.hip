/**
 * patch_embed.hip — Image Patch Embedding Kernel
 
 * Converts an image into a sequence of patch embeddings.
 
 * ViT-style patch extraction:
 *  Input:  [batch, channels, height, width] (e.g. [1, 3, 224, 224])
 *  Patches: H/P × W/P patches of size P×P  (e.g. 14×14 = 196 patches for P=16)
 *  Output: [batch, num_patches, embed_dim]  (e.g. [1, 196, 768])
 
 * Each patch is flattened (P*P*C values) and multiplied by a projection matrix.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

//  Patch Extraction Kernel 
// Extracts and flattens image patches from input image
__global__ void extract_patches_kernel(
    const float* image,     // [batch, C, H, W] — float32 normalized input
    __half* patches,        // [batch, num_patches, patch_size] — output patches
    int batch_size,
    int channels,           // 3 for RGB
    int height,
    int width,
    int patch_size_h,       // P (e.g. 16)
    int patch_size_w,       // P (e.g. 16)
    int num_patches_h,      // H/P
    int num_patches_w       // W/P
) {
    int b = blockIdx.z;
    int ph = blockIdx.y;  // patch row index
    int pw = blockIdx.x;  // patch col index
    int tid = threadIdx.x;

    if (b >= batch_size || ph >= num_patches_h || pw >= num_patches_w) return;

    int patch_flat_size = channels * patch_size_h * patch_size_w;
    int patch_idx = ph * num_patches_w + pw;

    // Each thread handles one element in the flattened patch
    for (int i = tid; i < patch_flat_size; i += blockDim.x) {
        int c   = i / (patch_size_h * patch_size_w);
        int rem = i % (patch_size_h * patch_size_w);
        int pi  = rem / patch_size_w;
        int pj  = rem % patch_size_w;

        int img_h = ph * patch_size_h + pi;
        int img_w = pw * patch_size_w + pj;

        // Read from [batch, C, H, W]
        float val = image[b * channels * height * width
                        + c * height * width
                        + img_h * width
                        + img_w];

        // Write to [batch, num_patches, patch_flat_size]
        int num_patches = num_patches_h * num_patches_w;
        patches[b * num_patches * patch_flat_size
              + patch_idx * patch_flat_size
              + i] = __float2half(val);
    }
}

//  Patch Projection Kernel 
// Projects each flattened patch to embedding dim via linear transform
__global__ void project_patches_kernel(
    const __half* patches,      // [batch, num_patches, patch_flat_size]
    const __half* proj_weight,  // [embed_dim, patch_flat_size]
    const __half* proj_bias,    // [embed_dim]
    __half* output,             // [batch, num_patches, embed_dim]
    int batch_size,
    int num_patches,
    int patch_flat_size,
    int embed_dim
) {
    int b    = blockIdx.z;
    int p    = blockIdx.y;   // patch index
    int e    = blockIdx.x * blockDim.x + threadIdx.x;  // embed dim index

    if (b >= batch_size || p >= num_patches || e >= embed_dim) return;

    // Dot product: patch[p] · proj_weight[e]
    float sum = 0.0f;
    int patch_off = b * num_patches * patch_flat_size + p * patch_flat_size;
    int weight_off = e * patch_flat_size;

    for (int i = 0; i < patch_flat_size; i++) {
        sum += __half2float(patches[patch_off + i])
             * __half2float(proj_weight[weight_off + i]);
    }

    if (proj_bias) sum += __half2float(proj_bias[e]);

    output[b * num_patches * embed_dim + p * embed_dim + e] = __float2half(sum);
}

//  Normalize Image Kernel 
// ImageNet normalization: (x - mean) / std
// mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
__global__ void normalize_image_kernel(
    const unsigned char* raw_pixels,  // [B, H, W, 3] uint8 RGB
    float* normalized,                // [B, 3, H, W] float32
    int batch_size,
    int height,
    int width
) {
    // ImageNet stats
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_v[3] = {0.229f, 0.224f, 0.225f};

    int b = blockIdx.z;
    int h = blockIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || h >= height || w >= width) return;

    int hw = height * width;
    int src_off = (b * height * width + h * width + w) * 3;

    for (int c = 0; c < 3; c++) {
        float val = (float)raw_pixels[src_off + c] / 255.0f;
        float normed = (val - mean[c]) / std_v[c];
        normalized[b * 3 * hw + c * hw + h * width + w] = normed;
    }
}

//  Add CLS Token Kernel
// Prepend learnable [CLS] token to patch sequence
__global__ void add_cls_token_kernel(
    const __half* patches,    // [batch, num_patches, embed_dim]
    const __half* cls_token,  // [embed_dim]
    __half* output,           // [batch, num_patches+1, embed_dim]
    int batch_size,
    int num_patches,
    int embed_dim
) {
    int b = blockIdx.z;
    int p = blockIdx.y;  // 0 = CLS, 1..num_patches+1 = patches
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || p > num_patches || e >= embed_dim) return;

    int out_off = b * (num_patches + 1) * embed_dim + p * embed_dim + e;

    if (p == 0) {
        // CLS token
        output[out_off] = cls_token[e];
    } else {
        // Patch embeddings
        int in_off = b * num_patches * embed_dim + (p - 1) * embed_dim + e;
        output[out_off] = patches[in_off];
    }
}

// Add Position Embeddings 
__global__ void add_position_embed_kernel(
    __half* tokens,           // [batch, seq_len, embed_dim] (in-place)
    const __half* pos_embed,  // [seq_len, embed_dim]
    int batch_size,
    int seq_len,
    int embed_dim
) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || s >= seq_len || e >= embed_dim) return;

    int tok_off = b * seq_len * embed_dim + s * embed_dim + e;
    tokens[tok_off] = __float2half(
        __half2float(tokens[tok_off]) + __half2float(pos_embed[s * embed_dim + e])
    );
}

//  C Exports 

extern "C" {

void launch_normalize_image(
    const void* raw_pixels,
    void* normalized,
    int batch_size,
    int height,
    int width,
    void* stream
) {
    dim3 block(32);
    dim3 grid((width + 31) / 32, height, batch_size);
    hipLaunchKernelGGL(normalize_image_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const unsigned char*)raw_pixels, (float*)normalized,
        batch_size, height, width);
}

void launch_extract_patches(
    const void* image,
    void* patches,
    int batch_size,
    int channels,
    int height,
    int width,
    int patch_size_h,
    int patch_size_w,
    int num_patches_h,
    int num_patches_w,
    void* stream
) {
    dim3 block(256);
    dim3 grid(num_patches_w, num_patches_h, batch_size);
    hipLaunchKernelGGL(extract_patches_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const float*)image, (__half*)patches,
        batch_size, channels, height, width,
        patch_size_h, patch_size_w,
        num_patches_h, num_patches_w);
}

void launch_project_patches(
    const void* patches,
    const void* proj_weight,
    const void* proj_bias,
    void* output,
    int batch_size,
    int num_patches,
    int patch_flat_size,
    int embed_dim,
    void* stream
) {
    dim3 block(64);
    dim3 grid((embed_dim + 63) / 64, num_patches, batch_size);
    hipLaunchKernelGGL(project_patches_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const __half*)patches, (const __half*)proj_weight,
        (const __half*)proj_bias, (__half*)output,
        batch_size, num_patches, patch_flat_size, embed_dim);
}

void launch_add_cls_token(
    const void* patches,
    const void* cls_token,
    void* output,
    int batch_size,
    int num_patches,
    int embed_dim,
    void* stream
) {
    dim3 block(64);
    dim3 grid((embed_dim + 63) / 64, num_patches + 1, batch_size);
    hipLaunchKernelGGL(add_cls_token_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const __half*)patches, (const __half*)cls_token, (__half*)output,
        batch_size, num_patches, embed_dim);
}

void launch_add_position_embed(
    void* tokens,
    const void* pos_embed,
    int batch_size,
    int seq_len,
    int embed_dim,
    void* stream
) {
    dim3 block(64);
    dim3 grid((embed_dim + 63) / 64, seq_len, batch_size);
    hipLaunchKernelGGL(add_position_embed_kernel,
        grid, block, 0, (hipStream_t)stream,
        (__half*)tokens, (const __half*)pos_embed,
        batch_size, seq_len, embed_dim);
}

} // extern "C"
