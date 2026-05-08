#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __constant__ float MXFP4_LUT[16] = { // 4 bits -> float : MXFP4 value look up
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// Decode FP8 E4M3 scale to float32
__device__ __forceinline__ float decode_fp8_e4m3(unsigned char byte) {
    if (byte == 0) return 0.0f;
    float sign = (byte & 0x80) ? -1.0f : 1.0f;
    int exp = ((byte >> 3) & 0x0F) - 7;
    float mantissa = 1.0f + (float)(byte & 0x07) / 8.0f;
    return sign * mantissa * __powf(2.0f, (float)exp);
}

__global__ void mxfp4_dequant_kernel(
    const unsigned char* __restrict__ blocks, // packed 4-bit values
    const unsigned char* __restrict__ scales,  // FP8 E4M3 per-block scales
    __half* __restrict__ output,   // dequantized f16 weights
    int total_elements,
    int block_size     // 32
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int block_idx = idx / block_size;
    float scale = decode_fp8_e4m3(scales[block_idx]);

    // Extract 4-bit nibble
    int byte_idx = idx / 2;
    unsigned char byte = blocks[byte_idx];
    int nibble = (idx % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);

    float value = MXFP4_LUT[nibble] * scale;
    output[idx] = __float2half(value);
}


__global__ void mxfp4_dequant_transpose_kernel(
    const unsigned char* __restrict__ blocks,
    const unsigned char* __restrict__ scales,
    __half* __restrict__ output,   // [num_experts, in_dim, out_dim] (transposed)
    int num_experts,
    int out_dim,
    int in_dim,
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_experts * out_dim * in_dim;
    if (idx >= total) return;

    // Original position: [expert, out, in]
    int expert = idx / (out_dim * in_dim);
    int rem    = idx % (out_dim * in_dim);
    int out_i  = rem / in_dim;
    int in_i   = rem % in_dim;

    // Dequantize
    int block_idx = idx / block_size;
    float scale = decode_fp8_e4m3(scales[block_idx]);
    int byte_idx = idx / 2;
    unsigned char byte = blocks[byte_idx];
    int nibble = (idx % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    float value = MXFP4_LUT[nibble] * scale;

    // Write transposed: [expert, in, out]
    int t_idx = expert * in_dim * out_dim + in_i * out_dim + out_i;
    output[t_idx] = __float2half(value);
}