#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MXFP4 format: microscaling FP4
// Each value is 4 bits (stored as packed uint8: 2 values per byte)
// Each block of 32 values shares a BF16 scale factor
// 
// Layout: tensor.blocks = packed uint8 (fp4 pairs)
//         tensor.scales = bf16 per-block scales

// FP4 E2M1 lookup table (4-bit to float)
__constant__ float fp4_e2m1_table[16] = {
     0.0f,    0.5f,    1.0f,    1.5f,
     2.0f,    3.0f,    4.0f,    6.0f,
    -0.0f,   -0.5f,   -1.0f,   -1.5f,
    -2.0f,   -3.0f,   -4.0f,   -6.0f
};

__global__ void mxfp4_dequant_kernel(
    const uint8_t* packed_data,   // Packed FP4 values (2 per byte)
    const __half* scales,         // BF16 scales (1 per 32 values)
    __half* output,               // Dequantized FP16 output
    int numel,                    // Total number of values
    int block_size                // Values per scale block (32)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx;  // Each thread handles one byte = 2 values
    int val_idx = pair_idx * 2;
    
    if (val_idx >= numel) return;
    
    uint8_t packed = packed_data[pair_idx];
    
    // Extract two 4-bit values
    int low_bits  = packed & 0x0F;
    int high_bits = (packed >> 4) & 0x0F;
    
    // Look up FP4 values
    float val_low  = fp4_e2m1_table[low_bits];
    float val_high = fp4_e2m1_table[high_bits];
    
    // Get block scale
    int block_idx = val_idx / block_size;
    float scale = __half2float(scales[block_idx]);
    
    // Dequantize
    output[val_idx] = __float2half(val_low * scale);
    if (val_idx + 1 < numel) {
        output[val_idx + 1] = __float2half(val_high * scale);
    }
}

extern "C" void launch_mxfp4_dequant(
    const void* packed_data,
    const void* scales,
    void* output,
    int numel,
    int block_size,
    void* stream
) {
    int num_pairs = (numel + 1) / 2;
    hipLaunchKernelGGL(mxfp4_dequant_kernel,
        dim3((num_pairs + 255) / 256), dim3(256), 0, (hipStream_t)stream,
        (const uint8_t*)packed_data,
        (const __half*)scales,
        (__half*)output,
        numel, block_size);
}
