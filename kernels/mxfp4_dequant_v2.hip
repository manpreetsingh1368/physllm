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
    const unsigned char* __restrict__ blocks,
    const
)