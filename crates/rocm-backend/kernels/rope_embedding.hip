
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
__global__ void rope_k(__half* x, int seq, int heads, int head_dim, int offset, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * head_dim / 2;
    if (idx >= total) return;
    int d = idx % (head_dim/2);
    int h = (idx / (head_dim/2)) % heads;
    int s = idx / (head_dim/2) / heads;
    float freq = (float)(s + offset) / powf(theta, (float)(2 * d) / (float)head_dim);
    float cos_f = cosf(freq), sin_f = sinf(freq);
    int i0 = ((s * heads + h) * head_dim) + d * 2;
    int i1 = i0 + 1;
    float x0 = __half2float(x[i0]), x1 = __half2float(x[i1]);
    x[i0] = __float2half(x0 * cos_f - x1 * sin_f);
    x[i1] = __float2half(x0 * sin_f + x1 * cos_f);
}
extern "C" void rope_embed(hipStream_t s, __half* x,
                            int seq, int heads, int head_dim, int offset, float theta) {
    int total = seq * heads * head_dim / 2;
    hipLaunchKernelGGL(rope_k, dim3((total+255)/256), dim3(256), 0, s,
                       x, seq, heads, head_dim, offset, theta);
}
