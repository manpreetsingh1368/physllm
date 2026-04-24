
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>
// Simplified flash-attention — production version would tile for shared memory.
// This is a direct attention implementation: O(N²) in memory but correct.
__global__ void flash_attn_k(const __half* Q, const __half* K, const __half* V,
                              __half* O, int seq, int d, float scale, int causal) {
    int row = blockIdx.x;
    if (row >= seq) return;
    extern __shared__ float s_scores[];
    float max_s = -FLT_MAX;
    for (int c = threadIdx.x; c < seq; c += blockDim.x) {
        if (causal && c > row) { s_scores[c] = -FLT_MAX; continue; }
        float dot = 0.f;
        for (int i = 0; i < d; i++) dot += __half2float(Q[row*d+i]) * __half2float(K[c*d+i]);
        dot *= scale;
        s_scores[c] = dot;
        max_s = fmaxf(max_s, dot);
    }
    __syncthreads();
    float sum = 0.f;
    for (int c = threadIdx.x; c < seq; c += blockDim.x) {
        s_scores[c] = expf(s_scores[c] - max_s);
        sum += s_scores[c];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.f;
        for (int c = 0; c < seq; c++) acc += s_scores[c] / sum * __half2float(V[c*d+i]);
        O[row*d+i] = __float2half(acc);
    }
}
extern "C" void flash_attention(hipStream_t s, const __half* Q, const __half* K,
                                 const __half* V, __half* O,
                                 int seq, int d, float scale, int causal) {
    size_t shm = seq * sizeof(float);
    hipLaunchKernelGGL(flash_attn_k, dim3(seq), dim3(128), shm, s,
                       Q, K, V, O, seq, d, scale, causal);
}
