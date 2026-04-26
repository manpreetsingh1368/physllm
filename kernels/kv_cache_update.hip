#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void kv_cache_update_kernel(
    __half* cache, const __half* new_kv,
    int position, int num_heads, int head_dim
) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;
    int cache_off = (h * 4096 + position) * head_dim + d;  // 4096 = max seq
    int new_off = h * head_dim + d;
    cache[cache_off] = new_kv[new_off];
}

extern "C" void launch_kv_cache_update(
    void* cache, const void* new_kv,
    int position, int num_heads, int head_dim, void* stream
) {
    hipLaunchKernelGGL(kv_cache_update_kernel,
        dim3(num_heads), dim3(head_dim), 0, (hipStream_t)stream,
        (__half*)cache, (const __half*)new_kv,
        position, num_heads, head_dim);
}
