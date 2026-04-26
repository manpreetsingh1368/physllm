// kernels/kv_cache.hip
//
// KV cache management kernels for autoregressive decoding.
//
// The KV cache stores all past key and value tensors so each decode step
// only needs to compute K,V for the single new token rather than re-computing
// the entire sequence. This reduces decode from O(n²) to O(n) per step.
//
// Layout: cache[batch, heads, max_seq, head_dim]   (f16)
//
// Kernels:
//   kv_cache_append — copy new K or V tokens into the cache at position offset
//   kv_cache_gather — (optional) gather a contiguous slice for attention

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Append kernel 
//
// Copies new_kv [batch, heads, new_tokens, head_dim] into
// cache         [batch, heads, max_seq,    head_dim]
// at position   cache[b, h, offset+tok, d] = new_kv[b, h, tok, d]

__global__ void kv_cache_append_kernel(
          __half* __restrict__ cache,
    const __half* __restrict__ new_kv,
    int batch,
    int heads,
    int max_seq,
    int head_dim,
    int offset,       // token position to start writing at
    int new_tokens    // number of new tokens to write (usually 1 during decode)
) {
    const long idx   = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)batch * heads * new_tokens * head_dim;
    if (idx >= total) return;

    // Decode flat index → (b, h, tok, d)
    const int d   = idx % head_dim;
    const int tok = (idx / head_dim) % new_tokens;
    const int h   = (idx / head_dim / new_tokens) % heads;
    const int b   =  idx / head_dim / new_tokens / heads;

    // Write position in cache
    const long cache_idx =
        ((long)(b * heads + h) * max_seq + (offset + tok)) * head_dim + d;

    cache[cache_idx] = new_kv[idx];
}

//  Gather kernel (copy a window of the cache for attention) 
//
// out   [batch, heads, seq_len, head_dim]  ← cache[b, h, start..start+seq_len, d]

__global__ void kv_cache_gather_kernel(
          __half* __restrict__ out,
    const __half* __restrict__ cache,
    int batch,
    int heads,
    int max_seq,
    int head_dim,
    int start,
    int seq_len
) {
    const long idx   = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)batch * heads * seq_len * head_dim;
    if (idx >= total) return;

    const int d   = idx % head_dim;
    const int tok = (idx / head_dim) % seq_len;
    const int h   = (idx / head_dim / seq_len) % heads;
    const int b   =  idx / head_dim / seq_len / heads;

    const long src_idx =
        ((long)(b * heads + h) * max_seq + (start + tok)) * head_dim + d;

    out[idx] = cache[src_idx];
}

//  C-linkage entry points called from Rust 
extern "C" void kv_cache_append(
    hipStream_t  stream,
    __half*      cache,
    const __half* new_kv,
    int batch, int heads, int max_seq, int head_dim,
    int offset, int new_tokens
) {
    const long total   = (long)batch * heads * new_tokens * head_dim;
    const int  threads = 256;
    const long blocks  = (total + threads - 1) / threads;

    hipLaunchKernelGGL(
        kv_cache_append_kernel,
        dim3((unsigned int)blocks), dim3(threads), 0, stream,
        cache, new_kv,
        batch, heads, max_seq, head_dim,
        offset, new_tokens
    );
}

extern "C" void kv_cache_gather(
    hipStream_t   stream,
    __half*       out,
    const __half* cache,
    int batch, int heads, int max_seq, int head_dim,
    int start, int seq_len
) {
    const long total   = (long)batch * heads * seq_len * head_dim;
    const int  threads = 256;
    const long blocks  = (total + threads - 1) / threads;

    hipLaunchKernelGGL(
        kv_cache_gather_kernel,
        dim3((unsigned int)blocks), dim3(threads), 0, stream,
        out, cache,
        batch, heads, max_seq, head_dim,
        start, seq_len
    );
}
