#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void embedding_lookup_kernel(
    const __half* embed_table, const int* token_ids,
    __half* output, int hidden_dim, int num_tokens
) {
    int tok_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    if (tok_idx >= num_tokens || dim_idx >= hidden_dim) return;

    int token_id = token_ids[tok_idx];
    output[tok_idx * hidden_dim + dim_idx] =
        embed_table[token_id * hidden_dim + dim_idx];
}

extern "C" void launch_embedding_lookup(
    const void* embed_table, const void* token_ids,
    void* output, int hidden_dim, int num_tokens, void* stream
) {
    dim3 grid(num_tokens);
    dim3 block(hidden_dim < 1024 ? hidden_dim : 1024);
    hipLaunchKernelGGL(embedding_lookup_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const __half*)embed_table, (const int*)token_ids,
        (__half*)output, hidden_dim, num_tokens);
}
