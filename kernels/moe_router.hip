#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MoE Router: compute gate scores and select top-k experts
// gate_logits = hidden @ router_weight^T → [batch, num_experts]
// Then select top-k experts with highest scores
__global__ void moe_router_kernel(
    const __half* hidden,           // [batch_size, hidden_dim]
    const __half* router_weight,    // [num_experts, hidden_dim]
    int* expert_indices,            // [batch_size, top_k] - output
    float* expert_weights,          // [batch_size, top_k] - output (softmax weights)
    int batch_size,
    int hidden_dim,
    int num_experts,
    int top_k
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float gate_scores[];  // [num_experts]
    
    int tid = threadIdx.x;
    
    // Phase 1: Compute gate logits for each expert
    if (tid < num_experts) {
        float score = 0.0f;
        int h_off = batch_idx * hidden_dim;
        int w_off = tid * hidden_dim;
        
        for (int d = 0; d < hidden_dim; d++) {
            score += __half2float(hidden[h_off + d]) * 
                     __half2float(router_weight[w_off + d]);
        }
        gate_scores[tid] = score;
    }
    __syncthreads();
    
    // Phase 2: Find top-k experts (single thread for simplicity)
    if (tid == 0) {
        // Simple selection sort for top-k
        int selected[16];  // Max top_k = 16
        float selected_scores[16];
        bool used[256] = {false};  // Max num_experts = 256
        
        for (int k = 0; k < top_k; k++) {
            float best_score = -1e38f;
            int best_idx = 0;
            
            for (int e = 0; e < num_experts; e++) {
                if (!used[e] && gate_scores[e] > best_score) {
                    best_score = gate_scores[e];
                    best_idx = e;
                }
            }
            
            selected[k] = best_idx;
            selected_scores[k] = best_score;
            used[best_idx] = true;
        }
        
        // Phase 3: Softmax over selected expert scores
        float max_score = -1e38f;
        for (int k = 0; k < top_k; k++) {
            max_score = fmaxf(max_score, selected_scores[k]);
        }
        
        float sum_exp = 0.0f;
        for (int k = 0; k < top_k; k++) {
            selected_scores[k] = expf(selected_scores[k] - max_score);
            sum_exp += selected_scores[k];
        }
        
        // Write outputs
        int out_off = batch_idx * top_k;
        for (int k = 0; k < top_k; k++) {
            expert_indices[out_off + k] = selected[k];
            expert_weights[out_off + k] = selected_scores[k] / sum_exp;
        }
    }
}

extern "C" void launch_moe_router(
    const void* hidden,
    const void* router_weight,
    void* expert_indices,
    void* expert_weights,
    int batch_size,
    int hidden_dim,
    int num_experts,
    int top_k,
    void* stream
) {
    dim3 grid(batch_size);
    dim3 block(num_experts < 256 ? num_experts : 256);
    size_t smem = num_experts * sizeof(float);
    
    hipLaunchKernelGGL(moe_router_kernel,
        grid, block, smem, (hipStream_t)stream,
        (const __half*)hidden,
        (const __half*)router_weight,
        (int*)expert_indices,
        (float*)expert_weights,
        batch_size, hidden_dim, num_experts, top_k);
}
