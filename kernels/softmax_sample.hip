#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void softmax_sample_kernel(
    float* logits, int* output_token,
    int vocab_size, float temperature, unsigned long long seed
) {
    if (threadIdx.x != 0) return;

    // Apply temperature
    for (int i = 0; i < vocab_size; i++)
        logits[i] /= temperature;

    // Find max
    float max_val = -1e38f;
    for (int i = 0; i < vocab_size; i++)
        max_val = fmaxf(max_val, logits[i]);

    // Exp and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum_exp += logits[i];
    }

    // Simple LCG random
    unsigned long long state = seed;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float r = (float)(state >> 33) / (float)(1ULL << 31) * sum_exp;

    // Cumulative sampling
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (r < cumsum) { *output_token = i; return; }
    }
    *output_token = vocab_size - 1;
}

extern "C" void launch_softmax_sample(
    void* logits, void* output_token,
    int vocab_size, float temperature, unsigned long long seed, void* stream
) {
    hipLaunchKernelGGL(softmax_sample_kernel,
        dim3(1), dim3(1), 0, (hipStream_t)stream,
        (float*)logits, (int*)output_token,
        vocab_size, temperature, seed);
}
