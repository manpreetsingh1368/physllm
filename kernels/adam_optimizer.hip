#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void adam_update_kernel(
    __half* param, const __half* grad, __half* m, __half* v,
    float lr, float beta1, float beta2, float epsilon,
    float weight_decay, int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float g = __half2float(grad[idx]);
    float p = __half2float(param[idx]);
    float m_val = __half2float(m[idx]);
    float v_val = __half2float(v[idx]);

    m_val = beta1 * m_val + (1.0f - beta1) * g;
    v_val = beta2 * v_val + (1.0f - beta2) * g * g;

    float update = m_val / (sqrtf(v_val) + epsilon);
    update += weight_decay * p;
    p -= lr * update;

    param[idx] = __float2half(p);
    m[idx] = __float2half(m_val);
    v[idx] = __float2half(v_val);
}

extern "C" void launch_adam_update(
    void* param, const void* grad, void* m, void* v,
    float lr, float beta1, float beta2, float epsilon,
    float weight_decay, int numel, void* stream
) {
    hipLaunchKernelGGL(adam_update_kernel,
        dim3((numel + 255) / 256), dim3(256), 0, (hipStream_t)stream,
        (__half*)param, (const __half*)grad, (__half*)m, (__half*)v,
        lr, beta1, beta2, epsilon, weight_decay, numel);
}
