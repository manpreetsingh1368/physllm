
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
// GEMM is handled by hipblasHgemm from Rust; this stub satisfies the build list.
extern "C" void gemm_f16_noop(void) {}
