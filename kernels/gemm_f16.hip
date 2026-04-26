// kernels/gemm_f16.hip
//
// f16 General Matrix Multiply.
//
// The actual GEMM computation is performed by hipblasHgemm, called directly
// from Rust in crates/rocm-backend/src/ops.rs. This file exists only to
// satisfy the build system's kernel list — it compiles cleanly to a no-op.
//
// If you want a custom tiled GEMM kernel (e.g. for non-standard strides),
// implement it here and declare it extern "C" so ops.rs can call it.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Sentinel symbol so the linker sees something from this translation unit.
extern "C" void gemm_f16_noop(void) {}
