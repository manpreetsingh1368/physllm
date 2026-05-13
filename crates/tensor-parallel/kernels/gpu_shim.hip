// kernels/gpu_shim.hip
//
// Single C shim exposing RCCL collectives and hipBLAS GEMMs to Rust.
// All operations work on device pointers — no host↔device copies here.
// Called from src/comm/ffi.rs via extern "C".

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <hipblas/hipblas.h>
#define BLAS_HANDLE  hipblasHandle_t
#define gpuStream_t  hipStream_t
#define gpuError_t   hipError_t
#define gpuSuccess   hipSuccess
#define gpuSetDevice hipSetDevice
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuGetErrorString hipGetErrorString
#define CollComm_t   ncclComm_t
#define CollResult_t ncclResult_t
#define CollSuccess  ncclSuccess
#define collGetError ncclGetErrorString
#define collAllReduce ncclAllReduce
#define collAllGather ncclAllGather
#define collBcast     ncclBcast
#define collCommInitAll ncclCommInitAll
#define collCommDestroy ncclCommDestroy
#define DTYPE_FP16   ncclFloat16
#define DTYPE_FP32   ncclFloat32
#define blasCreate   hipblasCreate
#define blasDestroy  hipblasDestroy
#define blasSetStream hipblasSetStream
#define blasHgemm    hipblasHgemm
#define blasSgemm    hipblasSgemm
#define BLAS_OP_N    HIPBLAS_OP_N
#define BLAS_OP_T    HIPBLAS_OP_T
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <nccl.h>
#include <cublas_v2.h>
#define BLAS_HANDLE  cublasHandle_t
#define gpuStream_t  cudaStream_t
#define gpuError_t   cudaError_t
#define gpuSuccess   cudaSuccess
#define gpuSetDevice cudaSetDevice
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuGetErrorString cudaGetErrorString
#define CollComm_t   ncclComm_t
#define CollResult_t ncclResult_t
#define CollSuccess  ncclSuccess
#define collGetError ncclGetErrorString
#define collAllReduce ncclAllReduce
#define collAllGather ncclAllGather
#define collBcast     ncclBcast
#define collCommInitAll ncclCommInitAll
#define collCommDestroy ncclCommDestroy
#define DTYPE_FP16   ncclFloat16
#define DTYPE_FP32   ncclFloat32
#define blasCreate   cublasCreate
#define blasDestroy  cublasDestroy
#define blasSetStream cublasSetStream
#define blasHgemm    cublasHgemm
#define blasSgemm    cublasSgemm
#define BLAS_OP_N    CUBLAS_OP_N
#define BLAS_OP_T    CUBLAS_OP_T
#endif

#include <stdio.h>
#include <stdlib.h>

//  Error macros 
#define GPU_CHECK(call) \
    do { gpuError_t _e = (call); \
         if (_e != gpuSuccess) { \
             fprintf(stderr, "[tp] GPU error %d at %s:%d — %s\n", \
                     _e, __FILE__, __LINE__, gpuGetErrorString(_e)); \
             return -1; } } while(0)

#define COLL_CHECK(call) \
    do { CollResult_t _r = (call); \
         if (_r != CollSuccess) { \
             fprintf(stderr, "[tp] RCCL/NCCL error %d at %s:%d — %s\n", \
                     _r, __FILE__, __LINE__, collGetError(_r)); \
             return -1; } } while(0)

// Comm handle 

typedef struct {
    CollComm_t  coll_comm;
    BLAS_HANDLE blas;
    gpuStream_t stream;
    int         rank;
    int         world_size;
    int         device_id;
} TpHandle;

extern "C" {

//  Lifecycle 

/// Initialise all GPU communicators.
/// Returns a malloc'd array of `world_size` TpHandle pointers.
/// Caller must free the outer array with free(); each handle with tp_destroy().
TpHandle** tp_init(int world_size, int* devices) {
    CollComm_t* comms = (CollComm_t*)malloc(world_size * sizeof(CollComm_t));
    CollResult_t r = collCommInitAll(comms, world_size, devices);
    if (r != CollSuccess) {
        fprintf(stderr, "[tp] collCommInitAll failed: %s\n", collGetError(r));
        free(comms); return NULL;
    }

    TpHandle** handles = (TpHandle**)malloc(world_size * sizeof(TpHandle*));
    for (int i = 0; i < world_size; i++) {
        TpHandle* h = (TpHandle*)malloc(sizeof(TpHandle));
        h->coll_comm  = comms[i];
        h->rank        = i;
        h->world_size  = world_size;
        h->device_id   = devices[i];

        gpuSetDevice(devices[i]);
        gpuStreamCreate(&h->stream);
        blasCreate(&h->blas);
        blasSetStream(h->blas, h->stream);

        handles[i] = h;
    }
    free(comms);
    return handles;
}

/// Destroy one handle and release all GPU resources.
int tp_destroy(TpHandle* h) {
    if (!h) return 0;
    COLL_CHECK(collCommDestroy(h->coll_comm));
    blasDestroy(h->blas);
    GPU_CHECK(gpuStreamDestroy(h->stream));
    free(h);
    return 0;
}

//  Collectives 

/// In-place AllReduce (sum) fp16 on device pointer.
int tp_allreduce_fp16(TpHandle* h, void* buf, size_t count) {
    COLL_CHECK(collAllReduce(buf, buf, count, DTYPE_FP16, ncclSum,
                             h->coll_comm, h->stream));
    GPU_CHECK(gpuStreamSynchronize(h->stream));
    return 0;
}

/// In-place AllReduce (sum) fp32.
int tp_allreduce_fp32(TpHandle* h, void* buf, size_t count) {
    COLL_CHECK(collAllReduce(buf, buf, count, DTYPE_FP32, ncclSum,
                             h->coll_comm, h->stream));
    GPU_CHECK(gpuStreamSynchronize(h->stream));
    return 0;
}

/// Non-blocking AllReduce fp16 — caller must call tp_sync() before using result.
int tp_allreduce_fp16_async(TpHandle* h, void* buf, size_t count) {
    COLL_CHECK(collAllReduce(buf, buf, count, DTYPE_FP16, ncclSum,
                             h->coll_comm, h->stream));
    return 0;
}

/// AllGather fp16: each rank contributes send_count elements.
/// recv_buf must be pre-allocated: world_size * send_count * sizeof(fp16).
int tp_allgather_fp16(TpHandle* h, void* send_buf, void* recv_buf, size_t send_count) {
    COLL_CHECK(collAllGather(send_buf, recv_buf, send_count, DTYPE_FP16,
                             h->coll_comm, h->stream));
    GPU_CHECK(gpuStreamSynchronize(h->stream));
    return 0;
}

/// Broadcast fp16 from rank 0.
int tp_broadcast_fp16(TpHandle* h, void* buf, size_t count) {
    COLL_CHECK(collBcast(buf, count, DTYPE_FP16, 0, h->coll_comm, h->stream));
    GPU_CHECK(gpuStreamSynchronize(h->stream));
    return 0;
}

/// Synchronise this rank's RCCL/CUDA stream.
int tp_sync(TpHandle* h) {
    GPU_CHECK(gpuStreamSynchronize(h->stream));
    return 0;
}

//  hipBLAS / cuBLAS GEMMs 

// All matrices are in row-major layout (C order).
// hipBLAS is column-major, so we compute C.T = B.T @ A.T which gives
// the row-major result C = A @ B without any transposition overhead.

// For fp16: uses tensor cores automatically on MI300X / A100+.
/// C = alpha * A @ B + beta * C   (all fp16, all device pointers)
/// A: [m, k]   B: [k, n]   C: [m, n]
int tp_gemm_fp16(
    TpHandle*  h,
    int        m, int n, int k,
    const void* A,        // device ptr [m, k] fp16
    const void* B,        // device ptr [k, n] fp16
    void*       C,        // device ptr [m, n] fp16 (output)
    float       alpha,
    float       beta
) {
    // hipBLAS uses __half for alpha/beta in Hgemm
    __half ha = __float2half(alpha);
    __half hb = __float2half(beta);

    // Row-major A[m,k] × B[k,n] → C[m,n]
    // In col-major terms: B.T[n,k] × A.T[k,m] → C.T[n,m]
    blasHgemm(
        h->blas,
        BLAS_OP_N, BLAS_OP_N,
        n, m, k,      // (N, M, K) in col-major = (n, m, k) for our row-major
        &ha,
        (const __half*)B, n,   // B is "A" in col-major, leading dim = n
        (const __half*)A, k,   // A is "B" in col-major, leading dim = k
        &hb,
        (__half*)C, n
    );
    return 0;  // stream is async; call tp_sync() or chain with collectives
}

/// C = alpha * A @ B.T + beta * C   (A: [m,k], B: [n,k], C: [m,n])
/// Used for row-parallel output projection where B is stored transposed.
int tp_gemm_fp16_bt(
    TpHandle*  h,
    int        m, int n, int k,
    const void* A,
    const void* B,        // stored as [n, k], i.e. B transposed
    void*       C,
    float       alpha,
    float       beta
) {
    __half ha = __float2half(alpha);
    __half hb = __float2half(beta);

    // Row-major A[m,k] × B.T[k,n]
    // Col-major: B[n,k] × A.T[k,m]  with B as OP_T
    blasHgemm(
        h->blas,
        BLAS_OP_T, BLAS_OP_N,
        n, m, k,
        &ha,
        (const __half*)B, k,
        (const __half*)A, k,
        &hb,
        (__half*)C, n
    );
    return 0;
}

// Accessors used by Rust to read handle fields without pointer arithmetic
int tp_rank(TpHandle* h)       { return h->rank; }
int tp_world_size(TpHandle* h) { return h->world_size; }
int tp_device_id(TpHandle* h)  { return h->device_id; }

} // extern "C"
