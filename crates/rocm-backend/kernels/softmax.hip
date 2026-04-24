
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>
#define WS 64
__device__ float wmax(float v){for(int m=WS/2;m>0;m>>=1)v=fmaxf(v,__shfl_xor(v,m));return v;}
__device__ float wsum(float v){for(int m=WS/2;m>0;m>>=1)v+=__shfl_xor(v,m);return v;}
__global__ void softmax_k(__half* x,int rows,int cols){
    int row=blockIdx.x; if(row>=rows)return;
    __half* r=x+(long)row*cols;
    float mx=-FLT_MAX;
    for(int i=threadIdx.x;i<cols;i+=blockDim.x)mx=fmaxf(mx,__half2float(r[i]));
    mx=wmax(mx); float s=0;
    for(int i=threadIdx.x;i<cols;i+=blockDim.x){float e=expf(__half2float(r[i])-mx);r[i]=__float2half(e);s+=e;}
    s=wsum(s);
    for(int i=threadIdx.x;i<cols;i+=blockDim.x)r[i]=__float2half(__half2float(r[i])/s);
}
extern "C" void softmax_f16(hipStream_t s,__half* x,int rows,int cols){
    hipLaunchKernelGGL(softmax_k,dim3(rows),dim3(WS),0,s,x,rows,cols);}
