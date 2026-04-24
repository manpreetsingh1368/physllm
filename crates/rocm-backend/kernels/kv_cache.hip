
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
__global__ void kv_append_k(__half* cache,const __half* nkv,
    int batch,int heads,int ms,int hd,int off,int nt){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=(long)batch*heads*nt*hd)return;
    int d=idx%hd,tok=(idx/hd)%nt,h=(idx/hd/nt)%heads,b=idx/hd/nt/heads;
    cache[((long)(b*heads+h)*ms+(off+tok))*hd+d]=nkv[idx];
}
extern "C" void kv_cache_append(hipStream_t s,__half* cache,const __half* nkv,
    int batch,int heads,int ms,int hd,int off,int nt){
    long tot=(long)batch*heads*nt*hd;
    hipLaunchKernelGGL(kv_append_k,dim3((tot+255)/256),dim3(256),0,s,cache,nkv,batch,heads,ms,hd,off,nt);}
