// benches/tp_bench.rs
//
// Run:  HIP_ARCH=gfx942 cargo bench --features rocm -p tensor-parallel

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "rocm")]
mod benches {
    use super::*;
    use tensor_parallel::{TpConfig, TpHandleGroup, config::TpStrategy};

    unsafe fn gpu_alloc_zeros(bytes: usize) -> *mut u8 {
        extern "C" {
            fn hipMalloc(ptr: *mut *mut u8, size: usize) -> i32;
            fn hipMemset(ptr: *mut u8, val: i32, size: usize) -> i32;
        }
        let mut ptr: *mut u8 = std::ptr::null_mut();
        hipMalloc(&mut ptr, bytes);
        hipMemset(ptr, 0, bytes);
        ptr
    }

    unsafe fn gpu_free(ptr: *mut u8) {
        extern "C" { fn hipFree(ptr: *mut u8); }
        hipFree(ptr);
    }

    pub fn bench_allreduce(c: &mut Criterion) {
        let cfg   = TpConfig::new(0, vec![0], TpStrategy::Megatron).unwrap();
        let group = TpHandleGroup::init(cfg).unwrap();
        let h     = group.handle(0);
        let handle = h.lock();

        let mut bg = c.benchmark_group("allreduce_fp16");

        // Test different hidden sizes: 2880 (GPT-OSS-20B), 4096 (7B), 8192 (70B)
        for &hidden in &[2880usize, 4096, 8192] {
            // 1 token, full hidden dim
            let numel = hidden;
            let bytes = numel * 2;
            bg.throughput(Throughput::Bytes(bytes as u64));

            let buf = unsafe { gpu_alloc_zeros(bytes) };

            bg.bench_with_input(
                BenchmarkId::new("hidden", hidden),
                &numel,
                |b, &n| {
                    b.iter(|| unsafe {
                        handle.allreduce_fp16(buf, n).unwrap();
                    });
                },
            );

            unsafe { gpu_free(buf) };
        }
        bg.finish();
    }

    pub fn bench_gemm(c: &mut Criterion) {
        let cfg    = TpConfig::new(0, vec![0], TpStrategy::Megatron).unwrap();
        let group  = TpHandleGroup::init(cfg).unwrap();
        let h      = group.handle(0);
        let handle = h.lock();

        let mut bg = c.benchmark_group("gemm_fp16");

        // Simulate Q projection: [128 tokens, 2880] @ [2880, 720].T
        // 720 = 2880/4 heads (4-GPU split)
        let (tokens, hidden, out_shard) = (128i32, 2880i32, 720i32);
        let bytes_a = (tokens * hidden * 2) as usize;
        let bytes_b = (out_shard * hidden * 2) as usize;
        let bytes_c = (tokens * out_shard * 2) as usize;

        bg.throughput(Throughput::Elements(
            (tokens as u64) * (out_shard as u64) * (hidden as u64) * 2, // FMAs
        ));

        let a = unsafe { gpu_alloc_zeros(bytes_a) };
        let b = unsafe { gpu_alloc_zeros(bytes_b) };
        let c = unsafe { gpu_alloc_zeros(bytes_c) };

        bg.bench_function("q_proj_128tok_2880h_720shard", |bench| {
            bench.iter(|| unsafe {
                handle.gemm_fp16_bt(tokens, out_shard, hidden,
                                    a as *const u8, b as *const u8, c,
                                    1.0, 0.0).unwrap();
                handle.sync().unwrap();
            });
        });

        unsafe { gpu_free(a); gpu_free(b); gpu_free(c); }
        bg.finish();
    }
}

#[cfg(feature = "rocm")]
criterion_group!(tp_benches, benches::bench_allreduce, benches::bench_gemm);

#[cfg(not(feature = "rocm"))]
criterion_group!(tp_benches,);

criterion_main!(tp_benches);
