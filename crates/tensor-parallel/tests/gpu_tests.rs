// tests/gpu_tests.rs
//
// Run on actual hardware:
//   HIP_ARCH=gfx942 cargo test --features rocm -p tensor-parallel -- --nocapture
//
// Each test allocates device buffers, runs a collective or GEMM, and checks
// the result by copying back to host.

#[cfg(feature = "rocm")]
mod rocm {
    use tensor_parallel::{
        TpConfig, TpHandleGroup,
        config::TpStrategy,
    };
    use std::sync::Arc;

    //  Helpers 

    /// hipMalloc + hipMemcpy host→device. Returns device pointer.
    unsafe fn gpu_alloc_copy(data: &[u8]) -> *mut u8 {
        let mut ptr: *mut u8 = std::ptr::null_mut();
        extern "C" {
            fn hipMalloc(ptr: *mut *mut u8, size: usize) -> i32;
            fn hipMemcpy(dst: *mut u8, src: *const u8, size: usize, kind: i32) -> i32;
        }
        let rc = unsafe { hipMalloc(&mut ptr, data.len()) };
        assert_eq!(rc, 0, "hipMalloc failed");
        let rc = unsafe { hipMemcpy(ptr, data.as_ptr(), data.len(), 1 /* H2D */) };
        assert_eq!(rc, 0, "hipMemcpy H2D failed");
        ptr
    }

    unsafe fn gpu_read(ptr: *mut u8, len: usize) -> Vec<u8> {
        extern "C" {
            fn hipMemcpy(dst: *mut u8, src: *const u8, size: usize, kind: i32) -> i32;
        }
        let mut buf = vec![0u8; len];
        let rc = unsafe { hipMemcpy(buf.as_mut_ptr(), ptr as *const u8, len, 2 /* D2H */) };
        assert_eq!(rc, 0, "hipMemcpy D2H failed");
        buf
    }

    unsafe fn gpu_free(ptr: *mut u8) {
        extern "C" { fn hipFree(ptr: *mut u8) -> i32; }
        unsafe { hipFree(ptr) };
    }

    fn f16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect()
    }

    fn bytes_to_f16(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()
    }

    // Tests 

    /// Verify AllReduce sums fp16 buffers across 2 GPUs.
    #[test]
    fn allreduce_fp16_2gpu() {
        let cfg = TpConfig::new(0, vec![0, 1], TpStrategy::Megatron)
            .expect("need 2 GPUs");
        let group = TpHandleGroup::init(cfg).expect("init failed");

        // Each rank puts [1.0, 2.0, 3.0, 4.0] into its buffer.
        // After AllReduce, every rank should see [2.0, 4.0, 6.0, 8.0].
        let input = f16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let expected = vec![2.0f32, 4.0, 6.0, 8.0];

        // Run on rank 0 for simplicity (in real usage: one thread per rank)
        let h0 = group.handle(0);
        let handle = h0.lock();

        let dev_ptr = unsafe { gpu_alloc_copy(&input) };
        unsafe { handle.allreduce_fp16(dev_ptr, 4).expect("allreduce failed") };

        let result_bytes = unsafe { gpu_read(dev_ptr, input.len()) };
        let result = bytes_to_f16(&result_bytes);

        for (got, exp) in result.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 0.1, "got {got}, expected {exp}");
        }

        unsafe { gpu_free(dev_ptr) };
        println!("✓ allreduce_fp16_2gpu passed");
    }

    /// Verify GEMM: C = A @ B  with A=Identity gives C=B.
    #[test]
    fn gemm_fp16_identity() {
        let cfg = TpConfig::new(0, vec![0], TpStrategy::Megatron).unwrap();
        let group = TpHandleGroup::init(cfg).unwrap();
        let h = group.handle(0);
        let handle = h.lock();

        // 4×4 identity A, 4×4 B = [[1..16]]
        let identity: Vec<f32> = (0..16)
            .map(|i| if i / 4 == i % 4 { 1.0 } else { 0.0 })
            .collect();
        let b_vals: Vec<f32> = (1..=16).map(|x| x as f32).collect();

        let a_ptr = unsafe { gpu_alloc_copy(&f16_bytes(&identity)) };
        let b_ptr = unsafe { gpu_alloc_copy(&f16_bytes(&b_vals)) };
        let c_ptr = unsafe { gpu_alloc_copy(&vec![0u8; 16 * 2]) };

        unsafe {
            handle.gemm_fp16(4, 4, 4, a_ptr as *const u8, b_ptr as *const u8, c_ptr, 1.0, 0.0)
                .expect("gemm failed");
            handle.sync().unwrap();
        }

        let result = bytes_to_f16(&unsafe { gpu_read(c_ptr, 16 * 2) });
        for (got, exp) in result.iter().zip(b_vals.iter()) {
            assert!((got - exp).abs() < 0.5, "GEMM mismatch: got {got}, expected {exp}");
        }

        unsafe { gpu_free(a_ptr); gpu_free(b_ptr); gpu_free(c_ptr); }
        println!("✓ gemm_fp16_identity passed");
    }

    /// Config validation — world_size must be power of 2.
    #[test]
    fn config_validation() {
        assert!(TpConfig::new(0, vec![0, 1, 2], TpStrategy::Megatron).is_err());
        assert!(TpConfig::new(0, vec![0, 1, 2, 3], TpStrategy::Megatron).is_ok());
    }

    /// Shard arithmetic correctness.
    #[test]
    fn shard_arithmetic() {
        let cfg = TpConfig::new(2, vec![0,1,2,3], TpStrategy::Megatron).unwrap();
        assert_eq!(cfg.shard_size(4096),  1024);
        assert_eq!(cfg.shard_start(4096), 2048);
        assert_eq!(cfg.shard_end(4096),   3072);
    }
}
