# tensor-parallel

GPU-only tensor parallelism for PhysLLM.  
RCCL AllReduce + hipBLAS GEMMs + expert-parallel MoE.  
No CPU path. No host-side compute. Everything lives on device.

---

## File layout

```
kernels/
  gpu_shim.hip          C shim: RCCL collectives + hipBLAS GEMMs → Rust FFI

src/
  comm/
    ffi.rs              raw extern "C" declarations
    mod.rs              TpHandle (one GPU), TpHandleGroup (all GPUs)
  config.rs             TpConfig — world_size, rank, devices, strategy
  error.rs              TpError
  layers/
    linear.rs           ColumnParallelLinear, RowParallelLinear
    attention.rs        ParallelAttention (col QKV → Flash Attn → row O → AllReduce)
    moe.rs              ParallelMoE (expert-parallel, zero AllToAll)
  loader.rs             WeightLoader — mmap safetensors, slice shard for rank
  lib.rs

tests/gpu_tests.rs      AllReduce + GEMM correctness tests (requires GPU)
benches/tp_bench.rs     Throughput benchmarks
```

---

## Build

```bash
# AMD ROCm (MI300X default)
HIP_ARCH=gfx942 cargo build --release --features rocm -p tensor-parallel

# NVIDIA (A100)
CUDA_ARCH=sm_80 cargo build --release --features cuda -p tensor-parallel
```

Override GPU arch:
```bash
HIP_ARCH=gfx90a   # MI250X
HIP_ARCH=gfx1100  # RX 7900 XTX
CUDA_ARCH=sm_90   # H100
CUDA_ARCH=sm_89   # RTX 4090
```

---

## Add to workspace

**`Cargo.toml` (root):**
```toml
members = [..., "crates/tensor-parallel"]
```

**`crates/llm-core/Cargo.toml`:**
```toml
tensor-parallel = { path = "../tensor-parallel", features = ["rocm"] }
```

---

## Usage in llm-core

```rust
use tensor_parallel::{TpConfig, TpHandleGroup, TpStrategy, WeightLoader};
use std::{sync::Arc, path::Path};

// 1. Config (4× MI300X, expert-parallel for GPT-OSS-20B)
let cfg = Arc::new(
    TpConfig::new(rank, vec![0, 1, 2, 3], TpStrategy::ExpertParallel)?
);

// 2. Init communicators (ncclCommInitAll + hipblasCreate per GPU)
let group = TpHandleGroup::init((*cfg).clone())?;
let handle = group.handle(rank);  // Arc<Mutex<TpHandle>>

// 3. Load sharded weights (mmap, zero RAM spike)
let loader = WeightLoader::from_dir(Path::new("/models/gpt-oss-20b"), cfg.clone())?;
let layer  = loader.load_layer(layer_idx)?;

// 4. Upload shards to GPU (your existing hipMemcpy from rocm-backend)
let q_ptr = hip_malloc_copy(layer["q_proj"].as_bytes());
let k_ptr = hip_malloc_copy(layer["k_proj"].as_bytes());
// ...

// 5. Build parallel attention
let attn = ParallelAttention::from_shards(
    &layer["q_proj"], q_ptr,
    &layer["k_proj"], k_ptr,
    &layer["v_proj"], v_ptr,
    &layer["o_proj"], o_ptr,
    attn_cfg, cfg.clone(), handle.clone(),
);

// 6. Forward (all async on GPU stream)
unsafe {
    attn.forward(
        x_ptr, q_buf, k_buf, v_buf, attn_buf, out_ptr, kv_cache,
        seq_offset, tokens,
        rope_fn, kv_cache_fn, flash_attn_fn,   // fn ptrs from rocm-backend
    )?;
}
// Overlap: next layer's norm/embed can start while AllReduce runs
handle.lock().sync()?;
```

---

## Overlap pattern (hide AllReduce latency)

The AllReduce in `RowParallelLinear::forward` is launched **async**.  
Chain operations to overlap it with independent work:

```
Layer N:
  QKV GEMMs          (compute)
  Flash Attention     (compute)
  O proj GEMM         (compute)
  AllReduce async  ←──── launched, NOT waited
    │
    ▼ (overlaps with ↓)
  RMSNorm             (can start — reads residual, not AllReduce output)
  Embedding lookup    (independent)
    │
  handle.sync()   ←── wait here, just before Layer N+1 reads the reduced output
Layer N+1:
  QKV GEMMs          (reads fully reduced output)
```

On MI300X Infinity Fabric, AllReduce for hidden=2880 takes ~1.5 µs.  
RMSNorm for 128 tokens takes ~3 µs → full overlap, zero exposed latency.

---

## Tests

```bash
# Correctness (2 GPUs required)
HIP_ARCH=gfx942 cargo test --features rocm -p tensor-parallel -- --nocapture

# Benchmarks
HIP_ARCH=gfx942 cargo bench --features rocm -p tensor-parallel
```

---

## Strategy guide

| Model | Recommended | Reason |
|---|---|---|
| GPT-OSS-20B (64 experts) | `ExpertParallel` | 16 experts/GPU, zero AllToAll |
| GPT-OSS-120B (MoE) | `Hybrid` | Expert-parallel + Megatron within group |
| Dense 7B–13B | `Megatron` | Head split is more balanced for dense |
