# PhysLLM — GPU-Accelerated Physics LLM

Custom-built 7B parameter LLM with native GPU acceleration for **AMD (ROCm)** and **NVIDIA (CUDA)** via unified HIP backend.

## Features

- **100% GPU Compute**: All 10 operations run on GPU (no CPU bottlenecks)
- **Dual GPU Support**: Same codebase for AMD MI300X and NVIDIA H100
- **Flash Attention v2**: Optimized attention kernel with tiling
- **7B Parameters**: Mistral 7B architecture (32 layers, GQA)
- **Custom HIP Kernels**: RMS Norm, RoPE, SiLU, Embedding, LM Head, KV Cache
- **Training Support**: AdamW optimizer kernel on GPU
- **Physics Domain**: Specialized tokenizer extensions for physics/chemistry

## Quick Start

### 1. Build

**AMD GPUs (MI300X, MI250X, RX 7900):**
```bash
./build_amd.sh
```

**NVIDIA GPUs (H100, A100, RTX 4090):**
```bash
./build_nvidia.sh
```

**CPU only (for testing):**
```bash
./build_cpu.sh
```

### 2. Download Model Weights

```bash
./scripts/download_model.sh
```

### 3. Run Inference

```bash
cargo run --release -p llm-core --example gpu_inference
```

## Manual Kernel Compilation

If the automatic build fails, compile kernels manually:

```bash
# Check your GPU architecture
rocminfo | grep "Name:" | grep gfx     # AMD
nvidia-smi                              # NVIDIA

# Compile for your GPU (example: MI300X = gfx942)
cd kernels
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c flash_attention_v2.hip -o flash_attention_v2.o
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c rms_norm_gpu.hip -o rms_norm_gpu.o
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c rope_gpu.hip -o rope_gpu.o
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c silu.hip -o silu.o
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c residual_add.hip -o residual_add.o
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c embedding.hip -o embedding.o
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c lm_head.hip -o lm_head.o
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c softmax_sample.hip -o softmax_sample.o
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c kv_cache_update.hip -o kv_cache_update.o
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O3 -fPIC -c adam_optimizer.hip -o adam_optimizer.o

# Create static library
ar rcs libphysllm_kernels.a *.o

# Copy to build output
cp libphysllm_kernels.a ../target/release/build/rocm-backend-*/out/
```

Or use the helper script:
```bash
./scripts/compile_kernels.sh
```

### GPU Architecture Targets

| GPU | Architecture Flag |
|-----|-------------------|
| AMD MI300X | `--offload-arch=gfx942` |
| AMD MI250X | `--offload-arch=gfx90a` |
| AMD RX 7900 XTX | `--offload-arch=gfx1100` |
| NVIDIA H100 | `--offload-arch=sm_90` |
| NVIDIA A100 | `--offload-arch=sm_80` |
| NVIDIA RTX 4090 | `--offload-arch=sm_89` |
| NVIDIA RTX 3090 | `--offload-arch=sm_86` |
| NVIDIA V100 | `--offload-arch=sm_70` |

## Project Structure

```
physllm/
├── kernels/                    # 16 HIP GPU kernels (work on AMD + NVIDIA)
│   ├── flash_attention_v2.hip  # Optimized attention
│   ├── rms_norm_gpu.hip        # RMS normalization
│   ├── rope_gpu.hip            # Rotary position embeddings
│   ├── silu.hip                # SiLU activation + multiply
│   ├── residual_add.hip        # Residual connections
│   ├── embedding.hip           # Token embedding lookup
│   ├── lm_head.hip             # Vocabulary projection
│   ├── softmax_sample.hip      # Softmax + token sampling
│   ├── kv_cache_update.hip     # KV cache management
│   ├── adam_optimizer.hip       # AdamW training optimizer
│   ├── gemm_f16.hip            # FP16 matrix multiply
│   ├── flash_attention.hip     # Original attention kernel
│   ├── softmax.hip             # Standalone softmax
│   ├── rope_embedding.hip      # Original RoPE
│   ├── layer_norm.hip          # Layer normalization
│   └── kv_cache.hip            # Original KV cache
├── crates/
│   ├── rocm-backend/           # GPU acceleration layer
│   │   ├── build.rs            # Compiles HIP kernels + generates FFI
│   │   └── src/
│   │       ├── device.rs       # GPU device management
│   │       ├── tensor.rs       # DeviceTensor (GPU memory)
│   │       ├── ops.rs          # ALL GPU operations (10 new + 4 original)
│   │       ├── kernels.rs      # Kernel dispatch helpers
│   │       ├── memory.rs       # Memory pool management
│   │       └── lib.rs          # Public exports
│   ├── llm-core/               # LLM implementation
│   │   └── src/
│   │       ├── config.rs       # Model hyperparameters (7B/13B)
│   │       ├── model.rs        # Transformer forward pass
│   │       ├── loader.rs       # Safetensors weight loading
│   │       ├── attention.rs    # Multi-head attention
│   │       ├── generate.rs     # Text generation loop
│   │       └── ...
│   ├── domain-physics/         # Physics constants & formulas
│   ├── sim-agent/              # Physics simulation engines
│   ├── trainer/                # Training pipeline (LoRA, AdamW)
│   ├── api-server/             # REST API
│   ├── voice-io/               # Voice I/O
│   └── web-search/             # Web search integration
├── build_amd.sh                # Build for AMD GPUs
├── build_nvidia.sh             # Build for NVIDIA GPUs
├── build_cpu.sh                # Build CPU-only
└── scripts/
    ├── download_model.sh       # Download Mistral 7B
    └── compile_kernels.sh      # Manual kernel compilation
```

## 100% GPU Pipeline

```
Token Input
    ↓
[GPU] Embedding Lookup         ← embedding.hip
    ↓
[GPU] RMS Norm                 ← rms_norm_gpu.hip
    ↓
[GPU] Q/K/V Projection        ← hipBLAS matmul
    ↓
[GPU] RoPE                     ← rope_gpu.hip
    ↓
[GPU] KV Cache Update          ← kv_cache_update.hip
    ↓
[GPU] Flash Attention v2       ← flash_attention_v2.hip
    ↓
[GPU] Output Projection        ← hipBLAS matmul
    ↓
[GPU] Residual Add             ← residual_add.hip
    ↓
[GPU] RMS Norm                 ← rms_norm_gpu.hip
    ↓
[GPU] MLP Gate/Up              ← hipBLAS matmul
    ↓
[GPU] SiLU × Up                ← silu.hip
    ↓
[GPU] MLP Down                 ← hipBLAS matmul
    ↓
[GPU] Residual Add             ← residual_add.hip
    ↓
    × 32 layers
    ↓
[GPU] Final RMS Norm           ← rms_norm_gpu.hip
    ↓
[GPU] LM Head                  ← lm_head.hip
    ↓
[GPU] Softmax + Sample         ← softmax_sample.hip
    ↓
Token Output
```

**Zero CPU/GPU transfers during inference!**

## Bug Fixes Applied

### ROCm 7.x FFI Compatibility
```rust
// Fixed: hipGetDeviceProperties_v2 → hipGetDevicePropertiesR0600
```

### Weight Matrix Transpose
```rust
// PyTorch stores: W = [out_features, in_features]
// Forward pass:   y = x @ W.T
// Fix: transpose weights after loading from safetensors
fn transpose_weight(w: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut t = vec![f16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = w[r * cols + c];
        }
    }
    t
}
```

### Dependency Version Pinning
```toml
uuid = "1.6.1"        # Avoid edition2024 requirement
moka = "0.11"
tempfile = "3.9.0"
reqwest = "0.11"       # Avoid rand 0.9
```

## Performance

| Metric | Value |
|--------|-------|
| Model Load | ~72 seconds |
| VRAM Usage | ~13 GB (FP16) |
| Free VRAM (MI300X) | 180 GB |
| Generation | ~1-20 tok/s (depends on optimization) |

## Requirements

- **OS**: Linux (Ubuntu 22.04+ recommended)
- **GPU**: AMD Instinct (MI300X/MI250X) or NVIDIA (A100/H100/RTX)
- **Rust**: 1.75+
- **ROCm**: 6.0+ (AMD) or CUDA 12+ with HIP (NVIDIA)

## License

MIT
