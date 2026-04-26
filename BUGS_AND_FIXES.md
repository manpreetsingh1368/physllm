# PhysLLM — Known Bugs That Cause Gibberish Output

This document lists **every bug identified** during the build session that causes
the model to generate nonsensical output instead of coherent text, along with
the fix for each.

---

## Bug #1: Weight Matrix Transpose (CRITICAL — Root Cause of Gibberish)

**Symptom:** Model generates random garbage like `inyourstomachlining.A-inorder...`

**Root Cause:** PyTorch/safetensors stores linear layer weights as `[out_features, in_features]`,
but the forward pass needs `y = x @ W^T`. Without transposing, every matmul produces wrong results.

**Where:** All weight loading code — `loader.rs`, `inference.rs`, any example that loads safetensors.

**The Bug:**
```rust
// WRONG: Using weights as-is from safetensors
let wq = load_tensor("model.layers.0.self_attn.q_proj.weight"); // Shape: [4096, 4096]
let output = matmul(input, wq);  // input @ W — WRONG! Should be input @ W^T
```

**The Fix:**
```rust
// CORRECT: Transpose weights after loading
fn transpose_weight(weight: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut transposed = vec![f16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = weight[r * cols + c];
        }
    }
    transposed
}

// When loading:
let wq_raw = load_tensor("model.layers.0.self_attn.q_proj.weight"); // [4096, 4096]
let wq_transposed = transpose_weight(&wq_raw, 4096, 4096);          // [4096, 4096] transposed
let wq = DeviceTensor::from_slice(&wq_transposed, &[4096, 4096])?;

// Now matmul is correct:
let output = matmul(input, wq);  // input @ W^T — CORRECT!
```

**Files to fix:**
- `crates/llm-core/src/inference.rs` — `transpose_weight()` function provided
- `crates/llm-core/src/loader.rs` — Apply transpose when loading from safetensors
- `crates/llm-core/src/model.rs` — Check `transformer_block()` projections
- Any example files (`gpu_generation.rs`, `coherent_generation.rs`, etc.)

**Applies to these weight tensors:**
- `q_proj.weight` → [4096, 4096] → transpose
- `k_proj.weight` → [1024, 4096] → transpose
- `v_proj.weight` → [1024, 4096] → transpose
- `o_proj.weight` → [4096, 4096] → transpose
- `gate_proj.weight` → [14336, 4096] → transpose
- `up_proj.weight` → [14336, 4096] → transpose
- `down_proj.weight` → [4096, 14336] → transpose

---

## Bug #2: RoPE Position Encoding Not Applied or Applied Incorrectly

**Symptom:** Output words are real but in wrong order, or model ignores position entirely.

**Root Cause:** RoPE (Rotary Position Embeddings) must be applied to Q and K tensors
AFTER projection but BEFORE attention. If RoPE is wrong, the model can't distinguish
token positions.

**The Bug:**
```rust
// WRONG: RoPE not applied, or applied to wrong dimensions
let q = matmul(normed, wq);   // Q projected
let k = matmul(normed, wk);   // K projected
// Missing: apply_rope(&mut q, ...) and apply_rope(&mut k, ...)
let attn = flash_attention(q, k, v);  // Attention without position info!
```

**The Fix:**
```rust
let mut q = matmul(normed, wq);
let mut k = matmul(normed, wk);

// Apply RoPE to Q and K (GPU kernel)
rope_gpu(&device, &mut q, num_heads, 1, head_dim, position, rope_theta)?;
rope_gpu(&device, &mut k, num_kv_heads, 1, head_dim, position, rope_theta)?;

let attn = flash_attention(q, k, v);  // Now with correct positional info
```

**Key parameters for Mistral 7B:**
- `rope_theta = 10000.0`
- `head_dim = 128` (rotation applied per pair of dimensions)
- Position = index in the KV cache (increments each token)

**Key parameters for GPT-OSS-20B:**
- `rope_theta = 150000.0`
- `head_dim = 64`
- YaRN scaling: `factor = 32.0`, `original_max_pos = 4096`

**Files to fix:**
- `crates/llm-core/src/inference.rs` — Lines where Q/K are projected
- `kernels/rope_gpu.hip` — Verify rotation formula matches reference

**Reference implementation (PyTorch):**
```python
# From transformers/models/mistral/modeling_mistral.py
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

---

## Bug #3: KV Cache Indexing Wrong

**Symptom:** First few tokens are OK, then output degrades or repeats.

**Root Cause:** The KV cache stores past key/value pairs for attention. If the
indexing is wrong, the model attends to garbage data from uninitialized memory.

**The Bug:**
```rust
// WRONG: Not updating cache position correctly
kv_cache.k[0..1024].copy_from_slice(&k);  // Always writes to position 0!
```

**The Fix:**
```rust
// CORRECT: Write to the correct position in the cache
let cache_offset = kv_cache.current_len * num_kv_heads * head_dim;
kv_cache.k[cache_offset..cache_offset + kv_dim].copy_from_slice(&k);
kv_cache.v[cache_offset..cache_offset + kv_dim].copy_from_slice(&v);
kv_cache.current_len += 1;

// When computing attention, use only the valid portion:
let total_seq = kv_cache.current_len;
flash_attention(q, &kv_cache.k[..total_seq * kv_dim], &kv_cache.v[..total_seq * kv_dim]);
```

**Files to fix:**
- `crates/llm-core/src/inference.rs` — `forward_layer_gpu()` KV cache section
- `kernels/kv_cache_update.hip` — Verify position offset calculation

---

## Bug #4: GQA (Grouped Query Attention) Head Mapping Wrong

**Symptom:** Attention scores are computed but output is scrambled.

**Root Cause:** Mistral 7B uses GQA with 32 query heads but only 8 KV heads.
Each KV head serves 4 query heads (32/8 = 4). The flash attention kernel must
map query heads to the correct KV head.

**The Bug:**
```cpp
// WRONG: Using same head index for Q and K
int q_offset = (head_idx * seq_q + q_pos) * head_dim;
int k_offset = (head_idx * seq_kv + kv_pos) * head_dim;  // head_idx is wrong for K!
```

**The Fix:**
```cpp
// CORRECT: Map query head to KV head
int kv_head_idx = head_idx * num_kv_heads / num_heads;  // Maps 0-31 → 0-7
int q_offset = (head_idx * seq_q + q_pos) * head_dim;
int k_offset = (kv_head_idx * seq_kv + kv_pos) * head_dim;  // Use kv_head_idx!
```

**Files to fix:**
- `kernels/flash_attention_v2.hip` — Line with K/V offset calculation
- `kernels/flash_attention.hip` — Same fix needed

---

## Bug #5: Residual Connections Missing or Wrong

**Symptom:** Output quality degrades with more layers, or model produces near-zero values.

**Root Cause:** Each transformer layer has two residual connections:
1. `hidden = hidden + attention_output`
2. `hidden = hidden + mlp_output`
If either is missing, gradients vanish and the model collapses.

**The Bug:**
```rust
// WRONG: Overwriting hidden instead of adding
hidden = attention_output;  // Lost the original hidden state!
```

**The Fix:**
```rust
// CORRECT: Add residual connection
// Option A: GPU kernel (fastest)
residual_add_gpu(&device, &mut hidden, &attention_output)?;

// Option B: CPU fallback
for i in 0..hidden.len() {
    hidden[i] = f16::from_f32(hidden[i].to_f32() + attention_output[i].to_f32());
}
```

**Files to fix:**
- `crates/llm-core/src/inference.rs` — Two residual adds per layer
- `crates/llm-core/src/model.rs` — `transformer_block()` residual connections

---

## Bug #6: SiLU Activation Wrong in MLP

**Symptom:** MLP output is too large or too small, causing numerical instability.

**Root Cause:** The MLP uses SwiGLU: `output = SiLU(gate) * up`. If SiLU is
implemented wrong, the activation values explode or collapse.

**The Bug:**
```rust
// WRONG: Missing SiLU, just multiplying gate * up
let inter = gate * up;  // No activation function!
```

**The Fix:**
```rust
// CORRECT: SiLU(gate) * up
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
let silu_gate = gate / (1.0 + (-gate).exp());
let inter = silu_gate * up;
```

**GPU kernel (kernels/silu.hip):**
```cpp
float g = __half2float(gate[idx]);
float u = __half2float(up[idx]);
float silu = g / (1.0f + expf(-g));
output[idx] = __float2half(silu * u);
```

**Files to fix:**
- `kernels/silu.hip` — Verify SiLU formula
- `crates/llm-core/src/model.rs` — `transformer_block()` SwiGLU section

---

## Bug #7: BF16 to F16 Conversion Wrong

**Symptom:** All weights are garbage values (NaN, infinity, or near-zero).

**Root Cause:** Mistral 7B safetensors stores weights in BF16 (Brain Float 16).
Your GPU uses F16 (IEEE Float 16). These are DIFFERENT formats with different
bit layouts. Direct reinterpretation corrupts all values.

**The Bug:**
```rust
// WRONG: Treating BF16 bytes as F16
let f16_data = unsafe {
    std::slice::from_raw_parts(raw_bytes.as_ptr() as *const f16, num_elements)
};
// All values are garbage because BF16 ≠ F16!
```

**The Fix:**
```rust
// CORRECT: Convert BF16 → F32 → F16
fn bf16_to_f16_vec(bf16_data: &[u8]) -> Vec<f16> {
    let bf16_slice: &[u16] = unsafe {
        std::slice::from_raw_parts(
            bf16_data.as_ptr() as *const u16,
            bf16_data.len() / 2,
        )
    };
    bf16_slice.iter().map(|&bf16| {
        // BF16 to F32: shift left 16 bits (BF16 = top 16 bits of F32)
        let f32_val = f32::from_bits((bf16 as u32) << 16);
        // F32 to F16: standard conversion
        f16::from_f32(f32_val)
    }).collect()
}
```

**Files to fix:**
- `crates/llm-core/src/inference.rs` — `bf16_to_f16_vec()` function provided
- `crates/llm-core/src/loader.rs` — Weight loading conversion

---

## Bug #8: ROCm 7.x FFI Function Name Mismatch

**Symptom:** Linker error: `undefined symbol: hipGetDeviceProperties_v2`

**Root Cause:** ROCm 7.2 renamed the function to `hipGetDevicePropertiesR0600`.
The old name `hipGetDeviceProperties_v2` doesn't exist.

**The Fix:**
```rust
// In crates/rocm-backend/src/device.rs:
// WRONG:
hipGetDeviceProperties_v2(&mut props, idx);

// CORRECT (ROCm 7.x):
hipGetDevicePropertiesR0600(&mut props, idx);
```

**Files to fix:**
- `crates/rocm-backend/src/device.rs`
- `crates/rocm-backend/build.rs` (if using stub bindings)

---

## Bug #9: Flash Attention Kernel Shared Memory Size Wrong

**Symptom:** GPU kernel crash, or silent corruption of attention scores.

**Root Cause:** The flash attention kernel uses shared memory to store attention
scores. If the shared memory size doesn't match `seq_kv * sizeof(float)`, the
kernel reads/writes out of bounds.

**The Fix:**
```cpp
// Ensure shared memory matches sequence length
size_t shared_mem = seq_kv * sizeof(float);  // NOT seq_q!
hipLaunchKernelGGL(flash_attention_kernel,
    grid, block, shared_mem, stream,  // shared_mem must be correct
    ...);
```

**Files to fix:**
- `kernels/flash_attention_v2.hip` — Launch configuration
- `crates/rocm-backend/src/attention_ops.rs` — Rust wrapper

---

## Bug #10: LM Head Not Transposed

**Symptom:** Token probabilities are uniform (random sampling from vocabulary).

**Root Cause:** The LM head weight `lm_head.weight` has shape `[vocab_size, hidden_dim]`.
The forward pass needs `logits = hidden @ lm_head.T`. Same transpose issue as Bug #1
but for the final output layer.

**The Fix:**
```rust
// The lm_head kernel already handles this correctly:
// logits[v] = sum_d(hidden[d] * lm_head[v * hidden_dim + d])
// This is equivalent to hidden @ lm_head.T

// But if using matmul_f16 directly, you need:
let lm_head_t = transpose_weight(&lm_head_raw, vocab_size, hidden_dim);
let logits = matmul(hidden, lm_head_t);  // [1, vocab_size]
```

**Files to fix:**
- `crates/llm-core/src/inference.rs` — LM head section
- `kernels/lm_head.hip` — Already handles transpose internally

---

## Debugging Checklist

Run these checks IN ORDER to find which bug is causing gibberish:

### Step 1: Verify Weight Loading
```rust
// Load one weight tensor and check values
let wq = load_tensor("model.layers.0.self_attn.q_proj.weight");
let sample = &wq[..10];
println!("Q proj first 10 values: {:?}", sample.iter().map(|x| x.to_f32()).collect::<Vec<_>>());
// Should see small values like [-0.01, 0.03, -0.02, ...]
// If you see NaN, inf, or very large values → Bug #7 (BF16 conversion)
```

### Step 2: Verify Matmul Correctness
```rust
// Test: [1,1,1,...] @ W should give column sums
let ones = vec![f16::from_f32(1.0); 4096];
let output = matmul(ones, wq);
println!("Output norm: {}", output.iter().map(|x| x.to_f32() * x.to_f32()).sum::<f32>().sqrt());
// Should be non-zero and finite
// If very large → Bug #1 (transpose)
```

### Step 3: Verify RoPE
```rust
// Apply RoPE and check values change
let mut q_copy = q.clone();
apply_rope(&mut q_copy, position=0, theta=10000.0);
assert!(q != q_copy, "RoPE didn't change Q!");
// If unchanged → Bug #2 (RoPE not applied)
```

### Step 4: Verify Attention
```rust
// Check attention weights sum to 1.0
let attn_weights = softmax(q @ k.T / sqrt(head_dim));
let row_sum: f32 = attn_weights.iter().sum();
assert!((row_sum - 1.0).abs() < 0.01, "Attention weights don't sum to 1!");
// If sum != 1 → Bug #4 (GQA) or Bug #9 (shared memory)
```

### Step 5: Verify Residuals
```rust
// Check hidden state grows through layers (not zero or explode)
for layer in 0..32 {
    let norm = hidden.iter().map(|x| x.to_f32().abs()).sum::<f32>() / 4096.0;
    println!("Layer {} hidden norm: {:.4}", layer, norm);
}
// Should be stable (0.5-2.0 range)
// If decreasing → Bug #5 (missing residual)
// If increasing rapidly → Bug #6 (SiLU wrong)
```

---

## Priority Order for Fixing

1. **Bug #7** — BF16 conversion (if weights are garbage, nothing else matters)
2. **Bug #1** — Weight transpose (most common cause of gibberish)
3. **Bug #2** — RoPE (causes position-unaware output)
4. **Bug #4** — GQA mapping (scrambles attention)
5. **Bug #3** — KV cache (degrades over sequence)
6. **Bug #5** — Residuals (layer quality degradation)
7. **Bug #6** — SiLU (numerical instability)
8. **Bug #10** — LM head (uniform token distribution)
9. **Bug #9** — Shared memory (kernel crash)
10. **Bug #8** — FFI name (compile error, not runtime)

---

## Reference Implementations for Validation

Compare your output against these proven implementations:

1. **HuggingFace Transformers (Python):**
   - `transformers/models/mistral/modeling_mistral.py`
   - Run same input, compare output token-by-token

2. **Candle (Rust):**
   - `candle-transformers/src/models/mistral.rs`
   - Same language, proven to work

3. **llama.cpp (C++):**
   - `llama.cpp` — attention computation
   - Battle-tested on millions of users

4. **GPT-OSS Reference (PyTorch):**
   - `github.com/openai/gpt-oss/gpt_oss/torch/model.py`
   - Official implementation from OpenAI
