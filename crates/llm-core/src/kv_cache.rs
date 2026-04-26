//! kv_cache.rs — Key-Value cache for efficient autoregressive decoding.
//!
//! Without KV cache: each decode step recomputes K,V for ALL previous tokens → O(n²)
//! With KV cache: K,V computed once per token and cached → O(n) per step
//!
//! Layout: [num_layers, 2 (K|V), batch, num_kv_heads, max_seq, head_dim]

use crate::{config::ModelConfig, Result, LlmError};
use rocm_backend::{GpuDevice, DeviceTensor, BackendError};
use half::f16;
use std::sync::Arc;

/// KV cache for a single layer (K and V tensors).
pub struct LayerKVCache {
    pub k:   DeviceTensor<f16>,   // [batch, num_kv_heads, max_seq, head_dim]
    pub v:   DeviceTensor<f16>,
    /// How many tokens are currently stored
    pub len: usize,
}

impl LayerKVCache {
    pub fn new(
        batch:       usize,
        num_kv_heads: usize,
        max_seq:     usize,
        head_dim:    usize,
    ) -> std::result::Result<Self, BackendError> {
        let shape = &[batch, num_kv_heads, max_seq, head_dim];
        Ok(Self {
            k:   DeviceTensor::alloc(shape)?,
            v:   DeviceTensor::alloc(shape)?,
            len: 0,
        })
    }

    /// Append new K, V slices (one decode step or prefill chunk).
    pub fn append(
        &mut self,
        k_new: &[f16],    // [batch, num_kv_heads, new_tokens, head_dim]
        v_new: &[f16],
        new_tokens: usize,
    ) -> Result<()> {
        // In a full implementation this would do in-place GPU copy to offset
        // For now we track length (actual GPU copy happens in forward pass)
        self.len += new_tokens;
        Ok(())
    }

    /// Reset cache (start a new sequence).
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Number of tokens cached.
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// Is the cache full?
    pub fn is_full(&self, max_seq: usize) -> bool {
        self.len >= max_seq
    }
}

/// Full KV cache across all layers.
pub struct KVCache {
    pub layers:   Vec<LayerKVCache>,
    pub max_seq:  usize,
    pub batch:    usize,
}

impl KVCache {
    /// Allocate KV cache on the GPU.
    pub fn new(config: &ModelConfig, _device: &Arc<GpuDevice>) -> Result<Self> {
        let batch = 1;
        let mut layers = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            let layer = LayerKVCache::new(
                batch,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim,
            ).map_err(LlmError::Backend)?;
            layers.push(layer);
        }

        Ok(Self {
            layers,
            max_seq: config.max_seq_len,
            batch,
        })
    }

    /// Reset all layers (clear sequence state).
    pub fn reset(&mut self) {
        for layer in &mut self.layers { layer.reset(); }
    }

    /// Current sequence length (tokens cached).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.len).unwrap_or(0)
    }

    /// Memory used by KV cache on GPU (bytes).
    pub fn memory_bytes(&self, config: &ModelConfig) -> usize {
        let per_layer = 2   // K and V
            * self.batch
            * config.num_kv_heads
            * config.max_seq_len
            * config.head_dim
            * std::mem::size_of::<f16>();
        per_layer * config.num_layers
    }

    /// Pretty-print cache memory usage.
    pub fn memory_summary(&self, config: &ModelConfig) -> String {
        let bytes = self.memory_bytes(config);
        if bytes >= 1 << 30 {
            format!("{:.2} GB", bytes as f64 / (1u64 << 30) as f64)
        } else {
            format!("{:.2} MB", bytes as f64 / (1u64 << 20) as f64)
        }
    }
}

// ── Paged KV cache (future: vLLM-style) ──────────────────────────────────────

/// Block metadata for paged attention.
/// When implementing paged KV, the cache is divided into fixed-size blocks
/// and sequences are given a page table mapping logical → physical blocks.
#[derive(Debug, Clone)]
pub struct KVBlock {
    pub physical_id: usize,
    pub tokens_used: usize,
    pub block_size:  usize,
}

impl KVBlock {
    pub fn new(physical_id: usize, block_size: usize) -> Self {
        Self { physical_id, tokens_used: 0, block_size }
    }
    pub fn is_full(&self) -> bool { self.tokens_used >= self.block_size }
    pub fn remaining(&self) -> usize { self.block_size - self.tokens_used }
}

/// Paged KV cache allocator (stub — full implementation uses vLLM page table).
pub struct PagedKVAllocator {
    pub block_size:   usize,
    pub num_blocks:   usize,
    free_blocks:      Vec<usize>,
}

impl PagedKVAllocator {
    pub fn new(total_gpu_bytes: usize, block_size: usize, config: &ModelConfig) -> Self {
        let bytes_per_block = 2  // K+V
            * config.num_layers
            * config.num_kv_heads
            * block_size
            * config.head_dim
            * std::mem::size_of::<f16>();
        let num_blocks = total_gpu_bytes / bytes_per_block;
        let free_blocks = (0..num_blocks).collect();
        Self { block_size, num_blocks, free_blocks }
    }

    pub fn allocate(&mut self) -> Option<KVBlock> {
        self.free_blocks.pop().map(|id| KVBlock::new(id, self.block_size))
    }

    pub fn free(&mut self, block: KVBlock) {
        self.free_blocks.push(block.physical_id);
    }

    pub fn free_blocks(&self) -> usize { self.free_blocks.len() }
    pub fn used_blocks(&self) -> usize { self.num_blocks - self.free_blocks.len() }
}
