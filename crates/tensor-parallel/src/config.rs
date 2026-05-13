// src/config.rs
use serde::{Deserialize, Serialize};
use crate::error::{TpError, TpResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpConfig {
    pub world_size: usize,
    pub rank: usize,
    /// HIP/CUDA device IDs, one per rank. Length == world_size.
    pub devices: Vec<i32>,
    pub strategy: TpStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpStrategy {
    /// Split attention heads + MLP columns/rows. Dense transformers.
    Megatron,
    /// Each GPU owns n_experts/world experts. Best for GPT-OSS MoE.
    ExpertParallel,
    /// Megatron within expert groups + expert-parallel across groups.
    Hybrid,
}

impl TpConfig {
    pub fn new(rank: usize, devices: Vec<i32>, strategy: TpStrategy) -> TpResult<Self> {
        let world_size = devices.len();
        if world_size == 0 || !world_size.is_power_of_two() {
            return Err(TpError::InvalidWorldSize(world_size));
        }
        if rank >= world_size {
            return Err(TpError::RankOutOfRange { rank, world_size });
        }
        Ok(Self { world_size, rank, devices, strategy })
    }

    pub fn local_device(&self) -> i32 { self.devices[self.rank] }

    /// Elements per shard for a dimension of total `n`.
    /// Pads to the next multiple of world_size when not evenly divisible.
    pub fn shard_size(&self, n: usize) -> usize {
        n.next_multiple_of(self.world_size) / self.world_size
    }

    /// Start element index for this rank.
    pub fn shard_start(&self, n: usize) -> usize {
        self.rank * self.shard_size(n)
    }

    /// End element index (exclusive, clamped to n).
    pub fn shard_end(&self, n: usize) -> usize {
        (self.shard_start(n) + self.shard_size(n)).min(n)
    }
}
