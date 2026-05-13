// src/error.rs
use thiserror::Error;

pub type TpResult<T> = Result<T, TpError>;

#[derive(Debug, Error)]
pub enum TpError {
    #[error("GPU collective failed (rank {rank}): {msg}")]
    Collective { rank: usize, msg: String },

    #[error("hipBLAS/cuBLAS GEMM failed (rank {rank}): code {code}")]
    Gemm { rank: usize, code: i32 },

    #[error("Weight shard load failed for '{name}': {source}")]
    WeightLoad { name: String, source: anyhow::Error },

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("world_size {0} must be a power of 2 and ≥ 1")]
    InvalidWorldSize(usize),

    #[error("Rank {rank} out of range for world_size {world_size}")]
    RankOutOfRange { rank: usize, world_size: usize },

    #[error("Communicator not initialised — call TpHandleGroup::init() first")]
    NotInitialised,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
