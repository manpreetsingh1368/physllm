// crates/tensor-parallel/src/lib.rs
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

pub mod comm;
pub mod config;
pub mod error;
pub mod layers;
pub mod loader;

pub use comm::{TpHandle, TpHandleGroup};
pub use config::{TpConfig, TpStrategy};
pub use error::{TpError, TpResult};
pub use layers::{ParallelAttention, ParallelMoE, ColumnParallelLinear, RowParallelLinear};
pub use loader::{ShardedWeight, WeightLoader};
