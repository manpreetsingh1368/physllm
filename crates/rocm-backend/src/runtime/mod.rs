//! runtime — GPU execution runtime: stream pool and CUDA graph capture.

pub mod stream_pool;
pub mod graph;

pub use stream_pool::StreamPool;
pub use graph::{GraphCapture, CapturedGraph};
