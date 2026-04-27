//! llm-core — Transformer LLM runtime in pure Rust.
//!
//! Architecture (Llama 3 / Mistral style):
//!   Embeddings → N × TransformerBlock → RMSNorm → LM Head → Logits
//!
//! Each TransformerBlock:
//!   RMSNorm → GQA (Grouped Query Attention) + RoPE → residual
//!   RMSNorm → SwiGLU FFN → residual

pub mod config;
pub mod model;
pub mod attention;
pub mod ffn;
pub mod embedding;
pub mod kv_cache;
pub mod tokenizer;
pub mod generate;
pub mod inference;
pub mod moe;
pub mod gpt_oss_loader;
pub mod loader;
// pub mod search_agent; // requires web-search crate
pub mod agent;

pub use config::ModelConfig;
pub use model::PhysLLM;
pub use generate::{GenerateRequest, GenerateResponse, SamplingParams};
pub use inference::{InferenceEngine, transpose_weight, bf16_to_f16_vec};
pub use moe::MoEInferenceEngine;
pub use gpt_oss_loader::load_gpt_oss_weights;
pub use tokenizer::PhysTokenizer;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Model load error: {0}")]
    Load(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("GPU backend error: {0}")]
    Backend(#[from] rocm_backend::BackendError),
    #[error("Context window overflow: {tokens} > {max}")]
    ContextOverflow { tokens: usize, max: usize },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, LlmError>;
