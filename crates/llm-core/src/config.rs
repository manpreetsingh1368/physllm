//! ModelConfig — all hyperparameters for the PhysLLM transformer.

use serde::{Deserialize, Serialize};

/// Full model hyperparameters.
/// Compatible with Llama 3 / Mistral config format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // ── Vocabulary ────────────────────────────────────────────────────────────
    /// Total vocabulary size (base + physics/chemistry extensions)
    pub vocab_size:          usize,
    /// Physics/chemistry domain token extension starting index
    pub domain_vocab_offset: usize,
    /// Domain-specific token count (formulas, constants, units)
    pub domain_vocab_size:   usize,

    // ── Architecture ──────────────────────────────────────────────────────────
    pub hidden_dim:          usize,   // e.g. 4096
    pub intermediate_dim:    usize,   // FFN intermediate, e.g. 11008 (SwiGLU)
    pub num_layers:          usize,   // e.g. 32
    pub num_heads:           usize,   // query heads, e.g. 32
    pub num_kv_heads:        usize,   // GQA key/value heads, e.g. 8
    pub head_dim:            usize,   // hidden_dim / num_heads
    pub max_seq_len:         usize,   // context window, e.g. 32768

    // ── Normalization ─────────────────────────────────────────────────────────
    pub rms_norm_eps:        f32,     // e.g. 1e-5
    pub use_rms_norm:        bool,    // true = RMSNorm, false = LayerNorm

    // ── Position Embeddings ───────────────────────────────────────────────────
    pub rope_theta:          f32,     // e.g. 500000.0 (Llama 3)
    pub rope_scaling:        Option<RopeScaling>,

    // ── Tokenizer ─────────────────────────────────────────────────────────────
    pub bos_token_id:        u32,
    pub eos_token_id:        u32,
    pub pad_token_id:        Option<u32>,
    pub tokenizer_path:      String,

    // ── Quantization ─────────────────────────────────────────────────────────
    pub quantization:        QuantizationConfig,

    // ── Domain-specific ───────────────────────────────────────────────────────
    pub domain:              DomainConfig,

    // ── Inference ─────────────────────────────────────────────────────────────
    pub dtype:               DType,
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub factor:       f32,
    pub low_freq_factor:  f32,
    pub high_freq_factor: f32,
    pub original_max_pos: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub method:       QuantMethod,
    pub bits:         u8,            // 4, 8, or 16
    pub group_size:   usize,         // e.g. 128
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantMethod {
    None,
    GPTQ,
    AWQ,
    BnB,    // bitsandbytes
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainConfig {
    /// Fields of specialisation
    pub domains: Vec<PhysicsDomain>,
    /// Path to NIST CODATA constants JSON
    pub constants_db: Option<String>,
    /// System prompt prefix injected automatically
    pub system_prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PhysicsDomain {
    ClassicalMechanics,
    Thermodynamics,
    Electromagnetism,
    QuantumMechanics,
    StatisticalMechanics,
    Optics,
    Relativity,
    ParticlePhysics,
    NuclearPhysics,
    CondensedMatter,
    // Chemistry
    PhysicalChemistry,
    OrganicChemistry,
    InorganicChemistry,
    Biochemistry,
    ComputationalChemistry,
    // Astrophysics / Astrochemistry
    Astrophysics,
    Astrochemistry,
    CosmicRayPhysics,
    Cosmology,
    PlanetaryScience,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DType { F16, BF16, F32 }

impl ModelConfig {
    /// 7B parameter config (default — fits on 8GB VRAM at f16)
    pub fn physllm_7b() -> Self {
        Self {
            vocab_size:          32768 + 4096,  // base + physics extensions
            domain_vocab_offset: 32768,
            domain_vocab_size:   4096,
            hidden_dim:          4096,
            intermediate_dim:    11008,
            num_layers:          32,
            num_heads:           32,
            num_kv_heads:        8,
            head_dim:            128,
            max_seq_len:         32768,
            rms_norm_eps:        1e-5,
            use_rms_norm:        true,
            rope_theta:          500_000.0,
            rope_scaling:        None,
            bos_token_id:        1,
            eos_token_id:        2,
            pad_token_id:        Some(0),
            tokenizer_path:      "configs/tokenizer.json".into(),
            quantization: QuantizationConfig {
                method:    QuantMethod::None,
                bits:      16,
                group_size: 128,
            },
            domain: DomainConfig {
                domains: vec![
                    PhysicsDomain::QuantumMechanics,
                    PhysicsDomain::Thermodynamics,
                    PhysicsDomain::Astrophysics,
                    PhysicsDomain::Astrochemistry,
                    PhysicsDomain::PhysicalChemistry,
                    PhysicsDomain::ComputationalChemistry,
                    PhysicsDomain::Cosmology,
                ],
                constants_db: Some("configs/nist_codata.json".into()),
                system_prompt: PHYSICS_SYSTEM_PROMPT.into(),
            },
            dtype: DType::F16,
            tie_word_embeddings: true,
        }
    }

    /// 13B config (fits on 24GB VRAM, e.g. RX 7900 XTX)
    pub fn physllm_13b() -> Self {
        let mut cfg = Self::physllm_7b();
        cfg.hidden_dim       = 5120;
        cfg.intermediate_dim = 13824;
        cfg.num_layers       = 40;
        cfg.num_heads        = 40;
        cfg.num_kv_heads     = 8;
        cfg.head_dim         = 128;
        cfg
    }

    pub fn attention_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

const PHYSICS_SYSTEM_PROMPT: &str = r#"You are PhysLLM, a specialist AI assistant with deep expertise in:
- Physics: classical mechanics, quantum mechanics, thermodynamics, electromagnetism,
  statistical mechanics, relativity, condensed matter, particle and nuclear physics
- Chemistry: physical chemistry, organic/inorganic chemistry, computational chemistry,
  reaction kinetics, molecular dynamics
- Astrophysics & Astrochemistry: stellar physics, galactic dynamics, interstellar
  medium chemistry, cosmic ray physics, cosmology, planetary science

You reason rigorously using mathematical formalism where appropriate. You cite
physical constants precisely (SI units by default), express uncertainties clearly,
and distinguish between well-established results and current research frontiers.
When solving problems, show dimensional analysis and sanity checks.
You have access to simulation tools that you can invoke to model physical systems."#;
