//! tts/mod.rs — Text-to-Speech with multiple backends.
//!
//! Backends:
//!   EspeakTts   — eSpeak-ng (offline, free, monotone but reliable, ~1ms latency)
//!   KokoroTts   — Kokoro TTS (Apache 2.0, high quality, runs on CPU/GPU)
//!   OpenAiTts   — OpenAI TTS API (paid, very natural)
//!   CoquiTts    — Coqui TTS via Python subprocess (MPL-2.0, good quality)
//!   PiperTts    — Piper TTS (offline, fast, good quality, Apache 2.0)
//!
//! For production:
//!   - Offline use:  Piper or Kokoro (fast, good quality, Apache 2.0)
//!   - Best quality: OpenAI TTS API
//!   - Lowest cost:  eSpeak-ng (zero cost, works anywhere)
//!
//! Physics note: eSpeak handles Greek letters and math notation reasonably
//! well. For better results, pre-process the text to expand abbreviations:
//!   "hbar" → "h-bar"
//!   "kB T" → "k sub B times T"
//!   "∇²ψ"  → "del squared psi"

pub mod espeak;
pub mod piper;
pub mod kokoro;
pub mod api;

use crate::Result;
use serde::{Deserialize, Serialize};

/// A chunk of synthesised speech audio.
#[derive(Debug, Clone)]
pub struct SpeechChunk {
    pub samples:     Vec<f32>,
    pub sample_rate: u32,
    /// The text that was synthesised
    pub text:        String,
    /// True if this is a partial streaming chunk
    pub is_final:    bool,
}

/// TTS voice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    pub voice_id:  String,
    pub speed:     f32,     // 1.0 = normal
    pub pitch:     f32,     // 1.0 = normal
    pub volume:    f32,     // 1.0 = normal
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            voice_id: "en".into(),
            speed:    1.0,
            pitch:    1.0,
            volume:   1.0,
        }
    }
}

/// Trait for TTS backends.
#[async_trait::async_trait]
pub trait TtsBackend: Send + Sync {
    /// Synthesise text to audio.
    async fn synthesise(&self, text: &str, voice: &VoiceConfig) -> Result<SpeechChunk>;

    /// Whether streaming synthesis is supported.
    fn supports_streaming(&self) -> bool { false }

    /// Backend name.
    fn name(&self) -> &'static str;
}

/// Pre-process text for better TTS output with physics content.
pub fn preprocess_for_tts(text: &str) -> String {
    let mut s = text.to_string();

    // Greek letters → spoken form
    let greek = [
        ("α", "alpha"), ("β", "beta"), ("γ", "gamma"), ("δ", "delta"),
        ("ε", "epsilon"), ("ζ", "zeta"), ("η", "eta"), ("θ", "theta"),
        ("λ", "lambda"), ("μ", "mu"), ("ν", "nu"), ("ξ", "xi"),
        ("π", "pi"), ("ρ", "rho"), ("σ", "sigma"), ("τ", "tau"),
        ("φ", "phi"), ("χ", "chi"), ("ψ", "psi"), ("ω", "omega"),
        ("Γ", "Gamma"), ("Δ", "Delta"), ("Θ", "Theta"), ("Λ", "Lambda"),
        ("Σ", "Sigma"), ("Φ", "Phi"), ("Ψ", "Psi"), ("Ω", "Omega"),
        ("ℏ", "h-bar"), ("ℓ", "ell"),
    ];
    for (sym, name) in &greek { s = s.replace(sym, name); }

    // Math operators
    let math = [
        ("∇²",  "del squared"),
        ("∇×",  "curl of"),
        ("∇·",  "divergence of"),
        ("∇",   "del"),
        ("∂/∂t","partial derivative with respect to t"),
        ("∂",   "partial"),
        ("∫",   "integral"),
        ("∑",   "sum"),
        ("∏",   "product"),
        ("∞",   "infinity"),
        ("≈",   "approximately equals"),
        ("≡",   "is defined as"),
        ("≤",   "less than or equal to"),
        ("≥",   "greater than or equal to"),
        ("→",   "yields"),
        ("←",   "comes from"),
        ("⟨",   "expectation of"),
        ("⟩",   "end expectation"),
        ("†",   "dagger"),
        ("★",   "star"),
    ];
    for (sym, word) in &math { s = s.replace(sym, &format!(" {word} ")); }

    // Units and subscripts
    let units = [
        ("cm⁻³", "per cubic centimetre"),
        ("m⁻³",  "per cubic metre"),
        ("s⁻¹",  "per second"),
        ("J·s",  "joule seconds"),
        ("eV",   "electron volts"),
        ("keV",  "kilo electron volts"),
        ("MeV",  "mega electron volts"),
        ("GeV",  "giga electron volts"),
        ("g/mol","grams per mole"),
        ("km/s", "kilometres per second"),
        ("M☉",   "solar masses"),
        ("L☉",   "solar luminosities"),
        ("R☉",   "solar radii"),
        ("AU",   "astronomical units"),
        ("ly",   "light years"),
        ("pc",   "parsecs"),
        ("kpc",  "kiloparsecs"),
        ("Mpc",  "megaparsecs"),
    ];
    for (sym, word) in &units { s = s.replace(sym, &format!(" {word} ")); }

    // Chemical formulae → spoken form
    let chem = [
        ("H₂O",  "water"),
        ("CO₂",  "carbon dioxide"),
        ("NH₃",  "ammonia"),
        ("H₂",   "molecular hydrogen"),
        ("O₂",   "molecular oxygen"),
        ("N₂",   "molecular nitrogen"),
        ("CH₄",  "methane"),
        ("H₂O₂", "hydrogen peroxide"),
        ("HCN",  "hydrogen cyanide"),
        ("H₃⁺",  "trihydrogen cation"),
    ];
    for (formula, name) in &chem { s = s.replace(formula, name); }

    // Clean up multiple spaces
    while s.contains("  ") { s = s.replace("  ", " "); }
    s.trim().to_string()
}
