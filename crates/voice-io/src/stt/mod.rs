//! stt/mod.rs — Speech-to-Text using Whisper.
//!
//! Two modes:
//!   whisper-local  → whisper.cpp via whisper-rs (runs on CPU or GPU)
//!   whisper-api    → any OpenAI-compatible /v1/audio/transcriptions endpoint
//!
//! Whisper model sizes and approximate performance on CPU:
//!   tiny.en   (~75MB)  —  ~5× real-time on modern CPU, English only
//!   base.en   (~145MB) — ~4× real-time, good accuracy for physics terms
//!   small.en  (~460MB) — ~2× real-time, better for technical vocabulary
//!   medium.en (~1.5GB) — ~1× real-time, excellent accuracy
//!   large-v3  (~3GB)   — ~0.5× real-time, best (use with GPU for real-time)
//!
//! For physics/chemistry, small.en or medium.en is recommended because
//! technical terms like "Hamiltonian", "eigenstate", "astrochemistry" need
//! the larger vocabulary coverage.
//!
//! Model download:
//!   bash scripts/download_whisper.sh small.en

pub mod local;
pub mod api;

use crate::Result;
use serde::{Deserialize, Serialize};

/// A transcribed speech segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptChunk {
    pub text:       String,
    pub start_ms:   u64,
    pub end_ms:     u64,
    pub confidence: f32,
    /// True if this is a partial/streaming result (may be revised)
    pub is_partial: bool,
    /// Language detected (e.g. "en")
    pub language:   String,
}

impl TranscriptChunk {
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Apply basic physics/chemistry post-processing corrections.
    /// Whisper sometimes mishears technical terms — we fix common ones.
    pub fn postprocess(&mut self) {
        let fixes: &[(&str, &str)] = &[
            // Common Whisper mishearings of physics terms
            ("eigen state",    "eigenstate"),
            ("eigen value",    "eigenvalue"),
            ("eigen vector",   "eigenvector"),
            ("Schrodinger",    "Schrödinger"),
            ("Schrodinger's",  "Schrödinger's"),
            ("hamilton Ian",   "Hamiltonian"),
            ("Lagrange Ian",   "Lagrangian"),
            ("Fay Newman",     "Feynman"),
            ("bows on",        "boson"),
            ("fair me on",     "fermion"),
            ("had ron",        "hadron"),
            ("new train oh",   "neutrino"),
            ("Kelvin",         "kelvin"),     // keep lowercase for unit
            ("dark matter",    "dark matter"),
            ("dark energy",    "dark energy"),
            ("astro chemistry","astrochemistry"),
            ("astro physics",  "astrophysics"),
            ("black whole",    "black hole"),
            ("red shift",      "redshift"),
            ("blue shift",     "blueshift"),
            ("Hubble's",       "Hubble"),
            ("Boltzmann",      "Boltzmann"),
            ("plank constant", "Planck constant"),
            ("avocados number","Avogadro's number"),
            ("mole ar",        "molar"),
            ("H 2 O",          "H₂O"),
            ("C O 2",          "CO₂"),
            ("N H 3",          "NH₃"),
        ];

        let mut text = self.text.clone();
        for &(wrong, right) in fixes {
            text = text.replace(wrong, right);
            // Also try title-case version
            let wrong_title: String = wrong.chars().enumerate()
                .map(|(i, c)| if i == 0 { c.to_uppercase().next().unwrap_or(c) } else { c })
                .collect();
            let right_title: String = right.chars().enumerate()
                .map(|(i, c)| if i == 0 { c.to_uppercase().next().unwrap_or(c) } else { c })
                .collect();
            text = text.replace(&wrong_title, &right_title);
        }
        self.text = text;
    }
}

/// Trait implemented by all STT backends.
#[async_trait::async_trait]
pub trait SttBackend: Send + Sync {
    /// Transcribe a full audio segment (16kHz mono f32).
    async fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<TranscriptChunk>;

    /// Whether this backend supports streaming partial results.
    fn supports_streaming(&self) -> bool { false }

    /// Name of this backend (for logging).
    fn name(&self) -> &'static str;
}
