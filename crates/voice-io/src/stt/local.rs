//! stt/local.rs — Local Whisper inference via whisper-rs (whisper.cpp bindings).
//!
//! whisper.cpp uses:
//!   - CPU: AVX2 / AVX512 acceleration (most modern x86)
//!   - GPU: CUDA / ROCm / Metal via OpenCL
//!   - OpenCL: works with AMD GPUs without full ROCm install
//!
//! Build whisper.cpp with ROCm (for AMD GPU acceleration):
//!   cd vendor/whisper.cpp
//!   cmake -B build -DWHISPER_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release
//!   cmake --build build --config Release -j$(nproc)

use crate::{Result, VoiceError};
use super::{SttBackend, TranscriptChunk};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use std::sync::Arc;
use tracing::{info, debug};

/// Whisper model sizes.
#[derive(Debug, Clone, PartialEq)]
pub enum WhisperModel {
    TinyEn,
    BaseEn,
    SmallEn,
    MediumEn,
    LargeV3,
    Custom(String),
}

impl WhisperModel {
    pub fn filename(&self) -> &str {
        match self {
            Self::TinyEn   => "ggml-tiny.en.bin",
            Self::BaseEn   => "ggml-base.en.bin",
            Self::SmallEn  => "ggml-small.en.bin",
            Self::MediumEn => "ggml-medium.en.bin",
            Self::LargeV3  => "ggml-large-v3.bin",
            Self::Custom(p) => p.as_str(),
        }
    }

    pub fn default_model_dir() -> String {
        "models/whisper".to_string()
    }
}

/// Local Whisper STT backend.
pub struct WhisperLocal {
    ctx:        Arc<WhisperContext>,
    model:      WhisperModel,
    use_gpu:    bool,
}

impl WhisperLocal {
    /// Load a Whisper model. `model_dir` should contain the .bin file.
    pub fn load(model: WhisperModel, model_dir: &str, use_gpu: bool) -> Result<Self> {
        let path = format!("{}/{}", model_dir, model.filename());
        if !std::path::Path::new(&path).exists() {
            return Err(VoiceError::ModelNotLoaded(format!(
                "Whisper model not found: {path}\n\
                 Download with: bash scripts/download_whisper.sh {}",
                model.filename()
            )));
        }

        info!("Loading Whisper model: {path} (gpu={use_gpu})");
        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(&path, params)
            .map_err(|e| VoiceError::ModelNotLoaded(e.to_string()))?;

        info!("Whisper model loaded OK");
        Ok(Self { ctx: Arc::new(ctx), model, use_gpu })
    }
}

#[async_trait::async_trait]
impl SttBackend for WhisperLocal {
    async fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<TranscriptChunk> {
        let ctx = self.ctx.clone();
        let lang = language.unwrap_or("en").to_string();
        let samples = samples.to_vec();

        // Whisper inference is CPU-bound — run in blocking thread pool
        let result = tokio::task::spawn_blocking(move || {
            let mut state = ctx.create_state()
                .map_err(|e| VoiceError::Stt(e.to_string()))?;

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_language(Some(&lang));
            params.set_translate(false);
            params.set_print_realtime(false);
            params.set_print_progress(false);
            params.set_print_timestamps(false);
            params.set_print_special(false);
            params.set_suppress_blank(true);
            params.set_suppress_non_speech_tokens(true);
            // Physics-domain initial prompt helps with technical vocabulary
            params.set_initial_prompt(
                "Physics, chemistry, astrophysics, astrochemistry. \
                 Technical terms: Schrödinger, Hamiltonian, eigenstate, \
                 Boltzmann, Avogadro, Planck, astrochemistry, photon, \
                 fermion, boson, quark, hadron, neutrino, redshift."
            );
            params.set_no_context(false);
            params.set_single_segment(false);
            params.set_temperature(0.0);  // greedy for better accuracy

            state.full(params, &samples)
                .map_err(|e| VoiceError::Stt(e.to_string()))?;

            let n_segments = state.full_n_segments()
                .map_err(|e| VoiceError::Stt(e.to_string()))?;

            let mut text = String::new();
            let mut t0 = 0i64;
            let mut t1 = 0i64;

            for i in 0..n_segments {
                let seg = state.full_get_segment_text(i)
                    .map_err(|e| VoiceError::Stt(e.to_string()))?;
                if i == 0 { t0 = state.full_get_segment_t0(i).unwrap_or(0); }
                t1 = state.full_get_segment_t1(i).unwrap_or(0);
                text.push_str(&seg);
            }

            // whisper.cpp timestamps are in units of 10ms
            let start_ms = (t0 * 10) as u64;
            let end_ms   = (t1 * 10) as u64;

            Ok::<TranscriptChunk, VoiceError>(TranscriptChunk {
                text:       text.trim().to_string(),
                start_ms,
                end_ms,
                confidence: 0.9,  // whisper.cpp doesn't expose token-level probs easily
                is_partial: false,
                language:   lang,
            })
        }).await.map_err(|e| VoiceError::Stt(e.to_string()))??;

        let mut chunk = result;
        chunk.postprocess();

        debug!("STT: {:?}", chunk.text);
        Ok(chunk)
    }

    fn name(&self) -> &'static str { "whisper-local" }
}
