//! pipeline.rs — The main voice interaction pipeline.
//!
//! Orchestrates all components into a continuous conversation loop:
//!
//!   ┌─────────────────────────────────────────────────────────┐
//!   │                   Voice Pipeline Loop                   │
//!   │                                                         │
//!   │  MicCapture ──► Resampler ──► VAD                      │
//!   │                                │ speech segment         │
//!   │                            Whisper STT                  │
//!   │                                │ transcript             │
//!   │                         PhysLLM + Search                │
//!   │                                │ response text          │
//!   │                            TTS Engine                   │
//!   │                                │ audio                  │
//!   │                           Speaker ◄───┘                 │
//!   └─────────────────────────────────────────────────────────┘

use crate::{
    Result, VoiceError,
    audio::{AudioDevices, MicCapture, Speaker, AudioChunk},
    vad::{SileroVad, SpeechSegment},
    stt::SttBackend,
    tts::{TtsBackend, VoiceConfig},
    session::{VoiceSession, SessionEvent},
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, debug, warn};
use serde::{Deserialize, Serialize};

/// Full configuration for the voice pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Whisper model path directory
    pub whisper_model_dir:  String,
    /// Whisper model size ("tiny.en", "base.en", "small.en", "medium.en")
    pub whisper_model:      String,
    /// Whether to use GPU for Whisper
    pub whisper_gpu:        bool,
    /// Silero VAD model path
    pub vad_model_path:     String,
    /// TTS backend to use
    pub tts_backend:        TtsBackendKind,
    /// TTS voice settings
    pub tts_voice:          crate::tts::VoiceConfig,
    /// Language code
    pub language:           String,
    /// Whether to search the web before answering
    pub use_web_search:     bool,
    /// Whether to mute the mic while TTS is playing (prevents echo)
    pub echo_cancellation:  bool,
    /// Maximum response length in tokens
    pub max_response_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TtsBackendKind {
    Espeak,
    Piper { model_path: String },
    Kokoro { host: String, port: u16 },
    OpenAi { api_key: String },
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            whisper_model_dir:   "models/whisper".into(),
            whisper_model:       "small.en".into(),
            whisper_gpu:         false,
            vad_model_path:      "models/silero_vad.onnx".into(),
            tts_backend:         TtsBackendKind::Piper {
                model_path: "models/piper/en_GB-alan-medium.onnx".into()
            },
            tts_voice:           crate::tts::VoiceConfig::default(),
            language:            "en".into(),
            use_web_search:      true,
            echo_cancellation:   true,
            max_response_tokens: 300,
        }
    }
}

/// The assembled voice pipeline.
pub struct VoicePipeline {
    config:  VoiceConfig,
    devices: AudioDevices,
    stt:     Arc<dyn SttBackend>,
    tts:     Arc<dyn TtsBackend>,
    vad:     SileroVad,
}

impl VoicePipeline {
    /// Build the pipeline from config, loading all models.
    pub fn build(config: VoiceConfig) -> Result<Self> {
        info!("Building voice pipeline...");

        // STT
        let stt: Arc<dyn SttBackend> = {
            use crate::stt::local::{WhisperLocal, WhisperModel};
            let model = match config.whisper_model.as_str() {
                "tiny.en"   => WhisperModel::TinyEn,
                "base.en"   => WhisperModel::BaseEn,
                "small.en"  => WhisperModel::SmallEn,
                "medium.en" => WhisperModel::MediumEn,
                "large-v3"  => WhisperModel::LargeV3,
                other       => WhisperModel::Custom(other.to_string()),
            };
            Arc::new(WhisperLocal::load(model, &config.whisper_model_dir, config.whisper_gpu)?)
        };

        // TTS
        let tts: Arc<dyn TtsBackend> = match &config.tts_backend {
            TtsBackendKind::Espeak => {
                Arc::new(crate::tts::espeak::EspeakTts::new())
            }
            TtsBackendKind::Piper { model_path } => {
                Arc::new(crate::tts::piper::PiperTts::new(model_path))
            }
            TtsBackendKind::Kokoro { host, port } => {
                Arc::new(crate::tts::piper::KokoroTts::new(host, *port))
            }
            TtsBackendKind::OpenAi { api_key } => {
                Arc::new(crate::tts::piper::OpenAiTts::openai(api_key))
            }
        };

        info!("STT: {}  TTS: {}", stt.name(), tts.name());

        // VAD
        let mut vad = SileroVad::new(&config.vad_model_path);
        match vad.load() {
            Ok(_)  => info!("VAD: Silero loaded"),
            Err(e) => warn!("VAD: falling back to energy-based ({e})"),
        }

        let devices = AudioDevices::default_devices()?;

        Ok(Self { config, devices, stt, tts, vad })
    }

    /// Run the voice pipeline indefinitely.
    /// `on_event` receives SessionEvents for UI/logging.
    pub async fn run(
        mut self,
        mut on_event: impl FnMut(SessionEvent) + Send + 'static,
    ) -> Result<()> {
        info!("Voice pipeline started. Speak to interact with PhysLLM.");
        on_event(SessionEvent::Ready);

        let mut mic = MicCapture::start(&self.devices)?;
        let speaker = Speaker::start(&self.devices)?;
        let mut is_speaking = false;   // true = TTS is playing

        loop {
            // Receive audio chunk from microphone
            let raw_chunk = match mic.rx.recv().await {
                Some(c) => c,
                None    => break,
            };

            // Skip mic input while TTS is playing (echo cancellation)
            if is_speaking && self.config.echo_cancellation {
                continue;
            }

            // Resample to 16kHz
            let chunk = mic.resample(&raw_chunk)?;

            // VAD: detect speech segments
            if let Some(segment) = self.vad.feed(&chunk)? {
                info!("Speech segment: {:.1}s", segment.duration_ms() as f32 / 1000.0);
                on_event(SessionEvent::SpeechDetected {
                    duration_ms: segment.duration_ms(),
                });

                // STT: transcribe
                on_event(SessionEvent::Transcribing);
                let transcript = self.stt.transcribe(
                    &segment.samples,
                    Some(&self.config.language),
                ).await?;

                if transcript.is_empty() {
                    debug!("Empty transcript — ignoring");
                    continue;
                }

                info!("Transcript: {:?}", transcript.text);
                on_event(SessionEvent::Transcribed { text: transcript.text.clone() });

                // LLM: generate response
                on_event(SessionEvent::Thinking);
                let response = self.generate_response(&transcript.text).await;

                on_event(SessionEvent::Responding { text: response.clone() });
                info!("Response: {:?}", &response[..response.len().min(100)]);

                // TTS: synthesise
                is_speaking = true;
                self.vad.reset(); // reset VAD so we don't pick up our own voice

                on_event(SessionEvent::Speaking);
                match self.tts.synthesise(&response, &self.config.tts_voice).await {
                    Ok(speech) => {
                        let sample_count = speech.samples.len();
                        speaker.play(speech.samples).await;
                        // Approximate playback duration
                        let playback_ms = (sample_count as u64 * 1000)
                            / speech.sample_rate as u64;
                        tokio::time::sleep(std::time::Duration::from_millis(playback_ms + 200)).await;
                    }
                    Err(e) => warn!("TTS failed: {e}"),
                }

                is_speaking = false;
                on_event(SessionEvent::Listening);
            }
        }

        Ok(())
    }

    /// Generate a response for the given user text.
    /// In production: calls PhysLLM model.forward() + generate().
    async fn generate_response(&self, user_text: &str) -> String {
        // For now: call the API server if running, else return placeholder
        let client = reqwest::Client::new();
        let endpoint = if self.config.use_web_search {
            "http://localhost:8080/v1/generate/search"
        } else {
            "http://localhost:8080/v1/generate"
        };

        let payload = serde_json::json!({
            "prompt":       user_text,
            "search_query": user_text,
            "sampling": {
                "temperature":    0.4,
                "max_new_tokens": self.config.max_response_tokens,
            }
        });

        match client.post(endpoint).json(&payload).send().await {
            Ok(resp) => {
                match resp.json::<serde_json::Value>().await {
                    Ok(body) => {
                        body["text"].as_str()
                            .unwrap_or("I couldn't generate a response.")
                            .to_string()
                    }
                    Err(e) => format!("Parse error: {e}"),
                }
            }
            Err(_) => {
                // Fallback if API server isn't running
                format!("You asked: {}. The PhysLLM API server is not running. \
                         Start it with: cargo run --release -p api-server", user_text)
            }
        }
    }
}
