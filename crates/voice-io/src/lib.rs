//! voice-io — Real-time voice interaction for PhysLLM.
//!
//! Full pipeline:
//!
//!   Microphone (cpal)
//!       │ raw f32 PCM @ device rate
//!   Resampler (rubato)
//!       │ 16kHz mono f32
//!   VAD (Silero)
//!       │ speech segments only
//!   Whisper STT
//!       │ transcript text
//!   PhysLLM + Web Search
//!       │ response text (streamed)
//!   TTS Engine
//!       │ audio chunks (f32 PCM)
//!   Speaker (cpal)
//!
//! Browser path (WebSocket):
//!   Browser mic (MediaRecorder / WebAudio)
//!       │ Opus/PCM chunks over WebSocket
//!   WS handler → VAD → Whisper → LLM → TTS → WS → Browser speaker

pub mod audio;
#[cfg_attr(not(feature = "vad"), path = "vad_stub.rs")]
pub mod vad;
pub mod stt;
pub mod tts;
pub mod pipeline;
pub mod ws_handler;
pub mod session;

pub use pipeline::{VoicePipeline, VoiceConfig};
pub use session::{VoiceSession, SessionEvent};
pub use stt::TranscriptChunk;
pub use tts::SpeechChunk;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VoiceError {
    #[error("Audio device error: {0}")]
    AudioDevice(String),
    #[error("STT error: {0}")]
    Stt(String),
    #[error("TTS error: {0}")]
    Tts(String),
    #[error("VAD error: {0}")]
    Vad(String),
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("WebSocket error: {0}")]
    WebSocket(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, VoiceError>;
