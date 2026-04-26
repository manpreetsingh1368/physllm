//! stt/api.rs — Whisper via any OpenAI-compatible /v1/audio/transcriptions endpoint.
//!
//! Compatible with:
//!   - OpenAI Whisper API (paid)
//!   - faster-whisper server (free, self-hosted, 4× faster than whisper.cpp)
//!   - Groq Whisper (free tier, very fast)
//!   - Local whisper-server (https://github.com/ahmetoner/whisper-asr-webservice)

use crate::{Result, VoiceError, audio::save_wav};
use super::{SttBackend, TranscriptChunk};
use serde::Deserialize;
use tracing::debug;

pub struct WhisperApi {
    client:   reqwest::Client,
    base_url: String,
    api_key:  Option<String>,
    model:    String,
}

#[derive(Debug, Deserialize)]
struct WhisperApiResponse {
    text: String,
    #[serde(default)]
    language: String,
}

impl WhisperApi {
    pub fn new(base_url: &str, api_key: Option<String>, model: &str) -> Self {
        Self {
            client:   reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model:    model.to_string(),
        }
    }

    /// OpenAI-compatible Whisper endpoint
    pub fn openai(api_key: &str) -> Self {
        Self::new("https://api.openai.com", Some(api_key.to_string()), "whisper-1")
    }

    /// Groq Whisper (free, extremely fast)
    pub fn groq(api_key: &str) -> Self {
        Self::new("https://api.groq.com/openai", Some(api_key.to_string()), "whisper-large-v3")
    }

    /// Local faster-whisper server (self-hosted, no API key)
    pub fn local_server(host: &str, port: u16) -> Self {
        Self::new(&format!("http://{host}:{port}"), None, "base")
    }
}

#[async_trait::async_trait]
impl SttBackend for WhisperApi {
    async fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<TranscriptChunk> {
        // Write samples to temp WAV file
        let tmp_path = format!("/tmp/physllm_stt_{}.wav", std::process::id());
        save_wav(samples, 16_000, &tmp_path)?;
        let audio_bytes = tokio::fs::read(&tmp_path).await
            .map_err(|e| VoiceError::Stt(e.to_string()))?;
        let _ = tokio::fs::remove_file(&tmp_path).await;

        // Build multipart form
        let part = reqwest::multipart::Part::bytes(audio_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| VoiceError::Stt(e.to_string()))?;

        let mut form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model",            self.model.clone())
            .text("response_format",  "json")
            .text("prompt",
                "Physics, chemistry, astrophysics. Schrödinger, Hamiltonian, eigenstate.");

        if let Some(lang) = language {
            form = form.text("language", lang.to_string());
        }

        let mut req = self.client
            .post(format!("{}/v1/audio/transcriptions", self.base_url))
            .multipart(form);

        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp: WhisperApiResponse = req.send().await
            .map_err(|e| VoiceError::Stt(e.to_string()))?
            .json().await
            .map_err(|e| VoiceError::Stt(e.to_string()))?;

        debug!("STT API: {:?}", resp.text);

        let mut chunk = TranscriptChunk {
            text:       resp.text.trim().to_string(),
            start_ms:   0,
            end_ms:     (samples.len() as u64 * 1000) / 16_000,
            confidence: 0.9,
            is_partial: false,
            language:   resp.language.unwrap_or_else(|| language.unwrap_or("en").to_string()),
        };
        chunk.postprocess();
        Ok(chunk)
    }

    fn name(&self) -> &'static str { "whisper-api" }
}
