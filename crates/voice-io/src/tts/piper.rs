//! tts/piper.rs — Piper TTS backend (Apache 2.0, fast, good quality offline).
//!
//! Piper is a fast, local neural TTS system by Rhasspy/OHF.
//! Voices are ONNX models (~40-100MB each), Apache 2.0 licensed.
//!
//! Install:
//!   pip install piper-tts
//!   # or download binary: https://github.com/rhasspy/piper/releases
//!
//! Download voices:
//!   python3 -m piper --download-dir models/piper \
//!     --model en_US-lessac-high   # high quality, natural
//!     --model en_GB-alan-medium   # British, good for science content

use crate::{Result, VoiceError};
use super::{TtsBackend, SpeechChunk, VoiceConfig, preprocess_for_tts};
use tracing::debug;

pub struct PiperTts {
    model_path: String,
}

impl PiperTts {
    pub fn new(model_path: &str) -> Self {
        Self { model_path: model_path.to_string() }
    }

    /// Recommended voice for physics content (British, clear, formal)
    pub fn alan_medium() -> Self {
        Self::new("models/piper/en_GB-alan-medium.onnx")
    }

    /// US English, high quality
    pub fn lessac_high() -> Self {
        Self::new("models/piper/en_US-lessac-high.onnx")
    }
}

#[async_trait::async_trait]
impl TtsBackend for PiperTts {
    async fn synthesise(&self, text: &str, voice: &VoiceConfig) -> Result<SpeechChunk> {
        let clean = preprocess_for_tts(text);
        debug!("Piper TTS: {:?}", &clean[..clean.len().min(80)]);

        let tmp_wav = format!("/tmp/physllm_piper_{}.wav", std::process::id());

        // Piper can read from stdin and write WAV to stdout
        let mut child = tokio::process::Command::new("piper")
            .args(["--model", &self.model_path, "--output_file", &tmp_wav,
                   "--length-scale", &(1.0 / voice.speed).to_string()])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| VoiceError::Tts(format!("Piper not found: {e}. Install: pip install piper-tts")))?;

        // Write text to stdin
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(clean.as_bytes()).await
                .map_err(|e| VoiceError::Tts(e.to_string()))?;
        }

        let status = child.wait().await
            .map_err(|e| VoiceError::Tts(e.to_string()))?;

        if !status.success() {
            return Err(VoiceError::Tts("Piper TTS failed".into()));
        }

        let wav_bytes = tokio::fs::read(&tmp_wav).await
            .map_err(|e| VoiceError::Tts(e.to_string()))?;
        let _ = tokio::fs::remove_file(&tmp_wav).await;

        let samples = wav_to_f32(&wav_bytes)?;

        Ok(SpeechChunk {
            samples,
            sample_rate: 22_050,
            text: text.to_string(),
            is_final: true,
        })
    }

    fn name(&self) -> &'static str { "piper" }
}

// ─────────────────────────────────────────────────────────────────────────────
// tts/kokoro.rs — Kokoro TTS (Apache 2.0, very high quality).
//
// Kokoro runs via Python subprocess. Install:
//   pip install kokoro-onnx soundfile
//   # Download model: https://github.com/remsky/Kokoro-FastAPI

pub struct KokoroTts {
    host: String,
    port: u16,
}

impl KokoroTts {
    pub fn new(host: &str, port: u16) -> Self {
        Self { host: host.to_string(), port }
    }
    pub fn local() -> Self { Self::new("127.0.0.1", 8880) }
}

#[async_trait::async_trait]
impl TtsBackend for KokoroTts {
    async fn synthesise(&self, text: &str, voice: &VoiceConfig) -> Result<SpeechChunk> {
        let clean = preprocess_for_tts(text);
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{}:{}/v1/audio/speech", self.host, self.port))
            .json(&serde_json::json!({
                "model": "kokoro",
                "voice": voice.voice_id,
                "input": clean,
                "speed": voice.speed,
                "response_format": "wav",
            }))
            .send().await
            .map_err(|e| VoiceError::Tts(format!("Kokoro unavailable: {e}")))?;

        let wav_bytes = resp.bytes().await
            .map_err(|e| VoiceError::Tts(e.to_string()))?.to_vec();
        let samples = wav_to_f32(&wav_bytes)?;

        Ok(SpeechChunk {
            samples,
            sample_rate: 24_000,
            text: text.to_string(),
            is_final: true,
        })
    }

    fn name(&self) -> &'static str { "kokoro" }
}

// ─────────────────────────────────────────────────────────────────────────────
// tts/api.rs — OpenAI-compatible TTS API backend.

pub struct OpenAiTts {
    client:   reqwest::Client,
    base_url: String,
    api_key:  Option<String>,
}

impl OpenAiTts {
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        Self {
            client:   reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        }
    }
    pub fn openai(api_key: &str) -> Self {
        Self::new("https://api.openai.com", Some(api_key.to_string()))
    }
}

#[async_trait::async_trait]
impl TtsBackend for OpenAiTts {
    async fn synthesise(&self, text: &str, voice: &VoiceConfig) -> Result<SpeechChunk> {
        let clean = preprocess_for_tts(text);
        let mut req = self.client
            .post(format!("{}/v1/audio/speech", self.base_url))
            .json(&serde_json::json!({
                "model": "tts-1",
                "voice": voice.voice_id,
                "input": clean,
                "speed": voice.speed,
                "response_format": "wav",
            }));

        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let wav_bytes = req.send().await
            .map_err(|e| VoiceError::Tts(e.to_string()))?
            .bytes().await
            .map_err(|e| VoiceError::Tts(e.to_string()))?
            .to_vec();

        let samples = wav_to_f32(&wav_bytes)?;
        Ok(SpeechChunk {
            samples, sample_rate: 24_000,
            text: text.to_string(), is_final: true,
        })
    }

    fn name(&self) -> &'static str { "openai-tts" }
}

// ── Shared helper ─────────────────────────────────────────────────────────────

fn wav_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let cursor = std::io::Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| VoiceError::Tts(format!("WAV parse: {e}")))?;
    let spec = reader.spec();
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().filter_map(|s| s.ok()).collect()
        }
        hound::SampleFormat::Int => {
            let max = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / max).collect()
        }
    };
    Ok(samples)
}
