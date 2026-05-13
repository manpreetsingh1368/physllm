//! tts/espeak.rs — eSpeak-ng TTS backend.
//!
//! eSpeak-ng is a compact open-source speech synthesiser.
//! Install: sudo apt install espeak-ng
//!
//! Advantages: zero latency, zero cost, works fully offline, handles
//! technical terms reasonably well, supports 100+ languages.
//! Disadvantage: robotic voice quality.

use crate::{Result, VoiceError};
use super::{TtsBackend, SpeechChunk, VoiceConfig, preprocess_for_tts};
use tracing::debug;

pub struct EspeakTts {
    pub voice:       String,
    pub sample_rate: u32,
}

impl EspeakTts {
    pub fn new() -> Self {
        Self { voice: "en".into(), sample_rate: 22_050 }
    }

    pub fn with_voice(voice: &str) -> Self {
        Self { voice: voice.to_string(), sample_rate: 22_050 }
    }

    fn check_installed() -> bool {
        std::process::Command::new("espeak-ng")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

#[async_trait::async_trait]
impl TtsBackend for EspeakTts {
    async fn synthesise(&self, text: &str, voice: &VoiceConfig) -> Result<SpeechChunk> {
        if !Self::check_installed() {
            return Err(VoiceError::Tts(
                "eSpeak-ng not found. Install with: sudo apt install espeak-ng".into()
            ));
        }

        let clean = preprocess_for_tts(text);
        debug!("eSpeak TTS: {:?}", &clean[..clean.len().min(80)]);

        let tmp_wav = format!("/tmp/physllm_tts_{}.wav", std::process::id());
        let speed_wpm = (175.0 * voice.speed) as u32;  // default ~175 WPM

        let status = tokio::process::Command::new("espeak-ng")
            .args([
                "-v", &format!("{}+m3", voice.voice_id),  // male voice variant 3
                "-s", &speed_wpm.to_string(),
                "-p", &((50.0 * voice.pitch) as u32).to_string(),   // pitch 0-99, default 50
                "-a", &((200.0 * voice.volume) as u32).to_string(), // amplitude 0-200
                "-w", &tmp_wav,
                &clean,
            ])
            .status().await
            .map_err(|e| VoiceError::Tts(e.to_string()))?;

        if !status.success() {
            return Err(VoiceError::Tts("eSpeak-ng process failed".into()));
        }

        // Read WAV file
        let wav_bytes = tokio::fs::read(&tmp_wav).await
            .map_err(|e| VoiceError::Tts(e.to_string()))?;
        let _ = tokio::fs::remove_file(&tmp_wav).await;

        let samples = parse_wav_to_f32(&wav_bytes)?;

        Ok(SpeechChunk {
            samples,
            sample_rate: self.sample_rate,
            text:        text.to_string(),
            is_final:    true,
        })
    }

    fn name(&self) -> &'static str { "espeak-ng" }
}

fn parse_wav_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    use std::io::Cursor;
    let cursor = Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| VoiceError::Tts(e.to_string()))?;
    let samples: Vec<f32> = match reader.spec().sample_format {
        hound::SampleFormat::Int => {
            let bits = reader.spec().bits_per_sample;
            let max = (1i32 << (bits - 1)) as f32;
            reader.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
    };
    Ok(samples)
}

impl Default for EspeakTts {
    fn default() -> Self { Self::new() }
}
