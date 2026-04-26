//! vad.rs — Voice Activity Detection using Silero VAD (ONNX Runtime).
//!
//! Silero VAD is a lightweight neural VAD model (~1MB) that runs in real-time
//! on CPU. It outputs a probability 0.0–1.0 of speech presence per 30ms frame.
//!
//! Model download:
//!   wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
//!   -O models/silero_vad.onnx
//!
//! The VAD segments audio into speech chunks and silence gaps, allowing Whisper
//! to only process actual speech — reducing latency and compute by 3–10×.

use crate::{Result, VoiceError, audio::AudioChunk};
use ndarray::{Array1, Array2, Array3};
use std::path::Path;
use tracing::{debug, trace};

/// Probability threshold above which a frame is considered speech.
const SPEECH_THRESHOLD: f32 = 0.5;
/// Minimum speech duration to consider a complete utterance (ms)
const MIN_SPEECH_MS:    u64 = 250;
/// How long silence must last before ending an utterance (ms)
const SILENCE_TIMEOUT_MS: u64 = 800;
/// Maximum utterance length before forcing a cut (ms)
const MAX_UTTERANCE_MS: u64 = 30_000;

/// State of the VAD state machine.
#[derive(Debug, Clone, PartialEq)]
pub enum VadState {
    Silence,
    Speech { started_ms: u64 },
    Trailing { speech_ended_ms: u64, started_ms: u64 },
}

/// A detected speech segment.
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub samples:    Vec<f32>,
    pub start_ms:   u64,
    pub end_ms:     u64,
    pub confidence: f32,
}

impl SpeechSegment {
    pub fn duration_ms(&self) -> u64 { self.end_ms - self.start_ms }
}

/// Silero VAD engine.
pub struct SileroVad {
    /// ONNX session (lazy-loaded)
    session:  Option<ort::Session>,
    model_path: String,
    /// Internal RNN state (h and c tensors)
    h: Array3<f32>,   // [2, 1, 64]
    c: Array3<f32>,   // [2, 1, 64]
    state:    VadState,
    /// Accumulated samples for current utterance
    utterance_buf: Vec<f32>,
    utterance_confidences: Vec<f32>,
}

impl SileroVad {
    pub fn new(model_path: &str) -> Self {
        Self {
            session:    None,
            model_path: model_path.to_string(),
            h:          Array3::zeros([2, 1, 64]),
            c:          Array3::zeros([2, 1, 64]),
            state:      VadState::Silence,
            utterance_buf: Vec::new(),
            utterance_confidences: Vec::new(),
        }
    }

    /// Load the ONNX model (call once before processing).
    pub fn load(&mut self) -> Result<()> {
        if !Path::new(&self.model_path).exists() {
            return Err(VoiceError::ModelNotLoaded(format!(
                "Silero VAD model not found at {}. Download with:\n\
                 wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx \\\n\
                 -O {}",
                self.model_path, self.model_path
            )));
        }

        let session = ort::Session::builder()
            .map_err(|e| VoiceError::Vad(e.to_string()))?
            .with_optimization_level(ort::GraphOptimizationLevel::All)
            .map_err(|e| VoiceError::Vad(e.to_string()))?
            .commit_from_file(&self.model_path)
            .map_err(|e| VoiceError::Vad(e.to_string()))?;

        self.session = Some(session);
        Ok(())
    }

    /// Process a 30ms audio chunk (480 samples at 16kHz).
    /// Returns a speech probability 0.0–1.0.
    pub fn process_chunk(&mut self, samples: &[f32]) -> Result<f32> {
        // If model not loaded, use simple energy-based VAD as fallback
        if self.session.is_none() {
            return Ok(self.energy_vad(samples));
        }

        let session = self.session.as_ref().unwrap();
        let n = samples.len();

        // Input: [1, n] f32
        let input = Array2::from_shape_vec([1, n], samples.to_vec())
            .map_err(|e| VoiceError::Vad(e.to_string()))?;

        // Run inference
        let outputs = session.run(ort::inputs![
            "input"   => input.view(),
            "sr"      => Array1::from_vec(vec![16000i64]).view(),
            "h"       => self.h.view(),
            "c"       => self.c.view(),
        ].map_err(|e| VoiceError::Vad(e.to_string()))?)
            .map_err(|e| VoiceError::Vad(e.to_string()))?;

        // Output: speech_prob [1, 1], hn [2, 1, 64], cn [2, 1, 64]
        let prob  = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| VoiceError::Vad(e.to_string()))?;
        let hn    = outputs[1].try_extract_tensor::<f32>()
            .map_err(|e| VoiceError::Vad(e.to_string()))?;
        let cn    = outputs[2].try_extract_tensor::<f32>()
            .map_err(|e| VoiceError::Vad(e.to_string()))?;

        // Update RNN state
        self.h = Array3::from_shape_vec([2, 1, 64], hn.view().iter().cloned().collect())
            .unwrap_or_else(|_| Array3::zeros([2, 1, 64]));
        self.c = Array3::from_shape_vec([2, 1, 64], cn.view().iter().cloned().collect())
            .unwrap_or_else(|_| Array3::zeros([2, 1, 64]));

        Ok(prob.view()[[0, 0]])
    }

    /// Simple RMS energy-based VAD (fallback when model not available).
    fn energy_vad(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() { return 0.0; }
        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        // Map RMS to 0–1 range (empirically tuned)
        (rms / 0.02).clamp(0.0, 1.0)
    }

    /// Feed audio chunk through VAD state machine.
    /// Returns a completed SpeechSegment if an utterance just ended.
    pub fn feed(&mut self, chunk: &AudioChunk) -> Result<Option<SpeechSegment>> {
        let prob = self.process_chunk(&chunk.samples)?;
        let is_speech = prob >= SPEECH_THRESHOLD;
        let now = chunk.timestamp_ms;

        trace!("VAD: t={now}ms prob={prob:.2} state={:?}", self.state);

        match &self.state.clone() {
            VadState::Silence => {
                if is_speech {
                    self.state = VadState::Speech { started_ms: now };
                    self.utterance_buf.clear();
                    self.utterance_confidences.clear();
                    // Include a short pre-buffer (not implemented here for brevity;
                    // in production keep a ring buffer of the last 200ms)
                }
            }

            VadState::Speech { started_ms } => {
                self.utterance_buf.extend_from_slice(&chunk.samples);
                self.utterance_confidences.push(prob);

                let duration = now - started_ms;

                if !is_speech {
                    // Start trailing silence timer
                    self.state = VadState::Trailing {
                        speech_ended_ms: now,
                        started_ms: *started_ms,
                    };
                } else if duration >= MAX_UTTERANCE_MS {
                    // Force cut — utterance too long
                    return Ok(self.emit_segment(*started_ms, now));
                }
            }

            VadState::Trailing { speech_ended_ms, started_ms } => {
                if is_speech {
                    // Speech resumed — go back to speech state
                    self.utterance_buf.extend_from_slice(&chunk.samples);
                    self.utterance_confidences.push(prob);
                    self.state = VadState::Speech { started_ms: *started_ms };
                } else {
                    let silence_duration = now - speech_ended_ms;
                    let total_speech = speech_ended_ms - started_ms;

                    if silence_duration >= SILENCE_TIMEOUT_MS && total_speech >= MIN_SPEECH_MS {
                        // Utterance complete
                        let seg = self.emit_segment(*started_ms, *speech_ended_ms);
                        self.state = VadState::Silence;
                        return Ok(seg);
                    } else if silence_duration >= SILENCE_TIMEOUT_MS {
                        // Too short — discard
                        self.state = VadState::Silence;
                        self.utterance_buf.clear();
                    }
                }
            }
        }

        Ok(None)
    }

    fn emit_segment(&mut self, start_ms: u64, end_ms: u64) -> Option<SpeechSegment> {
        if self.utterance_buf.is_empty() { return None; }
        let avg_confidence = if self.utterance_confidences.is_empty() { 0.8 }
            else { self.utterance_confidences.iter().sum::<f32>() / self.utterance_confidences.len() as f32 };
        let seg = SpeechSegment {
            samples:    std::mem::take(&mut self.utterance_buf),
            start_ms,
            end_ms,
            confidence: avg_confidence,
        };
        debug!("VAD: emitting segment {}ms–{}ms ({:.1}s) conf={:.2}",
               start_ms, end_ms, seg.duration_ms() as f32 / 1000.0, avg_confidence);
        Some(seg)
    }

    /// Reset VAD state (e.g. after TTS playback to avoid capturing own voice).
    pub fn reset(&mut self) {
        self.state = VadState::Silence;
        self.utterance_buf.clear();
        self.utterance_confidences.clear();
        self.h = Array3::zeros([2, 1, 64]);
        self.c = Array3::zeros([2, 1, 64]);
    }
}
