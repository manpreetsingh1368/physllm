//! Stub VAD used when the "vad" feature is disabled (no ONNX Runtime).
//! Falls back to simple energy-based detection.
use crate::{Result, audio::AudioChunk};

pub enum VadState { Silence, Speech { started_ms: u64 }, Trailing { speech_ended_ms: u64, started_ms: u64 } }
impl Clone for VadState { fn clone(&self) -> Self { match self { Self::Silence => Self::Silence, Self::Speech{started_ms:s} => Self::Speech{started_ms:*s}, Self::Trailing{speech_ended_ms:e,started_ms:s} => Self::Trailing{speech_ended_ms:*e,started_ms:*s} } } }
impl PartialEq for VadState { fn eq(&self, other: &Self) -> bool { std::mem::discriminant(self) == std::mem::discriminant(other) } }
impl std::fmt::Debug for VadState { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "VadState") } }

#[derive(Debug, Clone)]
pub struct SpeechSegment { pub samples: Vec<f32>, pub start_ms: u64, pub end_ms: u64, pub confidence: f32 }
impl SpeechSegment { pub fn duration_ms(&self) -> u64 { self.end_ms - self.start_ms } }

pub struct SileroVad { state: VadState, buf: Vec<f32>, started: u64, last_speech: u64 }

impl SileroVad {
    pub fn new(_path: &str) -> Self { Self { state: VadState::Silence, buf: vec![], started: 0, last_speech: 0 } }
    pub fn load(&mut self) -> Result<()> { Ok(()) }
    pub fn reset(&mut self) { self.state = VadState::Silence; self.buf.clear(); }
    pub fn feed(&mut self, chunk: &AudioChunk) -> Result<Option<SpeechSegment>> {
        let rms = if chunk.samples.is_empty() { 0.0 } else {
            (chunk.samples.iter().map(|&s| s*s).sum::<f32>() / chunk.samples.len() as f32).sqrt()
        };
        let is_speech = rms > 0.01;
        let now = chunk.timestamp_ms;
        match &self.state.clone() {
            VadState::Silence => { if is_speech { self.state = VadState::Speech{started_ms:now}; self.buf.clear(); self.started=now; } }
            VadState::Speech{started_ms} => {
                self.buf.extend_from_slice(&chunk.samples);
                if !is_speech { self.state = VadState::Trailing{speech_ended_ms:now,started_ms:*started_ms}; self.last_speech=now; }
                else if now - started_ms > 30_000 { let s=self.emit(self.started,now); self.state=VadState::Silence; return Ok(s); }
            }
            VadState::Trailing{speech_ended_ms,started_ms} => {
                if is_speech { self.buf.extend_from_slice(&chunk.samples); self.state=VadState::Speech{started_ms:*started_ms}; }
                else if now - speech_ended_ms > 800 {
                    let s = if now-self.started>250 { self.emit(self.started,*speech_ended_ms) } else { None };
                    self.state=VadState::Silence; self.buf.clear(); return Ok(s);
                }
            }
        }
        Ok(None)
    }
    fn emit(&mut self, start: u64, end: u64) -> Option<SpeechSegment> {
        if self.buf.is_empty() { return None; }
        Some(SpeechSegment { samples: std::mem::take(&mut self.buf), start_ms: start, end_ms: end, confidence: 0.8 })
    }
}
