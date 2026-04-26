//! ws_handler.rs — WebSocket handler for browser voice interaction.
//!
//! The browser sends raw PCM or Opus audio chunks over WebSocket.
//! This handler runs VAD + STT + LLM + TTS and streams events back.
//!
//! Browser-side (JavaScript):
//!   const ws = new WebSocket("ws://localhost:8080/v1/voice");
//!   const stream = await navigator.mediaDevices.getUserMedia({audio: true});
//!   // See scripts/voice_client.html for the full browser client

use crate::{
    Result, VoiceError,
    vad::SileroVad,
    stt::SttBackend,
    tts::{TtsBackend, VoiceConfig},
    session::SessionEvent,
    audio::TARGET_SAMPLE_RATE,
};
use std::sync::Arc;
use tokio_tungstenite::tungstenite::Message;
use tokio::sync::mpsc;
use tracing::{info, debug, warn};
use serde::{Deserialize, Serialize};
use futures::{SinkExt, StreamExt};

/// Message types from browser to server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Raw PCM f32 samples (little-endian bytes, 16kHz mono)
    Audio  { data: Vec<u8> },
    /// Opus-encoded audio chunk
    Opus   { data: Vec<u8>, sequence: u32 },
    /// Control messages
    Start  { language: Option<String> },
    Stop,
    SetVoice { voice_id: String, speed: f32 },
}

/// Message types from server to browser.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Event        { event: SessionEvent },
    /// TTS audio chunk (WAV bytes, base64-encoded for JSON transport)
    Audio        { data: String, sample_rate: u32, is_final: bool },
    /// Or send raw binary TTS audio (more efficient)
    Transcript   { text: String, is_partial: bool, confidence: f32 },
    Response     { text: String, sources: Vec<String> },
    Error        { message: String },
}

/// Per-connection voice handler state.
pub struct VoiceWsHandler {
    stt:     Arc<dyn SttBackend>,
    tts:     Arc<dyn TtsBackend>,
    vad:     SileroVad,
    voice:   VoiceConfig,
    language: String,
    // LLM API endpoint (PhysLLM API server)
    api_base: String,
    http:    reqwest::Client,
}

impl VoiceWsHandler {
    pub fn new(
        stt:      Arc<dyn SttBackend>,
        tts:      Arc<dyn TtsBackend>,
        vad_path: &str,
        api_base: &str,
    ) -> Self {
        let mut vad = SileroVad::new(vad_path);
        let _ = vad.load(); // best-effort
        Self {
            stt, tts, vad,
            voice:    VoiceConfig::default(),
            language: "en".into(),
            api_base: api_base.to_string(),
            http:     reqwest::Client::new(),
        }
    }

    /// Handle a WebSocket connection. Called from the Axum ws upgrade handler.
    pub async fn handle(
        mut self,
        ws: tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>
        >,
    ) {
        let (mut sink, mut stream) = ws.split();
        info!("Voice WebSocket connected");
        let _ = self.send_event(&mut sink, SessionEvent::Ready).await;
        let _ = self.send_event(&mut sink, SessionEvent::Listening).await;

        let mut pcm_buf: Vec<f32> = Vec::new();
        let chunk_samples = (TARGET_SAMPLE_RATE * 30 / 1000) as usize; // 30ms

        while let Some(msg) = stream.next().await {
            match msg {
                Ok(Message::Binary(bytes)) => {
                    // Raw binary PCM f32 LE from browser
                    let samples: Vec<f32> = bytes.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    pcm_buf.extend(samples);

                    // Process in 30ms chunks
                    while pcm_buf.len() >= chunk_samples {
                        let chunk_data: Vec<f32> = pcm_buf.drain(..chunk_samples).collect();
                        let chunk = crate::audio::AudioChunk {
                            samples: chunk_data,
                            sample_rate: TARGET_SAMPLE_RATE,
                            channels: 1,
                            timestamp_ms: 0,
                        };

                        match self.vad.feed(&chunk) {
                            Ok(Some(segment)) => {
                                info!("Browser speech: {:.1}s", segment.duration_ms() as f32 / 1000.0);
                                let _ = self.send_event(&mut sink, SessionEvent::Transcribing).await;

                                // STT
                                match self.stt.transcribe(&segment.samples, Some(&self.language)).await {
                                    Ok(t) if !t.is_empty() => {
                                        info!("Browser transcript: {:?}", t.text);
                                        let _ = sink.send(Message::Text(
                                            serde_json::to_string(&ServerMessage::Transcript {
                                                text:       t.text.clone(),
                                                is_partial: false,
                                                confidence: t.confidence,
                                            }).unwrap_or_default().into()
                                        )).await;

                                        let _ = self.send_event(&mut sink, SessionEvent::Thinking).await;

                                        // LLM
                                        let resp = self.call_llm(&t.text).await;
                                        let resp_text = resp.0;
                                        let sources   = resp.1;

                                        let _ = sink.send(Message::Text(
                                            serde_json::to_string(&ServerMessage::Response {
                                                text: resp_text.clone(), sources: sources.clone()
                                            }).unwrap_or_default().into()
                                        )).await;

                                        let _ = self.send_event(&mut sink, SessionEvent::Speaking).await;

                                        // TTS
                                        match self.tts.synthesise(&resp_text, &self.voice).await {
                                            Ok(speech) => {
                                                // Send audio as base64 JSON or raw binary
                                                let wav_bytes = f32_to_wav_bytes(&speech.samples, speech.sample_rate);
                                                let b64 = base64_encode(&wav_bytes);
                                                let _ = sink.send(Message::Text(
                                                    serde_json::to_string(&ServerMessage::Audio {
                                                        data:        b64,
                                                        sample_rate: speech.sample_rate,
                                                        is_final:    true,
                                                    }).unwrap_or_default().into()
                                                )).await;
                                            }
                                            Err(e) => warn!("TTS error: {e}"),
                                        }

                                        self.vad.reset();
                                        let _ = self.send_event(&mut sink, SessionEvent::Listening).await;
                                    }
                                    Ok(_)  => debug!("Empty transcript, ignoring"),
                                    Err(e) => warn!("STT error: {e}"),
                                }
                            }
                            Ok(None) => {}
                            Err(e) => warn!("VAD error: {e}"),
                        }
                    }
                }

                Ok(Message::Text(text)) => {
                    match serde_json::from_str::<serde_json::Value>(&text) {
                        Ok(v) => {
                            if let Some("set_voice") = v["type"].as_str() {
                                if let Some(speed) = v["speed"].as_f64() {
                                    self.voice.speed = speed as f32;
                                }
                                if let Some(vid) = v["voice_id"].as_str() {
                                    self.voice.voice_id = vid.to_string();
                                }
                            }
                            if let Some("set_language") = v["type"].as_str() {
                                if let Some(lang) = v["language"].as_str() {
                                    self.language = lang.to_string();
                                }
                            }
                        }
                        Err(_) => {}
                    }
                }

                Ok(Message::Close(_)) => {
                    info!("Voice WebSocket closed");
                    break;
                }

                Err(e) => { warn!("WS error: {e}"); break; }
                _ => {}
            }
        }
    }

    async fn send_event<S>(&self, sink: &mut S, event: SessionEvent) -> Result<()>
    where S: SinkExt<Message> + Unpin, S::Error: std::fmt::Display
    {
        let msg = serde_json::to_string(&ServerMessage::Event { event })
            .map_err(|e| VoiceError::WebSocket(e.to_string()))?;
        sink.send(Message::Text(msg.into())).await
            .map_err(|e| VoiceError::WebSocket(e.to_string()))
    }

    async fn call_llm(&self, text: &str) -> (String, Vec<String>) {
        let payload = serde_json::json!({
            "prompt": text, "search_query": text,
            "sampling": {"temperature": 0.4, "max_new_tokens": 300}
        });
        match self.http
            .post(format!("{}/v1/generate/search", self.api_base))
            .json(&payload).send().await
        {
            Ok(r) => {
                if let Ok(body) = r.json::<serde_json::Value>().await {
                    let text = body["text"].as_str().unwrap_or("").to_string();
                    let sources = body["search"]["sources_used"].as_array()
                        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    return (text, sources);
                }
            }
            Err(e) => warn!("LLM call failed: {e}"),
        }
        ("Could not reach the PhysLLM model.".into(), vec![])
    }
}

fn f32_to_wav_bytes(samples: &[f32], rate: u32) -> Vec<u8> {
    use std::io::Write;
    let mut buf = Vec::new();
    let data_len = (samples.len() * 2) as u32;
    // WAV header
    buf.write_all(b"RIFF").unwrap();
    buf.write_all(&(36 + data_len).to_le_bytes()).unwrap();
    buf.write_all(b"WAVE").unwrap();
    buf.write_all(b"fmt ").unwrap();
    buf.write_all(&16u32.to_le_bytes()).unwrap();   // chunk size
    buf.write_all(&1u16.to_le_bytes()).unwrap();    // PCM
    buf.write_all(&1u16.to_le_bytes()).unwrap();    // mono
    buf.write_all(&rate.to_le_bytes()).unwrap();
    buf.write_all(&(rate * 2).to_le_bytes()).unwrap(); // byte rate
    buf.write_all(&2u16.to_le_bytes()).unwrap();    // block align
    buf.write_all(&16u16.to_le_bytes()).unwrap();   // bits per sample
    buf.write_all(b"data").unwrap();
    buf.write_all(&data_len.to_le_bytes()).unwrap();
    for &s in samples {
        let i16_s = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        buf.write_all(&i16_s.to_le_bytes()).unwrap();
    }
    buf
}

fn base64_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 { chunk[1] as usize } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as usize } else { 0 };
        out.push(CHARS[(b0 >> 2)] as char);
        out.push(CHARS[((b0 & 3) << 4) | (b1 >> 4)] as char);
        out.push(if chunk.len() > 1 { CHARS[((b1 & 0xf) << 2) | (b2 >> 6)] as char } else { '=' });
        out.push(if chunk.len() > 2 { CHARS[b2 & 0x3f] as char } else { '=' });
    }
    out
}
