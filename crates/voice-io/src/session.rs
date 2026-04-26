//! session.rs — Voice session state and events.

use serde::{Deserialize, Serialize};

/// Events emitted by the voice pipeline (for UI, logging, WebSocket).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    Ready,
    Listening,
    SpeechDetected   { duration_ms: u64 },
    Transcribing,
    Transcribed      { text: String },
    Thinking,
    Responding       { text: String },
    Speaking,
    Error            { message: String },
    ToolCall         { tool: String },
    SearchComplete   { sources: Vec<String>, count: usize },
}

/// A complete voice conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSession {
    pub id:              String,
    pub turns:           Vec<ConversationTurn>,
    pub started_at:      chrono::DateTime<chrono::Utc>,
    pub total_speech_ms: u64,
    pub total_llm_ms:    u64,
    pub total_tts_ms:    u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub user_text:      String,
    pub assistant_text: String,
    pub sources_used:   Vec<String>,
    pub latency_ms:     u64,
    pub timestamp:      chrono::DateTime<chrono::Utc>,
}

impl VoiceSession {
    pub fn new() -> Self {
        Self {
            id:              uuid::Uuid::new_v4().to_string(),
            turns:           Vec::new(),
            started_at:      chrono::Utc::now(),
            total_speech_ms: 0,
            total_llm_ms:    0,
            total_tts_ms:    0,
        }
    }
}
