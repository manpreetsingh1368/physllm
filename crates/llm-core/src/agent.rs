//! crates/llm-core/src/agent.rs
//!
//! PhysLLM Agent Loop — orchestrates tool-use across web search and simulations.
//!
//! Flow:
//!   User message
//!     → LLM generates response (may contain tool calls)
//!     → Parse tool calls
//!     → Dispatch: web_search | simulate_* | lookup_constant | lookup_molecule
//!     → Inject tool results back as context
//!     → LLM generates final answer
//!     → Return to user
//!
//! Supports multi-turn tool use (the LLM can call multiple tools per query).

use crate::{GenerateRequest, GenerateResponse, SamplingParams, PhysTokenizer, Result, LlmError};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, debug, warn};

/// A parsed tool call from the LLM's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id:   String,
    pub name: String,
    pub args: serde_json::Value,
}

/// Result from executing a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub name:    String,
    pub content: String,
    pub error:   Option<String>,
}

/// A single turn in the agent conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTurn {
    pub role:         AgentRole,
    pub content:      String,
    pub tool_calls:   Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentRole { System, User, Assistant, Tool }

/// Full agent response including all intermediate steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub final_answer:   String,
    pub turns:          Vec<AgentTurn>,
    pub tools_called:   Vec<String>,
    pub total_tokens:   usize,
    pub wall_time_ms:   u64,
    pub search_results: Vec<SearchSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSummary {
    pub query:   String,
    pub sources: Vec<String>,
    pub count:   usize,
}

/// Configuration for the agent loop.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum tool-use rounds before forcing a direct answer
    pub max_rounds:        usize,
    /// Whether to auto-search for any physics query
    pub auto_search:       bool,
    /// Minimum confidence below which the agent searches before answering
    pub search_threshold:  f32,
    /// System prompt override (uses ModelConfig default if None)
    pub system_prompt:     Option<String>,
    pub sampling:          SamplingParams,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_rounds:       5,
            auto_search:      true,
            search_threshold: 0.7,
            system_prompt:    None,
            sampling: SamplingParams {
                temperature:       0.3,   // lower temp for factual physics
                max_new_tokens:    1024,
                top_p:             0.9,
                top_k:             40,
                repetition_penalty: 1.1,
                stop_sequences:    vec!["<|eot_id|>".into(), "</tool_call>".into()],
                seed:              None,
            },
        }
    }
}

/// The agent system prompt — tells the LLM how to use its tools.
pub const AGENT_SYSTEM_PROMPT: &str = r#"You are PhysLLM, a specialist AI for physics, chemistry, astrophysics, and astrochemistry with access to real-time web search and physics simulation tools.

## Available tools

### Web search tools
- `web_search(query)` — search arXiv, Semantic Scholar, NIST, NASA ADS, PubChem automatically
- `search_arxiv(query, category?)` — search arXiv preprints directly
- `fetch_url(url)` — fetch full text of any URL or arXiv paper
- `lookup_constant(symbol)` — NIST CODATA physical constant (hbar, kB, G, NA, e, me, c, sigma...)
- `lookup_molecule(name)` — molecular formula, MW, SMILES from PubChem + NIST

### Simulation tools
- `simulate_nbody(...)` — gravitational N-body (RK4), presets: solar_system, binary_star, three_body
- `simulate_quantum(...)` — 1D Schrödinger (Crank-Nicolson): infinite_well, harmonic, double_well, step
- `simulate_md(...)` — Lennard-Jones molecular dynamics
- `simulate_kinetics(...)` — chemical kinetics ODE, presets: ozone_depletion, h2_o2_combustion, ism_hydrogen
- `simulate_stellar(...)` — stellar evolution HR track
- `simulate_astrochem(...)` — ISM astrochemical network
- `simulate_thermodynamics(...)` — equation of state: ideal_gas, van_der_waals, blackbody

## Tool call format

When you need a tool, output EXACTLY this format (nothing else on that turn):
```
<tool_call>
{"name": "tool_name", "id": "call_1", "args": {...}}
</tool_call>
```

You may call multiple tools:
```
<tool_call>
{"name": "web_search", "id": "call_1", "args": {"query": "JWST detection of CO2 in exoplanet atmosphere"}}
</tool_call>
<tool_call>
{"name": "lookup_constant", "id": "call_2", "args": {"symbol": "hbar"}}
</tool_call>
```

## When to use tools

ALWAYS use `web_search` or `search_arxiv` when:
- Asked about specific papers, recent discoveries, or results from after 2024
- Uncertain about a specific numerical value (use `lookup_constant` instead if it's a fundamental constant)
- Asked about specific astronomical objects, missions, or experiments

ALWAYS use `lookup_constant` instead of reciting constant values from memory — your training may have rounding errors.

Use simulation tools when the user asks to model, simulate, or visualise a physical system.

## Answer format

After receiving tool results, synthesise a clear, rigorous answer:
- Show mathematical derivations when relevant
- Always give units (SI by default)
- Cite sources as [Source N] when drawing on search results
- Distinguish between established theory and recent/uncertain results
- Use LaTeX-style notation for equations: E = mc² or ∇²ψ = (2m/ℏ²)(V-E)ψ"#;

/// Parse tool calls from LLM output text.
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut remaining = text;

    while let Some(start) = remaining.find("<tool_call>") {
        let after_open = &remaining[start + "<tool_call>".len()..];
        if let Some(end) = after_open.find("</tool_call>") {
            let json_str = after_open[..end].trim();
            match serde_json::from_str::<serde_json::Value>(json_str) {
                Ok(v) => {
                    let id   = v["id"].as_str().unwrap_or("call_0").to_string();
                    let name = v["name"].as_str().unwrap_or("").to_string();
                    let args = v["args"].clone();
                    if !name.is_empty() {
                        calls.push(ToolCall { id, name, args });
                    }
                }
                Err(e) => warn!("Failed to parse tool call JSON: {e}\n{json_str}"),
            }
            remaining = &after_open[end + "</tool_call>".len()..];
        } else {
            break;
        }
    }

    calls
}

/// Extract the non-tool-call text from LLM output (the human-readable part).
pub fn extract_text(output: &str) -> String {
    let mut result = String::new();
    let mut remaining = output;

    loop {
        match remaining.find("<tool_call>") {
            Some(pos) => {
                result.push_str(&remaining[..pos]);
                if let Some(end) = remaining[pos..].find("</tool_call>") {
                    remaining = &remaining[pos + end + "</tool_call>".len()..];
                } else {
                    break;
                }
            }
            None => {
                result.push_str(remaining);
                break;
            }
        }
    }

    result.trim().to_string()
}

/// Format tool results back into the conversation context.
pub fn format_tool_results(results: &[ToolResult]) -> String {
    let mut s = String::new();
    for r in results {
        s.push_str(&format!("<tool_result id=\"{}\" name=\"{}\">\n", r.call_id, r.name));
        if let Some(err) = &r.error {
            s.push_str(&format!("ERROR: {err}\n"));
        } else {
            s.push_str(&r.content);
            s.push('\n');
        }
        s.push_str("</tool_result>\n\n");
    }
    s
}

/// Determine if a query likely needs a web search before answering.
pub fn needs_search(query: &str) -> bool {
    let q = query.to_lowercase();

    // Always search for recent/specific claims
    let search_triggers = [
        "latest", "recent", "2024", "2025", "2026",
        "new study", "new paper", "published", "discovered",
        "arxiv", "nature", "science", "physical review",
        "what is the current", "how many", "when was",
        "james webb", "jwst", "ligo", "cern", "lhc",
        "specific value", "measured", "experimental",
    ];
    if search_triggers.iter().any(|t| q.contains(t)) { return true; }

    // Never search for these (model knows them well)
    let no_search = [
        "explain", "derive", "prove", "what is the formula",
        "what does", "how does", "describe the concept",
        "what are the units", "what is the definition",
    ];
    if no_search.iter().any(|t| q.starts_with(t)) { return false; }

    // Search for specific numerical questions
    if q.contains("what is the value") || q.contains("give me the") {
        return true;
    }

    false
}

/// Build the full agent system prompt with tool descriptions.
pub fn build_system_prompt(base: Option<&str>) -> String {
    format!("{}\n\n{}", base.unwrap_or(""), AGENT_SYSTEM_PROMPT)
}

/// Format an agent conversation history into a prompt string.
pub fn format_conversation(turns: &[AgentTurn]) -> String {
    let mut prompt = String::new();
    for turn in turns {
        match turn.role {
            AgentRole::System => {
                prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                    turn.content
                ));
            }
            AgentRole::User => {
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
                    turn.content
                ));
            }
            AgentRole::Assistant => {
                let mut content = turn.content.clone();
                // Append any tool calls made in this turn
                for tc in &turn.tool_calls {
                    content.push_str(&format!(
                        "\n<tool_call>\n{}\n</tool_call>",
                        serde_json::json!({"name": tc.name, "id": tc.id, "args": tc.args})
                    ));
                }
                prompt.push_str(&format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
                    content
                ));
            }
            AgentRole::Tool => {
                // Tool results are injected as a user turn
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n\
                     <tool_results>\n{}\n</tool_results><|eot_id|>",
                    turn.content
                ));
            }
        }
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_calls() {
        let text = r#"I'll search for this.
<tool_call>
{"name": "web_search", "id": "call_1", "args": {"query": "dark matter detection 2024"}}
</tool_call>
And also look up the constant.
<tool_call>
{"name": "lookup_constant", "id": "call_2", "args": {"symbol": "G"}}
</tool_call>"#;

        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[1].name, "lookup_constant");
        assert_eq!(calls[1].args["symbol"].as_str(), Some("G"));
    }

    #[test]
    fn test_extract_text() {
        let text = "Here is my thinking.\n<tool_call>\n{\"name\":\"web_search\",\"id\":\"c1\",\"args\":{}}\n</tool_call>\nDone.";
        let extracted = extract_text(text);
        assert_eq!(extracted, "Here is my thinking.\nDone.");
    }

    #[test]
    fn test_needs_search() {
        assert!(needs_search("What is the latest JWST discovery?"));
        assert!(needs_search("arxiv 2401.12345"));
        assert!(!needs_search("explain the Schrödinger equation"));
        assert!(!needs_search("derive the Lorentz transformation"));
    }
}
