//! crates/llm-core/src/search_agent.rs
//!
//! Search-augmented generation — an agentic loop that:
//!   1. Sends the user message + search tool schema to the LLM
//!   2. If the LLM emits a tool_call, executes it via SearchRouter
//!   3. Injects the search results back as a tool_result message
//!   4. Repeats until the LLM generates a final text response
//!
//! This implements the standard Anthropic tool-use pattern but fully in Rust,
//! so the LLM can autonomously decide when to search and what to query.

use crate::{PhysLLM, PhysTokenizer, GenerateRequest, SamplingParams, Result, LlmError};
use web_search::{SearchRouter, SearchResponse};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn};

/// A message in the agentic conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub role:    AgentRole,
    pub content: AgentContent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AgentRole { System, User, Assistant, Tool }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentContent {
    Text(String),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

/// A tool invocation emitted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name:   String,
    pub id:     String,
    pub input:  serde_json::Value,
}

/// The result returned to the LLM after executing a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content:     String,
}

/// Output from the search-augmented generation loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub text:          String,
    pub tool_calls:    Vec<ToolCall>,
    pub search_results: Vec<SearchResponse>,
    pub iterations:    usize,
    pub tokens_in:     usize,
    pub tokens_out:    usize,
    pub elapsed_ms:    u64,
}

/// Configuration for the agentic loop.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum tool-use iterations before forcing a final answer
    pub max_iterations:   usize,
    /// Whether to always search before answering
    pub always_search:    bool,
    /// Keywords that trigger automatic search (even if LLM doesn't call the tool)
    pub auto_search_triggers: Vec<String>,
    pub sampling:         SamplingParams,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            always_search:  false,
            auto_search_triggers: vec![
                "latest".into(), "recent".into(), "2024".into(), "2025".into(),
                "current".into(), "new paper".into(), "arxiv".into(),
                "preprint".into(), "published".into(),
            ],
            sampling: SamplingParams {
                max_new_tokens:    1024,
                temperature:       0.3,
                top_p:             0.9,
                top_k:             40,
                repetition_penalty: 1.1,
                stop_sequences:    vec!["</s>".into(), "<|eot_id|>".into()],
                seed:              None,
            },
        }
    }
}

/// The search-augmented agent.
pub struct SearchAgent<'a> {
    model:         &'a PhysLLM,
    tokenizer:     &'a PhysTokenizer,
    search_router: &'a SearchRouter,
    config:        AgentConfig,
}

impl<'a> SearchAgent<'a> {
    pub fn new(
        model:         &'a PhysLLM,
        tokenizer:     &'a PhysTokenizer,
        search_router: &'a SearchRouter,
        config:        AgentConfig,
    ) -> Self {
        Self { model, tokenizer, search_router, config }
    }

    /// Run the full agentic loop for a user query.
    pub async fn run(&self, user_query: &str, system: Option<&str>) -> Result<AgentResponse> {
        let t0 = std::time::Instant::now();
        let sys = system.unwrap_or(&self.model.config.domain.system_prompt);

        let mut conversation: Vec<AgentMessage> = vec![
            AgentMessage {
                role:    AgentRole::System,
                content: AgentContent::Text(self.build_system_prompt(sys)),
            },
            AgentMessage {
                role:    AgentRole::User,
                content: AgentContent::Text(user_query.to_string()),
            },
        ];

        let mut tool_calls_made:  Vec<ToolCall>       = Vec::new();
        let mut search_results:   Vec<SearchResponse> = Vec::new();
        let mut total_tokens_in   = 0usize;
        let mut total_tokens_out  = 0usize;
        let mut final_text        = String::new();

        // Pre-search: if query has auto-search triggers, search before LLM call
        if self.config.always_search || self.should_auto_search(user_query) {
            info!("Auto-searching for: {user_query}");
            if let Ok(sr) = self.search_router.search(user_query).await {
                let ctx = sr.llm_context.clone();
                search_results.push(sr);
                // Inject search results into the conversation as a system note
                conversation.push(AgentMessage {
                    role:    AgentRole::Tool,
                    content: AgentContent::ToolResult(ToolResult {
                        tool_use_id: "auto_search".into(),
                        content:     ctx,
                    }),
                });
            }
        }

        // Agentic loop
        for iteration in 0..self.config.max_iterations {
            debug!("Agent iteration {iteration}");

            // Format conversation into a single prompt string
            let prompt = self.format_conversation(&conversation);
            total_tokens_in += prompt.split_whitespace().count();

            // Run LLM forward pass
            let tokens = self.tokenizer.encode(&prompt)?;
            let logits = self.model.forward(&tokens, 0)?;

            // Sample and decode (simplified — in production uses streaming generate())
            let response_text = self.sample_response(&logits, &tokens)?;
            total_tokens_out += response_text.split_whitespace().count();

            // Parse response: check if it contains a tool call
            match self.parse_tool_call(&response_text) {
                Some(tool_call) => {
                    info!("LLM called tool: {} with input: {}", tool_call.name, tool_call.input);
                    tool_calls_made.push(tool_call.clone());

                    // Execute the tool call
                    let tool_result = self.execute_tool(&tool_call, &mut search_results).await;

                    // Add assistant message (the tool call) to conversation
                    conversation.push(AgentMessage {
                        role:    AgentRole::Assistant,
                        content: AgentContent::ToolCall(tool_call.clone()),
                    });

                    // Add tool result to conversation
                    conversation.push(AgentMessage {
                        role:    AgentRole::Tool,
                        content: AgentContent::ToolResult(ToolResult {
                            tool_use_id: tool_call.id.clone(),
                            content:     tool_result,
                        }),
                    });
                }
                None => {
                    // No tool call — this is the final response
                    final_text = response_text;
                    conversation.push(AgentMessage {
                        role:    AgentRole::Assistant,
                        content: AgentContent::Text(final_text.clone()),
                    });
                    info!("Agent finished after {} iterations", iteration + 1);
                    break;
                }
            }
        }

        // If we hit max iterations without a final answer, generate one with full context
        if final_text.is_empty() {
            warn!("Max iterations reached — forcing final answer");
            let prompt = self.format_conversation(&conversation)
                + "\n\nBased on the search results above, provide a comprehensive answer:";
            let tokens = self.tokenizer.encode(&prompt)?;
            let logits = self.model.forward(&tokens, 0)?;
            final_text = self.sample_response(&logits, &tokens)?;
        }

        Ok(AgentResponse {
            text:           final_text,
            tool_calls:     tool_calls_made,
            search_results,
            iterations:     conversation.iter().filter(|m| m.role == AgentRole::Assistant).count(),
            tokens_in:      total_tokens_in,
            tokens_out:     total_tokens_out,
            elapsed_ms:     t0.elapsed().as_millis() as u64,
        })
    }

    /// Execute a tool call and return the result as a string.
    async fn execute_tool(
        &self,
        call: &ToolCall,
        search_results: &mut Vec<SearchResponse>,
    ) -> String {
        match call.name.as_str() {
            "web_search" | "search_arxiv" => {
                let query = call.input["query"].as_str().unwrap_or("").to_string();
                match self.search_router.search(&query).await {
                    Ok(sr) => {
                        let ctx = sr.llm_context.clone();
                        search_results.push(sr);
                        ctx
                    }
                    Err(e) => format!("Search failed: {e}"),
                }
            }
            "fetch_url" => {
                let url = call.input["url"].as_str().unwrap_or("").to_string();
                match self.search_router.search(&url).await {
                    Ok(sr) => {
                        let ctx = sr.llm_context.clone();
                        search_results.push(sr);
                        ctx
                    }
                    Err(e) => format!("Fetch failed: {e}"),
                }
            }
            "lookup_constant" => {
                let symbol = call.input["symbol"].as_str().unwrap_or("");
                let db = domain_physics::ConstantsDB::built_in();
                match db.get(symbol) {
                    Ok(c) => format!(
                        "{} ({}): {} ± {} {} [relative uncertainty: {:.2e}]",
                        c.name, c.symbol, c.value, c.uncertainty, c.unit,
                        c.relative_uncertainty()
                    ),
                    Err(e) => format!("Constant not found: {e}"),
                }
            }
            "lookup_molecule" => {
                let name = call.input["name"].as_str().unwrap_or("");
                let query = format!("molecular properties {name}");
                match self.search_router.search(&query).await {
                    Ok(sr) => sr.llm_context.clone(),
                    Err(e) => format!("Molecule lookup failed: {e}"),
                }
            }
            _ => format!("Unknown tool: {}", call.name),
        }
    }

    /// Build system prompt with tool schema injected.
    fn build_system_prompt(&self, base: &str) -> String {
        format!(
            "{base}\n\n\
             You have access to the following tools. Call them using this JSON format \
             when you need current information, specific papers, or data lookup:\n\n\
             <tool_call>\n\
             {{\"name\": \"tool_name\", \"id\": \"call_1\", \"input\": {{...}}}}\n\
             </tool_call>\n\n\
             Available tools:\n{tools}\n\n\
             When you have enough information, respond directly without using a tool.",
            tools = web_search::SEARCH_TOOLS
        )
    }

    /// Format the conversation history into a single prompt string.
    fn format_conversation(&self, msgs: &[AgentMessage]) -> String {
        let mut out = String::from("<|begin_of_text|>");
        for msg in msgs {
            let (role_tag, content) = match &msg.role {
                AgentRole::System => ("system", match &msg.content {
                    AgentContent::Text(t) => t.clone(),
                    _ => String::new(),
                }),
                AgentRole::User => ("user", match &msg.content {
                    AgentContent::Text(t) => t.clone(),
                    _ => String::new(),
                }),
                AgentRole::Assistant => ("assistant", match &msg.content {
                    AgentContent::Text(t) => t.clone(),
                    AgentContent::ToolCall(tc) => format!(
                        "<tool_call>\n{}\n</tool_call>",
                        serde_json::to_string_pretty(tc).unwrap_or_default()
                    ),
                    _ => String::new(),
                }),
                AgentRole::Tool => ("tool_result", match &msg.content {
                    AgentContent::ToolResult(tr) => tr.content.clone(),
                    _ => String::new(),
                }),
            };
            out.push_str(&format!(
                "<|start_header_id|>{role_tag}<|end_header_id|>\n\n{content}<|eot_id|>"
            ));
        }
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        out
    }

    /// Parse a tool call from the model's raw output.
    fn parse_tool_call(&self, text: &str) -> Option<ToolCall> {
        // Look for <tool_call>...</tool_call> block
        let start = text.find("<tool_call>")? + "<tool_call>".len();
        let end   = text[start..].find("</tool_call>")? + start;
        let json  = text[start..end].trim();

        let val: serde_json::Value = serde_json::from_str(json).ok()?;
        Some(ToolCall {
            name:  val["name"].as_str()?.to_string(),
            id:    val["id"].as_str().unwrap_or("call_1").to_string(),
            input: val["input"].clone(),
        })
    }

    /// Very simple sampling from logits (production uses full generate()).
    fn sample_response(&self, logits: &[f32], context_tokens: &[u32]) -> Result<String> {
        // Find argmax (greedy) — production replaces this with full sampling
        let next_id = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(2); // fallback to EOS

        // In production: run full autoregressive decode loop
        // For now return a placeholder that the tests can work with
        self.tokenizer.decode(&[next_id])
            .map(|s| if s.is_empty() {
                "[PhysLLM response — load model weights to activate]".into()
            } else { s })
    }

    /// Check if the query contains signals that warrant automatic pre-search.
    fn should_auto_search(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        self.config.auto_search_triggers
            .iter()
            .any(|trigger| q.contains(trigger.as_str()))
    }
}
