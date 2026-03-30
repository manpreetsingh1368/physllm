//! api-server — Axum REST + WebSocket server for PhysLLM.
//!
//! Endpoints:
//!   POST  /v1/generate          — text completion
//!   POST  /v1/simulate          — direct simulation request
//!   WS    /v1/stream            — streaming generation
//!   GET   /v1/models            — model info
//!   GET   /v1/health            — health check
//!   GET   /v1/constants/:symbol — NIST constant lookup
//!   POST  /v1/chemistry/mw      — molecular weight calculator

use axum::{
    extract::{Path, State, WebSocketUpgrade},
    extract::ws::{WebSocket, Message},
    routing::{get, post},
    Json, Router, response::IntoResponse,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{info, error};

use llm_core::{GenerateRequest, GenerateResponse, SamplingParams};
use sim_agent::{SimAgent, SimRequest};
use domain_physics::{ConstantsDB, ChemicalFormula};

// ── App state ─────────────────────────────────────────────────────────────────

pub struct AppState {
    pub sim_agent:    SimAgent,
    pub constants_db: ConstantsDB,
    // model: Arc<PhysLLM>,  // uncomment when weights are loaded
}

type SharedState = Arc<RwLock<AppState>>;

// ── Main entry point ──────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("physllm=debug,tower_http=info")
        .json()
        .init();

    let state = Arc::new(RwLock::new(AppState {
        sim_agent:    SimAgent::new(),
        constants_db: ConstantsDB::built_in(),
    }));

    let app = Router::new()
        .route("/v1/health",             get(health))
        .route("/v1/models",             get(list_models))
        .route("/v1/generate",           post(generate))
        .route("/v1/simulate",           post(simulate))
        .route("/v1/stream",             get(stream_ws))
        .route("/v1/constants/:symbol",  get(get_constant))
        .route("/v1/chemistry/mw",       post(molecular_weight))
        .route("/v1/tools",              get(list_tools))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:8080";
    info!("PhysLLM API server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "physllm",
        "version": env!("CARGO_PKG_VERSION"),
        "gpu_backend": if cfg!(feature = "rocm") { "ROCm/HIP" } else { "CPU" },
    }))
}

async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "models": [
            { "id": "physllm-7b",  "description": "7B parameter physics/chemistry LLM", "context": 32768 },
            { "id": "physllm-13b", "description": "13B parameter physics/chemistry LLM", "context": 32768 },
        ]
    }))
}

async fn list_tools() -> impl IntoResponse {
    (StatusCode::OK, sim_agent::SIMULATION_TOOLS)
}

#[derive(Debug, Deserialize)]
struct GenerateReq {
    prompt:   String,
    system:   Option<String>,
    sampling: Option<SamplingParams>,
}

async fn generate(
    State(state): State<SharedState>,
    Json(req): Json<GenerateReq>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    // In production this calls model.forward() + generate()
    // Here we return a stub that shows the request was parsed correctly
    info!("generate: prompt len={}", req.prompt.len());

    let resp = GenerateResponse {
        text: format!(
            "[PhysLLM stub] Received: '{}'. \
             In production this runs the transformer inference. \
             Load weights via model.safetensors and call llm_core::generate().",
            &req.prompt[..req.prompt.len().min(80)]
        ),
        tokens_in:     req.prompt.split_whitespace().count(),
        tokens_out:    20,
        finish_reason: llm_core::FinishReason::Eos,
        time_ms:       0,
    };
    Ok(Json(resp))
}

async fn simulate(
    State(state): State<SharedState>,
    Json(req): Json<SimRequest>,
) -> Result<Json<sim_agent::SimResult>, (StatusCode, String)> {
    let st = state.read().await;
    st.sim_agent.run(req).await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

async fn get_constant(
    State(state): State<SharedState>,
    Path(symbol): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let st = state.read().await;
    st.constants_db.get(&symbol)
        .map(|c| Json(serde_json::json!({
            "symbol":      c.symbol,
            "name":        c.name,
            "value":       c.value,
            "uncertainty": c.uncertainty,
            "unit":        c.unit,
            "relative_uncertainty": c.relative_uncertainty(),
        })))
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))
}

#[derive(Deserialize)]
struct MwReq { formula: String }

async fn molecular_weight(
    Json(req): Json<MwReq>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let formula = ChemicalFormula::parse(&req.formula)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let mw = formula.molecular_weight()
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(serde_json::json!({
        "formula":      req.formula,
        "hill":         formula.hill_notation(),
        "elements":     formula.elements,
        "molar_mass_g_per_mol": mw,
    })))
}

// ── WebSocket streaming ───────────────────────────────────────────────────────

async fn stream_ws(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: SharedState) {
    info!("WebSocket connection established");

    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
                let req: Result<GenerateReq, _> = serde_json::from_str(&text);
                match req {
                    Ok(r) => {
                        // Stream tokens back — in production this uses generate() with a channel
                        let response = format!(
                            "{{\"token\": \"[PhysLLM streaming for: {}...]\", \"done\": true}}",
                            &r.prompt[..r.prompt.len().min(40)]
                        );
                        let _ = socket.send(Message::Text(response.into())).await;
                    }
                    Err(e) => {
                        let _ = socket.send(Message::Text(
                            format!("{{\"error\": \"{e}\"}}").into()
                        )).await;
                    }
                }
            }
            Message::Close(_) => {
                info!("WebSocket closed by client");
                break;
            }
            _ => {}
        }
    }
}
