//! api-server/src/main.rs — PhysLLM REST + WebSocket server with web search.

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
use tracing::{info, warn};

use llm_core::{GenerateResponse, SamplingParams, FinishReason};
use sim_agent::{SimAgent, SimRequest};
use domain_physics::{ConstantsDB, ChemicalFormula};
use web_search::{SearchRouter, SearchConfig, SearchResponse};

pub struct AppState {
    pub sim_agent:     SimAgent,
    pub constants_db:  ConstantsDB,
    pub search_router: SearchRouter,
}

type SharedState = Arc<RwLock<AppState>>;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "physllm=debug,tower_http=info,web_search=debug".into())
        )
        .json()
        .init();

    let search_config = SearchConfig {
        brave_api_key:     std::env::var("BRAVE_API_KEY").ok(),
        serp_api_key:      std::env::var("SERP_API_KEY").ok(),
        nasa_ads_key:      std::env::var("NASA_ADS_KEY").ok(),
        max_results:       5,
        cache_ttl_s:       3600,
        request_timeout_s: 15,
        full_page_fetch:   std::env::var("FULL_PAGE_FETCH").is_ok(),
        max_page_chars:    8_000,
    };

    info!("Brave search: {}", search_config.brave_api_key.is_some());
    info!("NASA ADS:     {}", search_config.nasa_ads_key.is_some());

    let state = Arc::new(RwLock::new(AppState {
        sim_agent:     SimAgent::new(),
        constants_db:  ConstantsDB::built_in(),
        search_router: SearchRouter::new(search_config),
    }));

    let app = Router::new()
        .route("/v1/health",            get(health))
        .route("/v1/models",            get(list_models))
        .route("/v1/tools",             get(list_tools))
        .route("/v1/generate",          post(generate))
        .route("/v1/generate/search",   post(generate_with_search))
        .route("/v1/stream",            get(stream_ws))
        .route("/v1/simulate",          post(simulate))
        .route("/v1/search",            post(search))
        .route("/v1/search/arxiv",      post(search_arxiv))
        .route("/v1/fetch",             post(fetch_url_handler))
        .route("/v1/constants/:symbol", get(get_constant))
        .route("/v1/chemistry/mw",      post(molecular_weight))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = std::env::var("PHYSLLM_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    info!("PhysLLM API server on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok", "service": "physllm",
        "version": env!("CARGO_PKG_VERSION"),
        "gpu_backend": if cfg!(feature = "rocm") { "ROCm/HIP" } else { "CPU" },
        "features": { "web_search": true, "simulation": true }
    }))
}

async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "models": [
            {"id":"physllm-7b","context":32768},
            {"id":"physllm-13b","context":32768}
        ]
    }))
}

async fn list_tools() -> impl IntoResponse {
    let mut all: Vec<serde_json::Value> = Vec::new();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(sim_agent::SIMULATION_TOOLS) {
        if let Some(arr) = v.as_array() { all.extend(arr.clone()); }
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(web_search::SEARCH_TOOLS) {
        if let Some(arr) = v.as_array() { all.extend(arr.clone()); }
    }
    (StatusCode::OK, Json(all))
}

#[derive(Debug, Deserialize)]
struct GenerateReq {
    prompt:       String,
    system:       Option<String>,
    sampling:     Option<SamplingParams>,
    use_search:   Option<bool>,
    search_query: Option<String>,
}

#[derive(Debug, Serialize)]
struct SearchAugmentedResponse {
    text:       String,
    tokens_in:  usize,
    tokens_out: usize,
    search:     SearchResponse,
}

async fn generate(Json(req): Json<GenerateReq>) -> Json<GenerateResponse> {
    info!("generate: prompt_len={}", req.prompt.len());
    Json(stub_generate(&req.prompt))
}

async fn generate_with_search(
    State(state): State<SharedState>,
    Json(req): Json<GenerateReq>,
) -> Result<Json<SearchAugmentedResponse>, (StatusCode, String)> {
    let query = req.search_query.as_deref().unwrap_or(&req.prompt);
    info!("generate_with_search: query={query:?}");
    let st = state.read().await;
    let search_resp = st.search_router.search(query).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let augmented = format!("{}\n\n{}", search_resp.llm_context, req.prompt);
    let gen = stub_generate(&augmented);
    Ok(Json(SearchAugmentedResponse {
        text: gen.text, tokens_in: gen.tokens_in, tokens_out: gen.tokens_out,
        search: search_resp,
    }))
}

fn stub_generate(prompt: &str) -> GenerateResponse {
    GenerateResponse {
        text: format!(
            "[PhysLLM] Received {} chars. Load safetensors weights to activate generation.",
            prompt.len()
        ),
        tokens_in: prompt.split_whitespace().count(),
        tokens_out: 20,
        finish_reason: FinishReason::Eos,
        time_ms: 0,
    }
}

#[derive(Debug, Deserialize)]
struct SearchReq {
    query:           String,
    max_results:     Option<usize>,
    fetch_full_text: Option<bool>,
}

async fn search(
    State(state): State<SharedState>,
    Json(req): Json<SearchReq>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let st = state.read().await;
    st.search_router.search(&req.query).await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

#[derive(Debug, Deserialize)]
struct ArxivReq { query: String, category: Option<String>, max_results: Option<usize> }

async fn search_arxiv(
    State(state): State<SharedState>,
    Json(req): Json<ArxivReq>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let query = match &req.category {
        Some(cat) => format!("{} cat:{cat}", req.query),
        None      => req.query.clone(),
    };
    let st = state.read().await;
    st.search_router.search(&query).await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

#[derive(Debug, Deserialize)]
struct FetchReq { url: String, max_chars: Option<usize> }

async fn fetch_url_handler(
    State(state): State<SharedState>,
    Json(req): Json<FetchReq>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let st = state.read().await;
    st.search_router.search(&req.url).await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
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
            "symbol": c.symbol, "name": c.name, "value": c.value,
            "uncertainty": c.uncertainty, "unit": c.unit,
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
        "formula": req.formula, "hill": formula.hill_notation(),
        "elements": formula.elements, "molar_mass_g_per_mol": mw,
    })))
}

async fn stream_ws(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: SharedState) {
    info!("WebSocket connected");
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
                let req: Result<GenerateReq, _> = serde_json::from_str(&text);
                match req {
                    Ok(r) if r.use_search.unwrap_or(false) => {
                        let query = r.search_query.as_deref().unwrap_or(&r.prompt).to_string();
                        let st = state.read().await;
                        match st.search_router.search(&query).await {
                            Ok(sr) => {
                                let evt = serde_json::json!({
                                    "type": "search_done",
                                    "sources": sr.sources_used,
                                    "count": sr.results.len(),
                                    "elapsed_ms": sr.elapsed_ms,
                                });
                                let _ = socket.send(Message::Text(evt.to_string().into())).await;
                                let tok = serde_json::json!({"type":"token","text":"Generating answer...","done":true});
                                let _ = socket.send(Message::Text(tok.to_string().into())).await;
                            }
                            Err(e) => {
                                let err = serde_json::json!({"type":"error","message": e.to_string()});
                                let _ = socket.send(Message::Text(err.to_string().into())).await;
                            }
                        }
                    }
                    Ok(r) => {
                        let tok = serde_json::json!({"type":"token","text":format!("[PhysLLM] {}", &r.prompt[..r.prompt.len().min(40)]),"done":true});
                        let _ = socket.send(Message::Text(tok.to_string().into())).await;
                    }
                    Err(e) => {
                        let err = serde_json::json!({"type":"error","message": e.to_string()});
                        let _ = socket.send(Message::Text(err.to_string().into())).await;
                    }
                }
            }
            Message::Close(_) => { info!("WebSocket closed"); break; }
            _ => {}
        }
    }
}
