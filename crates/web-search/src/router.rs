//! router.rs — Intelligent search router.
//!
//! Detects the query type and dispatches to the best backend:
//!
//!   "arxiv 2401.xxxxx"           → ArxivSearch (direct lookup)
//!   "gravitational wave detection" → ArxivSearch + SemanticScholar
//!   "molecular weight of glucose"  → PubchemSearch + NistSearch
//!   "Hubble constant tension"      → NasaAdsSearch + ArxivSearch
//!   "latest news on JWST"          → GeneralSearch
//!   "https://..."                  → WebFetcher (direct URL)

use crate::{
    backends::{
        SearchResult, SourceKind,
        arxiv::ArxivBackend,
        semantic::SemanticScholarBackend,
        nist::NistBackend,
        nasa_ads::NasaAdsBackend,
        pubchem::PubchemBackend,
        brave::BraveBackend,
        ddg::DuckDuckGoBackend,
    },
    fetcher::WebFetcher,
    cache::SearchCache,
    ratelimit::RateLimiter,
    Result, SearchError,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, debug, warn};

/// Configuration for the search router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Brave Search API key (optional but recommended for general search)
    pub brave_api_key:    Option<String>,
    /// SerpAPI key (Google Search, alternative to Brave)
    pub serp_api_key:     Option<String>,
    /// NASA ADS API key (free at https://ui.adsabs.harvard.edu/user/settings/token)
    pub nasa_ads_key:     Option<String>,
    /// Max results to return per source
    pub max_results:      usize,
    /// Enable result caching (TTL in seconds)
    pub cache_ttl_s:      u64,
    /// Timeout per HTTP request
    pub request_timeout_s: u64,
    /// Whether to fetch and extract full page text (slower but richer)
    pub full_page_fetch:  bool,
    /// Max characters of page text to include in context
    pub max_page_chars:   usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            brave_api_key:     std::env::var("BRAVE_API_KEY").ok(),
            serp_api_key:      std::env::var("SERP_API_KEY").ok(),
            nasa_ads_key:      std::env::var("NASA_ADS_KEY").ok(),
            max_results:       5,
            cache_ttl_s:       3600,       // 1 hour
            request_timeout_s: 15,
            full_page_fetch:   false,
            max_page_chars:    8_000,
        }
    }
}

/// A fully processed search response ready for the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub query:        String,
    pub intent:       QueryIntent,
    pub results:      Vec<SearchResult>,
    pub total_found:  usize,
    pub sources_used: Vec<String>,
    pub elapsed_ms:   u64,
    /// Pre-formatted context string injected into the LLM prompt
    pub llm_context:  String,
}

/// Detected intent of a search query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryIntent {
    ArxivPaper,          // specific paper or preprint
    PhysicsResearch,     // general physics/astro research
    ChemistryData,       // molecular, thermodynamic data
    AstrophysicsObject,  // star, galaxy, nebula lookup
    GeneralWeb,          // current events, news
    DirectUrl,           // user gave a URL
    NistConstant,        // specific physical constant or measurement
}

/// The main search router.
pub struct SearchRouter {
    config:    SearchConfig,
    cache:     Arc<SearchCache>,
    limiter:   Arc<RateLimiter>,
    client:    reqwest::Client,
}

impl SearchRouter {
    pub fn new(config: SearchConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_s))
            .user_agent("PhysLLM/0.1 (physics AI assistant; +https://physllm.ai)")
            .gzip(true)
            .build()
            .expect("HTTP client build failed");

        Self {
            cache:   Arc::new(SearchCache::new(config.cache_ttl_s)),
            limiter: Arc::new(RateLimiter::new()),
            config,
            client,
        }
    }

    /// Main entry point: detect intent, dispatch, aggregate, format.
    pub async fn search(&self, query: &str) -> Result<SearchResponse> {
        let t0 = std::time::Instant::now();

        // Check cache first
        if let Some(cached) = self.cache.get(query) {
            debug!("Cache hit for: {query}");
            return Ok(cached);
        }

        let intent = self.detect_intent(query);
        info!("search: intent={intent:?} query={query:?}");

        // Dispatch to appropriate backends
        let mut all_results: Vec<SearchResult> = Vec::new();
        let mut sources_used: Vec<String> = Vec::new();

        match &intent {
            QueryIntent::DirectUrl => {
                let url = query.trim();
                let fetcher = WebFetcher::new(self.client.clone(), self.config.max_page_chars);
                match fetcher.fetch(url).await {
                    Ok(page) => {
                        all_results.push(SearchResult {
                            title:   page.title.clone(),
                            url:     url.to_string(),
                            snippet: page.text[..page.text.len().min(500)].to_string(),
                            full_text: Some(page.text),
                            source:  SourceKind::WebPage,
                            score:   1.0,
                            metadata: Default::default(),
                        });
                        sources_used.push("direct_fetch".into());
                    }
                    Err(e) => warn!("Direct fetch failed: {e}"),
                }
            }

            QueryIntent::ArxivPaper => {
                let arxiv = ArxivBackend::new(self.client.clone());
                match arxiv.search(query, self.config.max_results).await {
                    Ok(mut r) => { sources_used.push("arxiv".into()); all_results.append(&mut r); }
                    Err(e)   => warn!("arXiv: {e}"),
                }
                // Also try Semantic Scholar for citation context
                let sem = SemanticScholarBackend::new(self.client.clone());
                if let Ok(mut r) = sem.search(query, 3).await {
                    sources_used.push("semantic_scholar".into());
                    all_results.append(&mut r);
                }
            }

            QueryIntent::PhysicsResearch => {
                // arXiv + Semantic Scholar in parallel
                let arxiv = ArxivBackend::new(self.client.clone());
                let sem   = SemanticScholarBackend::new(self.client.clone());

                let (r1, r2) = tokio::join!(
                    arxiv.search(query, self.config.max_results),
                    sem.search(query, self.config.max_results),
                );
                if let Ok(mut r) = r1 { sources_used.push("arxiv".into()); all_results.append(&mut r); }
                if let Ok(mut r) = r2 { sources_used.push("semantic_scholar".into()); all_results.append(&mut r); }

                // Supplement with general web if we have a Brave key
                if self.config.brave_api_key.is_some() {
                    let brave = BraveBackend::new(self.client.clone(), self.config.brave_api_key.clone());
                    if let Ok(mut r) = brave.search(query, 3).await {
                        sources_used.push("brave".into());
                        all_results.append(&mut r);
                    }
                }
            }

            QueryIntent::ChemistryData => {
                let pubchem = PubchemBackend::new(self.client.clone());
                let nist    = NistBackend::new(self.client.clone());

                let (r1, r2) = tokio::join!(
                    pubchem.search(query, self.config.max_results),
                    nist.search(query, self.config.max_results),
                );
                if let Ok(mut r) = r1 { sources_used.push("pubchem".into()); all_results.append(&mut r); }
                if let Ok(mut r) = r2 { sources_used.push("nist".into()); all_results.append(&mut r); }

                // arXiv for recent computational chemistry papers
                let arxiv = ArxivBackend::new(self.client.clone());
                if let Ok(mut r) = arxiv.search(&format!("{query} chemistry", ), 3).await {
                    sources_used.push("arxiv".into());
                    all_results.append(&mut r);
                }
            }

            QueryIntent::AstrophysicsObject => {
                // NASA ADS is the gold standard for astronomical objects
                let ads = NasaAdsBackend::new(self.client.clone(), self.config.nasa_ads_key.clone());
                let arxiv = ArxivBackend::new(self.client.clone());

                let (r1, r2) = tokio::join!(
                    ads.search(query, self.config.max_results),
                    arxiv.search(&format!("astro-ph {query}"), self.config.max_results),
                );
                if let Ok(mut r) = r1 { sources_used.push("nasa_ads".into()); all_results.append(&mut r); }
                if let Ok(mut r) = r2 { sources_used.push("arxiv".into()); all_results.append(&mut r); }
            }

            QueryIntent::NistConstant => {
                let nist = NistBackend::new(self.client.clone());
                match nist.search(query, self.config.max_results).await {
                    Ok(mut r) => { sources_used.push("nist".into()); all_results.append(&mut r); }
                    Err(e)   => warn!("NIST: {e}"),
                }
            }

            QueryIntent::GeneralWeb => {
                // Try Brave first, fall back to DuckDuckGo
                if self.config.brave_api_key.is_some() {
                    let brave = BraveBackend::new(self.client.clone(), self.config.brave_api_key.clone());
                    match brave.search(query, self.config.max_results).await {
                        Ok(mut r) => { sources_used.push("brave".into()); all_results.append(&mut r); }
                        Err(e)   => warn!("Brave: {e}"),
                    }
                } else {
                    let ddg = DuckDuckGoBackend::new(self.client.clone());
                    match ddg.search(query, self.config.max_results).await {
                        Ok(mut r) => { sources_used.push("duckduckgo".into()); all_results.append(&mut r); }
                        Err(e)   => warn!("DDG: {e}"),
                    }
                }
            }
        }

        // Optional: fetch full text for top results
        if self.config.full_page_fetch && !all_results.is_empty() {
            let fetcher = WebFetcher::new(self.client.clone(), self.config.max_page_chars);
            for result in all_results.iter_mut().take(3) {
                if result.full_text.is_none() {
                    if let Ok(page) = fetcher.fetch(&result.url).await {
                        result.full_text = Some(page.text);
                    }
                }
            }
        }

        // Sort by relevance score
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.dedup_by(|a, b| a.url == b.url);

        let total = all_results.len();
        let llm_context = crate::context::format_for_llm(query, &all_results, &intent);

        let resp = SearchResponse {
            query:       query.to_string(),
            intent,
            results:     all_results,
            total_found: total,
            sources_used,
            elapsed_ms:  t0.elapsed().as_millis() as u64,
            llm_context,
        };

        self.cache.insert(query, resp.clone());
        Ok(resp)
    }

    /// Classify query intent using simple heuristics.
    fn detect_intent(&self, query: &str) -> QueryIntent {
        let q = query.trim().to_lowercase();

        // Direct URL
        if q.starts_with("http://") || q.starts_with("https://") {
            return QueryIntent::DirectUrl;
        }

        // arXiv paper ID pattern: 2401.12345 or arxiv:2401.12345
        let arxiv_id = regex_match_arxiv(&q);
        if arxiv_id || q.starts_with("arxiv") || q.contains("preprint") {
            return QueryIntent::ArxivPaper;
        }

        // NIST constant lookup
        let nist_signals = ["codata", "nist", "physical constant", "atomic weight",
                            "boltzmann", "planck constant", "elementary charge", "avogadro"];
        if nist_signals.iter().any(|s| q.contains(s)) {
            return QueryIntent::NistConstant;
        }

        // Chemistry data
        let chem_signals = ["molecular weight", "molar mass", "smiles", "cas number",
                            "boiling point", "melting point", "thermodynamic", "enthalpy",
                            "gibbs", "reaction rate", "kinetics", "solubility", "synthesis",
                            "compound", "molecule", "formula", "polymer", "enzyme"];
        if chem_signals.iter().any(|s| q.contains(s)) {
            return QueryIntent::ChemistryData;
        }

        // Astrophysics object
        let astro_obj = ["ngc ", "ic ", "messier", "m31", "m87", "andromeda",
                         "milky way", "black hole", "neutron star", "pulsar", "quasar",
                         "galaxy", "nebula", "supernova", "exoplanet", "kepler-",
                         "trappist", "gliese", "proxima", "alpha centauri", "betelgeuse",
                         "sagittarius", "andromeda", "virgo cluster"];
        if astro_obj.iter().any(|s| q.contains(s)) {
            return QueryIntent::AstrophysicsObject;
        }

        // Physics research
        let phys_signals = ["quantum", "relativity", "cosmology", "dark matter", "dark energy",
                            "gravitational wave", "higgs", "standard model", "string theory",
                            "condensed matter", "superconductor", "laser", "plasma",
                            "astrophysics", "astrochemistry", "spectroscopy", "photon",
                            "electron", "proton", "nuclear", "particle", "field theory",
                            "hamiltonian", "lagrangian", "entropy", "thermodynamics",
                            "schrödinger", "dirac", "feynman", "uncertainty principle"];
        if phys_signals.iter().any(|s| q.contains(s)) {
            return QueryIntent::PhysicsResearch;
        }

        QueryIntent::GeneralWeb
    }
}

fn regex_match_arxiv(q: &str) -> bool {
    // Match patterns like 2401.12345, 2401.123456, hep-ph/9901234
    let words: Vec<&str> = q.split_whitespace().collect();
    for w in words {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '/');
        // Modern ID: YYMM.NNNNN
        let parts: Vec<&str> = w.split('.').collect();
        if parts.len() == 2 {
            let yymm = parts[0];
            let nnnnn = parts[1];
            if yymm.len() == 4 && yymm.chars().all(|c| c.is_ascii_digit())
                && nnnnn.len() >= 4 && nnnnn.chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }
        // Old-style: hep-ph/9901234
        if w.contains('/') {
            let p: Vec<&str> = w.split('/').collect();
            if p.len() == 2 && p[1].len() == 7 && p[1].chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}
