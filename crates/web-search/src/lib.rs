//! web-search — Real-time web search and scientific database retrieval for PhysLLM.
//!
//! Architecture:
//!
//!   SearchRouter
//!     ├── GeneralSearch   (Brave API / SerpAPI / DuckDuckGo scrape)
//!     ├── ArxivSearch     (arXiv API — free, no key needed)
//!     ├── SemanticScholar (Semantic Scholar API — free)
//!     ├── NistSearch      (NIST WebBook / CODATA)
//!     ├── NasaAdsSearch   (NASA Astrophysics Data System)
//!     ├── PubchemSearch   (PubChem REST API)
//!     └── WebFetcher      (full-page fetch + text extraction)
//!
//! The LLM calls these as tools (tool-use JSON), the router dispatches,
//! results are formatted back as context the LLM sees before generating.

pub mod router;
pub mod backends;
pub mod fetcher;
pub mod cache;
pub mod tools;
pub mod context;
pub mod ratelimit;

pub use router::{SearchRouter, SearchConfig};
pub use backends::SearchResult;
pub use context::SearchContext;
pub use tools::SEARCH_TOOLS;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Rate limited by {source}: retry after {retry_after_s}s")]
    RateLimited { source: String, retry_after_s: u64 },
    #[error("API key missing for {0}")]
    MissingKey(String),
    #[error("No results found for: {0}")]
    NoResults(String),
    #[error("Backend unavailable: {0}")]
    Unavailable(String),
}

pub type Result<T> = std::result::Result<T, SearchError>;
