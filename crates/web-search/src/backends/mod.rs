//! backends/mod.rs — Common types and re-exports for all search backends.

pub mod arxiv;
pub mod semantic;
pub mod nist;
pub mod nasa_ads;
pub mod pubchem;
pub mod brave;
pub mod ddg;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single search result from any backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title:     String,
    pub url:       String,
    pub snippet:   String,
    /// Full extracted page text (populated by WebFetcher if enabled)
    pub full_text: Option<String>,
    pub source:    SourceKind,
    /// Relevance score 0.0–1.0
    pub score:     f32,
    /// Extra fields (authors, year, doi, formula, etc.)
    pub metadata:  HashMap<String, String>,
}

impl SearchResult {
    pub fn authors(&self) -> Option<&str> { self.metadata.get("authors").map(|s| s.as_str()) }
    pub fn year(&self)    -> Option<&str> { self.metadata.get("year").map(|s| s.as_str()) }
    pub fn doi(&self)     -> Option<&str> { self.metadata.get("doi").map(|s| s.as_str()) }
    pub fn journal(&self) -> Option<&str> { self.metadata.get("journal").map(|s| s.as_str()) }
    pub fn arxiv_id(&self) -> Option<&str> { self.metadata.get("arxiv_id").map(|s| s.as_str()) }
}

/// Which backend produced this result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceKind {
    Arxiv,
    SemanticScholar,
    NistWebBook,
    NasaAds,
    PubChem,
    BraveSearch,
    DuckDuckGo,
    WebPage,
}

impl SourceKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Arxiv           => "arXiv",
            Self::SemanticScholar => "Semantic Scholar",
            Self::NistWebBook     => "NIST WebBook",
            Self::NasaAds         => "NASA ADS",
            Self::PubChem         => "PubChem",
            Self::BraveSearch     => "Brave Search",
            Self::DuckDuckGo      => "DuckDuckGo",
            Self::WebPage         => "Web",
        }
    }
}
