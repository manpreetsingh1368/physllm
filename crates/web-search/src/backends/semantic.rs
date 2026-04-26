//! semantic.rs — Semantic Scholar API (free, no key needed)
//! https://api.semanticscholar.org/

use crate::{Result, SearchError};
use super::{SearchResult, SourceKind};
use std::collections::HashMap;

const S2_API: &str = "https://api.semanticscholar.org/graph/v1/paper/search";

pub struct SemanticScholarBackend { client: reqwest::Client }

impl SemanticScholarBackend {
    pub fn new(client: reqwest::Client) -> Self { Self { client } }

    pub async fn search(&self, query: &str, max: usize) -> Result<Vec<SearchResult>> {
        let resp: serde_json::Value = self.client
            .get(S2_API)
            .query(&[
                ("query",  query),
                ("limit",  &max.to_string()),
                ("fields", "title,abstract,authors,year,externalIds,venue,citationCount"),
            ])
            .header("User-Agent", "PhysLLM/0.1")
            .send()
            .await?
            .json()
            .await?;

        let papers = resp["data"].as_array()
            .ok_or_else(|| SearchError::Parse("S2: no data array".into()))?;

        let mut results = Vec::new();
        for p in papers {
            let title    = p["title"].as_str().unwrap_or("").to_string();
            let abstract_ = p["abstract"].as_str().unwrap_or("").to_string();
            let year     = p["year"].as_u64().map(|y| y.to_string()).unwrap_or_default();
            let venue    = p["venue"].as_str().unwrap_or("").to_string();
            let citations = p["citationCount"].as_u64().unwrap_or(0);

            let authors: Vec<&str> = p["authors"].as_array()
                .map(|a| a.iter().filter_map(|x| x["name"].as_str()).collect())
                .unwrap_or_default();
            let authors_str = if authors.len() > 3 {
                format!("{} et al.", authors[0])
            } else { authors.join(", ") };

            let doi     = p["externalIds"]["DOI"].as_str().unwrap_or("").to_string();
            let paper_id = p["paperId"].as_str().unwrap_or("").to_string();
            let url      = format!("https://www.semanticscholar.org/paper/{paper_id}");
            let snippet  = abstract_[..abstract_.len().min(500)].to_string();

            // Score boosted by citations
            let citation_boost = (citations as f32).ln_1p() / 20.0;
            let score = (0.75 + citation_boost).min(0.98);

            let mut meta = HashMap::new();
            meta.insert("authors".into(),   authors_str);
            meta.insert("year".into(),      year);
            meta.insert("journal".into(),   venue);
            meta.insert("doi".into(),       doi);
            meta.insert("citations".into(), citations.to_string());

            results.push(SearchResult {
                title, url, snippet,
                full_text: if abstract_.is_empty() { None } else { Some(abstract_) },
                source: SourceKind::SemanticScholar,
                score,
                metadata: meta,
            });
        }

        Ok(results)
    }
}
