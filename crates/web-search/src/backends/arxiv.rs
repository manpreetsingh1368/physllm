//! arxiv.rs — arXiv API backend.
//!
//! Uses the arXiv Atom feed API (no key required, free, reliable).
//! Docs: https://arxiv.org/help/api/user-manual
//!
//! Supports:
//!   - Full-text search across all physics/chemistry categories
//!   - Direct paper lookup by arXiv ID (e.g. "2401.12345")
//!   - Category filtering (astro-ph, quant-ph, cond-mat, hep, chem-ph, ...)
//!   - Sorting by relevance or submission date

use crate::{Result, SearchError};
use super::{SearchResult, SourceKind};
use std::collections::HashMap;
use tracing::{debug, warn};

const ARXIV_API: &str = "https://export.arxiv.org/api/query";
const MAX_WAIT_MS: u64 = 3_000;

pub struct ArxivBackend {
    client: reqwest::Client,
}

impl ArxivBackend {
    pub fn new(client: reqwest::Client) -> Self { Self { client } }

    /// Search arXiv. Handles both free-text and direct ID lookup.
    pub async fn search(&self, query: &str, max: usize) -> Result<Vec<SearchResult>> {
        let q = query.trim();

        // Direct ID lookup (faster, exact)
        if let Some(arxiv_id) = extract_arxiv_id(q) {
            debug!("arXiv direct ID lookup: {arxiv_id}");
            return self.fetch_by_id(&arxiv_id).await.map(|r| r.into_iter().collect());
        }

        // Add physics category filter if query doesn't already target a category
        let search_query = build_search_query(q);
        debug!("arXiv search: {search_query}");

        let resp = self.client
            .get(ARXIV_API)
            .query(&[
                ("search_query", search_query.as_str()),
                ("max_results",  &max.to_string()),
                ("sortBy",       "relevance"),
                ("sortOrder",    "descending"),
            ])
            .send()
            .await?
            .text()
            .await?;

        parse_arxiv_atom(&resp)
    }

    /// Fetch a specific paper by arXiv ID.
    pub async fn fetch_by_id(&self, id: &str) -> Result<Option<SearchResult>> {
        let resp = self.client
            .get(ARXIV_API)
            .query(&[("id_list", id), ("max_results", "1")])
            .send()
            .await?
            .text()
            .await?;

        let mut results = parse_arxiv_atom(&resp)?;
        Ok(results.into_iter().next())
    }

    /// Fetch the abstract page HTML and extract full abstract + metadata.
    pub async fn fetch_abstract(&self, arxiv_id: &str) -> Result<SearchResult> {
        let url = format!("https://arxiv.org/abs/{arxiv_id}");
        let html = self.client.get(&url).send().await?.text().await?;
        parse_arxiv_abstract_page(&html, &url)
    }
}

// ── Atom feed parser ──────────────────────────────────────────────────────────

fn parse_arxiv_atom(xml: &str) -> Result<Vec<SearchResult>> {
    // Minimal XML parse without pulling in a heavy dependency
    // arXiv atom format is regular enough for this
    let mut results = Vec::new();

    for entry in xml.split("<entry>").skip(1) {
        let end = entry.find("</entry>").unwrap_or(entry.len());
        let entry = &entry[..end];

        let title   = extract_xml_text(entry, "title")
            .unwrap_or_default()
            .replace('\n', " ")
            .trim()
            .to_string();
        let summary = extract_xml_text(entry, "summary")
            .unwrap_or_default()
            .replace('\n', " ")
            .trim()
            .to_string();
        let id_raw  = extract_xml_text(entry, "id").unwrap_or_default();
        let url     = id_raw.trim().to_string();
        let arxiv_id = url.rsplit('/').next().unwrap_or("").to_string();

        // Published date
        let published = extract_xml_text(entry, "published")
            .and_then(|s| s[..4].parse::<u32>().ok().map(|y| y.to_string()))
            .unwrap_or_default();

        // Authors
        let authors: Vec<String> = entry
            .split("<author>")
            .skip(1)
            .filter_map(|a| {
                let end = a.find("</author>")?;
                extract_xml_text(&a[..end], "name")
            })
            .collect();
        let authors_str = if authors.len() > 3 {
            format!("{} et al.", authors[0])
        } else {
            authors.join(", ")
        };

        // Primary category
        let category = entry
            .find("arxiv:primary_category")
            .and_then(|i| {
                let s = &entry[i..];
                let t_start = s.find("term=\"")? + 6;
                let t_end   = s[t_start..].find('"')?;
                Some(s[t_start..t_start + t_end].to_string())
            })
            .unwrap_or_default();

        if title.is_empty() || url.is_empty() { continue; }

        let snippet = if summary.len() > 600 {
            format!("{}...", &summary[..597])
        } else {
            summary.clone()
        };

        let mut meta = HashMap::new();
        meta.insert("authors".into(),  authors_str);
        meta.insert("year".into(),     published);
        meta.insert("category".into(), category);
        meta.insert("arxiv_id".into(), arxiv_id.clone());

        results.push(SearchResult {
            title,
            url:       format!("https://arxiv.org/abs/{arxiv_id}"),
            snippet,
            full_text: Some(summary),
            source:    SourceKind::Arxiv,
            score:     0.85,
            metadata:  meta,
        });
    }

    if results.is_empty() {
        return Err(SearchError::NoResults("arXiv".into()));
    }
    Ok(results)
}

fn parse_arxiv_abstract_page(html: &str, url: &str) -> Result<SearchResult> {
    // Extract key fields from the abstract HTML page
    let title = extract_between(html, "<h1 class=\"title mathjax\">", "</h1>")
        .or_else(|| extract_between(html, "<title>", "</title>"))
        .unwrap_or("Unknown title".into())
        .replace("Title:", "")
        .trim()
        .to_string();

    let abstract_text = extract_between(html, "<blockquote class=\"abstract mathjax\">", "</blockquote>")
        .unwrap_or_default()
        .replace("Abstract:", "")
        .trim()
        .to_string();

    Ok(SearchResult {
        title,
        url: url.to_string(),
        snippet: abstract_text[..abstract_text.len().min(500)].to_string(),
        full_text: Some(abstract_text),
        source: SourceKind::Arxiv,
        score: 0.9,
        metadata: Default::default(),
    })
}

// ── Query builder ─────────────────────────────────────────────────────────────

fn build_search_query(q: &str) -> String {
    // arXiv query syntax: field qualifiers + boolean operators
    // https://arxiv.org/help/api/user-manual#query_details

    let q_lower = q.to_lowercase();

    // Detect which categories to search
    let cats: Vec<&str> = if q_lower.contains("astro") || q_lower.contains("galaxy")
        || q_lower.contains("stellar") || q_lower.contains("cosmic") {
        vec!["astro-ph"]
    } else if q_lower.contains("quantum") || q_lower.contains("qubit") {
        vec!["quant-ph", "cond-mat"]
    } else if q_lower.contains("chem") || q_lower.contains("molecule") || q_lower.contains("reaction") {
        vec!["physics.chem-ph", "q-bio"]
    } else if q_lower.contains("nuclear") || q_lower.contains("particle") || q_lower.contains("higgs") {
        vec!["hep-ph", "hep-th", "nucl-th"]
    } else {
        // Broad physics search
        vec!["physics", "astro-ph", "quant-ph", "cond-mat", "hep-ph"]
    };

    let cat_filter = cats.iter()
        .map(|c| format!("cat:{c}"))
        .collect::<Vec<_>>()
        .join(" OR ");

    // Wrap user query and apply category filter
    format!("({}) AND ({})", q, cat_filter)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn extract_xml_text(xml: &str, tag: &str) -> Option<String> {
    let open  = format!("<{tag}");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    // Skip to end of opening tag
    let content_start = xml[start..].find('>')? + start + 1;
    let end = xml[content_start..].find(&close)? + content_start;
    Some(xml[content_start..end].to_string())
}

fn extract_between(html: &str, start_marker: &str, end_marker: &str) -> Option<String> {
    let start = html.find(start_marker)? + start_marker.len();
    let end   = html[start..].find(end_marker)? + start;
    // Strip HTML tags from the extracted text
    let raw = &html[start..end];
    Some(strip_html_tags(raw))
}

fn strip_html_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _   => if !in_tag { out.push(c); }
        }
    }
    out.trim().to_string()
}

fn extract_arxiv_id(q: &str) -> Option<String> {
    for word in q.split_whitespace() {
        let w = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '/');
        // YYMM.NNNNN(N)
        let parts: Vec<&str> = w.split('.').collect();
        if parts.len() == 2 {
            if parts[0].len() == 4 && parts[0].chars().all(|c| c.is_ascii_digit())
            && parts[1].len() >= 4 && parts[1].chars().all(|c| c.is_ascii_digit()) {
                return Some(w.to_string());
            }
        }
        // old-style hep-ph/9901234
        if w.contains('/') {
            let p: Vec<&str> = w.split('/').collect();
            if p.len() == 2 && p[1].len() == 7 && p[1].chars().all(|c| c.is_ascii_digit()) {
                return Some(w.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_arxiv_id() {
        assert_eq!(extract_arxiv_id("paper 2401.12345 please"), Some("2401.12345".into()));
        assert_eq!(extract_arxiv_id("hep-ph/9901234"),          Some("hep-ph/9901234".into()));
        assert_eq!(extract_arxiv_id("gravitational waves"),     None);
    }

    #[test]
    fn test_build_search_query() {
        let q = build_search_query("dark matter direct detection");
        assert!(q.contains("astro-ph") || q.contains("hep-ph"));
    }
}
