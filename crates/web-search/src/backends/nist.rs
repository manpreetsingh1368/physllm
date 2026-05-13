//! nist.rs — NIST WebBook and CODATA search.

use crate::{Result, SearchError};
use super::{SearchResult, SourceKind};
use std::collections::HashMap;

pub struct NistBackend { client: reqwest::Client }

impl NistBackend {
    pub fn new(client: reqwest::Client) -> Self { Self { client } }

    pub async fn search(&self, query: &str, max: usize) -> Result<Vec<SearchResult>> {
        // NIST WebBook search endpoint
        let resp = self.client
            .get("https://webbook.nist.gov/cgi/cbook.cgi")
            .query(&[("Name", query), ("Units", "SI"), ("cTG", "on")])
            .send()
            .await?
            .text()
            .await?;

        // Extract basic results from HTML (NIST doesn't have a JSON API)
        let mut results = Vec::new();

        // Check if we got a direct compound page
        if resp.contains("Molecular formula:") || resp.contains("CAS Registry Number") {
            let name = extract_html_value(&resp, "name of substance")
                .or_else(|| extract_between_tags(&resp, "<title>", "</title>"))
                .unwrap_or_else(|| query.to_string());
            let formula = extract_html_value(&resp, "Molecular formula")
                .unwrap_or_default();
            let mw = extract_html_value(&resp, "Molecular weight")
                .unwrap_or_default();
            let cas = extract_html_value(&resp, "CAS Registry Number")
                .unwrap_or_default();

            let snippet = format!(
                "Formula: {formula}. Molecular weight: {mw} g/mol. CAS: {cas}. \
                 Source: NIST WebBook (SI units)."
            );
            let mut meta = HashMap::new();
            meta.insert("formula".into(), formula);
            meta.insert("cas".into(), cas);
            meta.insert("mw".into(), mw);

            results.push(SearchResult {
                title:     name,
                url:       format!("https://webbook.nist.gov/cgi/cbook.cgi?Name={}&Units=SI", urlencoded(query)),
                snippet,
                full_text: Some(strip_html(&resp)[..strip_html(&resp).len().min(3000)].to_string()),
                source:    SourceKind::NistWebBook,
                score:     0.92,
                metadata:  meta,
            });
        }

        if results.is_empty() {
            // Fall back to NIST CODATA constants search
            let codata_url = format!(
                "https://physics.nist.gov/cgi-bin/cuu/Value?{}", 
                query.replace(' ', "+")
            );
            results.push(SearchResult {
                title:     format!("NIST CODATA: {query}"),
                url:       codata_url,
                snippet:   format!("Search NIST CODATA for physical constant: {query}"),
                full_text: None,
                source:    SourceKind::NistWebBook,
                score:     0.6,
                metadata:  Default::default(),
            });
        }

        Ok(results)
    }
}

fn extract_html_value(html: &str, label: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let label_lower = label.to_lowercase();
    let pos = lower.find(&label_lower)?;
    let after = &html[pos + label.len()..];
    let value_start = after.find('>')?  + 1;
    let value_end   = after[value_start..].find('<')? + value_start;
    Some(strip_html(&after[value_start..value_end]).trim().to_string())
}

fn extract_between_tags(html: &str, open: &str, close: &str) -> Option<String> {
    let start = html.find(open)? + open.len();
    let end   = html[start..].find(close)? + start;
    Some(strip_html(&html[start..end]).trim().to_string())
}

fn strip_html(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c { '<' => in_tag = true, '>' => in_tag = false, _ => if !in_tag { out.push(c); } }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn urlencoded(s: &str) -> String { s.replace(' ', "+") }

// ─────────────────────────────────────────────────────────────────────────────
// nasa_ads.rs — NASA Astrophysics Data System

pub struct NasaAdsBackend {
    client:  reqwest::Client,
    api_key: Option<String>,
}

impl NasaAdsBackend {
    pub fn new(client: reqwest::Client, api_key: Option<String>) -> Self { Self { client, api_key } }

    pub async fn search(&self, query: &str, max: usize) -> Result<Vec<SearchResult>> {
        let key = self.api_key.as_deref().unwrap_or("");
        if key.is_empty() {
            // Without a key, link to the ADS search page
            return Ok(vec![SearchResult {
                title:   format!("NASA ADS search: {query}"),
                url:     format!("https://ui.adsabs.harvard.edu/search/q={}&sort=score", urlencoded(query)),
                snippet: format!("Search NASA Astrophysics Data System for: {query}. Get a free API key at https://ui.adsabs.harvard.edu/user/settings/token"),
                full_text: None,
                source:  SourceKind::NasaAds,
                score:   0.5,
                metadata: Default::default(),
            }]);
        }

        let resp: serde_json::Value = self.client
            .get("https://api.adsabs.harvard.edu/v1/search/query")
            .bearer_auth(key)
            .query(&[
                ("q",      query),
                ("rows",   &max.to_string()),
                ("fl",     "title,abstract,author,year,doi,bibcode,citation_count,identifier"),
                ("sort",   "score desc"),
            ])
            .send()
            .await?
            .json()
            .await?;

        let docs = resp["response"]["docs"].as_array()
            .ok_or_else(|| SearchError::Parse("ADS: no docs".into()))?;

        let mut results = Vec::new();
        for doc in docs {
            let title   = doc["title"].as_array()
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let abstract_ = doc["abstract"].as_str().unwrap_or("").to_string();
            let year    = doc["year"].as_str().unwrap_or("").to_string();
            let bibcode = doc["bibcode"].as_str().unwrap_or("").to_string();
            let url     = format!("https://ui.adsabs.harvard.edu/abs/{bibcode}");
            let doi     = doc["doi"].as_array()
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let citations = doc["citation_count"].as_u64().unwrap_or(0);
            let authors: Vec<&str> = doc["author"].as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
                .unwrap_or_default();
            let authors_str = if authors.len() > 3 {
                format!("{} et al.", authors[0])
            } else { authors.join("; ") };

            let citation_boost = (citations as f32).ln_1p() / 15.0;
            let score = (0.8 + citation_boost).min(0.99);

            let mut meta = HashMap::new();
            meta.insert("authors".into(),   authors_str);
            meta.insert("year".into(),      year);
            meta.insert("doi".into(),       doi);
            meta.insert("bibcode".into(),   bibcode);
            meta.insert("citations".into(), citations.to_string());

            results.push(SearchResult {
                title, url,
                snippet:   abstract_[..abstract_.len().min(500)].to_string(),
                full_text: if abstract_.is_empty() { None } else { Some(abstract_) },
                source:    SourceKind::NasaAds,
                score,
                metadata:  meta,
            });
        }

        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// pubchem.rs — PubChem REST API

pub struct PubchemBackend { client: reqwest::Client }

impl PubchemBackend {
    pub fn new(client: reqwest::Client) -> Self { Self { client } }

    pub async fn search(&self, query: &str, max: usize) -> Result<Vec<SearchResult>> {
        // Search for compound by name
        let url = format!(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/JSON",
            urlencoded(query)
        );
        let resp: serde_json::Value = match self.client.get(&url).send().await?.json::<serde_json::Value>().await {
            Ok(v) => v,
            Err(_) => return Ok(vec![]),
        };

        let compounds = match resp["PC_Compounds"].as_array() {
            Some(c) => c,
            None    => return Ok(vec![]),
        };

        let mut results = Vec::new();
        for cmpd in compounds.iter().take(max) {
            let cid = cmpd["id"]["id"]["cid"].as_u64().unwrap_or(0);

            // Extract properties
            let mut formula = String::new();
            let mut mw      = String::new();
            let mut iupac   = String::new();
            let mut smiles  = String::new();
            let mut inchi   = String::new();

            if let Some(props) = cmpd["props"].as_array() {
                for prop in props {
                    let label = prop["urn"]["label"].as_str().unwrap_or("");
                    let name  = prop["urn"]["name"].as_str().unwrap_or("");
                    let val   = prop["value"]["sval"].as_str()
                        .or_else(|| prop["value"]["fval"].as_f64().map(|_| ""))
                        .unwrap_or("");

                    match (label, name) {
                        ("Molecular Formula", _)      => formula = val.to_string(),
                        ("Molecular Weight",  _)      => {
                            mw = prop["value"]["fval"].as_f64()
                                .map(|f| format!("{f:.3}"))
                                .unwrap_or_else(|| val.to_string());
                        }
                        ("IUPAC Name", "Preferred")   => iupac  = val.to_string(),
                        ("SMILES", "Canonical")       => smiles  = val.to_string(),
                        ("InChI", _)                  => inchi   = val.to_string(),
                        _ => {}
                    }
                }
            }

            let snippet = format!(
                "Formula: {formula}. MW: {mw} g/mol. SMILES: {smiles}. IUPAC: {iupac}."
            );
            let mut meta = HashMap::new();
            meta.insert("formula".into(), formula.clone());
            meta.insert("mw".into(),      mw);
            meta.insert("smiles".into(),  smiles);
            meta.insert("inchi".into(),   inchi);
            meta.insert("cid".into(),     cid.to_string());

            results.push(SearchResult {
                title:    if iupac.is_empty() { format!("Compound CID {cid}") } else { iupac.clone() },
                url:      format!("https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"),
                snippet,
                full_text: None,
                source:   SourceKind::PubChem,
                score:    0.88,
                metadata: meta,
            });
        }

        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// brave.rs — Brave Search API

pub struct BraveBackend {
    client:  reqwest::Client,
    api_key: Option<String>,
}

impl BraveBackend {
    pub fn new(client: reqwest::Client, api_key: Option<String>) -> Self { Self { client, api_key } }

    pub async fn search(&self, query: &str, max: usize) -> Result<Vec<SearchResult>> {
        let key = self.api_key.as_deref()
            .ok_or_else(|| SearchError::MissingKey("BRAVE_API_KEY".into()))?;

        let resp: serde_json::Value = self.client
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("Accept", "application/json")
            .header("Accept-Encoding", "gzip")
            .header("X-Subscription-Token", key)
            .query(&[("q", query), ("count", &max.to_string()), ("safesearch", "off")])
            .send()
            .await?
            .json()
            .await?;

        let web_results = resp["web"]["results"].as_array()
            .ok_or_else(|| SearchError::NoResults("Brave".into()))?;

        Ok(web_results.iter().map(|r| {
            let mut meta = HashMap::new();
            if let Some(age) = r["age"].as_str() { meta.insert("date".into(), age.to_string()); }

            SearchResult {
                title:     r["title"].as_str().unwrap_or("").to_string(),
                url:       r["url"].as_str().unwrap_or("").to_string(),
                snippet:   r["description"].as_str().unwrap_or("").to_string(),
                full_text: None,
                source:    SourceKind::BraveSearch,
                score:     0.7,
                metadata:  meta,
            }
        }).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ddg.rs — DuckDuckGo instant answers (no API key, limited)

pub struct DuckDuckGoBackend { client: reqwest::Client }

impl DuckDuckGoBackend {
    pub fn new(client: reqwest::Client) -> Self { Self { client } }

    pub async fn search(&self, query: &str, _max: usize) -> Result<Vec<SearchResult>> {
        // DDG Instant Answer API (free, no auth, limited to instant answers)
        let resp: serde_json::Value = self.client
            .get("https://api.duckduckgo.com/")
            .query(&[("q", query), ("format", "json"), ("no_html", "1"), ("skip_disambig", "1")])
            .send()
            .await?
            .json()
            .await?;

        let mut results = Vec::new();

        // Abstract result (main topic)
        if let Some(abstract_text) = resp["Abstract"].as_str() {
            if !abstract_text.is_empty() {
                results.push(SearchResult {
                    title:     resp["Heading"].as_str().unwrap_or(query).to_string(),
                    url:       resp["AbstractURL"].as_str().unwrap_or("").to_string(),
                    snippet:   abstract_text.to_string(),
                    full_text: None,
                    source:    SourceKind::DuckDuckGo,
                    score:     0.65,
                    metadata:  Default::default(),
                });
            }
        }

        // Related topics
        if let Some(topics) = resp["RelatedTopics"].as_array() {
            for topic in topics.iter().take(4) {
                if let Some(text) = topic["Text"].as_str() {
                    results.push(SearchResult {
                        title:    text[..text.len().min(80)].to_string(),
                        url:      topic["FirstURL"].as_str().unwrap_or("").to_string(),
                        snippet:  text.to_string(),
                        full_text: None,
                        source:   SourceKind::DuckDuckGo,
                        score:    0.5,
                        metadata: Default::default(),
                    });
                }
            }
        }

        if results.is_empty() {
            return Err(SearchError::NoResults("DuckDuckGo".into()));
        }

        Ok(results)
    }
}
