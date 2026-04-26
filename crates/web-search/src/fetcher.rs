//! fetcher.rs — Fetch and extract clean text from any URL.
//!
//! Strategy:
//!   1. HTTP GET with gzip + appropriate UA
//!   2. html2text for clean extraction (strips nav/ads)
//!   3. PDF: download and extract text via pdftotext if available
//!   4. arXiv: redirect /abs/ → /html/ for structured extraction

use crate::{Result, SearchError};
use tracing::debug;

pub struct FetchedPage {
    pub url:   String,
    pub title: String,
    pub text:  String,
}

pub struct WebFetcher {
    client:        reqwest::Client,
    max_chars:     usize,
}

impl WebFetcher {
    pub fn new(client: reqwest::Client, max_chars: usize) -> Self {
        Self { client, max_chars }
    }

    pub async fn fetch(&self, url: &str) -> Result<FetchedPage> {
        debug!("Fetching: {url}");

        // Redirect arXiv abstract URLs to HTML version (better text)
        let fetch_url = if url.contains("arxiv.org/abs/") {
            url.replace("/abs/", "/html/")
        } else {
            url.to_string()
        };

        let resp = self.client
            .get(&fetch_url)
            .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .send()
            .await?;

        let content_type = resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let body = resp.bytes().await?;

        let (title, text) = if content_type.contains("pdf") {
            self.extract_pdf(&body)
        } else {
            self.extract_html(&body)
        };

        let truncated = if text.len() > self.max_chars {
            format!("{}...[truncated at {} chars]", &text[..self.max_chars], self.max_chars)
        } else {
            text
        };

        Ok(FetchedPage { url: url.to_string(), title, text: truncated })
    }

    fn extract_html(&self, bytes: &[u8]) -> (String, String) {
        let html = String::from_utf8_lossy(bytes);

        // Extract title
        let title = extract_between(&html, "<title>", "</title>")
            .unwrap_or_else(|| "Untitled".into());

        // Use html2text for clean extraction
        let text = html2text::from_read(bytes, 120);

        // Clean up excessive whitespace
        let clean: String = text.lines()
            .filter(|l| l.trim().len() > 20)  // skip very short lines (nav items etc.)
            .collect::<Vec<_>>()
            .join("\n");

        (title, clean)
    }

    fn extract_pdf(&self, bytes: &[u8]) -> (String, String) {
        // Try pdftotext if available (poppler-utils)
        let tmp = std::env::temp_dir().join("physllm_fetch.pdf");
        if std::fs::write(&tmp, bytes).is_ok() {
            if let Ok(out) = std::process::Command::new("pdftotext")
                .args([tmp.to_str().unwrap(), "-"])
                .output()
            {
                let text = String::from_utf8_lossy(&out.stdout).to_string();
                let _ = std::fs::remove_file(&tmp);
                return ("PDF document".into(), text);
            }
        }
        ("PDF document".into(), "[PDF — install poppler-utils (pdftotext) for extraction]".into())
    }
}

fn extract_between(html: &str, open: &str, close: &str) -> Option<String> {
    let start = html.find(open)? + open.len();
    let end   = html[start..].find(close)? + start;
    let raw   = &html[start..end];
    let clean: String = {
        let mut out = String::new();
        let mut in_tag = false;
        for c in raw.chars() {
            match c { '<' => in_tag = true, '>' => in_tag = false, _ => if !in_tag { out.push(c); } }
        }
        out
    };
    Some(clean.trim().to_string())
}
