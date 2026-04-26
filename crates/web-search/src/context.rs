//! context.rs — Format search results into a prompt context block for the LLM.
//!
//! The LLM receives a structured context block BEFORE its generation step:
//!
//!   <search_results>
//!   [1] Title — Source (Year)
//!   URL: https://...
//!   Authors: X et al.
//!   Summary: ...
//!
//!   [2] ...
//!   </search_results>
//!
//!   Based on the above search results, answer: {original query}

use crate::backends::{SearchResult, SourceKind};
use crate::router::QueryIntent;

pub struct SearchContext {
    pub prompt_injection: String,  // injected before user message
    pub citations:        Vec<Citation>,
}

#[derive(Debug, Clone)]
pub struct Citation {
    pub index:  usize,
    pub title:  String,
    pub url:    String,
    pub source: String,
}

/// Format search results into a context block suitable for LLM injection.
pub fn format_for_llm(query: &str, results: &[SearchResult], intent: &QueryIntent) -> String {
    if results.is_empty() {
        return format!(
            "<search_results>\nNo results found for: {query}\n</search_results>\n\n\
             Note: Web search returned no results. Answer from your training knowledge, \
             and note any uncertainty."
        );
    }

    let mut ctx = String::from("<search_results>\n");

    for (i, r) in results.iter().enumerate() {
        ctx.push_str(&format!("\n[{}] {}\n", i + 1, r.title));
        ctx.push_str(&format!("Source: {} | URL: {}\n", r.source.label(), r.url));

        if let Some(authors) = r.authors() {
            if !authors.is_empty() {
                ctx.push_str(&format!("Authors: {authors}\n"));
            }
        }
        if let Some(year) = r.year() {
            if !year.is_empty() {
                ctx.push_str(&format!("Year: {year}\n"));
            }
        }
        if let Some(journal) = r.journal() {
            if !journal.is_empty() {
                ctx.push_str(&format!("Journal: {journal}\n"));
            }
        }
        if let Some(citations) = r.metadata.get("citations") {
            if citations != "0" && !citations.is_empty() {
                ctx.push_str(&format!("Citations: {citations}\n"));
            }
        }

        // Use full text if available, otherwise snippet
        let content = r.full_text.as_deref()
            .map(|t| &t[..t.len().min(1200)])
            .unwrap_or(&r.snippet);

        if !content.is_empty() {
            ctx.push_str(&format!("Content: {content}\n"));
        }
    }

    ctx.push_str("\n</search_results>\n\n");

    // Intent-specific instruction to the LLM
    let instruction = match intent {
        QueryIntent::ArxivPaper => {
            "The above search results are from arXiv. Summarise the paper(s), \
             highlight the key findings, methods, and conclusions. \
             Cite specific results with numbers where available."
        }
        QueryIntent::PhysicsResearch => {
            "The above search results are from physics literature databases. \
             Synthesise the information to answer the query. \
             Note the most recent findings and any ongoing debates. \
             Always cite result [N] when drawing on a specific source."
        }
        QueryIntent::ChemistryData => {
            "The above results contain chemical and thermodynamic data. \
             Present the molecular properties clearly in SI units. \
             Cross-reference NIST and PubChem values where both are present."
        }
        QueryIntent::AstrophysicsObject => {
            "The above results are from astronomical databases (NASA ADS, arXiv). \
             Describe the object's physical properties, distance, classification, \
             and any recent observational findings."
        }
        QueryIntent::NistConstant => {
            "The above results contain NIST CODATA physical constants. \
             Report the value with its full uncertainty in SI units, \
             and note whether the value is exact (defined) or measured."
        }
        QueryIntent::GeneralWeb => {
            "The above web search results are provided for context. \
             Use them to supplement your answer with current information. \
             Note the date of sources where relevant."
        }
        QueryIntent::DirectUrl => {
            "The above is the extracted content from the requested URL. \
             Summarise the key points relevant to the query."
        }
    };

    ctx.push_str(instruction);
    ctx.push_str(&format!("\n\nQuery: {query}"));
    ctx
}
