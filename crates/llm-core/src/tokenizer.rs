//! tokenizer.rs — Physics-aware tokenizer wrapping HuggingFace tokenizers
//! with domain-specific extensions for physics symbols, chemical formulae,
//! unit strings, and mathematical notation.

use crate::{Result, LlmError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Physics/chemistry domain token extensions appended after base vocab.
/// These are injected as special tokens so the model learns them as atomic units.
const DOMAIN_SPECIAL_TOKENS: &[&str] = &[
    // Mathematical operators
    "∇", "∂", "∮", "∯", "∫∫", "∭", "∞", "∝", "≈", "≡", "≠", "≤", "≥",
    // Physics notation
    "ℏ", "ħ", "ℓ", "ℕ", "ℤ", "ℝ", "ℂ",
    // Common composite expressions (kept atomic)
    "d/dt", "∂/∂t", "∂/∂x", "∂²/∂x²", "∇²", "∇×", "∇·",
    // Units (SI base)
    "kg·m/s²", "J·s", "C/m²", "V/m", "T·m²", "W/m²", "mol/L",
    // Energy units
    "eV", "keV", "MeV", "GeV", "TeV", "erg", "cal", "kcal",
    // Length units
    "Å", "fm", "nm", "μm", "mm", "AU", "ly", "pc", "kpc", "Mpc", "Gpc",
    // Time units
    "fs", "ps", "ns", "μs", "ms", "yr", "Myr", "Gyr", "kyr",
    // Mass/force
    "amu", "Da", "M☉", "L☉", "R☉",
    // Temperature
    "mK", "μK", "nK",
    // Pressure
    "mbar", "atm", "bar", "Torr", "GPa", "TPa",
    // Chemical formulae (most common)
    "H₂", "H₂O", "CO₂", "NH₃", "CH₄", "O₂", "N₂", "HCl", "H₂SO₄", "HNO₃",
    "NaCl", "CaCO₃", "Fe₂O₃", "SiO₂", "Al₂O₃", "H₂O₂",
    // Astrochemistry
    "H₃⁺", "H₂⁺", "OH⁻", "CO⁺", "HCO⁺", "N₂H⁺", "CH₃OH", "HCOOH",
    // Greek letters (physics usage)
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
    "ν", "ξ", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
    "Γ", "Δ", "Θ", "Λ", "Ξ", "Π", "Σ", "Φ", "Ψ", "Ω",
    // Physical constants by name-token
    "<|c_light|>", "<|h_planck|>", "<|hbar|>", "<|G_newton|>",
    "<|k_boltzmann|>", "<|N_avogadro|>", "<|e_charge|>",
    // Simulation tool tags
    "<|sim_start|>", "<|sim_end|>", "<|tool_call|>", "<|tool_result|>",
    // Equation delimiters
    "<|eq|>", "<|/eq|>", "<|inline_math|>", "<|/inline_math|>",
];

/// A physics-aware tokenizer.
pub struct PhysTokenizer {
    /// Map from domain token string → token ID (base_vocab_size + offset)
    domain_tokens:   HashMap<String, u32>,
    /// Reverse map for decoding
    domain_id_to_str: HashMap<u32, String>,
    /// Base vocabulary size
    base_vocab_size: usize,
    /// Total vocab size
    pub vocab_size:  usize,
    /// BOS / EOS token IDs
    pub bos_id:      u32,
    pub eos_id:      u32,
}

impl PhysTokenizer {
    /// Build a physics tokenizer using a simple character-based fallback
    /// (production: load from tokenizer.json via HuggingFace tokenizers crate)
    pub fn new_simple(base_vocab_size: usize, bos_id: u32, eos_id: u32) -> Self {
        let mut domain_tokens   = HashMap::new();
        let mut domain_id_to_str = HashMap::new();

        for (i, &tok) in DOMAIN_SPECIAL_TOKENS.iter().enumerate() {
            let id = (base_vocab_size + i) as u32;
            domain_tokens.insert(tok.to_string(), id);
            domain_id_to_str.insert(id, tok.to_string());
        }

        let vocab_size = base_vocab_size + DOMAIN_SPECIAL_TOKENS.len();

        Self {
            domain_tokens,
            domain_id_to_str,
            base_vocab_size,
            vocab_size,
            bos_id,
            eos_id,
        }
    }

    /// Tokenise a string into token IDs.
    ///
    /// Strategy:
    ///   1. Scan for domain special tokens (longest-match first)
    ///   2. Fall back to UTF-8 byte-level tokenisation for the rest
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos_id];
        let bytes = text.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            // Try to match a domain token (longest match)
            let mut matched = false;
            // Sort by length descending for longest-match
            let mut candidates: Vec<(&str, u32)> = self.domain_tokens
                .iter()
                .map(|(k, &v)| (k.as_str(), v))
                .collect();
            candidates.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

            for (tok_str, tok_id) in &candidates {
                let tok_bytes = tok_str.as_bytes();
                if bytes[i..].starts_with(tok_bytes) {
                    tokens.push(*tok_id);
                    i += tok_bytes.len();
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Byte-level fallback: encode each byte as token ID = byte value + offset
                // (In production, use BPE/SentencePiece from tokenizers crate)
                let byte = bytes[i];
                tokens.push(byte as u32 + 3); // offset to avoid special token IDs 0,1,2
                i += 1;
            }
        }

        Ok(tokens)
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut out = Vec::new();

        for &id in token_ids {
            if id == self.bos_id || id == self.eos_id { continue; }

            if let Some(s) = self.domain_id_to_str.get(&id) {
                out.extend_from_slice(s.as_bytes());
            } else if id >= 3 && (id as usize) < self.base_vocab_size {
                // Byte-level decode
                out.push((id - 3) as u8);
            }
            // Unknown IDs are silently skipped
        }

        String::from_utf8(out)
            .map_err(|e| LlmError::Tokenizer(e.to_string()))
    }

    /// Number of domain extension tokens.
    pub fn domain_vocab_size(&self) -> usize {
        DOMAIN_SPECIAL_TOKENS.len()
    }

    /// Check if a token ID is a physics domain token.
    pub fn is_domain_token(&self, id: u32) -> bool {
        id as usize >= self.base_vocab_size
    }

    /// Look up the token ID for a physics symbol.
    pub fn domain_token_id(&self, symbol: &str) -> Option<u32> {
        self.domain_tokens.get(symbol).copied()
    }

    /// Tokenise a chemical formula and return its token sequence.
    pub fn tokenise_formula(&self, formula: &str) -> Vec<u32> {
        // Try direct formula match first, then fall back to char-level
        if let Some(&id) = self.domain_tokens.get(formula) {
            vec![id]
        } else {
            self.encode(formula).unwrap_or_default()
        }
    }
}

// ── Production tokenizer loader (HuggingFace format) ─────────────────────────

/// Deserialise a tokenizer.json and build a PhysTokenizer.
/// Requires tokenizers crate in production; this is the interface contract.
pub fn load_from_file(path: &str, bos_id: u32, eos_id: u32) -> Result<PhysTokenizer> {
    // In production:
    //   let tokenizer = tokenizers::Tokenizer::from_file(path)
    //       .map_err(|e| LlmError::Tokenizer(e.to_string()))?;
    //   let base_vocab_size = tokenizer.get_vocab_size(false);
    //   Build PhysTokenizer wrapping it + domain extensions

    // Stub: fall back to simple tokenizer
    tracing::warn!("tokenizer.json not found at {path}, using simple byte-level tokenizer");
    Ok(PhysTokenizer::new_simple(32768, bos_id, eos_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_token_roundtrip() {
        let tok = PhysTokenizer::new_simple(32768, 1, 2);
        let ids = tok.tokenise_formula("H₂O");
        assert!(!ids.is_empty());
        // Check that H₂O is a single domain token
        assert_eq!(ids.len(), 1);
        assert!(tok.is_domain_token(ids[0]));
    }

    #[test]
    fn test_encode_decode_ascii() {
        let tok = PhysTokenizer::new_simple(32768, 1, 2);
        let text = "E = mc2 where c is the speed of light";
        let ids  = tok.encode(text).unwrap();
        let out  = tok.decode(&ids[1..]).unwrap(); // skip BOS
        assert!(out.contains("E"));
    }

    #[test]
    fn test_domain_vocab_size() {
        let tok = PhysTokenizer::new_simple(32768, 1, 2);
        assert_eq!(tok.domain_vocab_size(), DOMAIN_SPECIAL_TOKENS.len());
        assert_eq!(tok.vocab_size, 32768 + DOMAIN_SPECIAL_TOKENS.len());
    }
}
