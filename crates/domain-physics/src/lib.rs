//! domain-physics — Physics/chemistry knowledge and formula handling.
//!
//! Provides:
//!   - NIST CODATA physical constants database
//!   - Chemical formula parser and molecular weight calculator
//!   - Unit conversion engine (SI)
//!   - Domain vocabulary extensions for the tokenizer
//!   - Equation of state lookups

pub mod constants;
pub mod chemistry;
pub mod units;
pub mod vocab;
pub mod equations;
pub mod astrophysics;

pub use constants::{PhysicalConstant, ConstantsDB};
pub use chemistry::{Molecule, ChemicalFormula};
pub use units::{Quantity, Unit, UnitSystem};
pub use astrophysics::AstroObject;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DomainError {
    #[error("Unknown element: {0}")]
    UnknownElement(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Unit mismatch: cannot convert {from} to {to}")]
    UnitMismatch { from: String, to: String },
    #[error("Constant not found: {0}")]
    ConstantNotFound(String),
}

pub type Result<T> = std::result::Result<T, DomainError>;
