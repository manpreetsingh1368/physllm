// units.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantity { pub value: f64, pub unit: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Unit { pub name: String, pub si_factor: f64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnitSystem { SI, CGS, Gaussian, Natural }
