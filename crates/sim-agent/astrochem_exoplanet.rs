//! astrochem_exoplanet.rs - ISM astro_Exoplanet NASA network simulation
use std::f64::consts::E;
use crate::{Result,dispatcher::{SimResult,SimType,PlotSpec,PlotKind,SeriesSpec}};
 use serde::{Deserialize, Serialize}

 #[derive (Debug,Clone,Serialize,Deserialize)]
  
 pub struct astrochem_exoplanetParams{


    pub  mut density_cm3;
    pub mut  temperature_K as f64;
    pub  distant_Earth as f64;

 }
 struct CloudSpecies {
    name: String,
    delta_h_sub: f64, // J/mol
    c_const: f64,
    molar_mass: f64, // g/mol
}

impl CloudSpecies {
    fn calculate_p_sat(&self, temperature_k: f64) -> f64 {
        let r_gas_constant = 8.314; // J/(mol*K) – corrected value
        let exponent = -(self.delta_h_sub / (r_gas_constant * temperature_k)) + self.c_const;
        E.powf(exponent) // gives pressure in bar or Pa depending on C constant
    }

    fn check_condensation(&self, partial_pressure: f64, temperature_k: f64) -> bool {
        let p_sat = self.calculate_p_sat(temperature_k);
        partial_pressure >= p_sat
    }
}