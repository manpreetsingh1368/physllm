//! astrochem_exoplanet.rs - ISM astro_Exoplanet NASA network simulation

use crate::{Result,dispatcher::{SimResult,SimType,PlotSpec,PlotKind,SeriesSpec}};
 use serde::{Deserialize, Serialize}

 #[derive (Debug,Clone,Serialize,Deserialize)]
  
 pub struct astrochem_exoplanetParams{


    pub  mut density_cm3;
    pub mut  temperature_K as f64;
    pub  distant_Earth as f64;
    
 }