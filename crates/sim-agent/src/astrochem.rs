//! astrochem.rs — ISM astrochemical network simulation.

use crate::{Result, dispatcher::{SimResult, SimType, PlotSpec, PlotKind, SeriesSpec}};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstrochemParams {
    pub density_cm3:    f64,   // number density H nuclei cm^³
    pub temperature_K:  f64,
    pub uv_field:       f64,   // in Habing units (G0)
    pub cosmic_ray_rate: f64,  // ζ(lbm), s^¹ per H nucleus (~1.3e-17 standard)
    pub t_end_yr:       f64,
    pub av:             f64,   // visual extinction (mag)
}

pub struct AstrochemSim { params: AstrochemParams }

impl AstrochemSim {
    pub fn new(params: AstrochemParams) -> Self { Self { params } }

    pub fn run(&self, max_steps: usize) -> Result<SimResult> {
        let p = &self.params;
        // Delegate to the kinetics engine with UMIST-simplified ISM network
        use crate::reaction_kinetics::{KineticsSim, KineticsParams, KineticsPreset};
        let params = KineticsParams {
            species: vec![], initial_conc: vec![], reactions: vec![],
            temperature: p.temperature_K,
            t_end: p.t_end_yr * 3.156e7,
            dt_init: p.t_end_yr * 3.156e7 / max_steps as f64,
            preset: Some(KineticsPreset::ISMHydrogenChemistry),
            rel_tol: 1e-6, abs_tol: 1e-20,
        };
        let mut result = KineticsSim::new(params).run(max_steps)?;

        result.sim_type = SimType::AstrochemNetwork;
        result.description = format!(
            "ISM astrochemistry: n_H={:.2e}cm⁻³ T={:.1}K UV_G0={:.2} ζ={:.2e}s⁻¹ Av={:.2}",
            p.density_cm3, p.temperature_K, p.uv_field, p.cosmic_ray_rate, p.av
        );
        result.llm_context = format!(
            "Astrochemical network simulation of the interstellar medium. \
             Conditions: hydrogen nuclei density {:.2e} cm⁻³, temperature {:.1}K, \
             UV field G0={:.2} (Habing units), cosmic ray ionisation rate ζ={:.2e} s⁻¹, \
             visual extinction Av={:.2} mag. \
             Evolved from t=0 to {:.2e} yr. {}",
            p.density_cm3, p.temperature_K, p.uv_field, p.cosmic_ray_rate,
            p.av, p.t_end_yr, result.llm_context
        );
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────