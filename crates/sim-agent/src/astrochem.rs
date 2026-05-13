//! astrochem.rs — ISM astrochemical network simulation.

use crate::{Result, dispatcher::{SimResult, SimType, PlotSpec, PlotKind, SeriesSpec}};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstrochemParams {
    pub density_cm3:    f64,   // number density H nuclei cm⁻³
    pub temperature_K:  f64,
    pub uv_field:       f64,   // in Habing units (G0)
    pub cosmic_ray_rate: f64,  // ζ, s⁻¹ per H nucleus (~1.3e-17 standard)
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

// thermodynamics.rs — Equation of state and phase diagram.


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicsParams {
    pub system:      ThermSystem,
    pub t_range:     [f64; 2],   // K
    pub p_range:     [f64; 2],   // Pa
    pub n_points:    usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermSystem {
    IdealGas { n_mol: f64 },
    VanDerWaals { a: f64, b: f64, n_mol: f64 },
    Blackbody,
    StellarAtmosphere { log_g: f64, t_eff: f64 },
}

pub struct ThermodynamicsSim { params: ThermodynamicsParams }

impl ThermodynamicsSim {
    pub fn new(params: ThermodynamicsParams) -> Self { Self { params } }

    pub fn run(&self, _max_steps: usize) -> Result<SimResult> {
        let p = &self.params;
        const R: f64 = 8.314_462_618;

        let ts: Vec<f64> = (0..p.n_points).map(|i| {
            p.t_range[0] + i as f64 * (p.t_range[1] - p.t_range[0]) / (p.n_points - 1) as f64
        }).collect();

        let (pressures, label, context): (Vec<f64>, String, String) = match &p.system {
            ThermSystem::IdealGas { n_mol } => {
                let ps: Vec<f64> = ts.iter().map(|&T| n_mol * R * T / 1.0).collect();
                (ps, "Ideal gas P vs T (V=1m³)".into(),
                 format!("Ideal gas law PV=nRT. n={:.3} mol. P scales linearly with T.", n_mol))
            }
            ThermSystem::VanDerWaals { a, b, n_mol } => {
                let v = 1.0; // 1 m³
                let ps: Vec<f64> = ts.iter().map(|&T| {
                    n_mol * R * T / (v - n_mol * b) - a * n_mol * n_mol / (v * v)
                }).collect();
                (ps, "Van der Waals P vs T".into(),
                 format!("Van der Waals gas: a={a:.4} Pa⋅m⁶/mol², b={b:.6} m³/mol. n={n_mol:.3} mol."))
            }
            ThermSystem::Blackbody => {
                const SIGMA: f64 = 5.670_374_419e-8;
                // Return power spectral peak wavelength via Wien's law
                let lambdas: Vec<f64> = ts.iter().map(|&T| 2.897_771_955e-3 / T).collect();
                let powers: Vec<f64>  = ts.iter().map(|&T| SIGMA * T.powi(4)).collect();
                let _ = lambdas;
                (powers, "Blackbody radiated power vs T".into(),
                 "Stefan-Boltzmann law: P = σT⁴. Wien's law: λ_max = 2.898mm/T.".into())
            }
            ThermSystem::StellarAtmosphere { log_g, t_eff } => {
                let ks: Vec<f64> = ts.iter().map(|&T| {
                    // Simple Kramers opacity approximation
                    0.02 * (1.0 + 0.003 * T.powi(3)) * 10_f64.powf(-log_g)
                }).collect();
                (ks, format!("Stellar opacity κ(T), log g={log_g:.2}"),
                 format!("Stellar atmosphere model: T_eff={t_eff:.0}K, log g={log_g:.2}. \
                          Kramers opacity approximation."))
            }
        };

        let summary = format!("Thermodynamics EoS: {:?}\nT range: {:.1}–{:.1}K",
            p.system, p.t_range[0], p.t_range[1]);

        let plot = PlotSpec {
            kind: PlotKind::Line, title: label,
            x_label: "T (K)".into(), y_label: "P or power".into(),
            series: vec![SeriesSpec { label: "result".into(), x: ts, y: pressures }],
        };

        Ok(SimResult {
            sim_type: SimType::ThermodynamicsEos,
            description: "Equation of state".into(),
            steps_run: p.n_points, wall_time_ms: 0,
            summary, data: serde_json::json!({}),
            plots: vec![plot], llm_context: context,
        })
    }
}
