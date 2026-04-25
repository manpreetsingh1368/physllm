//! reaction_kinetics.rs — Chemical kinetics ODE system via RK45 (adaptive step).
//!
//! Models:
//!   - Arbitrary reaction network (stoichiometry + rate constants)
//!   - Arrhenius temperature dependence
//!   - Interstellar medium (ISM) astrochemical network (Woodall UMIST)
//!   - Combustion (H2-O2 ignition)

use crate::{Result, SimError, dispatcher::{SimResult, SimType, PlotSpec, PlotKind, SeriesSpec}};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reaction {
    pub name:       String,
    pub reactants:  Vec<(String, u32)>,  // (species, stoichiometry)
    pub products:   Vec<(String, u32)>,
    pub rate_type:  RateType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateType {
    FirstOrder  { k: f64 },
    SecondOrder { k: f64 },
    Arrhenius   { A: f64, Ea: f64 },     // k = A * exp(-Ea/RT)
    PowerLaw    { k: f64, n: f64 },      // k * [A]^n
    ISM         { alpha: f64, beta: f64, gamma: f64 }, // k = α(T/300)^β exp(-γ/T)
}

impl RateType {
    pub fn rate(&self, conc: f64, conc2: Option<f64>, T: f64) -> f64 {
        const R: f64 = 8.314_462_618;
        match self {
            Self::FirstOrder  { k }        => k * conc,
            Self::SecondOrder { k }        => k * conc * conc2.unwrap_or(conc),
            Self::Arrhenius   { A, Ea }    => A * (-Ea / (R * T)).exp() * conc,
            Self::PowerLaw    { k, n }     => k * conc.powf(*n),
            Self::ISM { alpha, beta, gamma } => alpha * (T / 300.0).powf(*beta) * (-gamma / T).exp() * conc,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KineticsParams {
    pub species:     Vec<String>,              // all species names
    pub initial_conc: Vec<f64>,               // initial concentrations (mol/L or cm^-3)
    pub reactions:   Vec<Reaction>,
    pub temperature: f64,                      // K
    pub t_end:       f64,                      // total time (s)
    pub dt_init:     f64,                      // initial time step
    pub preset:      Option<KineticsPreset>,
    pub rel_tol:     f64,                      // RK45 relative tolerance
    pub abs_tol:     f64,                      // RK45 absolute tolerance
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KineticsPreset {
    SimpsonOscillator,    // classic oscillating reaction (Brusselator)
    H2O2Combustion,
    OzoneDepletion,
    ISMHydrogenChemistry,
}

pub struct KineticsSim { params: KineticsParams }

impl KineticsSim {
    pub fn new(mut params: KineticsParams) -> Self {
        if let Some(preset) = &params.preset.clone() {
            let (species, concs, reactions) = preset_network(preset, params.temperature);
            params.species = species;
            params.initial_conc = concs;
            params.reactions = reactions;
        }
        Self { params }
    }

    pub fn run(&self, max_steps: usize) -> Result<SimResult> {
        let p = &self.params;
        let ns = p.species.len();

        if p.initial_conc.len() != ns {
            return Err(SimError::InvalidParameter(
                format!("species count {} != initial_conc count {}", ns, p.initial_conc.len())
            ));
        }

        // Build stoichiometry matrix (net: products - reactants)
        // stoich[reaction][species] = net change per unit reaction
        let mut stoich = vec![vec![0i32; ns]; p.reactions.len()];
        let species_idx: std::collections::HashMap<&str, usize> =
            p.species.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

        for (ri, rxn) in p.reactions.iter().enumerate() {
            for (name, coeff) in &rxn.reactants {
                if let Some(&si) = species_idx.get(name.as_str()) {
                    stoich[ri][si] -= *coeff as i32;
                }
            }
            for (name, coeff) in &rxn.products {
                if let Some(&si) = species_idx.get(name.as_str()) {
                    stoich[ri][si] += *coeff as i32;
                }
            }
        }

        // RHS: dy/dt = sum_r stoich[r][i] * rate_r(y, T)
        let rhs = |conc: &[f64]| -> Vec<f64> {
            let mut dydt = vec![0.0; ns];
            for (ri, rxn) in p.reactions.iter().enumerate() {
                // Compute reactant concentrations
                let rc: Vec<f64> = rxn.reactants.iter()
                    .filter_map(|(n, _)| species_idx.get(n.as_str()).map(|&i| conc[i].max(0.0)))
                    .collect();
                if rc.is_empty() { continue; }
                let r = rxn.rate_type.rate(rc[0], rc.get(1).copied(), p.temperature);
                if r.is_nan() || r.is_infinite() { continue; }
                for (si, &s) in stoich[ri].iter().enumerate() {
                    dydt[si] += s as f64 * r;
                }
            }
            dydt
        };

        // Adaptive RK45 (Dormand-Prince)
        let mut t = 0.0f64;
        let mut y = p.initial_conc.clone();
        let mut dt = p.dt_init;
        let rtol = p.rel_tol.max(1e-12);
        let atol = p.abs_tol.max(1e-15);

        let mut times: Vec<f64>                = vec![t];
        let mut history: Vec<Vec<f64>>         = vec![y.clone()];
        let mut step = 0;

        while t < p.t_end && step < max_steps {
            dt = dt.min(p.t_end - t);
            if dt < 1e-30 { break; }

            // RK45 Dormand-Prince tableau
            let k1 = rhs(&y);
            let y2: Vec<f64> = y.iter().zip(k1.iter()).map(|(&yi, &k)| yi + dt * k / 5.0).collect();
            let k2 = rhs(&y2);
            let y3: Vec<f64> = y.iter().zip(k1.iter()).zip(k2.iter())
                .map(|((&yi, &k1i), &k2i)| yi + dt * (3.0*k1i + 9.0*k2i) / 40.0).collect();
            let k3 = rhs(&y3);
            let y4: Vec<f64> = y.iter().enumerate()
                .map(|(i, &yi)| yi + dt * (44.0*k1[i] - 168.0*k2[i] + 160.0*k3[i]) / 45.0).collect();
            let k4 = rhs(&y4);
            let y5: Vec<f64> = y.iter().enumerate()
                .map(|(i, &yi)| yi + dt * (19372.0*k1[i] - 76080.0*k2[i]
                    + 64448.0*k3[i] - 1908.0*k4[i]) / 6561.0).collect();
            let k5 = rhs(&y5);

            // 5th order update
            let y_new: Vec<f64> = y.iter().enumerate()
                .map(|(i, &yi)| yi + dt * (35.0*k1[i]/384.0 + 500.0*k3[i]/1113.0
                    + 125.0*k4[i]/192.0 - 2187.0*k5[i]/6784.0)).collect();

            // 4th order (for error estimate)
            let y_err: Vec<f64> = y.iter().enumerate()
                .map(|(i, &yi)| yi + dt * (5179.0*k1[i]/57600.0 + 7571.0*k3[i]/16695.0
                    - 415.0*k4[i]/3456.0 - 9.0*k5[i]/35.0)).collect();

            // Error estimate
            let err: f64 = y_new.iter().zip(y_err.iter()).zip(y.iter())
                .map(|((&yn, &ye), &yi)| {
                    let scale = atol + rtol * yn.abs().max(yi.abs());
                    ((yn - ye) / scale).powi(2)
                }).sum::<f64>().sqrt() / ns as f64;

            if err <= 1.0 || dt < 1e-30 {
                t += dt;
                y = y_new.iter().map(|&v| v.max(0.0)).collect(); // clamp negative concentrations
                step += 1;
                times.push(t);
                history.push(y.clone());
            }

            // Adjust step size (PI controller)
            let factor = if err > 0.0 { 0.9 * err.powf(-0.2).min(5.0).max(0.1) } else { 2.0 };
            dt *= factor;
        }

        let summary = format!(
            "Reaction kinetics: {} species, {} reactions, T={:.1}K\n\
             Integrated to t={:.3e}s in {} steps\n\
             Final concentrations:\n{}",
            ns, p.reactions.len(), p.temperature, t, step,
            p.species.iter().zip(y.iter())
                .map(|(s, &c)| format!("  {s}: {c:.4e}"))
                .collect::<Vec<_>>().join("\n")
        );

        let llm_context = format!(
            "Chemical kinetics simulation: {} species, {} reactions at T={:.1}K. \
             Integrated from t=0 to t={:.3e}s using adaptive RK45. \
             Final concentrations: {}.",
            ns, p.reactions.len(), p.temperature, t,
            p.species.iter().zip(y.iter())
                .map(|(s, &c)| format!("{s}={c:.3e}"))
                .collect::<Vec<_>>().join(", ")
        );

        // Build time series plots per species
        let series: Vec<SeriesSpec> = p.species.iter().enumerate().map(|(si, name)| {
            SeriesSpec {
                label: name.clone(),
                x:     times.clone(),
                y:     history.iter().map(|h| h[si]).collect(),
            }
        }).collect();

        let conc_plot = PlotSpec {
            kind:    PlotKind::Line,
            title:   "Species concentrations over time".into(),
            x_label: "Time (s)".into(),
            y_label: "Concentration".into(),
            series,
        };

        let data = serde_json::json!({
            "times": times,
            "species": p.species,
            "final_concentrations": y,
            "steps": step,
        });

        Ok(SimResult {
            sim_type:    SimType::ReactionKinetics,
            description: format!("Chemical kinetics ({} species)", ns),
            steps_run:   step,
            wall_time_ms: 0,
            summary,
            data,
            plots: vec![conc_plot],
            llm_context,
        })
    }
}

fn preset_network(preset: &KineticsPreset, T: f64) -> (Vec<String>, Vec<f64>, Vec<Reaction>) {
    match preset {
        KineticsPreset::SimpsonOscillator => (
            vec!["A".into(), "B".into(), "X".into(), "Y".into()],
            vec![1.0, 3.0, 0.01, 0.01],
            vec![
                Reaction { name: "r1".into(), reactants: vec![("A".into(),1),("X".into(),2)], products: vec![("X".into(),3)], rate_type: RateType::SecondOrder { k: 1.0 } },
                Reaction { name: "r2".into(), reactants: vec![("X".into(),1),("B".into(),1)], products: vec![("Y".into(),1)], rate_type: RateType::SecondOrder { k: 1.0 } },
                Reaction { name: "r3".into(), reactants: vec![("Y".into(),1)],                products: vec![],              rate_type: RateType::FirstOrder  { k: 1.0 } },
                Reaction { name: "r4".into(), reactants: vec![("X".into(),1)],                products: vec![],              rate_type: RateType::FirstOrder  { k: 1.0 } },
            ]
        ),
        KineticsPreset::OzoneDepletion => (
            vec!["O3".into(), "O".into(), "O2".into(), "Cl".into(), "ClO".into()],
            vec![3e12, 1e8, 5e14, 1e6, 0.0],
            vec![
                Reaction { name: "photolysis".into(),    reactants: vec![("O3".into(),1)],               products: vec![("O".into(),1),("O2".into(),1)], rate_type: RateType::FirstOrder  { k: 3e-4 } },
                Reaction { name: "ozone_reform".into(),  reactants: vec![("O".into(),1),("O2".into(),1)], products: vec![("O3".into(),1)],               rate_type: RateType::SecondOrder { k: 6e-34 } },
                Reaction { name: "Cl_attack".into(),     reactants: vec![("Cl".into(),1),("O3".into(),1)],products: vec![("ClO".into(),1),("O2".into(),1)],rate_type: RateType::SecondOrder { k: 3e-11 } },
                Reaction { name: "ClO_O_react".into(),   reactants: vec![("ClO".into(),1),("O".into(),1)],products: vec![("Cl".into(),1),("O2".into(),1)], rate_type: RateType::SecondOrder { k: 5e-11 } },
            ]
        ),
        KineticsPreset::ISMHydrogenChemistry => (
            vec!["H".into(), "H2".into(), "H+".into(), "H2+".into(), "H3+".into(), "e-".into()],
            vec![100.0, 10.0, 0.1, 0.0, 0.0, 0.1],
            vec![
                Reaction { name: "H2_form_dust".into(),    reactants: vec![("H".into(),2)],              products: vec![("H2".into(),1)],             rate_type: RateType::ISM { alpha: 3e-17, beta: 0.0,  gamma: 0.0 } },
                Reaction { name: "H2_photodiss".into(),     reactants: vec![("H2".into(),1)],             products: vec![("H".into(),2)],              rate_type: RateType::FirstOrder { k: 5e-11 } },
                Reaction { name: "H_ionise".into(),         reactants: vec![("H".into(),1)],              products: vec![("H+".into(),1),("e-".into(),1)], rate_type: RateType::FirstOrder { k: 1e-12 } },
                Reaction { name: "H2_ionise".into(),        reactants: vec![("H2".into(),1)],             products: vec![("H2+".into(),1),("e-".into(),1)], rate_type: RateType::ISM { alpha: 1.2e-17, beta: 0.0, gamma: 0.0 } },
                Reaction { name: "H2+_react".into(),        reactants: vec![("H2+".into(),1),("H2".into(),1)], products: vec![("H3+".into(),1),("H".into(),1)], rate_type: RateType::SecondOrder { k: 2e-9 } },
                Reaction { name: "H3+_e_recomb".into(),     reactants: vec![("H3+".into(),1),("e-".into(),1)], products: vec![("H".into(),3)],         rate_type: RateType::SecondOrder { k: 2.3e-7 } },
            ]
        ),
        KineticsPreset::H2O2Combustion => (
            vec!["H2".into(), "O2".into(), "H".into(), "O".into(), "OH".into(), "H2O".into(), "HO2".into(), "H2O2".into()],
            vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![
                Reaction { name: "H2_initiation".into(), reactants: vec![("H2".into(),1),("O2".into(),1)], products: vec![("H".into(),1),("HO2".into(),1)], rate_type: RateType::Arrhenius { A: 7.4e13, Ea: 209_000.0 } },
                Reaction { name: "H_O2_chain".into(),   reactants: vec![("H".into(),1),("O2".into(),1)],  products: vec![("O".into(),1),("OH".into(),1)],   rate_type: RateType::Arrhenius { A: 5.1e16, Ea: 68_900.0  } },
                Reaction { name: "O_H2_chain".into(),   reactants: vec![("O".into(),1),("H2".into(),1)],  products: vec![("H".into(),1),("OH".into(),1)],   rate_type: RateType::Arrhenius { A: 1.8e10, Ea: 36_900.0  } },
                Reaction { name: "OH_H2_chain".into(),  reactants: vec![("OH".into(),1),("H2".into(),1)], products: vec![("H2O".into(),1),("H".into(),1)],  rate_type: RateType::Arrhenius { A: 2.2e13, Ea: 21_700.0  } },
                Reaction { name: "termination".into(),  reactants: vec![("H".into(),1),("OH".into(),1)],  products: vec![("H2O".into(),1)],                 rate_type: RateType::SecondOrder { k: 2.2e22 } },
            ]
        ),
    }
}