//! stellar.rs — Simplified stellar evolution model (HR diagram track).

use crate::{Result, dispatcher::{SimResult, SimType, PlotSpec, PlotKind, SeriesSpec}};
use serde::{Deserialize, Serialize};

const MSUN: f64 = 1.989e30;
const LSUN: f64 = 3.828e26;
const TSUN: f64 = 5778.0;
const SIGMA: f64 = 5.670_374_419e-8;
const RSUN: f64 = 6.957e8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StellarParams {
    pub mass_solar:    f64,    // stellar mass in solar masses
    pub metallicity:   f64,    // Z (0.02 = solar)
    pub initial_x_h:   f64,    // initial hydrogen fraction (~0.74)
    pub t_end_yr:      f64,    // time to evolve (years)
}

pub struct StellarSim { params: StellarParams }

impl StellarSim {
    pub fn new(params: StellarParams) -> Self { Self { params } }

    pub fn run(&self, max_steps: usize) -> Result<SimResult> {
        let p = &self.params;
        let m = p.mass_solar;

        // Main sequence lifetime (Schoenberg-Chandrasekhar approximation)
        let t_ms_yr = 1e10 * (m / (m.powf(4.0) * LSUN / (m * LSUN))).max(1e-3);
        let t_ms_yr = 1e10 / m.powf(2.5);  // simple approximation: t_MS ∝ M^{-2.5}

        // Zero-age main sequence luminosity & temperature (empirical)
        let l_zams  = LSUN * m.powf(4.0);
        let t_zams  = TSUN * m.powf(0.57);
        let r_zams  = (l_zams / (4.0 * std::f64::consts::PI * SIGMA * t_zams.powi(4))).sqrt();

        let dt_yr = p.t_end_yr / max_steps as f64;
        let yr_to_s = 3.156e7;

        let mut times_yr: Vec<f64>  = Vec::new();
        let mut lum: Vec<f64>       = Vec::new();
        let mut teff: Vec<f64>      = Vec::new();
        let mut radius: Vec<f64>    = Vec::new();
        let mut x_h: Vec<f64>       = Vec::new();

        let mut xh = p.initial_x_h;

        for step in 0..max_steps {
            let t_yr = step as f64 * dt_yr;
            let frac = (t_yr / t_ms_yr).min(1.0);

            // Simple evolutionary track
            let l_now  = l_zams * (1.0 + 0.5 * frac);           // luminosity increases on MS
            let xh_now = xh * (1.0 - frac);                     // hydrogen burns
            let t_now  = if frac < 0.9 {
                t_zams * (1.0 - 0.05 * frac)                    // slight cooling on ZAMS
            } else {
                t_zams * (0.955 - 2.0 * (frac - 0.9))           // rapid cooling at turn-off
            };
            let r_now  = (l_now / (4.0 * std::f64::consts::PI * SIGMA * t_now.powi(4))).sqrt();

            times_yr.push(t_yr);
            lum.push(l_now / LSUN);
            teff.push(t_now);
            radius.push(r_now / RSUN);
            x_h.push(xh_now);

            if step % (max_steps / 20).max(1) == 0 {
                xh = xh_now;
            }

            if frac >= 1.0 { break; }
        }

        let l_fin = lum.last().copied().unwrap_or(0.0);
        let t_fin = teff.last().copied().unwrap_or(0.0);
        let r_fin = radius.last().copied().unwrap_or(0.0);

        let post_ms = if m > 8.0 { "→ Type II supernova → neutron star / black hole"
        } else if m > 0.8 { "→ Red giant → planetary nebula → white dwarf"
        } else { "→ Red dwarf (very slow evolution, Hubble time exceeded)" };

        let summary = format!(
            "Stellar evolution: {:.2} M☉, Z={:.4}\n\
             ZAMS: L={:.3}L☉  T_eff={:.0}K  R={:.3}R☉\n\
             MS lifetime: {:.3e} yr\n\
             Evolved to {:.3e} yr → L={:.3}L☉  T={:.0}K  R={:.3}R☉\n\
             Next stage: {post_ms}",
            m, p.metallicity,
            l_zams/LSUN, t_zams, r_zams/RSUN,
            t_ms_yr, p.t_end_yr,
            l_fin, t_fin, r_fin
        );

        let llm_context = format!(
            "Stellar evolution simulation for a {:.2} solar mass star (Z={:.4}). \
             Zero-age main sequence: L={:.3}L☉, T_eff={:.0}K, R={:.3}R☉. \
             Main sequence lifetime ≈ {:.2e} yr. \
             After {:.2e} yr: L={:.3}L☉, T_eff={:.0}K, R={:.3}R☉. \
             Post-main-sequence fate: {post_ms}",
            m, p.metallicity, l_zams/LSUN, t_zams, r_zams/RSUN,
            t_ms_yr, p.t_end_yr, l_fin, t_fin, r_fin
        );

        let hr_plot = PlotSpec {
            kind:    PlotKind::Scatter,
            title:   "HR diagram evolutionary track".into(),
            x_label: "T_eff (K)".into(),
            y_label: "L / L☉".into(),
            series:  vec![SeriesSpec {
                label: format!("{:.1}M☉", m),
                x: teff.iter().rev().cloned().collect(),   // HR: T increases right to left
                y: lum.clone(),
            }],
        };

        let data = serde_json::json!({
            "times_yr": times_yr, "luminosity_lsun": lum,
            "teff_K": teff, "radius_rsun": radius,
            "hydrogen_fraction": x_h,
            "ms_lifetime_yr": t_ms_yr,
        });

        Ok(SimResult {
            sim_type: SimType::StellarEvolution,
            description: format!("{:.2} M☉ stellar evolution", m),
            steps_run: times_yr.len(),
            wall_time_ms: 0,
            summary, data, plots: vec![hr_plot], llm_context,
        })
    }
}
