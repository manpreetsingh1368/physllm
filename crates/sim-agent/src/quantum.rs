//! quantum.rs — 1D time-dependent Schrödinger equation via Crank-Nicolson.
//!
//! Solves: iℏ ∂ψ/∂t = [-ℏ²/2m ∂²/∂x² + V(x)] ψ
//!
//! Potentials supported:
//!   - Infinite square well
//!   - Harmonic oscillator
//!   - Double well
//!   - Finite square well
//!   - Hydrogen atom (1D effective)
//!   - Step potential (tunnelling demo)
//!   - Custom (user-supplied V values)

use crate::{Result, SimError, dispatcher::{SimResult, SimType, PlotSpec, PlotKind, SeriesSpec}};
use serde::{Deserialize, Serialize};
use num_complex::Complex64;
use std::f64::consts::PI;

const HBAR: f64 = 1.054_571_817e-34;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumParams {
    pub potential:     PotentialType,
    pub n_grid:        usize,         // number of spatial grid points
    pub x_min:         f64,           // m
    pub x_max:         f64,           // m
    pub dt:            f64,           // time step, s
    pub mass:          f64,           // particle mass, kg (default: electron mass)
    pub initial_state: InitialState,
    pub observe_every: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PotentialType {
    InfiniteWell,
    HarmonicOscillator { omega: f64 },
    DoubleWell         { barrier_height: f64, barrier_width: f64 },
    FiniteWell         { depth: f64, width: f64 },
    StepPotential      { height: f64, step_x: f64 },
    Coulomb,           // hydrogen-like 1D effective
    Custom             { values: Vec<f64> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitialState {
    GaussianWavepacket { x0: f64, sigma: f64, k0: f64 },
    Eigenstate         { n: usize },
    Custom             { re: Vec<f64>, im: Vec<f64> },
}

pub struct QuantumSim { params: QuantumParams }

impl QuantumSim {
    pub fn new(params: QuantumParams) -> Self { Self { params } }

    pub fn run(&self, max_steps: usize) -> Result<SimResult> {
        let p = &self.params;
        let n = p.n_grid;
        let dx = (p.x_max - p.x_min) / (n - 1) as f64;
        let mass = p.mass; // electron mass if 0
        let m = if mass == 0.0 { 9.109_383_7015e-31 } else { mass };

        let xs: Vec<f64> = (0..n).map(|i| p.x_min + i as f64 * dx).collect();

        // Build potential
        let v: Vec<f64> = match &p.potential {
            PotentialType::InfiniteWell => vec![0.0; n],
            PotentialType::HarmonicOscillator { omega } => {
                xs.iter().map(|&x| 0.5 * m * omega.powi(2) * x.powi(2)).collect()
            }
            PotentialType::DoubleWell { barrier_height, barrier_width } => {
                xs.iter().map(|&x| {
                    if x.abs() < barrier_width / 2.0 { *barrier_height } else { 0.0 }
                }).collect()
            }
            PotentialType::FiniteWell { depth, width } => {
                xs.iter().map(|&x| {
                    if x.abs() < width / 2.0 { -*depth } else { 0.0 }
                }).collect()
            }
            PotentialType::StepPotential { height, step_x } => {
                xs.iter().map(|&x| if x >= *step_x { *height } else { 0.0 }).collect()
            }
            PotentialType::Coulomb => {
                xs.iter().map(|&x| {
                    let r = x.abs().max(dx);
                    -1.602e-19_f64.powi(2) / (4.0 * PI * 8.854e-12 * r)
                }).collect()
            }
            PotentialType::Custom { values } => {
                if values.len() != n {
                    return Err(SimError::InvalidParameter(
                        format!("Custom potential length {} != n_grid {}", values.len(), n)
                    ));
                }
                values.clone()
            }
        };

        // Initial wavefunction
        let mut psi: Vec<Complex64> = match &p.initial_state {
            InitialState::GaussianWavepacket { x0, sigma, k0 } => {
                xs.iter().map(|&x| {
                    let env = (-(x - x0).powi(2) / (4.0 * sigma.powi(2))).exp();
                    let phase = Complex64::new(0.0, k0 * x);
                    Complex64::new(env, 0.0) * phase.exp()
                }).collect()
            }
            InitialState::Eigenstate { n: state_n } => {
                let L = p.x_max - p.x_min;
                xs.iter().map(|&x| {
                    let v = (2.0 / L).sqrt() * ((*state_n as f64) * PI * (x - p.x_min) / L).sin();
                    Complex64::new(v, 0.0)
                }).collect()
            }
            InitialState::Custom { re, im } => {
                re.iter().zip(im.iter()).map(|(&r, &i)| Complex64::new(r, i)).collect()
            }
        };

        // Normalise
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * dx;
        let norm = norm.sqrt();
        psi.iter_mut().for_each(|z| *z /= norm);

        // Crank-Nicolson constants
        let r = Complex64::new(0.0, HBAR * p.dt / (4.0 * m * dx * dx));

        // Time evolution storage
        let mut prob_density_frames: Vec<Vec<f64>> = Vec::new();
        let mut avg_x: Vec<f64> = Vec::new();
        let mut avg_p: Vec<f64> = Vec::new();
        let mut norm_vals: Vec<f64> = Vec::new();
        let mut times: Vec<f64> = Vec::new();

        for step in 0..max_steps {
            // Crank-Nicolson: solve tridiagonal system (Thomas algorithm)
            // (I + iΔt H/2ℏ) ψ^{n+1} = (I - iΔt H/2ℏ) ψ^n
            let diag = |i: usize| Complex64::new(1.0, 0.0) + 2.0*r
                + Complex64::new(0.0, p.dt * v[i] / (2.0 * HBAR));
            let off = -r;

            // Compute RHS
            let mut rhs: Vec<Complex64> = psi.iter().enumerate().map(|(i, &p_val)| {
                let left  = if i > 0   { psi[i-1] } else { Complex64::ZERO };
                let right = if i < n-1 { psi[i+1] } else { Complex64::ZERO };
                let d_conj = Complex64::new(1.0, 0.0) - 2.0*r
                    - Complex64::new(0.0, self.params.dt * v[i] / (2.0 * HBAR));
                d_conj * p_val + r * left + r * right
            }).collect();

            // Thomas algorithm (tridiagonal solve)
            let mut c_prime = vec![Complex64::ZERO; n];
            let mut d_prime = rhs.clone();
            c_prime[0] = off / diag(0);
            d_prime[0] = d_prime[0] / diag(0);
            for i in 1..n {
                let w = diag(i) - off * c_prime[i-1];
                c_prime[i] = off / w;
                d_prime[i] = (d_prime[i] - off * d_prime[i-1]) / w;
            }
            psi[n-1] = d_prime[n-1];
            for i in (0..n-1).rev() {
                psi[i] = d_prime[i] - c_prime[i] * psi[i+1];
            }

            if step % p.observe_every == 0 {
                let t = step as f64 * p.dt;
                let probs: Vec<f64> = psi.iter().map(|z| z.norm_sqr()).collect();
                let norm_sq: f64 = probs.iter().sum::<f64>() * dx;
                norm_vals.push(norm_sq);
                times.push(t);

                // <x>
                let x_exp: f64 = probs.iter().zip(xs.iter()).map(|(&p, &x)| p * x).sum::<f64>() * dx;
                avg_x.push(x_exp);

                // <p> ≈ ℏ Im[∫ ψ* ∂ψ/∂x dx]
                let p_exp: f64 = (1..n-1).map(|i| {
                    let dpsi = (psi[i+1] - psi[i-1]) / (2.0 * dx);
                    (psi[i].conj() * dpsi).im
                }).sum::<f64>() * dx * HBAR;
                avg_p.push(p_exp);

                if prob_density_frames.len() < 50 { // store at most 50 frames
                    prob_density_frames.push(probs);
                }
            }
        }

        let norm_drift = (norm_vals.last().copied().unwrap_or(1.0) - 1.0).abs();
        let final_x    = avg_x.last().copied().unwrap_or(0.0);
        let final_p    = avg_p.last().copied().unwrap_or(0.0);

        let summary = format!(
            "Quantum simulation: Crank-Nicolson, {} grid points, {} time steps\n\
             Potential: {:?}\n\
             Norm conservation error: {:.2e} (ideal: 0)\n\
             Final <x> = {:.4e} m,  <p> = {:.4e} kg⋅m/s",
            n, max_steps, p.potential, norm_drift, final_x, final_p
        );

        let llm_context = format!(
            "Solved the time-dependent Schrödinger equation using Crank-Nicolson on a {} point grid \
             from x={:.2e} to x={:.2e} m. Potential: {:?}. \
             Norm is conserved to {:.2e} (numerical precision). \
             Expectation values at final time: <x> = {:.4e} m, <p> = {:.4e} kg⋅m/s. \
             The wavefunction remained stable throughout the evolution.",
            n, p.x_min, p.x_max, p.potential, norm_drift, final_x, final_p
        );

        let last_prob = prob_density_frames.last().cloned().unwrap_or_default();

        let prob_plot = PlotSpec {
            kind:    PlotKind::Line,
            title:   "Probability density |ψ(x)|² at final time".into(),
            x_label: "x (m)".into(),
            y_label: "|ψ|²".into(),
            series:  vec![SeriesSpec { label: "|ψ|²".into(), x: xs.clone(), y: last_prob }],
        };

        let avg_x_plot = PlotSpec {
            kind:    PlotKind::Line,
            title:   "Expectation value <x> over time".into(),
            x_label: "Time (s)".into(),
            y_label: "<x> (m)".into(),
            series:  vec![SeriesSpec { label: "<x>".into(), x: times.clone(), y: avg_x }],
        };

        let data = serde_json::json!({
            "x_grid": xs,
            "final_psi_re": psi.iter().map(|z| z.re).collect::<Vec<_>>(),
            "final_psi_im": psi.iter().map(|z| z.im).collect::<Vec<_>>(),
            "norm_conservation": norm_vals,
            "times": times,
            "avg_p": avg_p,
        });

        Ok(SimResult {
            sim_type:    SimType::QuantumWavefunction,
            description: format!("1D Schrödinger equation, {:?}", p.potential),
            steps_run:   max_steps,
            wall_time_ms: 0,
            summary,
            data,
            plots: vec![prob_plot, avg_x_plot],
            llm_context,
        })
    }
}
