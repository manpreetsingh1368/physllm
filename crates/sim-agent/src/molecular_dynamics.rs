//! molecular_dynamics.rs — Lennard-Jones MD simulation stub.

use crate::{Result, SimError, dispatcher::{SimResult, SimType, PlotSpec, PlotKind, SeriesSpec}};
use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MDParams {
    pub n_particles:  usize,
    pub box_length:   f64,      // Å
    pub temperature:  f64,      // K
    pub dt:           f64,      // fs (femtoseconds)
    pub eps:          f64,      // LJ well depth, eV
    pub sigma:        f64,      // LJ size parameter, Å
    pub ensemble:     Ensemble,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Ensemble { NVE, NVT }

pub struct MDSimulation { params: MDParams }

impl MDSimulation {
    pub fn new(params: MDParams) -> Self { Self { params } }

    pub fn run(&self, max_steps: usize) -> Result<SimResult> {
        let p = &self.params;
        let n = p.n_particles;
        const KB: f64 = 8.617_333_262e-5; // eV/K
        let fs_to_s = 1e-15;
        let ang_to_m = 1e-10;
        let m_Ar = 39.948 * 1.6605e-27; // argon mass kg

        if n > 5000 {
            return Err(SimError::InvalidParameter("CPU MD limited to 5000 particles".into()));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let L = p.box_length;

        // FCC-ish initial positions (simple cubic grid)
        let nx = (n as f64).cbrt().ceil() as usize;
        let dl = L / nx as f64;
        let mut pos: Vec<[f64; 3]> = (0..n).map(|i| {
            let ix = i % nx; let iy = (i / nx) % nx; let iz = i / (nx * nx);
            [(ix as f64 + 0.5) * dl, (iy as f64 + 0.5) * dl, (iz as f64 + 0.5) * dl]
        }).collect();

        // Maxwell-Boltzmann velocities
        let kT = KB * p.temperature;
        let v_std = (kT / m_Ar * ang_to_m * ang_to_m / fs_to_s / fs_to_s * 1e-20).sqrt();
        let mut vel: Vec<[f64; 3]> = (0..n).map(|_| {
            [rng.gen::<f64>() * v_std - v_std/2.0,
             rng.gen::<f64>() * v_std - v_std/2.0,
             rng.gen::<f64>() * v_std - v_std/2.0]
        }).collect();

        // Remove COM drift
        for dim in 0..3 {
            let cm: f64 = vel.iter().map(|v| v[dim]).sum::<f64>() / n as f64;
            vel.iter_mut().for_each(|v| v[dim] -= cm);
        }

        let lj_force = |r2: f64| -> f64 {
            let s2 = (p.sigma * p.sigma) / r2;
            let s6 = s2 * s2 * s2;
            24.0 * p.eps * (2.0 * s6 * s6 - s6) / r2
        };

        let mut ke_series: Vec<f64> = Vec::new();
        let mut pe_series: Vec<f64> = Vec::new();
        let mut temp_series: Vec<f64> = Vec::new();
        let mut times: Vec<f64> = Vec::new();
        let record = max_steps / 100;

        for step in 0..max_steps {
            // Compute forces (O(N²) with minimum image)
            let mut forces = vec![[0.0f64; 3]; n];
            let mut pe = 0.0f64;

            for i in 0..n {
                for j in (i+1)..n {
                    let mut dr = [0.0f64; 3];
                    let mut r2 = 0.0;
                    for d in 0..3 {
                        let mut dxd = pos[j][d] - pos[i][d];
                        dxd -= L * (dxd / L).round();  // minimum image
                        dr[d] = dxd;
                        r2 += dxd * dxd;
                    }
                    let rc = 2.5 * p.sigma;
                    if r2 < rc * rc {
                        let s2 = (p.sigma * p.sigma) / r2;
                        let s6 = s2 * s2 * s2;
                        pe += 4.0 * p.eps * (s6 * s6 - s6);
                        let f = lj_force(r2);
                        for d in 0..3 {
                            forces[i][d] += f * dr[d];
                            forces[j][d] -= f * dr[d];
                        }
                    }
                }
            }

            // Velocity Verlet
            let dt = p.dt;
            let m_inv = 1.0 / (m_Ar / (ang_to_m * ang_to_m / (fs_to_s * fs_to_s)));
            for i in 0..n {
                for d in 0..3 {
                    vel[i][d] += 0.5 * forces[i][d] * m_inv * dt;
                    pos[i][d] = (pos[i][d] + vel[i][d] * dt).rem_euclid(L);
                    vel[i][d] += 0.5 * forces[i][d] * m_inv * dt;
                }
            }

            // NVT velocity rescaling (Berendsen)
            if matches!(p.ensemble, Ensemble::NVT) {
                let ke: f64 = vel.iter().map(|v| 0.5 / m_inv * v.iter().map(|&vi| vi*vi).sum::<f64>()).sum();
                let t_inst = 2.0 * ke / (3.0 * n as f64 * KB / (ang_to_m * ang_to_m / (fs_to_s * fs_to_s)));
                if t_inst > 0.0 {
                    let scale = (p.temperature / t_inst).sqrt();
                    vel.iter_mut().for_each(|v| v.iter_mut().for_each(|vi| *vi *= scale));
                }
            }

            if step % record.max(1) == 0 {
                let ke: f64 = vel.iter().map(|v| v.iter().map(|&vi| vi*vi).sum::<f64>() * 0.5 / m_inv).sum();
                ke_series.push(ke);
                pe_series.push(pe);
                temp_series.push(2.0 * ke / (3.0 * n as f64 * KB / (ang_to_m.powi(2) / fs_to_s.powi(2))));
                times.push(step as f64 * dt);
            }
        }

        let t_avg = temp_series.iter().copied().sum::<f64>() / temp_series.len() as f64;

        let summary = format!(
            "MD simulation: {} particles, LJ fluid, T_target={:.1}K, T_avg={:.1}K\n\
             Box={:.2}Å³, dt={:.2}fs, {} steps ({:?} ensemble)",
            n, p.temperature, t_avg, L, p.dt, max_steps, p.ensemble
        );

        let llm_context = format!(
            "Molecular dynamics simulation of {} Lennard-Jones particles in a {:.1}Å cubic box. \
             Target temperature {:.1}K, achieved average temperature {:.1}K using {:?} ensemble. \
             Time step {:.2}fs, ran {} steps.",
            n, L, p.temperature, t_avg, p.ensemble, p.dt, max_steps
        );

        let e_plot = PlotSpec {
            kind: PlotKind::Line, title: "Potential energy vs time".into(),
            x_label: "Time (fs)".into(), y_label: "PE (eV)".into(),
            series: vec![SeriesSpec { label: "PE".into(), x: times.clone(), y: pe_series }],
        };
        let t_plot = PlotSpec {
            kind: PlotKind::Line, title: "Instantaneous temperature".into(),
            x_label: "Time (fs)".into(), y_label: "T (K)".into(),
            series: vec![SeriesSpec { label: "T".into(), x: times, y: temp_series }],
        };

        Ok(SimResult {
            sim_type: SimType::MolecularDynamics,
            description: format!("LJ-MD {} particles", n),
            steps_run: max_steps, wall_time_ms: 0,
            summary, data: serde_json::json!({"ke": ke_series}),
            plots: vec![e_plot, t_plot], llm_context,
        })
    }
}
