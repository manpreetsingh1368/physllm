//! nbody.rs — Gravitational N-body simulation using 4th-order Runge-Kutta.
//!
//! Supports:
//!   - Arbitrary number of bodies with mass, position, velocity
//!   - Optional softening length (avoids singularities)
//!   - Energy and angular momentum tracking
//!   - Presets: Solar system, binary star, three-body, galaxy core

use crate::{Result, SimError, dispatcher::{SimResult, SimType, PlotSpec, PlotKind, SeriesSpec}};
use serde::{Deserialize, Serialize};
use nalgebra::Vector3;

/// A single body in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Body {
    pub name:  String,
    pub mass:  f64,                    // kg
    pub pos:   [f64; 3],               // m
    pub vel:   [f64; 3],               // m/s
    pub color: Option<String>,
}

impl Body {
    pub fn pos_v(&self) -> Vector3<f64> { Vector3::new(self.pos[0], self.pos[1], self.pos[2]) }
    pub fn vel_v(&self) -> Vector3<f64> { Vector3::new(self.vel[0], self.vel[1], self.vel[2]) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NBodyParams {
    pub bodies:        Vec<Body>,
    pub G:             Option<f64>,    // gravitational constant (default NIST)
    pub softening:     f64,            // softening length ε, m
    pub dt:            f64,            // time step, s
    pub total_time:    f64,            // total simulation time, s
    pub record_every:  usize,          // record state every N steps
    pub preset:        Option<NBodyPreset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NBodyPreset {
    SolarSystem,
    BinaryStar,
    ThreeBody,      // figure-eight orbit
    GalaxyCore,
    PlanetaryMoons,
}

pub struct NBodySimulation { pub params: NBodyParams }

impl NBodySimulation {
    pub fn new(mut params: NBodyParams) -> Self {
        if let Some(preset) = &params.preset {
            params.bodies = preset_bodies(preset);
        }
        Self { params }
    }

    pub fn run(&self, max_steps: usize) -> Result<SimResult> {
        let G = self.params.G.unwrap_or(6.674_30e-11);
        let dt = self.params.dt;
        let eps2 = self.params.softening.powi(2);

        let n = self.params.bodies.len();
        if n < 2 {
            return Err(SimError::InvalidParameter("N-body needs ≥ 2 bodies".into()));
        }

        let mut pos: Vec<Vector3<f64>> = self.params.bodies.iter().map(|b| b.pos_v()).collect();
        let mut vel: Vec<Vector3<f64>> = self.params.bodies.iter().map(|b| b.vel_v()).collect();
        let masses: Vec<f64>           = self.params.bodies.iter().map(|b| b.mass).collect();

        let total_steps  = (self.params.total_time / dt).ceil() as usize;
        let actual_steps = total_steps.min(max_steps);
        let record_every = self.params.record_every.max(1);

        // Time series storage
        let mut times: Vec<f64>            = Vec::new();
        let mut energies: Vec<f64>         = Vec::new();
        let mut positions: Vec<Vec<[f64;3]>> = vec![Vec::new(); n];
        let mut ang_mom: Vec<f64>          = Vec::new();

        let accel = |pos: &[Vector3<f64>]| -> Vec<Vector3<f64>> {
            (0..n).map(|i| {
                let mut a = Vector3::zeros();
                for j in 0..n {
                    if i == j { continue; }
                    let r = pos[j] - pos[i];
                    let r2 = r.norm_squared() + eps2;
                    let r3 = r2.sqrt() * r2;
                    a += r * (G * masses[j] / r3);
                }
                a
            }).collect()
        };

        let kinetic_energy = |vel: &[Vector3<f64>]| -> f64 {
            vel.iter().zip(masses.iter()).map(|(v, &m)| 0.5 * m * v.norm_squared()).sum()
        };

        let potential_energy = |pos: &[Vector3<f64>]| -> f64 {
            let mut pe = 0.0;
            for i in 0..n {
                for j in (i+1)..n {
                    let r = (pos[j] - pos[i]).norm() + self.params.softening;
                    pe -= G * masses[i] * masses[j] / r;
                }
            }
            pe
        };

        let total_angular_momentum = |pos: &[Vector3<f64>], vel: &[Vector3<f64>]| -> f64 {
            pos.iter().zip(vel.iter()).zip(masses.iter())
                .map(|((p, v), &m)| (p.cross(v) * m).norm())
                .sum()
        };

        for step in 0..actual_steps {
            // RK4 integration
            let a1 = accel(&pos);
            let p2: Vec<_> = pos.iter().zip(vel.iter()).zip(a1.iter())
                .map(|((p, v), a)| p + v * (dt/2.0) + a * (dt*dt/8.0)).collect();
            let v2: Vec<_> = vel.iter().zip(a1.iter())
                .map(|(v, a)| v + a * (dt/2.0)).collect();
            let a2 = accel(&p2);

            let p3: Vec<_> = pos.iter().zip(vel.iter()).zip(a2.iter())
                .map(|((p, v), a)| p + v * (dt/2.0) + a * (dt*dt/8.0)).collect();
            let v3: Vec<_> = vel.iter().zip(a2.iter())
                .map(|(v, a)| v + a * (dt/2.0)).collect();
            let a3 = accel(&p3);

            let p4: Vec<_> = pos.iter().zip(vel.iter()).zip(a3.iter())
                .map(|((p, v), a)| p + v * dt + a * (dt*dt/2.0)).collect();
            let v4: Vec<_> = vel.iter().zip(a3.iter())
                .map(|(v, a)| v + a * dt).collect();
            let a4 = accel(&p4);

            for i in 0..n {
                pos[i] += (vel[i] + v2[i]*2.0 + v3[i]*2.0 + v4[i]) * (dt/6.0)
                         + (a1[i] + a2[i]*2.0 + a3[i]*2.0 + a4[i]) * (dt*dt/12.0);
                vel[i] += (a1[i] + a2[i]*2.0 + a3[i]*2.0 + a4[i]) * (dt/6.0);
            }

            // Energy check (divergence detection)
            if step % record_every == 0 {
                let t = step as f64 * dt;
                let ke = kinetic_energy(&vel);
                let pe = potential_energy(&pos);
                let e  = ke + pe;

                if e.is_nan() || e.is_infinite() {
                    return Err(SimError::Diverged { step, msg: "Energy diverged (reduce dt or increase softening)".into() });
                }

                times.push(t);
                energies.push(e);
                ang_mom.push(total_angular_momentum(&pos, &vel));
                for (i, p) in pos.iter().enumerate() {
                    positions[i].push([p.x, p.y, p.z]);
                }
            }
        }

        // Build result
        let e0 = energies.first().copied().unwrap_or(0.0);
        let ef = energies.last().copied().unwrap_or(0.0);
        let drift = if e0.abs() > 1e-30 { (ef - e0).abs() / e0.abs() } else { 0.0 };

        let summary = format!(
            "N-body simulation: {} bodies, {actual_steps} steps, dt={:.2e}s\n\
             Total time: {:.3e}s | Energy drift: {:.2e} ({:.4}%)\n\
             Bodies: {}",
            n, dt, actual_steps as f64 * dt,
            (ef - e0).abs(), drift * 100.0,
            self.params.bodies.iter().map(|b| b.name.as_str()).collect::<Vec<_>>().join(", ")
        );

        let llm_context = format!(
            "Ran N-body simulation with {} bodies for {:.3e} seconds of simulated time. \
             Energy conservation error: {:.4}% (lower is better; <0.1% is good). \
             The bodies were: {}. \
             Initial total energy: {:.4e} J, Final: {:.4e} J.",
            n, actual_steps as f64 * dt,
            drift * 100.0,
            self.params.bodies.iter().map(|b| format!("{} (M={:.2e}kg)", b.name, b.mass)).collect::<Vec<_>>().join(", "),
            e0, ef
        );

        // Plot: energy conservation over time
        let energy_plot = PlotSpec {
            kind: PlotKind::Line,
            title: "Total mechanical energy vs time".into(),
            x_label: "Time (s)".into(),
            y_label: "Energy (J)".into(),
            series: vec![SeriesSpec { label: "E_total".into(), x: times.clone(), y: energies }],
        };

        // Plot: x-y trajectory for each body
        let traj_plot = PlotSpec {
            kind: PlotKind::Scatter,
            title: "Orbital trajectories (x-y plane)".into(),
            x_label: "x (m)".into(),
            y_label: "y (m)".into(),
            series: positions.iter().enumerate().map(|(i, pos_series)| SeriesSpec {
                label: self.params.bodies[i].name.clone(),
                x: pos_series.iter().map(|p| p[0]).collect(),
                y: pos_series.iter().map(|p| p[1]).collect(),
            }).collect(),
        };

        let data = serde_json::json!({
            "times": times,
            "angular_momentum": ang_mom,
            "final_positions": pos.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
            "final_velocities": vel.iter().map(|v| [v.x, v.y, v.z]).collect::<Vec<_>>(),
            "energy_drift_fraction": drift,
        });

        Ok(SimResult {
            sim_type:    SimType::NBody,
            description: format!("{}-body gravitational simulation", n),
            steps_run:   actual_steps,
            wall_time_ms: 0,
            summary,
            data,
            plots: vec![energy_plot, traj_plot],
            llm_context,
        })
    }
}

fn preset_bodies(preset: &NBodyPreset) -> Vec<Body> {
    match preset {
        NBodyPreset::SolarSystem => vec![
            Body { name: "Sun".into(),   mass: 1.989e30, pos: [0.0; 3],       vel: [0.0; 3],                   color: Some("#FFD700".into()) },
            Body { name: "Mercury".into(), mass: 3.301e23, pos: [5.791e10, 0.0, 0.0], vel: [0.0, 47_870.0, 0.0], color: Some("#A0522D".into()) },
            Body { name: "Venus".into(),   mass: 4.867e24, pos: [1.082e11, 0.0, 0.0], vel: [0.0, 35_020.0, 0.0], color: Some("#DEB887".into()) },
            Body { name: "Earth".into(),   mass: 5.972e24, pos: [1.496e11, 0.0, 0.0], vel: [0.0, 29_780.0, 0.0], color: Some("#4169E1".into()) },
            Body { name: "Mars".into(),    mass: 6.417e23, pos: [2.279e11, 0.0, 0.0], vel: [0.0, 24_070.0, 0.0], color: Some("#CD5C5C".into()) },
            Body { name: "Jupiter".into(), mass: 1.898e27, pos: [7.783e11, 0.0, 0.0], vel: [0.0, 13_070.0, 0.0], color: Some("#DAA520".into()) },
            Body { name: "Saturn".into(),  mass: 5.683e26, pos: [1.432e12, 0.0, 0.0], vel: [0.0,  9_690.0, 0.0], color: Some("#F4A460".into()) },
        ],
        NBodyPreset::BinaryStar => {
            let m1 = 2.0e30;
            let m2 = 1.5e30;
            let sep = 1.0e11;
            let v = ((6.674e-11_f64 * (m1 + m2) / sep).sqrt()) * 0.5;
            vec![
                Body { name: "Star A".into(), mass: m1, pos: [-sep/2.0, 0.0, 0.0], vel: [0.0, -v * m2/(m1+m2), 0.0], color: Some("#FF6347".into()) },
                Body { name: "Star B".into(), mass: m2, pos: [ sep/2.0, 0.0, 0.0], vel: [0.0,  v * m1/(m1+m2), 0.0], color: Some("#87CEEB".into()) },
            ]
        }
        NBodyPreset::ThreeBody => {
            // Chenciner-Montgomery figure-eight solution
            let scale = 1e11;
            let v_scale = 1e3;
            vec![
                Body { name: "Body 1".into(), mass: 1e30, pos: [-0.97000436*scale, 0.24308753*scale, 0.0], vel: [0.93240737*v_scale/2.0, 0.86473146*v_scale/2.0, 0.0], color: Some("#FF4500".into()) },
                Body { name: "Body 2".into(), mass: 1e30, pos: [ 0.97000436*scale,-0.24308753*scale, 0.0], vel: [0.93240737*v_scale/2.0, 0.86473146*v_scale/2.0, 0.0], color: Some("#32CD32".into()) },
                Body { name: "Body 3".into(), mass: 1e30, pos: [0.0, 0.0, 0.0],                           vel: [-0.93240737*v_scale,   -0.86473146*v_scale,   0.0], color: Some("#1E90FF".into()) },
            ]
        }
        NBodyPreset::GalaxyCore => {
            use rand::{Rng, SeedableRng};
            use rand_chacha::ChaCha8Rng;
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            let mut bodies = vec![
                Body { name: "SMBH".into(), mass: 4e36, pos: [0.0; 3], vel: [0.0; 3], color: Some("#000000".into()) },
            ];
            for i in 0..19 {
                let r   = rng.gen_range(1e15..5e16_f64);
                let phi = rng.gen_range(0.0_f64..std::f64::consts::TAU);
                let v_c = (6.674e-11 * 4e36 / r).sqrt();
                bodies.push(Body {
                    name:  format!("Star {}", i+1),
                    mass:  rng.gen_range(1e29..3e30),
                    pos:   [r*phi.cos(), r*phi.sin(), 0.0],
                    vel:   [-v_c*phi.sin()*0.9, v_c*phi.cos()*0.9, 0.0],
                    color: None,
                });
            }
            bodies
        }
        NBodyPreset::PlanetaryMoons => vec![
            Body { name: "Planet".into(), mass: 1e26,  pos: [0.0; 3],           vel: [0.0; 3],           color: Some("#4169E1".into()) },
            Body { name: "Moon 1".into(), mass: 1e22,  pos: [4e8, 0.0, 0.0],    vel: [0.0, 1000.0, 0.0], color: Some("#C0C0C0".into()) },
            Body { name: "Moon 2".into(), mass: 5e21,  pos: [-7e8, 0.0, 0.0],   vel: [0.0,-750.0,  0.0], color: Some("#808080".into()) },
            Body { name: "Moon 3".into(), mass: 2e21,  pos: [0.0, 6e8, 0.0],    vel: [800.0, 0.0,  0.0], color: Some("#A9A9A9".into()) },
        ],
    }
}