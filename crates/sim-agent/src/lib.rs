//! sim-agent — Physics simulation side-agent for PhysLLM.
//!
//! The agent receives structured simulation requests from the LLM (via tool-use JSON)
//! and dispatches to the appropriate engine:
//!
//!   - NBodySimulation    — gravitational N-body (Runge-Kutta 4th order)
//!   - QuantumSim         — 1D Schrödinger equation (Crank-Nicolson)
//!   - MolecularDynamics  — classical MD with Lennard-Jones potential
//!   - ReactionKinetics   — ODE system for chemical kinetics
//!   - StellarEvolution   — simplified Hertzsprung-Russell track simulation
//!   - AstrochemSim       — interstellar medium chemical network
//!   - ThermodynamicsSim  — equation of state + phase diagram calculations
//!   - PlasmaSimulation   — magnetohydrodynamics (MHD) basics

pub mod nbody;
pub mod quantum;
pub mod molecular_dynamics;
pub mod reaction_kinetics;
pub mod stellar;
pub mod astrochem;
pub mod dispatcher;
pub mod tools;

pub use dispatcher::{SimAgent, SimRequest, SimResult, SimType};
pub use tools::SIMULATION_TOOLS;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("Simulation diverged at step {step}: {msg}")]
    Diverged { step: usize, msg: String },
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Unknown simulation type: {0}")]
    UnknownType(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}

pub type Result<T> = std::result::Result<T, SimError>;
