//! dispatcher.rs — Routes LLM tool-use calls to simulation engines.

use crate::{
    nbody::{NBodySimulation, NBodyParams},
    quantum::{QuantumSim, QuantumParams},
    molecular_dynamics::{MDSimulation, MDParams},
    reaction_kinetics::{KineticsSim, KineticsParams},
    stellar::{StellarSim, StellarParams},
    astrochem::{AstrochemSim, AstrochemParams},
    Result, SimError
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Enum of all supported simulation types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SimType {
    NBody,
    QuantumWavefunction,
    MolecularDynamics,
    ReactionKinetics,
    StellarEvolution,
    AstrochemNetwork,
Eos
}

/// Input to the simulation agent (from LLM tool call).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimRequest {
    pub sim_type:   SimType,
    pub params:     serde_json::Value,
    pub max_steps:  Option<usize>,
    pub output_fmt: OutputFormat,
    pub description: String,  // natural-language description of what to simulate
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    #[default]
    Summary,
    TimeSeries,
    Hdf5,
    Json
}

/// Result returned from a simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimResult {
    pub sim_type:       SimType,
    pub description:    String,
    pub steps_run:      usize,
    pub wall_time_ms:   u64,
    pub summary:        String,                            // human-readable summary
    pub data:           serde_json::Value,                 // structured numeric results
    pub plots:          Vec<PlotSpec>,                     // suggested plot configs
    pub llm_context:    String,                            // text the LLM should include in reply
}

/// Specification for a plot the front-end can render.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotSpec {
    pub kind:   PlotKind,
    pub title:  String,
    pub x_label: String,
    pub y_label: String,
    pub series: Vec<SeriesSpec>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlotKind { Line, Scatter, Heatmap, Histogram, Phase2D }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesSpec {
    pub label: String,
    pub x:     Vec<f64>,
    pub y:     Vec<f64>
}

/// The simulation agent — receives requests, runs simulations, returns results.
pub struct SimAgent;

impl SimAgent {
    pub fn new() -> Self { Self }

    /// Dispatch a simulation request and return structured results.
    pub async fn run(&self, req: SimRequest) -> Result<SimResult> {
        let t0 = std::time::Instant::now();
        info!("sim-agent: running {:?} — {}", req.sim_type, req.description);

        let result = match &req.sim_type {
            SimType::NBody => {
                let params: NBodyParams = serde_json::from_value(req.params.clone())
                    .map_err(|e| SimError::Parse(e.to_string()))?;
                let sim = NBodySimulation::new(params);
                sim.run(req.max_steps.unwrap_or(10_000))
            }

            SimType::QuantumWavefunction => {
                let params: QuantumParams = serde_json::from_value(req.params.clone())
                    .map_err(|e| SimError::Parse(e.to_string()))?;
                let sim = QuantumSim::new(params);
                sim.run(req.max_steps.unwrap_or(5_000))
            }

            SimType::MolecularDynamics => {
                let params: MDParams = serde_json::from_value(req.params.clone())
                    .map_err(|e| SimError::Parse(e.to_string()))?;
                let sim = MDSimulation::new(params);
                sim.run(req.max_steps.unwrap_or(50_000))
            }

            SimType::ReactionKinetics => {
                let params: KineticsParams = serde_json::from_value(req.params.clone())
                    .map_err(|e| SimError::Parse(e.to_string()))?;
                let sim = KineticsSim::new(params);
                sim.run(req.max_steps.unwrap_or(1_000))
            }

            SimType::StellarEvolution => {
                let params: StellarParams = serde_json::from_value(req.params.clone())
                    .map_err(|e| SimError::Parse(e.to_string()))?;
                let sim = StellarSim::new(params);
                sim.run(req.max_steps.unwrap_or(2_000))
            }

            SimType::AstrochemNetwork => {
                let params: AstrochemParams = serde_json::from_value(req.params.clone())
                    .map_err(|e| SimError::Parse(e.to_string()))?;
                let sim = AstrochemSim::new(params);
                sim.run(req.max_steps.unwrap_or(5_000))
            }
            SimType::Eos => {
                return Err(SimError::InvalidParameter("Equation of state simulation not yet implemented".into()));
            }

            
        }?;

        let wall_ms = t0.elapsed().as_millis() as u64;
        info!("sim-agent: {:?} completed in {wall_ms}ms ({} steps)", req.sim_type, result.steps_run);

        Ok(SimResult { wall_time_ms: wall_ms, ..result })
    }

    /// Parse a tool-call JSON payload from the LLM and dispatch.
    pub async fn handle_tool_call(&self, tool_name: &str, args: serde_json::Value) -> Result<SimResult> {
        let sim_type = match tool_name {
            "simulate_nbody"           => SimType::NBody,
            "simulate_quantum"         => SimType::QuantumWavefunction,
            "simulate_md"              => SimType::MolecularDynamics,
            "simulate_kinetics"        => SimType::ReactionKinetics,
            "simulate_stellar"         => SimType::StellarEvolution,
            "simulate_astrochem"       => SimType::AstrochemNetwork,
            "simulate_thermodynamics"  => SimType::Eos,
            other => {
                warn!("Unknown tool: {other}");
                return Err(SimError::UnknownType(other.into()));
            }
        };

        let description = args.get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("Physics simulation")
            .to_string();

        self.run(SimRequest {
            sim_type,
            params: args,
            max_steps: None,
            output_fmt: OutputFormat::Summary,
            description
        }).await
    }
}