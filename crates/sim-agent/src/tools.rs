//! tools.rs — JSON tool definitions exposed to the LLM for tool-use.

/// All simulation tools the LLM can call, in Anthropic tool-use format.
pub const SIMULATION_TOOLS: &str = r#"[
  {
    "name": "simulate_nbody",
    "description": "Run a gravitational N-body simulation using 4th-order Runge-Kutta. Returns orbital trajectories, energy conservation, and angular momentum. Use for: solar systems, binary stars, three-body problems, galaxy dynamics.",
    "input_schema": {
      "type": "object",
      "properties": {
        "description": { "type": "string", "description": "What you are simulating" },
        "preset": {
          "type": "string",
          "enum": ["solar_system", "binary_star", "three_body", "galaxy_core", "planetary_moons"],
          "description": "Use a preset configuration"
        },
        "bodies": {
          "type": "array",
          "description": "Custom bodies (if not using preset)",
          "items": {
            "type": "object",
            "properties": {
              "name": { "type": "string" },
              "mass": { "type": "number", "description": "Mass in kg" },
              "pos":  { "type": "array", "items": {"type":"number"}, "description": "Position [x,y,z] in metres" },
              "vel":  { "type": "array", "items": {"type":"number"}, "description": "Velocity [vx,vy,vz] in m/s" }
            }
          }
        },
        "dt":           { "type": "number", "description": "Time step in seconds" },
        "total_time":   { "type": "number", "description": "Total simulation time in seconds" },
        "softening":    { "type": "number", "description": "Gravitational softening length in metres (default 1e6)" },
        "record_every": { "type": "integer", "description": "Record state every N steps" }
      },
      "required": ["description"]
    }
  },
  {
    "name": "simulate_quantum",
    "description": "Solve the 1D time-dependent Schrödinger equation using Crank-Nicolson. Models quantum tunnelling, wavepacket dynamics, energy eigenstates. Supports: infinite well, harmonic oscillator, double well, step potential, Coulomb.",
    "input_schema": {
      "type": "object",
      "properties": {
        "description": { "type": "string" },
        "potential":   { "type": "string", "enum": ["infinite_well","harmonic","double_well","step","coulomb"], "description": "Potential type" },
        "omega":       { "type": "number", "description": "Angular frequency for harmonic oscillator (rad/s)" },
        "step_height": { "type": "number", "description": "Step height in eV (for step potential)" },
        "x_min":       { "type": "number", "description": "Left boundary in Angstroms" },
        "x_max":       { "type": "number", "description": "Right boundary in Angstroms" },
        "n_grid":      { "type": "integer", "description": "Number of grid points (default 512)" },
        "dt":          { "type": "number", "description": "Time step in attoseconds (1e-18 s)" },
        "k0":          { "type": "number", "description": "Initial wave vector for Gaussian wavepacket (1/m)" },
        "x0":          { "type": "number", "description": "Initial wavepacket centre in Angstroms" },
        "sigma":       { "type": "number", "description": "Initial wavepacket width in Angstroms" }
      },
      "required": ["description", "potential"]
    }
  },
  {
    "name": "simulate_md",
    "description": "Molecular dynamics simulation using Lennard-Jones potential (CPU, up to 5000 particles). Computes thermodynamic properties, radial distribution function, diffusion coefficient.",
    "input_schema": {
      "type": "object",
      "properties": {
        "description":   { "type": "string" },
        "n_particles":   { "type": "integer", "description": "Number of particles (max 5000)" },
        "temperature_K": { "type": "number",  "description": "Target temperature in Kelvin" },
        "box_angstrom":  { "type": "number",  "description": "Cubic box side length in Angstroms" },
        "dt_fs":         { "type": "number",  "description": "Time step in femtoseconds" },
        "n_steps":       { "type": "integer", "description": "Number of MD steps" },
        "ensemble":      { "type": "string",  "enum": ["NVE","NVT"], "description": "Statistical ensemble" },
        "eps_eV":        { "type": "number",  "description": "LJ well depth in eV (default 0.0104 for Ar)" },
        "sigma_ang":     { "type": "number",  "description": "LJ sigma in Angstroms (default 3.4 for Ar)" }
      },
      "required": ["description", "n_particles", "temperature_K"]
    }
  },
  {
    "name": "simulate_kinetics",
    "description": "Solve chemical kinetics ODE system using adaptive RK45. Handles Arrhenius, ISM, and custom rate laws. Presets: ozone depletion, H2/O2 combustion, ISM hydrogen chemistry, Brusselator oscillator.",
    "input_schema": {
      "type": "object",
      "properties": {
        "description": { "type": "string" },
        "preset": { "type": "string", "enum": ["ozone_depletion","h2_o2_combustion","ism_hydrogen","brusselator"] },
        "temperature_K": { "type": "number" },
        "t_end_s":       { "type": "number", "description": "Integration end time in seconds" },
        "species": { "type": "array", "items": {"type":"string"} },
        "initial_conc": { "type": "array",  "items": {"type":"number"} }
      },
      "required": ["description"]
    }
  },
  {
    "name": "simulate_stellar",
    "description": "Simulate stellar evolution track on the Hertzsprung-Russell diagram. Computes luminosity, effective temperature, radius, and main-sequence lifetime as a function of time. Predicts post-MS fate.",
    "input_schema": {
      "type": "object",
      "properties": {
        "description":  { "type": "string" },
        "mass_solar":   { "type": "number", "description": "Stellar mass in solar masses" },
        "metallicity":  { "type": "number", "description": "Metal fraction Z (0.02 = solar)" },
        "t_end_yr":     { "type": "number", "description": "Evolve for this many years" }
      },
      "required": ["description", "mass_solar"]
    }
  },
  {
    "name": "simulate_astrochem",
    "description": "Astrochemical network simulation of the interstellar medium (ISM). Models molecular abundances including H, H2, CO, HCN, H2O, complex organics under UV radiation and cosmic rays.",
    "input_schema": {
      "type": "object",
      "properties": {
        "description":     { "type": "string" },
        "density_cm3":     { "type": "number", "description": "H nuclei number density cm⁻³" },
        "temperature_K":   { "type": "number" },
        "uv_g0":           { "type": "number", "description": "UV field in Habing units (G0=1 is standard ISRF)" },
        "cosmic_ray_rate": { "type": "number", "description": "Cosmic ray ionisation rate ζ per H nucleus (s⁻¹)" },
        "visual_extinction": { "type": "number", "description": "Av in magnitudes" },
        "t_end_yr":        { "type": "number", "description": "Integration end time in years" }
      },
      "required": ["description", "density_cm3", "temperature_K"]
    }
  },
  {
    "name": "simulate_thermodynamics",
    "description": "Compute thermodynamic equations of state: ideal gas, Van der Waals, blackbody radiation, stellar atmosphere opacity. Returns P-T or power-T curves.",
    "input_schema": {
      "type": "object",
      "properties": {
        "description":  { "type": "string" },
        "system":       { "type": "string", "enum": ["ideal_gas","van_der_waals","blackbody","stellar_atmosphere"] },
        "n_mol":        { "type": "number" },
        "a_vdw":        { "type": "number", "description": "Van der Waals a constant Pa⋅m⁶/mol²" },
        "b_vdw":        { "type": "number", "description": "Van der Waals b constant m³/mol" },
        "t_min_K":      { "type": "number" },
        "t_max_K":      { "type": "number" },
        "log_g":        { "type": "number", "description": "Stellar surface gravity log g (cm/s²)" },
        "t_eff_K":      { "type": "number", "description": "Effective temperature for stellar atmosphere" }
      },
      "required": ["description", "system"]
    }
  }
]"#;
