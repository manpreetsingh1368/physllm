#!/usr/bin/env bash
# scripts/example_sim.sh — Example simulation requests to the PhysLLM API
# Start the server first: cargo run --release -p api-server

BASE="http://localhost:8080/v1"

echo "═══════════════════════════════════════"
echo " PhysLLM Simulation API Examples"
echo "═══════════════════════════════════════"

#  1. Health check 
echo -e "\n[1] Health check:"
curl -s "$BASE/health" | python3 -m json.tool

#  2. Look up a physical constant 
echo -e "\n[2] Planck constant:"
curl -s "$BASE/constants/hbar" | python3 -m json.tool

#  3. Molecular weight of glucose 
echo -e "\n[3] Molecular weight of glucose (C6H12O6):"
curl -s -X POST "$BASE/chemistry/mw" \
  -H "Content-Type: application/json" \
  -d '{"formula": "C6H12O6"}' | python3 -m json.tool

#  4. N-body: Solar system simulation 
echo -e "\n[4] Solar system N-body (1 Earth year):"
curl -s -X POST "$BASE/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "sim_type": "n_body",
    "description": "Inner solar system for 1 Earth year",
    "params": {
      "preset": "SolarSystem",
      "dt": 3600.0,
      "total_time": 31557600.0,
      "softening": 1000000.0,
      "record_every": 24
    },
    "max_steps": 8766,
    "output_fmt": "summary"
  }' | python3 -m json.tool

#  5. Quantum tunnelling 
echo -e "\n[5] Quantum tunnelling through a step potential:"
curl -s -X POST "$BASE/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "sim_type": "quantum_wavefunction",
    "description": "Electron tunnelling through a 2eV step",
    "params": {
      "potential": {"StepPotential": {"height": 3.204e-19, "step_x": 0.0}},
      "n_grid": 512,
      "x_min": -5e-9,
      "x_max":  5e-9,
      "dt": 1e-19,
      "mass": 9.109e-31,
      "initial_state": {"GaussianWavepacket": {"x0": -2e-9, "sigma": 5e-10, "k0": 5e10}},
      "observe_every": 10
    },
    "max_steps": 2000,
    "output_fmt": "summary"
  }' | python3 -m json.tool

# 6. Stellar evolution: 5 solar mass star 
echo -e "\n[6] Stellar evolution: 5 M_sun star over 100 Myr:"
curl -s -X POST "$BASE/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "sim_type": "stellar_evolution",
    "description": "5 solar mass B-type star evolution",
    "params": {
      "mass_solar": 5.0,
      "metallicity": 0.02,
      "initial_x_h": 0.74,
      "t_end_yr": 1e8
    },
    "max_steps": 1000,
    "output_fmt": "summary"
  }' | python3 -m json.tool

#  7. Astrochemistry: cold molecular cloud 
echo -e "\n[7] ISM astrochemistry: cold dark cloud (1 Myr):"
curl -s -X POST "$BASE/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "sim_type": "astrochem_network",
    "description": "Cold dense molecular cloud chemistry",
    "params": {
      "density_cm3": 1e4,
      "temperature_K": 10.0,
      "uv_field": 0.01,
      "cosmic_ray_rate": 1.3e-17,
      "t_end_yr": 1e6,
      "av": 10.0
    },
    "max_steps": 2000,
    "output_fmt": "summary"
  }' | python3 -m json.tool

# ── 8. H2/O2 combustion kinetics ─────────────────────────────────────────────
echo -e "\n[8] H2/O2 combustion at 1500K:"
curl -s -X POST "$BASE/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "sim_type": "reaction_kinetics",
    "description": "Hydrogen-oxygen combustion ignition",
    "params": {
      "preset": "H2O2Combustion",
      "temperature": 1500.0,
      "t_end": 1e-4,
      "dt_init": 1e-10,
      "rel_tol": 1e-6,
      "abs_tol": 1e-15
    },
    "max_steps": 5000,
    "output_fmt": "summary"
  }' | python3 -m json.tool

echo -e "\n═══════════════════════════════════════"
echo " Done! See physllm-server logs for details."
echo "═══════════════════════════════════════"
