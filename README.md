# PhysLLM — Physics & Chemistry LLM in Rust with AMD ROCm

A production-quality transformer LLM runtime written entirely in Rust, with native AMD GPU acceleration via ROCm/HIP, purpose-built for physics, chemistry, astrophysics, and astrochemistry. Includes a full simulation agent with 7 physics engines the LLM can invoke via tool-use.

---

## Architecture overview

```
User (CLI / REST / WebSocket)
          │
    API Server (Axum)
          │
   PhysLLM Orchestrator
    ┌─────┴──────┐
    │            │
LLM Core    Simulation Agent
(Rust)       (7 engines)
    │            │
ROCm/HIP     Engines:
Backend        • N-body (RK4)
    │            • Quantum (Crank-Nicolson)
AMD GPU        • Molecular Dynamics (LJ)
(RDNA3/CDNA)   • Reaction Kinetics (RK45)
               • Stellar Evolution (HR track)
               • Astrochemistry (ISM network)
               • Thermodynamics (EoS)
```

---

## Hardware requirements

| GPU tier | Model | VRAM | Supports |
|----------|-------|------|---------|
| High-end consumer | RX 7900 XTX | 24 GB | 13B f16 |
| Mid consumer | RX 7900 XT / 6900 XT | 20 GB | 13B f16 |
| Entry consumer | RX 7800 XT / 6800 XT | 16 GB | 7B f16 |
| Data centre | MI250 / MI300 | 64–192 GB | 70B f16 |
| CPU fallback | Any CPU | RAM | 7B (slow) |

**Software:** ROCm ≥ 6.0, Rust ≥ 1.78, Ubuntu 22.04+

---

## Quick start

### 1. Install ROCm

```bash
# Ubuntu 22.04
wget https://repo.radeon.com/amdgpu-install/6.2/ubuntu/jammy/amdgpu-install_6.2.60200-1_all.deb
sudo apt install ./amdgpu-install_6.2.60200-1_all.deb
sudo amdgpu-install --usecase=rocm,hiplibsdk
sudo usermod -aG render,video $USER
# Log out and back in
```

### 2. Verify your GPU

```bash
rocm-smi
rocminfo | grep "gfx"    # note your arch, e.g. gfx1100 (RX 7900)
```

### 3. Clone and build

```bash
git clone https://github.com/manpreetsingh1368/physllm.git
cd physllm

# Verify ROCm setup
chmod +x scripts/setup_rocm.sh
./scripts/setup_rocm.sh

# Build with AMD GPU support
export ROCM_PATH=/opt/rocm
cargo build --release --features rocm

# Or CPU-only (no GPU required)
cargo build --release --features cpu-only
```

### 4. Run the API server

```bash
export ROCM_PATH=/opt/rocm
cargo run --release -p api-server

# Server starts at http://localhost:8080
```

### 5. Try it out

```bash
# Health check
curl http://localhost:8080/v1/health

# Look up a physical constant
curl http://localhost:8080/v1/constants/hbar

# Compute molecular weight
curl -X POST http://localhost:8080/v1/chemistry/mw \
  -H "Content-Type: application/json" \
  -d '{"formula": "C6H12O6"}'

# Run all example simulations
./scripts/example_sim.sh
```

---

## Loading pretrained weights

PhysLLM loads weights from any HuggingFace-compatible safetensors checkpoint. To use a pretrained Llama 3 / Mistral base model (and fine-tune on physics data):

```bash
# Download a base model (example: Llama-3-8B)
huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --local-dir models/llama3-8b \
  --include "*.safetensors" "config.json" "tokenizer.json"

# Start server with weights
ROCM_PATH=/opt/rocm \
MODEL_DIR=models/llama3-8b \
cargo run --release --features rocm -p api-server
```

The loader (`crates/llm-core/src/loader.rs`) automatically:
- Detects safetensors vs GGUF format
- Memory-maps weights (avoids RAM duplication)
- Maps HuggingFace weight names to internal layout
- Converts f32 → f16 on load if needed

---

## Simulation agent

The simulation agent lets the LLM call physics engines via tool-use. All 7 engines are available via REST:

### N-body gravitational simulation

```bash
curl -X POST http://localhost:8080/v1/simulate \
  -H "Content-Type: application/json" -d '{
  "sim_type": "n_body",
  "description": "Binary star system",
  "params": { "preset": "BinaryStar", "dt": 3600, "total_time": 31557600, "softening": 1e6, "record_every": 24 },
  "max_steps": 10000
}'
```

### Quantum wavefunction (Schrödinger equation)

```bash
curl -X POST http://localhost:8080/v1/simulate \
  -H "Content-Type: application/json" -d '{
  "sim_type": "quantum_wavefunction",
  "description": "Harmonic oscillator ground state",
  "params": {
    "potential": {"HarmonicOscillator": {"omega": 1e15}},
    "n_grid": 512, "x_min": -5e-9, "x_max": 5e-9, "dt": 1e-19,
    "mass": 9.109e-31,
    "initial_state": {"Eigenstate": {"n": 1}},
    "observe_every": 20
  },
  "max_steps": 3000
}'
```

### Available presets

| Simulation | Presets |
|---|---|
| N-body | `SolarSystem`, `BinaryStar`, `ThreeBody`, `GalaxyCore`, `PlanetaryMoons` |
| Reaction kinetics | `OzoneDepletion`, `H2O2Combustion`, `ISMHydrogenChemistry`, `Brusselator` |
| Thermodynamics | `IdealGas`, `VanDerWaals`, `Blackbody`, `StellarAtmosphere` |
| Stellar | Custom mass 0.1–150 M☉ |

---

## Project structure

```
physllm/
├── Cargo.toml                    # Workspace manifest
├── configs/
│   ├── physllm_7b.json           # 7B model hyperparameters
│   └── physllm_13b.json          # 13B model hyperparameters
├── kernels/                      # HIP GPU kernels
│   ├── flash_attention.hip       # Flash Attention 2 (AMD)
│   ├── rope_embedding.hip        # Rotary position embeddings
│   └── layer_norm.hip            # RMSNorm / LayerNorm
├── scripts/
│   ├── setup_rocm.sh             # ROCm environment checker
│   ├── train.sh                  # Training launcher
│   └── example_sim.sh            # API usage examples
└── crates/
    ├── rocm-backend/             # AMD GPU abstraction layer
    │   ├── build.rs              # Compiles HIP kernels, generates FFI bindings
    │   └── src/
    │       ├── lib.rs            # Public API
    │       ├── device.rs         # GpuDevice lifecycle
    │       ├── tensor.rs         # DeviceTensor<T> (typed GPU buffer)
    │       ├── ops.rs            # matmul_f16, flash_attention, rope, rms_norm
    │       ├── memory.rs         # MemoryPool (bump allocator)
    │       └── kernels.rs        # Kernel registry
    ├── llm-core/                 # Transformer model
    │   └── src/
    │       ├── config.rs         # ModelConfig (all hyperparameters)
    │       ├── model.rs          # PhysLLM forward pass
    │       ├── attention.rs      # Grouped Query Attention (GQA)
    │       ├── ffn.rs            # SwiGLU feed-forward network
    │       ├── embedding.rs      # Token + positional embeddings
    │       ├── kv_cache.rs       # KV cache + paged allocator
    │       ├── tokenizer.rs      # Physics-aware tokenizer
    │       ├── generate.rs       # Autoregressive generation + sampling
    │       └── loader.rs         # safetensors / GGUF weight loader
    ├── domain-physics/           # Physics knowledge layer
    │   └── src/
    │       ├── constants.rs      # NIST CODATA 2022 database
    │       ├── chemistry.rs      # Formula parser + IUPAC atomic weights
    │       ├── units.rs          # SI unit system
    │       ├── astrophysics.rs   # Astro objects + Planck/Schwarzschild/etc.
    │       └── vocab.rs          # Domain token extensions
    ├── sim-agent/                # Physics simulation agent
    │   └── src/
    │       ├── dispatcher.rs     # Tool-use router
    │       ├── tools.rs          # JSON tool schema for LLM
    │       ├── nbody.rs          # Gravitational N-body (RK4)
    │       ├── quantum.rs        # 1D Schrödinger (Crank-Nicolson)
    │       ├── molecular_dynamics.rs  # Lennard-Jones MD
    │       ├── reaction_kinetics.rs   # ODE kinetics (RK45 adaptive)
    │       ├── stellar.rs        # Stellar evolution (HR track)
    │       └── astrochem.rs      # ISM astrochemical network
    └── api-server/               # REST + WebSocket server
        └── src/main.rs           # Axum endpoints
```

---

## ROCm GPU kernel details

### Supported targets
```
gfx1100 — RX 7900 XTX, RX 7900 XT (RDNA3)
gfx1030 — RX 6900 XT, RX 6800 XT  (RDNA2)
gfx906  — Vega 20                  (RDNA1 / Instinct MI50)
gfx90a  — Instinct MI200 series    (CDNA2)
```

### Compiled kernels

| Kernel | Description |
|--------|-------------|
| `flash_attention.hip` | Flash Attention 2, f16, causal + non-causal, wavefront-64 |
| `rope_embedding.hip` | Rotary Position Embeddings, in-place on Q and K |
| `layer_norm.hip` | RMSNorm and LayerNorm, f16 I/O, f32 accumulation |

### Adding a new kernel

1. Write `kernels/your_kernel.hip`
2. Add its name to `KERNELS` array in `build.rs`
3. Add `extern "C"` declaration in `ops.rs`
4. Call from Rust via `unsafe { your_kernel_hip(...) }`

---

## Simulation engines — physics details

### N-body (RK4)
- 4th-order Runge-Kutta time integration
- Softened gravitational potential: `F = G·m1·m2 / (r² + ε²)^(3/2)`
- Tracks total energy E and angular momentum L (conservation check)
- Energy drift < 0.01% over 1 year (solar system, dt=1h)

### Quantum (Crank-Nicolson)
- Implicit, unconditionally stable
- Second-order accurate in both time and space
- Thomas algorithm tridiagonal solver: O(N) per step
- Norm conserved to machine precision

### Molecular Dynamics (Lennard-Jones)
- Velocity Verlet integrator
- Minimum image periodic boundary conditions
- Cutoff radius: 2.5σ
- NVE and NVT (Berendsen velocity rescaling) ensembles

### Reaction Kinetics (RK45)
- Dormand-Prince adaptive step size
- PI controller for step-size selection
- Concentration clamping at zero (no negative species)
- Arrhenius, power-law, and ISM rate laws

### Stellar Evolution
- Simplified main-sequence evolutionary track
- Schoenberg-Chandrasekhar MS lifetime: t_MS ∝ M^(-2.5)
- Post-MS fate: white dwarf (M < 8 M☉), neutron star/BH (M > 8 M☉)

### Astrochemistry (ISM network)
- UMIST-based rate network
- Cosmic ray ionisation, UV photodissociation, gas-phase reactions
- Species: H, H₂, H⁺, H₂⁺, H₃⁺, CO, OH, H₂O, HCN, e⁻, and more

---

## Physics domain vocabulary

The tokenizer extends the base vocabulary with 100+ domain tokens:

- **Mathematical:** `∇`, `∂`, `∮`, `∇²`, `∂/∂t`, `d/dt`
- **Units:** `eV`, `MeV`, `Å`, `fm`, `AU`, `ly`, `pc`, `M☉`, `L☉`
- **Chemical formulae:** `H₂`, `CO₂`, `NH₃`, `H₂O`, `H₃⁺`, `HCO⁺`
- **Greek symbols:** α β γ δ ε ζ η θ κ λ μ ν ξ π ρ σ τ φ χ ψ ω
- **Special:** `<|sim_start|>`, `<|tool_call|>`, `<|eq|>`, `<|/eq|>`
- **Constants:** `<|c_light|>`, `<|h_planck|>`, `<|k_boltzmann|>`

---

## Physical constants (NIST CODATA 2022)

All accessible via `/v1/constants/:symbol`:

| Symbol | Name | Value |
|--------|------|-------|
| `c` | Speed of light | 299 792 458 m/s (exact) |
| `h` | Planck constant | 6.626 070 15 × 10⁻³⁴ J·s (exact) |
| `hbar` | Reduced Planck | 1.054 571 817 × 10⁻³⁴ J·s |
| `G` | Gravitational constant | 6.674 30 × 10⁻¹¹ m³ kg⁻¹ s⁻² |
| `kB` | Boltzmann constant | 1.380 649 × 10⁻²³ J/K (exact) |
| `NA` | Avogadro constant | 6.022 140 76 × 10²³ mol⁻¹ (exact) |
| `e` | Elementary charge | 1.602 176 634 × 10⁻¹⁹ C (exact) |
| `me` | Electron mass | 9.109 383 7015 × 10⁻³¹ kg |
| `sigma` | Stefan-Boltzmann | 5.670 374 419 × 10⁻⁸ W m⁻² K⁻⁴ |
| `Msun` | Solar mass | 1.988 416 × 10³⁰ kg |
| `H0` | Hubble constant | 67.4 ± 0.5 km/s/Mpc (Planck 2018) |

---

## Fine-tuning on physics data

To fine-tune on physics papers/textbooks:

```bash
# Requirements
pip install pandas pyarrow requests tqdm feedparser PyMuPDF chemspipy
# 1. Prepare data (arXiv physics, NIST, textbooks)
python scripts/prepare_data.py \
  --sources arxiv_physics,nist_webbook,chemspider \
  --output data/physics_corpus \
  --format parquet

# 2. Launch fine-tuning
./scripts/train.sh \
  --model 7b \
  --data  data/physics_corpus \
  --lr    2e-5 \
  --batch 4

# 3. Monitor
tail -f logs/last_run.log | python scripts/parse_metrics.py
```

Recommended training data sources:
- arXiv (physics, astro-ph, cond-mat, quant-ph, hep) — 2M+ papers
- NIST WebBook (thermodynamic data)
- ChemSpider / PubChem (molecular data)
- SIMBAD / NED (astronomical object catalogs)
- OpenStax physics textbooks (CC-BY)

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/health` | Service health and GPU info |
| GET | `/v1/models` | Available model configs |
| POST | `/v1/generate` | Text generation |
| WS | `/v1/stream` | Streaming generation |
| POST | `/v1/simulate` | Run a physics simulation |
| GET | `/v1/constants/:symbol` | NIST constant lookup |
| POST | `/v1/chemistry/mw` | Molecular weight from formula |
| GET | `/v1/tools` | Tool schema for LLM tool-use |

---

## Roadmap

- [ ] Multi-GPU tensor parallelism (hipMPI)
- [ ] PagedAttention (vLLM-style KV cache)
- [ ] GPTQ / AWQ INT4 quantisation
- [ ] Continuous batching
- [ ] LoRA fine-tuning support
- [ ] GGUF export for llama.cpp compatibility
- [ ] Plasma / MHD simulation engine
- [ ] Neural equation discovery (SINDy integration)
- [ ] OpenCL fallback for non-ROCm AMD GPUs
- [ ] Docker + ROCm container image

---

## License

MIT — see LICENSE file.

---

## Citing

```bibtex
@software{physllm2025,
  title  = {PhysLLM: A Physics and Chemistry LLM Runtime in Rust with AMD ROCm},
  year   = {2026},
  url    = {https://github.com/manpreetsingh1368/physllm}
}
```
