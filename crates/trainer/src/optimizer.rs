// crates/trainer/src/optimizer.rs
//
// Production AdamW optimizer.
//
// Implements the decoupled weight decay variant from
// "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019).
//
// Update rule:
//   m_t  = β₁·m_{t-1} + (1-β₁)·g_t                    (1st moment)
//   v_t  = β₂·v_{t-1} + (1-β₂)·g_t²                   (2nd moment)
//   m̂_t  = m_t  / (1 - β₁ᵗ)                            (bias correction)
//   v̂_t  = v_t  / (1 - β₂ᵗ)                            (bias correction)
//   θ_t  = θ_{t-1} - lr · (m̂_t / (√v̂_t + ε) + λ·θ_{t-1})
//
// Weight decay is applied to θ directly (not to the gradient),
// which is the key difference from L2 regularisation in Adam.

use anyhow::{Context, Result};
use candle_core::{Tensor, Var, DType};
use tracing::debug;

/// AdamW configuration.
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub lr:           f64,
    pub beta1:        f64,
    pub beta2:        f64,
    pub eps:          f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr:           2e-4,
            beta1:        0.9,
            beta2:        0.999,
            eps:          1e-8,
            weight_decay: 0.01,
        }
    }
}

/// State for a single parameter.
struct ParamState {
    /// Exponential moving average of gradients (1st moment)
    m: Tensor,
    /// Exponential moving average of squared gradients (2nd moment)
    v: Tensor,
}

/// AdamW optimizer with bias correction.
pub struct AdamW {
    vars:   Vec<Var>,
    states: Vec<Option<ParamState>>,
    config: AdamWConfig,
    step:   usize,
}

impl AdamW {
    pub fn new(vars: Vec<Var>, config: AdamWConfig) -> Result<Self> {
        let n = vars.len();
        Ok(Self {
            vars,
            states: (0..n).map(|_| None).collect(),
            config,
            step: 0,
        })
    }

    /// Set learning rate (called every step from the LR schedule).
    pub fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    /// Zero all gradients.
    pub fn zero_grad(&self) {
        for var in &self.vars {
            let _ = var.set_grad(None);
        }
    }

    /// Compute the global L2 gradient norm across all parameters.
    pub fn global_grad_norm(&self) -> Result<f64> {
        let mut sq_sum = 0.0f64;
        let mut count  = 0;
        for var in &self.vars {
            if let Some(g) = var.grad() {
                let g_f32 = g.to_dtype(DType::F32)?;
                let sq = g_f32.sqr()?.sum_all()?.to_scalar::<f32>()?;
                sq_sum += sq as f64;
                count  += 1;
            }
        }
        Ok(sq_sum.sqrt())
    }

    /// Clip gradients by global L2 norm. Returns the norm before clipping.
    pub fn clip_grad_norm(&self, max_norm: f64) -> Result<f64> {
        let norm = self.global_grad_norm()?;
        if norm > max_norm {
            let scale = max_norm / (norm + 1e-6);
            for var in &self.vars {
                if let Some(g) = var.grad() {
                    let clipped = (g * scale)?;
                    var.set_grad(Some(clipped));
                }
            }
            debug!("Grad clipped: {norm:.4} → {max_norm:.4}");
        }
        Ok(norm)
    }

    /// Perform one AdamW update step.
    pub fn step(&mut self) -> Result<()> {
        self.step += 1;
        let t      = self.step as f64;
        let cfg    = &self.config;

        // Bias correction factors
        let bc1 = 1.0 - cfg.beta1.powf(t);
        let bc2 = 1.0 - cfg.beta2.powf(t);
        // Effective learning rate (with bias correction folded in)
        let alpha = cfg.lr * bc2.sqrt() / bc1;

        for (i, var) in self.vars.iter().enumerate() {
            let grad = match var.grad() {
                Some(g) => g.to_dtype(DType::F32)?,
                None    => continue,
            };

            let param = var.as_tensor().to_dtype(DType::F32)?;

            // Initialise moment estimates on first step
            if self.states[i].is_none() {
                self.states[i] = Some(ParamState {
                    m: Tensor::zeros_like(&grad)?,
                    v: Tensor::zeros_like(&grad)?,
                });
            }
            let state = self.states[i].as_mut().unwrap();

            // m_t = β₁·m + (1-β₁)·g
            state.m = ((cfg.beta1 * &state.m)? + ((1.0 - cfg.beta1) * &grad)?)?;

            // v_t = β₂·v + (1-β₂)·g²
            let g_sq = grad.sqr()?;
            state.v = ((cfg.beta2 * &state.v)? + ((1.0 - cfg.beta2) * &g_sq)?)?;

            // Adaptive term: m_hat / (√v_hat + ε)
            // = (m / bc1) / (√(v / bc2) + ε)
            // = m / (bc1 · (√(v/bc2) + ε))
            // Simplified: alpha·m / (√v + eps·√bc2) [numerically equivalent]
            let denom = (state.v.sqrt()? + cfg.eps)?;
            let update = (&state.m / &denom)?;

            // θ_new = θ - alpha·update - lr·λ·θ   (decoupled weight decay)
            let decay_factor = 1.0 - cfg.lr * cfg.weight_decay;
            let new_param = ((decay_factor * &param)? - (alpha * &update)?)?;

            // Write back to the Var (which holds the actual gradient-tracked tensor)
            var.set(&new_param.to_dtype(var.dtype())?)?;
        }

        Ok(())
    }

    /// Serialise optimizer state to a map for checkpointing.
    pub fn state_dict(&self) -> Result<std::collections::HashMap<String, Tensor>> {
        let mut map = std::collections::HashMap::new();
        map.insert("step".to_string(), Tensor::new(&[self.step as f32], &candle_core::Device::Cpu)?);

        for (i, state) in self.states.iter().enumerate() {
            if let Some(s) = state {
                map.insert(format!("m_{i}"), s.m.clone());
                map.insert(format!("v_{i}"), s.v.clone());
            }
        }
        Ok(map)
    }

    /// Restore optimizer state from a map.
    pub fn load_state_dict(&mut self, map: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        if let Some(step_t) = map.get("step") {
            self.step = step_t.to_scalar::<f32>()? as usize;
        }
        for i in 0..self.vars.len() {
            if let (Some(m), Some(v)) = (
                map.get(&format!("m_{i}")),
                map.get(&format!("v_{i}")),
            ) {
                self.states[i] = Some(ParamState {
                    m: m.clone(),
                    v: v.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn current_step(&self) -> usize { self.step }
    pub fn current_lr(&self) -> f64    { self.config.lr }
}
