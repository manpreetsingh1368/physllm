// astrophysics.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstroObject {
    pub name:       String,
    pub kind:       AstroKind,
    pub ra_deg:     Option<f64>,
    pub dec_deg:    Option<f64>,
    pub distance:   Option<f64>,  // parsecs
    pub redshift:   Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AstroKind {
    Star, Galaxy, Nebula, BlackHole, NeutronStar,
    Planet, Exoplanet, Pulsar, Quasar, Cluster,
    DarkMatterHalo, CosmicFilament,
}

// equations.rs
// Common named equations and their implementations
pub fn schwarzschild_radius(mass_kg: f64) -> f64 {
    // R_s = 2GM/c²
    2.0 * 6.674_30e-11 * mass_kg / (299_792_458.0_f64).powi(2)
}

pub fn hubble_distance(z: f64, h0_km_s_mpc: f64) -> f64 {
    // Simple Hubble law (non-relativistic): d = cz/H0
    let c_km_s = 299_792.458;
    let mpc_km = 3.085_677_581e19;
    c_km_s * z / h0_km_s_mpc * mpc_km  // metres
}

pub fn planck_function(wavelength_m: f64, temperature_K: f64) -> f64 {
    // B_λ(T) = 2hc²/λ⁵ × 1/(exp(hc/λkT) - 1)  [W/m³/sr]
    let h = 6.626_070_15e-34;
    let c = 299_792_458.0_f64;
    let k = 1.380_649e-23;
    let exp = (h * c / (wavelength_m * k * temperature_K)).exp() - 1.0;
    if exp <= 0.0 { return 0.0; }
    2.0 * h * c * c / wavelength_m.powi(5) / exp
}

pub fn escape_velocity(mass_kg: f64, radius_m: f64) -> f64 {
    // v_esc = sqrt(2GM/r)
    (2.0 * 6.674_30e-11 * mass_kg / radius_m).sqrt()
}

pub fn roche_limit(primary_mass: f64, secondary_density: f64) -> f64 {
    // d = R_p * (2 ρ_p / ρ_s)^(1/3)  — fluid body approximation
    // Simplified version without R_p: returns ratio
    (2.0 * primary_mass / secondary_density).powf(1.0 / 3.0) * 2.44
}
