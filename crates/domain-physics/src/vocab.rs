// vocab.rs — physics domain vocabulary extensions
pub fn domain_vocab() -> Vec<String> {
    vec![
        // Physical constants as tokens
        "ℏ", "∇²", "∂/∂t", "∮", "∯", "⊗", "⊕",
        // Common physics expressions
        "E=mc²", "F=ma", "PV=nRT", "H₂O", "CO₂", "NH₃",
        // Units
        "eV", "MeV", "GeV", "TeV", "fm", "Å", "nm", "μm",
        "kPa", "MPa", "GPa", "mbar", "atm",
        "mol/L", "cm⁻³", "g/mol",
        // Operators
        "∇", "∆", "∂", "∫", "Σ", "Π", "∞",
        // Greek letters used as variables
        "α", "β", "γ", "δ", "ε", "ζ", "η", "θ",
        "ι", "κ", "λ", "μ", "ν", "ξ", "π", "ρ",
        "σ", "τ", "φ", "χ", "ψ", "ω", "Γ", "Δ",
        "Θ", "Λ", "Ξ", "Π", "Σ", "Φ", "Ψ", "Ω",
        // Astrophysics
        "M☉", "L☉", "R☉", "pc", "kpc", "Mpc", "Gpc",
        "yr", "Myr", "Gyr", "AU", "ly",
        // Element symbols (common)
        "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si",
        "Ar", "Ca", "Fe", "Ni", "Cu", "Zn", "Kr", "Xe",
    ].into_iter().map(String::from).collect()
}
