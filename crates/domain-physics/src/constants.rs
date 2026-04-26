//! constants.rs — NIST CODATA 2022 physical constants database.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{DomainError, Result};

/// A single physical constant with value, uncertainty, and units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalConstant {
    pub symbol:      String,
    pub name:        String,
    pub value:       f64,
    pub uncertainty: f64,
    pub unit:        String,
    pub dimension:   String,
}

impl PhysicalConstant {
    pub fn relative_uncertainty(&self) -> f64 {
        self.uncertainty / self.value.abs()
    }
}

/// Database of physical constants — loaded at startup.
pub struct ConstantsDB {
    by_symbol: HashMap<String, PhysicalConstant>,
    by_name:   HashMap<String, String>,  // name → symbol
}

impl ConstantsDB {
    /// Build the built-in NIST CODATA 2022 database.
    pub fn built_in() -> Self {
        let constants = vec![
            //  Universal 
            PhysicalConstant {
                symbol: "c".into(), name: "speed of light in vacuum".into(),
                value: 299_792_458.0, uncertainty: 0.0, // exact
                unit: "m s^-1".into(), dimension: "velocity".into(),
            },
            PhysicalConstant {
                symbol: "h".into(), name: "Planck constant".into(),
                value: 6.626_070_15e-34, uncertainty: 0.0, // exact (2019 SI)
                unit: "J s".into(), dimension: "action".into(),
            },
            PhysicalConstant {
                symbol: "hbar".into(), name: "reduced Planck constant".into(),
                value: 1.054_571_817e-34, uncertainty: 0.0,
                unit: "J s".into(), dimension: "action".into(),
            },
            PhysicalConstant {
                symbol: "G".into(), name: "Newtonian constant of gravitation".into(),
                value: 6.674_30e-11, uncertainty: 0.000_15e-11,
                unit: "m^3 kg^-1 s^-2".into(), dimension: "gravitational constant".into(),
            },
            PhysicalConstant {
                symbol: "eps0".into(), name: "vacuum electric permittivity".into(),
                value: 8.854_187_8128e-12, uncertainty: 0.000_000_0013e-12,
                unit: "F m^-1".into(), dimension: "electric permittivity".into(),
            },
            PhysicalConstant {
                symbol: "mu0".into(), name: "vacuum magnetic permeability".into(),
                value: 1.256_637_062_12e-6, uncertainty: 0.000_000_000_19e-6,
                unit: "N A^-2".into(), dimension: "magnetic permeability".into(),
            },
            //  Electromagnetic 
            PhysicalConstant {
                symbol: "e".into(), name: "elementary charge".into(),
                value: 1.602_176_634e-19, uncertainty: 0.0, // exact
                unit: "C".into(), dimension: "electric charge".into(),
            },
            PhysicalConstant {
                symbol: "me".into(), name: "electron mass".into(),
                value: 9.109_383_7015e-31, uncertainty: 0.000_000_0028e-31,
                unit: "kg".into(), dimension: "mass".into(),
            },
            PhysicalConstant {
                symbol: "mp".into(), name: "proton mass".into(),
                value: 1.672_621_923_69e-27, uncertainty: 0.000_000_000_51e-27,
                unit: "kg".into(), dimension: "mass".into(),
            },
            PhysicalConstant {
                symbol: "mn".into(), name: "neutron mass".into(),
                value: 1.674_927_498_04e-27, uncertainty: 0.000_000_000_95e-27,
                unit: "kg".into(), dimension: "mass".into(),
            },
            PhysicalConstant {
                symbol: "alpha".into(), name: "fine-structure constant".into(),
                value: 7.297_352_5693e-3, uncertainty: 0.000_000_0011e-3,
                unit: "1".into(), dimension: "dimensionless".into(),
            },
            //  Thermodynamic 
            PhysicalConstant {
                symbol: "kB".into(), name: "Boltzmann constant".into(),
                value: 1.380_649e-23, uncertainty: 0.0, // exact
                unit: "J K^-1".into(), dimension: "entropy".into(),
            },
            PhysicalConstant {
                symbol: "NA".into(), name: "Avogadro constant".into(),
                value: 6.022_140_76e23, uncertainty: 0.0, // exact
                unit: "mol^-1".into(), dimension: "amount of substance".into(),
            },
            PhysicalConstant {
                symbol: "R".into(), name: "molar gas constant".into(),
                value: 8.314_462_618, uncertainty: 0.0, // exact (= NA * kB)
                unit: "J mol^-1 K^-1".into(), dimension: "molar entropy".into(),
            },
            PhysicalConstant {
                symbol: "sigma".into(), name: "Stefan–Boltzmann constant".into(),
                value: 5.670_374_419e-8, uncertainty: 0.0, // exact
                unit: "W m^-2 K^-4".into(), dimension: "power per area per temperature^4".into(),
            },
            //  Atomic & Nuclear 
            PhysicalConstant {
                symbol: "a0".into(), name: "Bohr radius".into(),
                value: 5.291_772_109_03e-11, uncertainty: 0.000_000_000_80e-11,
                unit: "m".into(), dimension: "length".into(),
            },
            PhysicalConstant {
                symbol: "Ry".into(), name: "Rydberg energy".into(),
                value: 2.179_872_361_1035e-18, uncertainty: 0.000_000_000_0042e-18,
                unit: "J".into(), dimension: "energy".into(),
            },
            PhysicalConstant {
                symbol: "muB".into(), name: "Bohr magneton".into(),
                value: 9.274_010_0783e-24, uncertainty: 0.000_000_0028e-24,
                unit: "J T^-1".into(), dimension: "magnetic dipole moment".into(),
            },
            //  Astrophysics 
            PhysicalConstant {
                symbol: "pc".into(), name: "parsec".into(),
                value: 3.085_677_581_49e16, uncertainty: 0.000_000_000_06e16,
                unit: "m".into(), dimension: "length".into(),
            },
            PhysicalConstant {
                symbol: "ly".into(), name: "light year".into(),
                value: 9.460_730_472_5808e15, uncertainty: 0.0,
                unit: "m".into(), dimension: "length".into(),
            },
            PhysicalConstant {
                symbol: "au".into(), name: "astronomical unit".into(),
                value: 1.495_978_707e11, uncertainty: 0.0, // exact
                unit: "m".into(), dimension: "length".into(),
            },
            PhysicalConstant {
                symbol: "Msun".into(), name: "solar mass".into(),
                value: 1.988_416e30, uncertainty: 0.000_003e30,
                unit: "kg".into(), dimension: "mass".into(),
            },
            PhysicalConstant {
                symbol: "Rsun".into(), name: "solar radius".into(),
                value: 6.957e8, uncertainty: 0.001e8,
                unit: "m".into(), dimension: "length".into(),
            },
            PhysicalConstant {
                symbol: "Lsun".into(), name: "solar luminosity".into(),
                value: 3.828e26, uncertainty: 0.005e26,
                unit: "W".into(), dimension: "power".into(),
            },
            PhysicalConstant {
                symbol: "H0".into(), name: "Hubble constant (Planck 2018)".into(),
                value: 67.4, uncertainty: 0.5,
                unit: "km s^-1 Mpc^-1".into(), dimension: "inverse time".into(),
            },
            PhysicalConstant {
                symbol: "Omega_Lambda".into(), name: "cosmological constant density".into(),
                value: 0.6847, uncertainty: 0.0073,
                unit: "1".into(), dimension: "dimensionless".into(),
            },
        ];

        let mut by_symbol = HashMap::new();
        let mut by_name   = HashMap::new();

        for c in constants {
            by_name.insert(c.name.clone(), c.symbol.clone());
            by_symbol.insert(c.symbol.clone(), c);
        }

        Self { by_symbol, by_name }
    }

    pub fn get(&self, symbol: &str) -> Result<&PhysicalConstant> {
        self.by_symbol.get(symbol)
            .ok_or_else(|| DomainError::ConstantNotFound(symbol.into()))
    }

    pub fn search(&self, query: &str) -> Vec<&PhysicalConstant> {
        let q = query.to_lowercase();
        self.by_symbol.values()
            .filter(|c| c.name.to_lowercase().contains(&q) || c.symbol.to_lowercase().contains(&q))
            .collect()
    }

    pub fn all(&self) -> impl Iterator<Item = &PhysicalConstant> {
        self.by_symbol.values()
    }
}

//  Common aliases 

impl ConstantsDB {
    /// Speed of light (exact, SI 2019)
    pub fn c(&self) -> f64   { self.by_symbol["c"].value }
    pub fn h(&self) -> f64   { self.by_symbol["h"].value }
    pub fn hbar(&self) -> f64 { self.by_symbol["hbar"].value }
    pub fn G(&self) -> f64   { self.by_symbol["G"].value }
    pub fn kB(&self) -> f64  { self.by_symbol["kB"].value }
    pub fn NA(&self) -> f64  { self.by_symbol["NA"].value }
    pub fn e(&self) -> f64   { self.by_symbol["e"].value }
    pub fn me(&self) -> f64  { self.by_symbol["me"].value }
    pub fn mp(&self) -> f64  { self.by_symbol["mp"].value }
    pub fn sigma(&self) -> f64 { self.by_symbol["sigma"].value }
}
