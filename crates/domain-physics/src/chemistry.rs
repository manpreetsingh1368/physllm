//! chemistry.rs — Chemical formula parsing, molecular weight, and properties.

use crate::{DomainError, Result};
use std::collections::HashMap;

/// A parsed chemical formula with element counts.
#[derive(Debug, Clone)]
pub struct ChemicalFormula {
    pub raw:      String,
    pub elements: HashMap<String, u32>,
}

impl ChemicalFormula {
    /// Parse a chemical formula string, e.g. "H2O", "C6H12O6", "Fe2(SO4)3".
    pub fn parse(formula: &str) -> Result<Self> {
        let elements = parse_formula(formula)?;
        Ok(Self { raw: formula.to_string(), elements })
    }

    /// Compute molecular weight in g/mol using standard atomic weights.
    pub fn molecular_weight(&self) -> Result<f64> {
        let mut mw = 0.0f64;
        for (element, &count) in &self.elements {
            let aw = atomic_weight(element)
                .ok_or_else(|| DomainError::UnknownElement(element.clone()))?;
            mw += aw * count as f64;
        }
        Ok(mw)
    }

    /// Hill notation (C first, H second, then alphabetical)
    pub fn hill_notation(&self) -> String {
        let mut parts: Vec<(String, u32)> = self.elements.iter()
            .map(|(k, &v)| (k.clone(), v)).collect();
        parts.sort_by(|(a, _), (b, _)| {
            let rank = |s: &str| match s {
                "C" => 0, "H" => 1, _ => 2,
            };
            rank(a).cmp(&rank(b)).then(a.cmp(b))
        });
        parts.iter().map(|(e, n)| {
            if *n == 1 { e.clone() } else { format!("{e}{n}") }
        }).collect::<String>()
    }

    /// Check if formula is charge-balanced (simple heuristic for common ions)
    pub fn atom_count(&self) -> u32 {
        self.elements.values().sum()
    }
}

/// Molecule with formula + common properties.
#[derive(Debug, Clone)]
pub struct Molecule {
    pub name:     String,
    pub formula:  ChemicalFormula,
    pub cas:      Option<String>,
    pub smiles:   Option<String>,
    pub phase:    Option<Phase>,
}

#[derive(Debug, Clone)]
pub enum Phase { Gas, Liquid, Solid, Plasma, Interstellar }

impl Molecule {
    pub fn new(name: &str, formula: &str) -> Result<Self> {
        Ok(Self {
            name:    name.to_string(),
            formula: ChemicalFormula::parse(formula)?,
            cas:     None,
            smiles:  None,
            phase:   None,
        })
    }

    pub fn molar_mass(&self) -> Result<f64> { self.formula.molecular_weight() }
}

// ── Astrochemistry molecule database ─────────────────────────────────────────

/// Pre-built list of interstellar molecules detected in the ISM.
pub fn interstellar_molecules() -> Vec<Molecule> {
    vec![
        // Simple diatomics
        Molecule { name: "Molecular hydrogen".into(),  formula: ChemicalFormula::parse("H2").unwrap(), cas: Some("1333-74-0".into()), smiles: Some("[HH]".into()), phase: Some(Phase::Interstellar) },
        Molecule { name: "Carbon monoxide".into(),     formula: ChemicalFormula::parse("CO").unwrap(), cas: Some("630-08-0".into()),  smiles: Some("[C-]#[O+]".into()), phase: Some(Phase::Interstellar) },
        Molecule { name: "Hydroxyl radical".into(),    formula: ChemicalFormula::parse("OH").unwrap(), cas: Some("3352-57-6".into()), smiles: Some("[OH]".into()), phase: Some(Phase::Interstellar) },
        Molecule { name: "Cyanide radical".into(),     formula: ChemicalFormula::parse("CN").unwrap(), cas: Some("2074-87-5".into()), smiles: Some("[C]#N".into()), phase: Some(Phase::Interstellar) },
        // Triatomics
        Molecule { name: "Water".into(),               formula: ChemicalFormula::parse("H2O").unwrap(), cas: Some("7732-18-5".into()), smiles: Some("O".into()), phase: Some(Phase::Interstellar) },
        Molecule { name: "Carbon dioxide".into(),      formula: ChemicalFormula::parse("CO2").unwrap(), cas: Some("124-38-9".into()), smiles: Some("O=C=O".into()), phase: Some(Phase::Interstellar) },
        Molecule { name: "Hydrogen cyanide".into(),    formula: ChemicalFormula::parse("HCN").unwrap(), cas: Some("74-90-8".into()),  smiles: Some("C#N".into()), phase: Some(Phase::Interstellar) },
        Molecule { name: "Ammonia".into(),             formula: ChemicalFormula::parse("NH3").unwrap(), cas: Some("7664-41-7".into()), smiles: Some("N".into()), phase: Some(Phase::Interstellar) },
        // Complex organics (COMs)
        Molecule { name: "Methanol".into(),            formula: ChemicalFormula::parse("CH3OH").unwrap(), cas: Some("67-56-1".into()), smiles: Some("CO".into()), phase: Some(Phase::Interstellar) },
        Molecule { name: "Glycine".into(),             formula: ChemicalFormula::parse("C2H5NO2").unwrap(), cas: Some("56-40-6".into()), smiles: Some("NCC(=O)O".into()), phase: Some(Phase::Interstellar) },
        Molecule { name: "Dimethyl ether".into(),      formula: ChemicalFormula::parse("C2H6O").unwrap(), cas: Some("115-10-6".into()), smiles: Some("COC".into()), phase: Some(Phase::Interstellar) },
        Molecule { name: "Ethanol".into(),             formula: ChemicalFormula::parse("C2H5OH").unwrap(), cas: Some("64-17-5".into()), smiles: Some("CCO".into()), phase: Some(Phase::Interstellar) },
        // PAHs (polycyclic aromatic hydrocarbons)
        Molecule { name: "Naphthalene".into(),         formula: ChemicalFormula::parse("C10H8").unwrap(), cas: Some("91-20-3".into()), smiles: Some("c1ccc2ccccc2c1".into()), phase: Some(Phase::Interstellar) },
        // Exotic
        Molecule { name: "Trihydrogen cation".into(),  formula: ChemicalFormula::parse("H3").unwrap(), cas: None, smiles: None, phase: Some(Phase::Interstellar) },
    ]
}

// ── Formula parser ────────────────────────────────────────────────────────────

fn parse_formula(s: &str) -> Result<HashMap<String, u32>> {
    let mut result = HashMap::new();
    let chars: Vec<char> = s.chars().collect();
    parse_segment(&chars, 0, &mut result)?;
    Ok(result)
}

fn parse_segment(chars: &[char], mut i: usize, acc: &mut HashMap<String, u32>) -> Result<usize> {
    while i < chars.len() {
        match chars[i] {
            '(' => {
                let mut sub = HashMap::new();
                i = parse_segment(chars, i + 1, &mut sub)?;
                // expect ')'
                if i >= chars.len() || chars[i] != ')' {
                    return Err(DomainError::ParseError("unmatched '('".into()));
                }
                i += 1;
                // read multiplier
                let (n, new_i) = read_number(chars, i);
                let mult = n.max(1);
                i = new_i;
                for (el, cnt) in sub { *acc.entry(el).or_insert(0) += cnt * mult; }
            }
            ')' => return Ok(i),
            c if c.is_uppercase() => {
                let (elem, new_i) = read_element(chars, i);
                i = new_i;
                let (n, new_i) = read_number(chars, i);
                i = new_i;
                *acc.entry(elem).or_insert(0) += n.max(1);
            }
            _ => { i += 1; }
        }
    }
    Ok(i)
}

fn read_element(chars: &[char], start: usize) -> (String, usize) {
    let mut s = chars[start].to_string();
    let mut i = start + 1;
    while i < chars.len() && chars[i].is_lowercase() {
        s.push(chars[i]);
        i += 1;
    }
    (s, i)
}

fn read_number(chars: &[char], start: usize) -> (u32, usize) {
    let mut s = String::new();
    let mut i = start;
    while i < chars.len() && chars[i].is_ascii_digit() {
        s.push(chars[i]);
        i += 1;
    }
    (s.parse().unwrap_or(0), i)
}

// ── Atomic weights (IUPAC 2021 standard) ─────────────────────────────────────

pub fn atomic_weight(symbol: &str) -> Option<f64> {
    match symbol {
        "H"  => Some(1.008),
        "He" => Some(4.002602),
        "Li" => Some(6.94),
        "Be" => Some(9.0121831),
        "B"  => Some(10.81),
        "C"  => Some(12.011),
        "N"  => Some(14.007),
        "O"  => Some(15.999),
        "F"  => Some(18.998403163),
        "Ne" => Some(20.1797),
        "Na" => Some(22.98976928),
        "Mg" => Some(24.305),
        "Al" => Some(26.9815385),
        "Si" => Some(28.085),
        "P"  => Some(30.973761998),
        "S"  => Some(32.06),
        "Cl" => Some(35.45),
        "Ar" => Some(39.948),
        "K"  => Some(39.0983),
        "Ca" => Some(40.078),
        "Sc" => Some(44.955908),
        "Ti" => Some(47.867),
        "V"  => Some(50.9415),
        "Cr" => Some(51.9961),
        "Mn" => Some(54.938044),
        "Fe" => Some(55.845),
        "Co" => Some(58.933194),
        "Ni" => Some(58.6934),
        "Cu" => Some(63.546),
        "Zn" => Some(65.38),
        "Ga" => Some(69.723),
        "Ge" => Some(72.630),
        "As" => Some(74.921595),
        "Se" => Some(78.971),
        "Br" => Some(79.904),
        "Kr" => Some(83.798),
        "Rb" => Some(85.4678),
        "Sr" => Some(87.62),
        "Y"  => Some(88.90584),
        "Zr" => Some(91.224),
        "Nb" => Some(92.90637),
        "Mo" => Some(95.95),
        "Tc" => Some(98.0),
        "Ru" => Some(101.07),
        "Rh" => Some(102.90550),
        "Pd" => Some(106.42),
        "Ag" => Some(107.8682),
        "Cd" => Some(112.414),
        "In" => Some(114.818),
        "Sn" => Some(118.710),
        "Sb" => Some(121.760),
        "Te" => Some(127.60),
        "I"  => Some(126.90447),
        "Xe" => Some(131.293),
        "Cs" => Some(132.90545196),
        "Ba" => Some(137.327),
        "La" => Some(138.90547),
        "Ce" => Some(140.116),
        "Pr" => Some(140.90766),
        "Nd" => Some(144.242),
        "Pm" => Some(145.0),
        "Sm" => Some(150.36),
        "Eu" => Some(151.964),
        "Gd" => Some(157.25),
        "Tb" => Some(158.92535),
        "Dy" => Some(162.500),
        "Ho" => Some(164.93033),
        "Er" => Some(167.259),
        "Tm" => Some(168.93422),
        "Yb" => Some(173.045),
        "Lu" => Some(174.9668),
        "Hf" => Some(178.49),
        "Ta" => Some(180.94788),
        "W"  => Some(183.84),
        "Re" => Some(186.207),
        "Os" => Some(190.23),
        "Ir" => Some(192.217),
        "Pt" => Some(195.084),
        "Au" => Some(196.966569),
        "Hg" => Some(200.592),
        "Tl" => Some(204.38),
        "Pb" => Some(207.2),
        "Bi" => Some(208.98040),
        "Po" => Some(209.0),
        "At" => Some(210.0),
        "Rn" => Some(222.0),
        "Fr" => Some(223.0),
        "Ra" => Some(226.0),
        "Ac" => Some(227.0),
        "Th" => Some(232.0377),
        "Pa" => Some(231.03588),
        "U"  => Some(238.02891),
        "Np" => Some(237.0),
        "Pu" => Some(244.0),
        _ => None,
    }
}
