//! tools.rs — JSON tool definitions for LLM web search tool-use.

pub const SEARCH_TOOLS: &str = r#"[
  {
    "name": "web_search",
    "description": "Search the web and scientific databases for current information. Automatically routes to the best source: arXiv for physics papers, Semantic Scholar for citations, NIST for constants/chemistry data, NASA ADS for astrophysics objects, PubChem for molecules, or general web search. Use this when you need current information, specific papers, or data you are not certain about.",
    "input_schema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query. Can be: a natural language question, an arXiv ID (e.g. '2401.12345'), a molecule name, a physical constant name, an astronomical object name, or a URL to fetch."
        },
        "focus": {
          "type": "string",
          "enum": ["auto", "arxiv", "semantic_scholar", "nist", "nasa_ads", "pubchem", "web"],
          "description": "Force a specific backend. Default 'auto' lets the router decide based on query content."
        },
        "max_results": {
          "type": "integer",
          "description": "Maximum results to return (default: 5, max: 10)"
        },
        "fetch_full_text": {
          "type": "boolean",
          "description": "Whether to fetch and include full page text for top results (slower, more context). Default false."
        }
      },
      "required": ["query"]
    }
  },
  {
    "name": "fetch_url",
    "description": "Fetch and extract the full text content of a specific URL. Use for arXiv papers, journal articles, NIST pages, NASA ADS records, or any web page. Automatically extracts clean text from HTML or PDF.",
    "input_schema": {
      "type": "object",
      "properties": {
        "url": {
          "type": "string",
          "description": "The URL to fetch. Supports HTTP/HTTPS, arXiv abstract pages, PDFs."
        },
        "max_chars": {
          "type": "integer",
          "description": "Maximum characters to return (default: 8000)"
        }
      },
      "required": ["url"]
    }
  },
  {
    "name": "search_arxiv",
    "description": "Search arXiv preprint server directly. Best for: finding recent physics/chemistry/astro papers, looking up a specific arXiv ID, or searching within a specific category.",
    "input_schema": {
      "type": "object",
      "properties": {
        "query":    { "type": "string",  "description": "Search terms or arXiv ID" },
        "category": {
          "type": "string",
          "description": "arXiv category filter",
          "enum": ["astro-ph", "astro-ph.GA", "astro-ph.SR", "astro-ph.CO", "astro-ph.HE",
                   "quant-ph", "cond-mat", "hep-ph", "hep-th", "nucl-th",
                   "physics.chem-ph", "physics.atom-ph", "physics.flu-dyn", "math-ph"]
        },
        "max_results": { "type": "integer" },
        "sort_by": {
          "type": "string",
          "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
          "description": "Sort order for results"
        }
      },
      "required": ["query"]
    }
  },
  {
    "name": "lookup_constant",
    "description": "Look up a physical constant from the NIST CODATA 2022 database. Returns the value, uncertainty, and units.",
    "input_schema": {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "Constant symbol or name. Examples: 'hbar', 'kB', 'G', 'e', 'me', 'NA', 'c', 'sigma', 'alpha', 'a0', 'Msun', 'H0'"
        }
      },
      "required": ["symbol"]
    }
  },
  {
    "name": "lookup_molecule",
    "description": "Look up molecular properties from PubChem and NIST. Returns formula, molecular weight, SMILES, InChI, boiling/melting points, and thermodynamic data.",
    "input_schema": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "Molecule name, IUPAC name, CAS number, or chemical formula"
        },
        "properties": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Specific properties to retrieve: ['formula', 'mw', 'smiles', 'boiling_point', 'melting_point', 'density', 'thermodynamic']"
        }
      },
      "required": ["name"]
    }
  }
]"#;
