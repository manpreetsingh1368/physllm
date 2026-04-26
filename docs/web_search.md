# Web Search Integration — PhysLLM

PhysLLM includes a full web search layer (`crates/web-search/`) that lets the LLM
autonomously retrieve current papers, chemical data, and astronomical records.

## Architecture

```
User query
    │
SearchRouter (detects intent)
    ├── arXiv API          → physics/astro preprints  (free, no key)
    ├── Semantic Scholar   → citations + abstracts     (free, no key)
    ├── NIST WebBook       → chemical + constant data  (free, no key)
    ├── NASA ADS           → astronomical literature   (free key required)
    ├── PubChem REST       → molecular properties      (free, no key)
    ├── Brave Search       → general web               (paid key, ~$3/mo)
    └── DuckDuckGo         → general web fallback      (free, limited)
         │
    SearchResponse
    (title · url · snippet · full_text · score · metadata)
         │
    context.rs  →  <search_results>...</search_results>  →  LLM prompt
```

## Intent detection (automatic routing)

The router detects the type of query and picks the right backend(s):

| Query contains | Routes to |
|---|---|
| `2401.12345` (arXiv ID) | arXiv direct lookup |
| `dark matter`, `quantum`, `Hamiltonian` | arXiv + Semantic Scholar |
| `molecular weight`, `boiling point`, `CAS` | PubChem + NIST |
| `NGC`, `galaxy`, `neutron star`, `JWST` | NASA ADS + arXiv |
| `NIST`, `CODATA`, `Boltzmann`, `Planck constant` | NIST WebBook |
| `https://...` | WebFetcher (full text extraction) |
| Anything else | Brave → DuckDuckGo fallback |

## Quick start

```bash
# No API keys needed for scientific search
cargo run --release -p api-server

# Search arXiv for recent papers
curl -X POST http://localhost:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "gravitational wave detection 2024"}'

# Direct arXiv paper lookup by ID
curl -X POST http://localhost:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "2401.04088"}'

# Chemistry data (auto-routes to PubChem + NIST)
curl -X POST http://localhost:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "molecular weight caffeine"}'

# Generate with search augmentation (RAG)
curl -X POST http://localhost:8080/v1/generate/search \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the Hubble tension?", "search_query": "Hubble tension 2024"}'

# Fetch full text of any URL (HTML or PDF)
curl -X POST http://localhost:8080/v1/fetch \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/abs/2301.04309"}'
```

## API keys (optional but recommended)

```bash
# Brave Search — best general web search ($3/month for 2000 queries/month)
export BRAVE_API_KEY=your_key   # https://api.search.brave.com/

# NASA ADS — required for astrophysics object queries (free)
export NASA_ADS_KEY=your_key    # https://ui.adsabs.harvard.edu/user/settings/token

# Enable full page text extraction for top results (slower)
export FULL_PAGE_FETCH=1
```

Without any API keys, the router still works for arXiv, Semantic Scholar, NIST, and PubChem — which covers the majority of physics and chemistry queries.

## Python client

```bash
python3 scripts/client.py "What are the latest LIGO gravitational wave detections?"
python3 scripts/client.py --arxiv "2401.04088"
python3 scripts/client.py --molecule "glycine"
python3 scripts/client.py --constant "G"
python3 scripts/client.py --simulate nbody --preset solar_system
python3 scripts/client.py --fetch "https://arxiv.org/abs/2301.04309"
python3 scripts/client.py --no-search "Derive the uncertainty principle"
```

## Search-augmented generation (RAG)

The `POST /v1/generate/search` endpoint:
1. Runs the search router on the `search_query`
2. Formats results into a `<search_results>` block
3. Injects the block before the user prompt
4. Runs transformer inference on the augmented context
5. Returns both the generated text and the search metadata

The agentic loop (`crates/llm-core/src/search_agent.rs`) goes further:
the LLM can itself decide to call search tools mid-generation, execute
them, receive the results, and continue reasoning — up to `max_iterations`
tool-use cycles before producing a final answer.

## Caching

Results are cached in-memory with a configurable TTL (default 1 hour).
This means repeated queries to the same paper or molecule are instant.
The cache holds up to 1000 entries and evicts expired items automatically.

## Rate limiting

Per-domain token buckets prevent hammering any single API:

| Backend | Rate limit |
|---|---|
| arXiv | 3 req/s (burst 5) |
| Semantic Scholar | 5 req/s (burst 10) |
| NIST WebBook | 2 req/s (burst 3) |
| NASA ADS | 5 req/s (burst 10) |
| PubChem | 5 req/s (burst 10) |
| Brave | 1 req/s (burst 3) |
