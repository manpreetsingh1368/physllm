#!/usr/bin/env python3
"""
client.py — Python client for PhysLLM with web search.

Usage:
  python3 scripts/client.py "What are the latest findings on dark matter detection?"
  python3 scripts/client.py --no-search "Derive the Schrödinger equation"
  python3 scripts/client.py --arxiv "2401.04088"
  python3 scripts/client.py --molecule "caffeine"
  python3 scripts/client.py --constant "kB"
  python3 scripts/client.py --simulate nbody --preset solar_system
"""

import argparse, json, sys, textwrap, urllib.request, urllib.error

BASE = "http://localhost:8080/v1"

def post(endpoint: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{BASE}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)

def get(endpoint: str) -> dict:
    req = urllib.request.Request(f"{BASE}{endpoint}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())

def print_sources(search: dict):
    print(f"\n{'─'*60}")
    print(f"  Sources: {', '.join(search.get('sources_used', []))}")
    print(f"  Results: {search.get('total_found', 0)}  "
          f"({search.get('elapsed_ms', 0)}ms)")
    for i, r in enumerate(search.get('results', [])[:3], 1):
        print(f"  [{i}] {r['title'][:65]}")
        meta = r.get('metadata', {})
        parts = []
        if meta.get('authors'): parts.append(meta['authors'][:40])
        if meta.get('year'):    parts.append(meta['year'])
        if parts:               print(f"       {' · '.join(parts)}")
        print(f"       {r['url']}")

def main():
    p = argparse.ArgumentParser(description="PhysLLM client")
    p.add_argument("query", nargs="?", help="Query or prompt")
    p.add_argument("--no-search",  action="store_true", help="Skip web search")
    p.add_argument("--arxiv",      metavar="ID_OR_QUERY", help="Search arXiv")
    p.add_argument("--category",   default=None,          help="arXiv category")
    p.add_argument("--molecule",   metavar="NAME",         help="Molecule lookup")
    p.add_argument("--constant",   metavar="SYMBOL",       help="Physical constant")
    p.add_argument("--simulate",   metavar="TYPE",         help="Run simulation")
    p.add_argument("--preset",     metavar="NAME",         help="Simulation preset")
    p.add_argument("--fetch",      metavar="URL",          help="Fetch URL")
    p.add_argument("--temp",       type=float, default=0.3, help="Temperature")
    p.add_argument("--max-tokens", type=int,   default=500, help="Max tokens")
    args = p.parse_args()

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        h = get("/health")
        print(f"PhysLLM {h['version']} — {h['backend']} backend")
    except Exception as e:
        print(f"Server not reachable: {e}\nRun: cargo run --release -p api-server", file=sys.stderr)
        sys.exit(1)

    # ── Physical constant ─────────────────────────────────────────────────────
    if args.constant:
        c = get(f"/constants/{args.constant}")
        print(f"\n{c['name']} ({c['symbol']})")
        print(f"  Value:       {c['value']:.6e} {c['unit']}")
        print(f"  Uncertainty: ±{c['uncertainty']:.2e}")
        print(f"  Rel. uncert: {c['relative_uncertainty']:.2e}")
        return

    # ── Molecule lookup ───────────────────────────────────────────────────────
    if args.molecule:
        # First check formula parser
        try:
            r = post("/chemistry/mw", {"formula": args.molecule})
            print(f"\nMolecular formula: {r['formula']} ({r['hill']})")
            print(f"Molar mass: {r['molar_mass_g_per_mol']:.4f} g/mol")
            print(f"Elements: {r['elements']}")
        except Exception:
            pass
        # Then search for more data
        r = post("/search", {"query": f"molecular properties {args.molecule}"})
        print_sources(r)
        ctx = r.get('llm_context', '')
        if ctx:
            print(f"\n{ctx[:800]}")
        return

    # ── URL fetch ─────────────────────────────────────────────────────────────
    if args.fetch:
        r = post("/fetch", {"url": args.fetch})
        for res in r.get('results', [])[:1]:
            print(f"\nTitle: {res['title']}")
            text = (res.get('full_text') or res.get('snippet', ''))
            print(textwrap.fill(text[:2000], width=80))
        return

    # ── arXiv search ─────────────────────────────────────────────────────────
    if args.arxiv:
        payload = {"query": args.arxiv}
        if args.category:
            payload["category"] = args.category
        r = post("/search/arxiv", payload)
        print(f"\narXiv search: {args.arxiv}")
        print_sources(r)
        for res in r.get('results', [])[:3]:
            print(f"\n{'='*60}")
            print(f"Title: {res['title']}")
            meta = res.get('metadata', {})
            if meta.get('authors'): print(f"Authors: {meta['authors']}")
            if meta.get('year'):    print(f"Year: {meta['year']}")
            abstract = (res.get('full_text') or res.get('snippet', ''))
            print(f"\n{textwrap.fill(abstract[:600], width=76)}")
        return

    # ── Simulation ────────────────────────────────────────────────────────────
    if args.simulate:
        sim_map = {
            "nbody":    "n_body",
            "quantum":  "quantum_wavefunction",
            "md":       "molecular_dynamics",
            "kinetics": "reaction_kinetics",
            "stellar":  "stellar_evolution",
            "astrochem":"astrochem_network",
            "thermo":   "thermodynamics_eos",
        }
        sim_type = sim_map.get(args.simulate, args.simulate)
        params   = {}
        if args.preset:
            preset_map = {
                "solar_system":    "SolarSystem",
                "binary_star":     "BinaryStar",
                "three_body":      "ThreeBody",
                "galaxy_core":     "GalaxyCore",
            }
            params["preset"] = preset_map.get(args.preset, args.preset)
            params["dt"]            = 3600.0
            params["total_time"]    = 31557600.0
            params["softening"]     = 1e6
            params["record_every"]  = 24

        r = post("/simulate", {
            "sim_type":    sim_type,
            "description": f"Testing {args.simulate}",
            "params":      params,
            "max_steps":   1000,
            "output_fmt":  "summary",
        })
        print(f"\n{'='*60}")
        print(r.get('summary', r.get('llm_context', '')))
        print(f"\nSteps run: {r.get('steps_run', '?')}")
        print(f"Wall time: {r.get('wall_time_ms', '?')}ms")
        if r.get('plots'):
            print(f"Plots available: {[p['title'] for p in r['plots']]}")
        return

    # ── Main: search-augmented generation ────────────────────────────────────
    query = args.query
    if not query:
        p.print_help()
        return

    print(f"\nQuery: {query}")

    if args.no_search:
        r = post("/generate", {
            "prompt": query,
            "sampling": {"temperature": args.temp, "max_new_tokens": args.max_tokens},
        })
        print(f"\n{'='*60}")
        print(textwrap.fill(r['text'], width=80))
        print(f"\nTokens: {r['tokens_in']} in · {r['tokens_out']} out · {r['time_ms']}ms")
    else:
        r = post("/generate/search", {
            "prompt":       query,
            "search_query": query,
            "sampling":     {"temperature": args.temp, "max_new_tokens": args.max_tokens},
        })
        # Print search info first
        print_sources(r['search'])
        # Then the generated answer
        print(f"\n{'='*60}")
        print("Answer:")
        print(textwrap.fill(r['text'], width=80))
        print(f"\nTokens: {r['tokens_in']} in · {r['tokens_out']} out")
        # Show top result snippets
        if r['search'].get('results'):
            print(f"\n{'─'*60}")
            print("Top result context used:")
            for res in r['search']['results'][:2]:
                snippet = (res.get('full_text') or res.get('snippet', ''))[:250]
                print(f"\n[{res['source']}] {res['title'][:60]}")
                print(textwrap.fill(snippet, width=76, initial_indent="  ", subsequent_indent="  "))

if __name__ == "__main__":
    main()
