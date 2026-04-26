#!/usr/bin/env bash
# scripts/example_search.sh — Web search API examples for PhysLLM
# Start the server first: cargo run --release -p api-server

BASE="http://localhost:8080/v1"


echo " PhysLLM Web Search API Examples"


#  1. Auto-search routing 
echo -e "\n[1] Physics research search (auto-routes to arXiv + Semantic Scholar):"
curl -s -X POST "$BASE/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "gravitational wave detection LIGO sensitivity 2024"}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Intent: {d[\"intent\"]}')
print(f'Sources: {d[\"sources_used\"]}')
print(f'Results: {d[\"total_found\"]}  ({d[\"elapsed_ms\"]}ms)')
for r in d['results'][:2]:
    print(f'  [{r[\"source\"]}] {r[\"title\"][:70]}')
    print(f'   {r[\"url\"]}')
"

#  2. Direct arXiv paper lookup 
echo -e "\n[2] Direct arXiv paper by ID:"
curl -s -X POST "$BASE/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "2401.04088"}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Intent: {d[\"intent\"]}')
for r in d['results'][:1]:
    print(f'Title: {r[\"title\"]}')
    authors = r.get('metadata', {}).get('authors', 'N/A')
    year    = r.get('metadata', {}).get('year', 'N/A')
    print(f'Authors: {authors}  Year: {year}')
    print(f'URL: {r[\"url\"]}')
    print(f'Abstract: {r[\"snippet\"][:200]}...')
"

# 3. arXiv search with category filter 
echo -e "\n[3] arXiv search — astrochemistry category:"
curl -s -X POST "$BASE/search/arxiv" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "interstellar medium molecular cloud chemistry",
    "category": "astro-ph.GA",
    "max_results": 3
  }' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Sources: {d[\"sources_used\"]}  Results: {d[\"total_found\"]}')
for i, r in enumerate(d['results'][:3], 1):
    print(f'  {i}. {r[\"title\"][:65]}')
    print(f'     {r[\"metadata\"].get(\"authors\",\"\")}  ({r[\"metadata\"].get(\"year\",\"\")})')
"

#  4. Chemistry data lookup 
echo -e "\n[4] Chemistry search (auto-routes to PubChem + NIST):"
curl -s -X POST "$BASE/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "molecular weight boiling point ethanol"}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Intent: {d[\"intent\"]}')
print(f'Sources: {d[\"sources_used\"]}')
for r in d['results'][:2]:
    print(f'  [{r[\"source\"]}] {r[\"title\"]}')
    print(f'  {r[\"snippet\"][:150]}')
"

#  5. Astrophysics object lookup 
echo -e "\n[5] Astrophysics object (auto-routes to NASA ADS + arXiv):"
curl -s -X POST "$BASE/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Sagittarius A* black hole mass measurement"}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Intent: {d[\"intent\"]}')
print(f'Sources: {d[\"sources_used\"]}')
for r in d['results'][:3]:
    print(f'  [{r[\"source\"]}] {r[\"title\"][:60]}')
    print(f'   {r[\"url\"]}')
"

#  6. Full page fetch 
echo -e "\n[6] Fetch full arXiv paper abstract:"
curl -s -X POST "$BASE/fetch" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/abs/2301.04309", "max_chars": 2000}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
for r in d['results'][:1]:
    print(f'Title: {r[\"title\"]}')
    text = (r.get('full_text') or r['snippet'])[:500]
    print(f'Content:\n{text}...')
"

#  7. Search-augmented generation 
echo -e "\n[7] Generate with automatic web search (RAG):"
curl -s -X POST "$BASE/generate/search" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the latest measurements of the Hubble constant and what is the Hubble tension?",
    "search_query": "Hubble constant tension 2024 measurement",
    "sampling": {"temperature": 0.3, "max_new_tokens": 500}
  }' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('=== Generation ===')
print(d['text'][:300])
print('\n=== Search used ===')
s = d['search']
print(f'Sources: {s[\"sources_used\"]}')
print(f'Results: {s[\"total_found\"]}  ({s[\"elapsed_ms\"]}ms)')
"

#  8. Physical constant via search 
echo -e "\n[8] NIST constant search:"
curl -s -X POST "$BASE/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "NIST Boltzmann constant CODATA 2022"}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Intent: {d[\"intent\"]}')
for r in d['results'][:1]:
    print(f'  {r[\"title\"]}')
    print(f'  {r[\"snippet\"]}')
"

echo -e ""
echo " All examples complete."
echo " Set BRAVE_API_KEY for general web search."
echo " Set NASA_ADS_KEY  for astrophysics object queries."
echo ""
