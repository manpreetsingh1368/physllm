#!/usr/bin/env python3
import argparse, os, re, asyncio, aiohttp, time, hashlib
import pandas as pd
from tqdm.asyncio import tqdm
from chemspipy import ChemSpider
from sklearn.model_selection import train_test_split
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor


ARXIV_API = "http://export.arxiv.org/api/query"
PHYSICS_CATEGORIES = ["astro-ph","cond-mat","quant-ph","hep-th","hep-ph","hep-ex","gr-qc","nucl-th","nucl-ex"]
DOMAIN_TOKENS = [
    "∇","∂","∮","∇²","∂/∂t","d/dt",
    "eV","MeV","Å","fm","AU","ly","pc","M☉","L☉",
    "H₂","CO₂","NH₃","H₂O","H₃⁺","HCO⁺",
    "α","β","γ","δ","ε","ζ","η","θ","κ","λ","μ","ν","ξ","π","ρ","σ","τ","φ","χ","ψ","ω",
    "<|sim_start|>","<|tool_call|>","<|eq|>","<|/eq|>",
    "<|c_light|>","<|h_planck|>","<|k_boltzmann|>"
]

PDF_CACHE_DIR = "./pdf_cache"
os.makedirs(PDF_CACHE_DIR, exist_ok=True)


def tokenize(text):
    for token in DOMAIN_TOKENS:
        text = text.replace(token, f" {token} ")
    return " ".join(text.split())


def normalize(records):
    clean = []
    for r in records:
        text = r.get("text","")
        if not text: continue
        text = re.sub(r"\s+"," ",text)
        clean.append({
            "source": r["source"],
            "id": r.get("id"),
            "title": r.get("title"),
            "text": tokenize(text),
            "length": len(text)
        })
    return clean

def pdf_cache_path(url):
    h = hashlib.sha256(url.encode()).hexdigest()
    return os.path.join(PDF_CACHE_DIR, f"{h}.pdf")

def parse_pdf_bytes(content_bytes):
    try:
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        return text.strip()
    except Exception as e:
        print(f"[PDF] Parse failed → {e}")
        return None

# ---------------- Async PDF Fetch with Cache & Retry ----------------
async def fetch_pdf(session, url, executor, retries=3):
    cache_file = pdf_cache_path(url)
    if os.path.exists(cache_file):
        with open(cache_file,"rb") as f:
            content = f.read()
        return await asyncio.get_event_loop().run_in_executor(executor, parse_pdf_bytes, content)
    
    attempt = 0
    while attempt < retries:
        try:
            async with session.get(url, timeout=30) as r:
                if r.status == 200:
                    content = await r.read()
                    with open(cache_file,"wb") as f:
                        f.write(content)
                    return await asyncio.get_event_loop().run_in_executor(executor, parse_pdf_bytes, content)
                else:
                    print(f"[PDF] HTTP {r.status} for {url}")
        except Exception as e:
            print(f"[PDF] Fetch error {url} → {e}")
        attempt += 1
        await asyncio.sleep(2**attempt)
    print(f"[PDF] Failed after {retries} retries: {url}")
    return None


import feedparser
async def fetch_arxiv(max_results=100, concurrency=5):
    entries = []
    sem = asyncio.Semaphore(concurrency)
    executor = ThreadPoolExecutor(max_workers=8)

    async with aiohttp.ClientSession() as session:
        async def fetch_category(cat):
            start = 0
            batch_size = 50
            while start < max_results:
                url = f"{ARXIV_API}?search_query=cat:{cat}&start={start}&max_results={batch_size}"
                async with sem:
                    async with session.get(url) as resp:
                        feed = feedparser.parse(await resp.text())
                if not feed.entries: break
                for e in feed.entries:
                    entry = {
                        "source": "arxiv",
                        "id": e.get("id"),
                        "title": e.get("title"),
                        "text": e.get("summary"),
                        "authors": [a.name for a in e.authors],
                        "category": cat,
                        "pdf_url": next((l.href for l in e.links if l.type=="application/pdf"), None)
                    }
                    if entry["pdf_url"]:
                        entry["pdf_text_task"] = asyncio.create_task(fetch_pdf(session, entry["pdf_url"], executor))
                    else:
                        entry["pdf_text_task"] = asyncio.create_task(asyncio.sleep(0, result=""))
                    entries.append(entry)
                start += batch_size
                await asyncio.sleep(1)

        await asyncio.gather(*[fetch_category(cat) for cat in PHYSICS_CATEGORIES])

        for e in tqdm(entries, desc="PDFs"):
            pdf_text = await e["pdf_text_task"]
            if pdf_text:
                e["text"] += "\n\n" + pdf_text
            del e["pdf_text_task"]

    executor.shutdown(wait=True)
    return entries


async def fetch_chemspider(cs_api_key, query_list=None):
    cs = ChemSpider(cs_api_key)
    results = []
    if not query_list: query_list = ["H2O","CO2","C6H12O6"]
    for mol in query_list:
        try:
            compounds = cs.search(mol)
            for c in compounds:
                results.append({
                    "source":"chemspider",
                    "id":str(c.csid),
                    "title":c.common_name or mol,
                    "text":c.molecular_formula,
                    "mass":c.molecular_weight
                })
        except Exception as e:
            print(f"[ChemSpider] Failed {mol} → {e}")
    return results


async def fetch_nist_stub():
    print("[NIST] Stub — implement scraper/API later")
    return []

=
def save_dataset(data, output_dir, fmt, split=True):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data)
    if split:
        train, val = train_test_split(df, test_size=0.1, random_state=42)
        train_path = os.path.join(output_dir, f"train.{fmt}")
        val_path = os.path.join(output_dir, f"val.{fmt}")
        if fmt=="parquet":
            train.to_parquet(train_path)
            val.to_parquet(val_path)
        elif fmt=="jsonl":
            train.to_json(train_path, orient="records", lines=True)
            val.to_json(val_path, orient="records", lines=True)
        print(f"[✓] Saved train → {train_path} ({len(train)})")
        print(f"[✓] Saved val   → {val_path} ({len(val)})")
    else:
        path = os.path.join(output_dir, f"dataset.{fmt}")
        if fmt=="parquet":
            df.to_parquet(path)
        elif fmt=="jsonl":
            df.to_json(path, orient="records", lines=True)
        print(f"[✓] Saved dataset → {path} ({len(df)})")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="parquet", choices=["parquet","jsonl"])
    parser.add_argument("--max_arxiv", type=int, default=100)
    parser.add_argument("--chemspider_api", type=str, default="")
    parser.add_argument("--chem_queries", type=str, default="")
    args = parser.parse_args()

    sources = args.sources.split(",")
    all_data = []

    if "arxiv_physics" in sources:
        all_data.extend(await fetch_arxiv(max_results=args.max_arxiv))

    if "nist_webbook" in sources:
        all_data.extend(await fetch_nist_stub())

    if "chemspider" in sources:
        queries = args.chem_queries.split(",") if args.chem_queries else None
        all_data.extend(await fetch_chemspider(args.chemspider_api, queries))

    clean = normalize(all_data)
    save_dataset(clean, args.output, args.format, split=True)

if __name__=="__main__":
    asyncio.run(main())