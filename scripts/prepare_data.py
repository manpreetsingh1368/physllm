#!/usr/bin/env python3
import argparse, os, time, requests, feedparser
import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF
from chemspipy import ChemSpider

ARXIV_API = "http://export.arxiv.org/api/query"
PHYSICS_CATEGORIES = [
    "astro-ph", "cond-mat", "quant-ph", "hep-th",
    "hep-ph", "hep-ex", "gr-qc", "nucl-th", "nucl-ex"
]

def fetch_arxiv(max_results=100):
    entries = []
    for cat in PHYSICS_CATEGORIES:
        print(f"[arXiv] Fetching {cat}...")
        start = 0
        batch_size = 50
        while start < max_results:
            url = f"{ARXIV_API}?search_query=cat:{cat}&start={start}&max_results={batch_size}"
            feed = feedparser.parse(requests.get(url).text)
            if not feed.entries:
                break
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
                # fetch PDF full text
                if entry["pdf_url"]:
                    pdf_text = parse_pdf(entry["pdf_url"])
                    if pdf_text:
                        entry["text"] += "\n\n" + pdf_text
                entries.append(entry)
            start += batch_size
            time.sleep(1)
    return entries

def parse_pdf(url):
    try:
        r = requests.get(url)
        tmp_path = "/tmp/temp.pdf"
        with open(tmp_path, "wb") as f: f.write(r.content)
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"[PDF] Failed {url} → {e}")
        return None


def fetch_chemspider(cs_api_key, query_list=None):
    cs = ChemSpider(cs_api_key)
    results = []
    if not query_list:
        query_list = ["H2O", "CO2", "C6H12O6"]  # default molecules
    for mol in query_list:
        try:
            compounds = cs.search(mol)
            for c in compounds:
                results.append({
                    "source": "chemspider",
                    "id": str(c.csid),
                    "title": c.common_name or mol,
                    "text": c.molecular_formula,
                    "mass": c.molecular_weight
                })
        except Exception as e:
            print(f"[ChemSpider] Failed {mol} → {e}")
    return results


def fetch_nist_stub():
    print("[NIST] Stub — implement scraper/API later")
    return []


def normalize(records):
    clean = []
    for r in records:
        text = r.get("text", "")
        if not text: continue
        clean.append({
            "source": r["source"],
            "id": r.get("id"),
            "title": r.get("title"),
            "text": text.strip(),
            "length": len(text)
        })
    return clean


def save_dataset(data, output_dir, fmt):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data)
    path = os.path.join(output_dir, f"dataset.{fmt}")
    if fmt == "parquet":
        df.to_parquet(path)
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True)
    print(f"[✓] Saved dataset → {path} ({len(df)} samples)")


def main():
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
        all_data.extend(fetch_arxiv(max_results=args.max_arxiv))

    if "nist_webbook" in sources:
        all_data.extend(fetch_nist_stub())

    if "chemspider" in sources:
        queries = args.chem_queries.split(",") if args.chem_queries else None
        all_data.extend(fetch_chemspider(args.chemspider_api, queries))

    clean = normalize(all_data)
    save_dataset(clean, args.output, args.format)

if __name__ == "__main__":
    main()