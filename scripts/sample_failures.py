import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

base = Path(__file__).resolve().parents[1]

paths = {
    "bm25":     base / "data/results/bm25/results.jsonl",
    "bge-m3":   base / "data/results/results/bge-m3/results.jsonl",
    "colqwen2": base / "data/results/results/colqwen2/results.jsonl",
    "colpali":  base / "data/results/results/colpali/results.jsonl",
}

by_model = {}
for model, path in paths.items():
    by_model[model] = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        by_model[model][r["question_id"]] = r

# Load page texts
pages = {}
for doc_dir in (base / "data/interim").iterdir():
    p = doc_dir / "pages.jsonl"
    if p.exists():
        for line in p.open(encoding="utf-8"):
            pg = json.loads(line)
            pages[(doc_dir.name, pg["page_num"])] = pg.get("text", "")

all_ids = set(by_model["bm25"].keys())
failures = []
for qid in all_ids:
    rows = {m: by_model[m][qid] for m in by_model}
    ranks = {m: rows[m]["rank"] for m in rows}
    if all(r != 1 for r in ranks.values()):
        failures.append((qid, rows, ranks))

print(f"Questions where all models fail R@1: {len(failures)}")

random.seed(42)
sample = random.sample(failures, min(5, len(failures)))

for qid, rows, ranks in sample:
    r = rows["bm25"]
    doc_id = r["doc_id"]
    valid_pages = r["valid_pages"]
    print(f"\n{'='*60}")
    print(f"{qid}  [{doc_id}]")
    print(f"Question : {r['question']}")
    print(f"Valid pages: {valid_pages}")
    print(f"Ranks    — BM25:{ranks['bm25']} | BGE-M3:{ranks['bge-m3']} | ColQwen2:{ranks['colqwen2']} | ColPali:{ranks['colpali']}")
    print(f"BM25 top3    : {r['retrieved_pages'][:3]}")
    print(f"BGE-M3 top3  : {rows['bge-m3']['retrieved_pages'][:3]}")
    print(f"ColQwen2 top3: {rows['colqwen2']['retrieved_pages'][:3]}")

    for vp in valid_pages:
        text = pages.get((doc_id, vp), "[no text]")
        print(f"\n  -- Gold page {vp} --")
        print(text[:500])

    top_page = r["retrieved_pages"][0]
    top_text = pages.get((doc_id, top_page), "[no text]")
    print(f"\n  -- BM25 rank-1 page {top_page} (wrong) --")
    print(top_text[:500])
