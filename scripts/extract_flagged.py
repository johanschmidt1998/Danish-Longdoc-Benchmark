"""Extract all flagged questions with their additional page content.

Produces data/processed/qa/flagged_review_data.json for Claude to review.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INTERIM = BASE / "data" / "interim"
QA_DIR = BASE / "data" / "processed" / "qa"

DOC_IDS = [
    "finance_nationalbank_2024",
    "finance_statens_laantagning_2023",
    "healthcare_sundhedsstyrelsen_2023",
    "healthcare_sundhedsprofil_2023",
    "legal_rigsrevisionen_2023",
    "energy_energistatistik_2023",
    "energy_forsyningspolitisk_2024",
    "municipal_kbh_2023",
]

MAX_CHARS = 1200  # per page excerpt for review

results = []

for doc_id in DOC_IDS:
    checked_path = QA_DIR / doc_id / "questions_checked.jsonl"
    pages_path = INTERIM / doc_id / "pages.jsonl"

    if not checked_path.exists():
        print(f"[SKIP] {doc_id} — no questions_checked.jsonl")
        continue

    # Load pages
    pages: dict[int, str] = {}
    if pages_path.exists():
        for line in pages_path.open(encoding="utf-8"):
            p = json.loads(line)
            pages[p["page_num"]] = p.get("text", "")

    flagged_count = 0
    for line in checked_path.open(encoding="utf-8"):
        q = json.loads(line)
        if not q.get("flagged"):
            continue

        flagged_count += 1
        source_page = q["source_page"]
        additional = [p for p in q["valid_pages"] if p != source_page]

        page_excerpts = {}
        for pn in additional:
            text = pages.get(pn, "")
            page_excerpts[pn] = text[:MAX_CHARS] + ("…" if len(text) > MAX_CHARS else "")

        results.append({
            "doc_id": doc_id,
            "id": q["id"],
            "question_da": q["question_da"],
            "answer_da": q["answer_da"],
            "source_page": source_page,
            "additional_pages": additional,
            "page_excerpts": page_excerpts,
        })

    print(f"[{doc_id}] {flagged_count} flagged questions")

out_path = QA_DIR / "flagged_review_data.json"
out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nWrote {len(results)} flagged questions to {out_path}")
