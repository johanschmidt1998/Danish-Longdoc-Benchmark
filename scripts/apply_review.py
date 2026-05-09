"""Apply Claude's review verdicts and produce questions_final.jsonl.

Reads per-document review.md files (KEEP / REMOVE judgments per additional page),
updates valid_pages in questions_checked.jsonl accordingly, and writes
questions_final.jsonl as the definitive ground-truth file used by the benchmark.

Logic:
  - Unflagged questions  → copied unchanged (valid_pages = [source_page])
  - Flagged, page KEEP   → page retained in valid_pages
  - Flagged, page REMOVE → page dropped from valid_pages
  - Flagged, no verdict  → conservative fallback: page retained + warning printed

Usage:
    python scripts/apply_review.py
    python scripts/apply_review.py --doc-ids finance_nationalbank_2024
    python scripts/apply_review.py --dry-run        # stats only, no files written
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
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


# ---------------------------------------------------------------------------
# Review parser
# ---------------------------------------------------------------------------

def _parse_review(text: str) -> dict[str, dict[int, str]]:
    """Parse review.md → {question_id: {page_num: 'KEEP' | 'REMOVE'}}.

    Handles two formats produced by the reviewer:

    Single additional page
    ----------------------
    **Additional page:** 7
    **Verdict: KEEP**

    Multiple additional pages
    -------------------------
    **Additional pages:** 5, 25, 50, 62
    **Verdicts:**
    - Page 5: KEEP — reason...
    - Page 25: KEEP — reason...
    - Page 62: REMOVE — reason...
    """
    results: dict[str, dict[int, str]] = {}

    # Split on question headers, keeping the ID in each block.
    blocks = re.split(r"\n## Q ", "\n" + text)

    for block in blocks[1:]:
        lines = block.splitlines()
        if not lines:
            continue
        q_id = lines[0].strip()
        page_verdicts: dict[int, str] = {}

        # --- Pattern A: multi-page list "- Page N: KEEP/REMOVE — ..."
        for m in re.finditer(
            r"[-•]\s*Page\s+(\d+)\s*:\s*\**(KEEP|REMOVE)\**",
            block, re.IGNORECASE,
        ):
            page_verdicts[int(m.group(1))] = m.group(2).upper()

        # --- Pattern B: single verdict "**Verdict: KEEP/REMOVE**"
        #     Only used when Pattern A found nothing (single-page questions).
        if not page_verdicts:
            vm = re.search(r"\*\*Verdict:\s*(KEEP|REMOVE)\*\*", block, re.IGNORECASE)
            if vm:
                ap = re.search(r"\*\*Additional pages?:\*\*\s*(\d+)", block)
                if ap:
                    page_verdicts[int(ap.group(1))] = vm.group(1).upper()

        results[q_id] = page_verdicts

    return results


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

def _apply(doc_id: str, dry_run: bool) -> dict:
    checked_path = QA_DIR / doc_id / "questions_checked.jsonl"
    review_path  = QA_DIR / doc_id / "review.md"
    output_path  = QA_DIR / doc_id / "questions_final.jsonl"

    if not checked_path.exists():
        print(f"[{doc_id}] SKIP — missing questions_checked.jsonl")
        return {}
    if not review_path.exists():
        print(f"[{doc_id}] SKIP — missing review.md")
        return {}

    verdicts = _parse_review(review_path.read_text(encoding="utf-8"))
    questions = [json.loads(l) for l in checked_path.open(encoding="utf-8")]

    kept_pages = removed_pages = warnings = 0
    output_rows = []

    for q in questions:
        if not q.get("flagged"):
            output_rows.append(q)
            continue

        q_id        = q["id"]
        source_page = q["source_page"]
        additional  = [p for p in q["valid_pages"] if p != source_page]
        pv          = verdicts.get(q_id, {})

        confirmed: list[int] = []
        for pn in additional:
            verdict = pv.get(pn)
            if verdict == "KEEP":
                confirmed.append(pn)
                kept_pages += 1
            elif verdict == "REMOVE":
                removed_pages += 1
            else:
                # No verdict found — conservative: retain page, emit warning.
                confirmed.append(pn)
                warnings += 1
                print(f"  [WARN] {q_id} page {pn} — no verdict in review, retaining")

        new_valid = [source_page] + sorted(confirmed)
        output_rows.append({**q, "valid_pages": new_valid, "flagged": len(confirmed) > 0})

    multi  = sum(1 for r in output_rows if len(r["valid_pages"]) > 1)
    single = len(output_rows) - multi

    print(
        f"[{doc_id}] {len(output_rows)} questions | "
        f"{single} single-page  {multi} multi-page | "
        f"+{kept_pages} pages kept  -{removed_pages} removed"
        + (f"  {warnings} warnings" if warnings else "")
    )

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  -> {output_path}")

    return {"total": len(output_rows), "single": single, "multi": multi}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply review verdicts and write questions_final.jsonl."
    )
    parser.add_argument(
        "--doc-ids", nargs="+", default=None,
        help="Document IDs to process. Defaults to all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats but do not write output files.",
    )
    args = parser.parse_args()

    doc_ids = args.doc_ids or DOC_IDS
    totals = {"total": 0, "single": 0, "multi": 0}

    for doc_id in doc_ids:
        result = _apply(doc_id, dry_run=args.dry_run)
        for k in totals:
            totals[k] += result.get(k, 0)

    print(
        f"\nTotal: {totals['total']} questions | "
        f"{totals['single']} single-page  {totals['multi']} multi-page"
    )
    if args.dry_run:
        print("(dry-run — no files written)")


if __name__ == "__main__":
    main()
