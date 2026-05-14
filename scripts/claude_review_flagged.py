"""Automated false-negative review using the Anthropic Claude API.

For each flagged question in questions_checked.jsonl, calls Claude to judge
whether each additional candidate page genuinely answers the question.
Writes review.md files per document in the format expected by apply_review.py.

This script automates the review step that was performed interactively during
the original DanRAG-Bench construction. Run apply_review.py afterwards to
produce questions_final.jsonl.

Usage:
    python scripts/claude_review_flagged.py
    python scripts/claude_review_flagged.py --doc-ids energy_energistatistik_2023
    python scripts/claude_review_flagged.py --model claude-haiku-3-5 --concurrency 10
    python scripts/claude_review_flagged.py --force   # overwrite existing review.md files

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=your-key
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths and document registry
# ---------------------------------------------------------------------------

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

MAX_PAGE_CHARS = 6000  # ~1 500 tokens per page — consistent with false_negative_check.py

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a strict relevance judge for an information retrieval benchmark. "
    "Decide whether a document page genuinely answers a given question. "
    "Reply with exactly one line: KEEP — <reason>  or  REMOVE — <reason>."
)

_TEMPLATE = """\
Decide whether the page below contains sufficient information to answer the \
question by itself.

Rules:
- Reply KEEP only if the page contains the specific fact, figure, or statement \
that directly and completely answers the question. The expected answer must be \
clearly present or directly derivable from this page alone.
- Reply REMOVE if the page merely discusses a related topic, contains similar \
but not identical information, or only partially addresses the question.
- Reply REMOVE if any inference, combination with other pages, or background \
knowledge is needed to arrive at the answer.
- When in doubt, reply REMOVE.

Question: {question}
Expected answer: {answer}

--- PAGE CONTENT BEGIN ---
{page_text}
--- PAGE CONTENT END ---

Reply with exactly one line: KEEP — <reason>  or  REMOVE — <reason>."""

# ---------------------------------------------------------------------------
# Single judge call
# ---------------------------------------------------------------------------


async def _judge_page(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    model: str,
    question: str,
    answer: str,
    page_text: str,
) -> tuple[str, str]:
    """Returns (verdict, reason) where verdict is 'KEEP' or 'REMOVE'."""
    prompt = _TEMPLATE.format(
        question=question,
        answer=answer,
        page_text=page_text[:MAX_PAGE_CHARS],
    )
    for attempt in range(8):
        try:
            async with sem:
                response = await client.messages.create(
                    model=model,
                    max_tokens=80,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
            raw = response.content[0].text.strip()
            verdict = "KEEP" if raw.upper().startswith("KEEP") else "REMOVE"
            # Extract reason after the em-dash or regular dash
            if "—" in raw:
                reason = raw.split("—", 1)[1].strip()
            elif "-" in raw:
                reason = raw.split("-", 1)[1].strip()
            else:
                reason = raw
            return verdict, reason
        except anthropic.RateLimitError:
            wait = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait)
    raise RuntimeError("Max retries exceeded for judge call.")


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------


async def _review_document(
    doc_id: str,
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    model: str,
    force: bool,
) -> None:
    checked_path = QA_DIR / doc_id / "questions_checked.jsonl"
    pages_path = INTERIM / doc_id / "pages.jsonl"
    review_path = QA_DIR / doc_id / "review.md"

    if review_path.exists() and not force:
        print(f"[{doc_id}] review.md already exists — skipping. Use --force to re-run.")
        return
    if not checked_path.exists():
        print(f"[{doc_id}] SKIP — missing questions_checked.jsonl")
        return
    if not pages_path.exists():
        print(f"[{doc_id}] SKIP — missing pages.jsonl")
        return

    pages: dict[int, str] = {}
    for line in pages_path.open(encoding="utf-8"):
        p = json.loads(line)
        pages[p["page_num"]] = p.get("text", "")

    questions = [json.loads(line) for line in checked_path.open(encoding="utf-8")]
    flagged = [q for q in questions if q.get("flagged")]

    if not flagged:
        print(f"[{doc_id}] No flagged questions — nothing to review.")
        return

    print(f"[{doc_id}] Reviewing {len(flagged)} flagged questions...")

    blocks: list[str] = []

    for q in flagged:
        source_page: int = q["source_page"]
        additional = [pn for pn in q["valid_pages"] if pn != source_page]

        tasks = [
            _judge_page(
                client, sem, model,
                q["question_da"], q["answer_da"],
                pages.get(pn, ""),
            )
            for pn in additional
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build a block in Pattern A format (compatible with apply_review.py)
        page_list = ", ".join(str(pn) for pn in additional)
        label = "page" if len(additional) == 1 else "pages"
        lines = [
            f"## Q {q['id']}",
            f"**Additional {label}:** {page_list}",
            "**Verdicts:**",
        ]
        for pn, result in zip(additional, results):
            if isinstance(result, Exception):
                print(f"  [ERROR] {q['id']} page {pn} — {result}")
                verdict, reason = "KEEP", "Judge call failed — retained conservatively."
            else:
                verdict, reason = result
            lines.append(f"- Page {pn}: {verdict} — {reason}")

        blocks.append("\n".join(lines))

    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"[{doc_id}] {len(blocks)} question blocks written to {review_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _main(doc_ids: list[str], model: str, concurrency: int, force: bool) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    for doc_id in doc_ids:
        await _review_document(doc_id, client, sem, model, force)

    print("\nDone. Run  python scripts/apply_review.py  to produce questions_final.jsonl.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-review flagged false negatives using the Claude API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--doc-ids", nargs="+", default=None,
        help="Document IDs to process. Defaults to all.",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Anthropic model to use for judging.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Maximum simultaneous API calls.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing review.md files.",
    )
    args = parser.parse_args()

    doc_ids = args.doc_ids or DOC_IDS
    asyncio.run(_main(doc_ids, args.model, args.concurrency, args.force))


if __name__ == "__main__":
    main()
