"""False negative detection for the retrieval benchmark.

For each question, checks every page in the document (excluding the gold
source_page) to determine whether that page can also answer the question.
Pages that pass are added to `valid_pages`. Questions with any additional
valid pages are marked `flagged = True`.

Output is written to questions_checked.jsonl alongside the original
questions.jsonl. The original file is never modified.

Usage:
    python -m danlonben.qa.false_negative_check
    python -m danlonben.qa.false_negative_check --doc-ids energy_energistatistik_2023
    python -m danlonben.qa.false_negative_check --force   # re-run even if output exists
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import random

import openai
from loguru import logger

from danlonben.config import DOCUMENTS, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "Du er en streng relevansdommer for et informationssøgnings-benchmark. "
    "Svar kun med det enkelte ord Yes eller No."
)

_TEMPLATE = """\
Din opgave: afgør om siden nedenfor indeholder tilstrækkelig information til at \
besvare spørgsmålet alene.

Vær meget streng:
- Svar Yes KUN hvis siden indeholder den specifikke kendsgerning, det tal eller \
den udtalelse, der direkte og fuldstændigt besvarer spørgsmålet. Det forventede \
svar skal være tydeligt tilstede eller direkte afledt af denne side alene.
- Svar No hvis siden blot diskuterer et relateret emne, indeholder lignende men \
ikke identisk information, eller kun delvist adresserer spørgsmålet.
- Svar No hvis der kræves slutninger, kombination med andre sider eller \
baggrundsviden for at nå frem til svaret.
- Svar No i tvivlstilfælde.

Spørgsmål: {question}
Forventet svar: {answer}

--- SIDEINDHOLD BEGYNDER ---
{page_text}
--- SIDEINDHOLD SLUTTER ---

Indeholder denne side alene tilstrækkelig information til at besvare spørgsmålet? \
Svar kun Yes eller No."""

_MAX_PAGE_CHARS = 6000  # roughly 1500 tokens — keeps cost low per call


# ---------------------------------------------------------------------------
# Single judge call
# ---------------------------------------------------------------------------

async def _judge_page(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    question: str,
    answer: str,
    page_text: str,
) -> bool:
    prompt = _TEMPLATE.format(
        question=question,
        answer=answer,
        page_text=page_text[:_MAX_PAGE_CHARS],
    )
    for attempt in range(8):
        try:
            async with sem:
                response = await client.chat.completions.create(
                    model=model,
                    max_tokens=4,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                )
            return response.choices[0].message.content.strip().lower().startswith("yes")
        except openai.RateLimitError:
            wait = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait)
    raise RuntimeError("Max retries exceeded for judge call")


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

async def _check_document(
    doc_id: str,
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    force: bool,
) -> None:
    pages_path = INTERIM_DATA_DIR / doc_id / "pages.jsonl"
    questions_path = PROCESSED_DATA_DIR / "qa" / doc_id / "questions.jsonl"
    output_path = PROCESSED_DATA_DIR / "qa" / doc_id / "questions_checked.jsonl"

    if output_path.exists() and not force:
        logger.info(f"[{doc_id}] Already checked — skipping. Use --force to re-run.")
        return

    if not pages_path.exists():
        logger.warning(f"[{doc_id}] Missing pages.jsonl — skipping.")
        return
    if not questions_path.exists():
        logger.warning(f"[{doc_id}] Missing questions.jsonl — skipping.")
        return

    pages: dict[int, str] = {}
    for line in pages_path.open(encoding="utf-8"):
        p = json.loads(line)
        pages[p["page_num"]] = p.get("text", "")

    questions: list[dict] = [
        json.loads(line) for line in questions_path.open(encoding="utf-8")
    ]

    logger.info(
        f"[{doc_id}] Checking {len(questions)} questions × "
        f"{len(pages)} pages = {len(questions) * (len(pages) - 1)} calls"
    )

    flagged_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for q in questions:
            source_page: int = q["source_page"]
            question_text: str = q["question_da"]
            answer_text: str = q["answer_da"]

            # Check every page except the known-good source page
            candidate_pages = [pn for pn in pages if pn != source_page]

            tasks = [
                _judge_page(
                    client, sem, model,
                    question_text, answer_text,
                    pages[pn],
                )
                for pn in candidate_pages
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            additional_valid: list[int] = []
            for page_num, result in zip(candidate_pages, results):
                if isinstance(result, Exception):
                    logger.error(
                        f"[{doc_id}] q={q['id']} page={page_num} — {result}"
                    )
                    continue
                if result:
                    additional_valid.append(page_num)

            flagged = len(additional_valid) > 0
            if flagged:
                flagged_count += 1
                logger.warning(
                    f"[{doc_id}] FLAGGED q={q['id']} "
                    f"— also answerable on pages {sorted(additional_valid)}"
                )

            row = {
                **q,
                "valid_pages": [source_page] + sorted(additional_valid),
                "flagged": flagged,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    logger.success(
        f"[{doc_id}] Done — {flagged_count}/{len(questions)} questions flagged. "
        f"Written to {output_path}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main(
    doc_ids: list[str],
    model: str,
    concurrency: int,
    force: bool,
) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = openai.AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    for doc_id in doc_ids:
        await _check_document(doc_id, client, sem, model, force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag questions where pages other than source_page can also answer."
    )
    parser.add_argument(
        "--doc-ids",
        nargs="+",
        default=None,
        help="Document IDs to check. Defaults to all documents in config.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use for judging.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of simultaneous API calls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if questions_checked.jsonl already exists.",
    )
    args = parser.parse_args()

    doc_ids = args.doc_ids or [d["doc_id"] for d in DOCUMENTS]
    asyncio.run(_main(doc_ids, args.model, args.concurrency, args.force))


if __name__ == "__main__":
    main()
