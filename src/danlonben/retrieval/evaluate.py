from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from loguru import logger

from danlonben.config import DOCUMENTS, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from danlonben.retrieval.retrievers import BaseRetriever

# Sector lookup built from the document registry.
_DOC_TO_SECTOR: dict[str, str] = {d["doc_id"]: d["sector"] for d in DOCUMENTS}


# ---------------------------------------------------------------------------
# Metrics  (all operate on a list of valid pages, not a single true page)
# ---------------------------------------------------------------------------

def _best_rank_of(valid_pages: list[int], retrieved: list[tuple[int, float]]) -> int | None:
    """Return the rank of the highest-ranked valid page, or None if none retrieved."""
    valid_set = set(valid_pages)
    for i, (page, _) in enumerate(retrieved, start=1):
        if page in valid_set:
            return i
    return None


def _recall_at_k(valid_pages: list[int], retrieved: list[tuple[int, float]], k: int) -> bool:
    top_k = {page for page, _ in retrieved[:k]}
    return bool(top_k & set(valid_pages))


def _ndcg_at_k(valid_pages: list[int], retrieved: list[tuple[int, float]], k: int) -> float:
    rank = _best_rank_of(valid_pages, retrieved)
    if rank is None or rank > k:
        return 0.0
    # ideal DCG = 1.0 (best possible: relevant page at rank 1)
    return 1.0 / math.log2(rank + 1)


def _mrr(valid_pages: list[int], retrieved: list[tuple[int, float]]) -> float:
    rank = _best_rank_of(valid_pages, retrieved)
    return 1.0 / rank if rank is not None else 0.0


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    retriever: BaseRetriever,
    retriever_name: str,
    output_dir: Path,
    doc_ids: list[str] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Run retrieval evaluation across all documents.

    Args:
        retriever:       Any BaseRetriever instance (BM25, BGE-M3, ColPali).
        retriever_name:  Used for logging and output folder name.
        output_dir:      Root results directory (e.g. data/results/).
        doc_ids:         Subset of doc IDs to evaluate. Defaults to all documents.
        k:               Cutoff for Recall@k and NDCG@k.

    Returns:
        Summary dict with overall, per-document, and per-sector metrics.

    Ground-truth source:
        Prefers questions_final.jsonl (with reviewed valid_pages lists).
        Falls back to questions.jsonl (single source_page) if not present.
    """
    results_dir = output_dir / retriever_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"
    summary_path = results_dir / "summary.json"

    if doc_ids is None:
        doc_ids = [doc["doc_id"] for doc in DOCUMENTS]

    all_rows: list[dict[str, Any]] = []
    per_doc_metrics: dict[str, dict[str, float]] = {}

    with open(results_path, "w", encoding="utf-8") as results_file:
        for doc_id in doc_ids:
            pages_path = INTERIM_DATA_DIR / doc_id / "pages.jsonl"

            # Prefer reviewed ground-truth; fall back to raw questions.
            final_path = PROCESSED_DATA_DIR / "qa" / doc_id / "questions_final.jsonl"
            legacy_path = PROCESSED_DATA_DIR / "qa" / doc_id / "questions.jsonl"
            if final_path.exists():
                questions_path = final_path
            elif legacy_path.exists():
                questions_path = legacy_path
                logger.warning(
                    f"[{doc_id}] questions_final.jsonl not found — "
                    "falling back to questions.jsonl (no false-negative correction)."
                )
            else:
                logger.warning(f"[{doc_id}] No questions file found — skipping.")
                continue

            if not pages_path.exists():
                logger.warning(f"[{doc_id}] Missing pages.jsonl — skipping.")
                continue

            pages: list[dict[str, Any]] = [
                json.loads(line) for line in pages_path.open(encoding="utf-8")
            ]
            questions: list[dict[str, Any]] = [
                json.loads(line) for line in questions_path.open(encoding="utf-8")
            ]

            logger.info(
                f"[{doc_id}] Indexing {len(pages)} pages for {retriever_name}..."
            )
            retriever.index(pages)

            doc_rows: list[dict[str, Any]] = []
            for question in questions:
                # valid_pages is present in questions_final.jsonl;
                # fall back to [source_page] for legacy questions.jsonl.
                valid_pages: list[int] = question.get(
                    "valid_pages", [question["source_page"]]
                )
                query = question["question_da"]

                retrieved = retriever.retrieve(query, k=k)

                rank = _best_rank_of(valid_pages, retrieved)
                row = {
                    "question_id": question["id"],
                    "doc_id": doc_id,
                    "sector": _DOC_TO_SECTOR.get(doc_id, "unknown"),
                    "question": query,
                    "valid_pages": valid_pages,
                    "retrieved_pages": [p for p, _ in retrieved],
                    "rank": rank,
                    "recall@1": _recall_at_k(valid_pages, retrieved, 1),
                    f"recall@{k}": _recall_at_k(valid_pages, retrieved, k),
                    f"ndcg@{k}": _ndcg_at_k(valid_pages, retrieved, k),
                    "mrr": _mrr(valid_pages, retrieved),
                }
                doc_rows.append(row)
                results_file.write(json.dumps(row, ensure_ascii=False) + "\n")

            doc_metrics = _aggregate(doc_rows, k)
            per_doc_metrics[doc_id] = doc_metrics
            all_rows.extend(doc_rows)
            logger.success(
                f"[{doc_id}] recall@1={doc_metrics['recall@1']:.3f}  "
                f"recall@{k}={doc_metrics[f'recall@{k}']:.3f}  "
                f"ndcg@{k}={doc_metrics[f'ndcg@{k}']:.3f}  "
                f"mrr={doc_metrics['mrr']:.3f}"
            )

    # --- Per-sector aggregation -------------------------------------------
    per_sector_metrics: dict[str, dict[str, float]] = {}
    sectors = sorted({_DOC_TO_SECTOR.get(d, "unknown") for d in per_doc_metrics})
    for sector in sectors:
        sector_rows = [r for r in all_rows if r["sector"] == sector]
        per_sector_metrics[sector] = _aggregate(sector_rows, k)

    overall = _aggregate(all_rows, k)
    summary = {
        "retriever": retriever_name,
        "overall": overall,
        "by_sector": per_sector_metrics,
        "by_document": per_doc_metrics,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if overall:
        logger.success(
            f"[overall] recall@1={overall['recall@1']:.3f}  "
            f"recall@{k}={overall[f'recall@{k}']:.3f}  "
            f"ndcg@{k}={overall[f'ndcg@{k}']:.3f}  "
            f"mrr={overall['mrr']:.3f}"
        )
        for sector, sm in per_sector_metrics.items():
            logger.info(
                f"  [{sector}] recall@1={sm['recall@1']:.3f}  "
                f"recall@{k}={sm[f'recall@{k}']:.3f}  "
                f"ndcg@{k}={sm[f'ndcg@{k}']:.3f}"
            )
    else:
        logger.warning("No documents evaluated — all skipped.")
    logger.info(f"Results written to {results_dir}")
    return summary


def _aggregate(rows: list[dict[str, Any]], k: int) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    return {
        "n_questions": n,
        "recall@1": sum(r["recall@1"] for r in rows) / n,
        f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / n,
        f"ndcg@{k}": sum(r[f"ndcg@{k}"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
    }
