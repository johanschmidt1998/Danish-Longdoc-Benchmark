# DanRAG-Bench

The first Danish multimodal document retrieval benchmark, spanning five sectors (energy, finance, health, legal, and municipalities) across 349 pages and 471 verified queries.

## Overview

DanRAG-Bench evaluates page-level retrieval over publicly available Danish PDF documents containing mixed formats — text, tables, and diagrams. Four retrieval systems are benchmarked representing sparse, dense, and visual retrieval paradigms.

| Model | Paradigm | NDCG@5 | Recall@5 |
|---|---|---|---|
| BGE-M3 | Dense | 0.836 | 0.924 |
| ColQwen2 | Visual | 0.832 | 0.945 |
| BM25 | Sparse | 0.774 | 0.867 |
| ColPali | Visual | 0.094 | 0.144 |

## Corpus

| Sector | Documents | Pages |
|---|---|---|
| Energy | 2 | 79 |
| Finance | 2 | 122 |
| Health | 2 | 74 |
| Legal | 1 | 33 |
| Municipality | 1 | 41 |
| **Total** | **8** | **349** |

All documents are publicly available Danish government and institutional reports.

## Pipeline

Query generation follows a four-stage automated pipeline:

1. **Ingestion** — PDFs are split into pages; text and 300 DPI images are extracted per page
2. **Generation** — GPT-4o generates two question-answer pairs per page (pages with fewer than 80 characters are skipped)
3. **Rewriting** — A second GPT-4o instance rephrases each question into natural Danish
4. **Critique** — A third GPT-4o instance verifies answerability and filters structural references
5. **Heuristic false negative check** — Each query-answer pair is cross-referenced against all other pages using token overlap (≥6 tokens, ≥55% ratio); conflicts are discarded

After the pipeline, all 482 pairs were manually verified: 11 deleted and 74 modified, yielding 471 final queries.

A second false negative verification stage uses GPT-4o to cross-reference every query against all pages in its source document, with Claude Sonnet 4.6 acting as judge. Of 471 queries, 134 were flagged and 84 confirmed as genuine false negatives. Additional valid pages are promoted to positive labels rather than discarding queries.

## Project Structure

```
├── data/
│   ├── interim/          <- Per-document pages.jsonl and page images
│   ├── processed/
│   │   └── qa/           <- questions.jsonl, questions_checked.jsonl, questions_final.jsonl
│   └── results/          <- Evaluation results per retrieval model
├── scripts/
│   ├── apply_review.py   <- Applies manual KEEP/REMOVE verdicts to produce questions_final.jsonl
│   ├── extract_flagged.py <- Extracts flagged questions for manual review
│   └── sample_failures.py <- Samples queries all models failed on for error analysis
├── slurm/                <- SLURM job scripts for LUMI HPC evaluation
└── src/danlonben/
    ├── config.py         <- Document registry and path configuration
    ├── ingestion.py      <- PDF ingestion (PyMuPDF, 300 DPI images)
    ├── pipeline/
    │   ├── graph.py      <- LangGraph pipeline definition and CLI
    │   ├── nodes.py      <- Pipeline node implementations
    │   └── state.py      <- Pipeline state types
    ├── qa/
    │   └── false_negative_check.py  <- GPT-4o false negative detection
    └── retrieval/
        ├── retrievers.py <- BM25, BGE-M3, ColPali, ColQwen2 implementations
        ├── run_eval.py   <- Evaluation entry point
        └── evaluate.py  <- Metrics (NDCG@5, Recall@5, MRR) and aggregation
```

## Setup

```bash
pip install -e .
export OPENAI_API_KEY=your_key_here
```

## Running the Pipeline

Generate questions for all documents:

```bash
python -m danlonben.pipeline.graph run-all
```

Generate for a single document:

```bash
python -m danlonben.pipeline.graph run finance_nationalbank_2024
```

Run false negative detection:

```bash
python -m danlonben.qa.false_negative_check
```

Apply manual review verdicts:

```bash
python scripts/apply_review.py
```

## Running Evaluation

```bash
python -m danlonben.retrieval.run_eval --retriever bm25
python -m danlonben.retrieval.run_eval --retriever bge-m3
python -m danlonben.retrieval.run_eval --retriever colpali
python -m danlonben.retrieval.run_eval --retriever colqwen2
```

Results are written to `data/results/<model>/summary.json` and `results.jsonl`.

## Configuration

Key defaults in `src/danlonben/pipeline/graph.py`:

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `gpt-4o` | OpenAI model for generation, rewriting, and critique |
| `questions_per_page` | `2` | Questions generated per page |
| `temperature` | `0.2` | Generation temperature |
| `verification_mode` | `heuristic` | False negative check mode (`heuristic` or `llm`) |

## Requirements

- Python 3.12+
- CUDA GPU required for BGE-M3, ColPali, ColQwen2
- `OPENAI_API_KEY` for pipeline execution
