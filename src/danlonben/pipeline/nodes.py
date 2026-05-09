from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

_LLM_CALL_TIMEOUT = 45  # seconds per LLM call


def _invoke_with_timeout(llm: Any, prompt: str) -> Any:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(llm.invoke, prompt)
    try:
        return future.result(timeout=_LLM_CALL_TIMEOUT)
    except FuturesTimeoutError:
        raise RuntimeError(f"LLM call timed out after {_LLM_CALL_TIMEOUT}s")
    finally:
        executor.shutdown(wait=False)

from loguru import logger

from danlonben.config import DOCUMENTS, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from danlonben.pipeline.state import (
    DocumentMeta,
    ExportRecord,
    PageRecord,
    PipelineState,
    QuestionCandidate,
    RunConfig,
)

DEFAULT_MODEL_NAME = "gpt-4o-mini"
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9æøåÆØÅ]{3,}")
_DANISH_STOPWORDS = {
    "der",
    "det",
    "den",
    "til",
    "for",
    "med",
    "som",
    "har",
    "kan",
    "ikke",
    "fra",
    "ved",
    "på",
    "af",
    "og",
    "en",
    "et",
    "de",
    "i",
    "at",
}


def _merge_stats(state: PipelineState, **updates: int) -> dict[str, int]:
    stats = dict(state.get("stats", {}))
    stats.update({k: int(v) for k, v in updates.items()})
    return stats


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _shorten(text: str, max_chars: int = 1400) -> str:
    clean = _normalize_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + " …"


def _tokenize(text: str) -> set[str]:
    tokens = {t.lower() for t in _TOKEN_PATTERN.findall(text)}
    return {t for t in tokens if t not in _DANISH_STOPWORDS}


def _extract_text_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk["text"]))
            else:
                parts.append(str(chunk))
        return "\n".join(parts)
    return str(content)


def _parse_json_from_llm(text: str) -> Any:
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    starts = [i for i, ch in enumerate(raw) if ch in "[{"]
    for start in starts:
        closing = "}" if raw[start] == "{" else "]"
        end = raw.rfind(closing)
        if end <= start:
            continue
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON from model response.")


def _build_llm(run_config: RunConfig) -> Any:
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it or run with mock_mode=True."
        )

    import httpx
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=run_config.get("model_name", DEFAULT_MODEL_NAME),
        temperature=float(run_config.get("temperature", 0.2)),
        max_tokens=int(run_config.get("max_tokens", 1200)),
        timeout=httpx.Timeout(30.0),
        max_retries=0,
    )


def _doc_meta_by_id(doc_id: str) -> DocumentMeta:
    for doc in DOCUMENTS:
        if doc["doc_id"] == doc_id:
            return doc
    return {
        "doc_id": doc_id,
        "sector": "unknown",
        "title": doc_id,
        "filename": "",
    }


def load_pages_node(state: PipelineState) -> dict[str, Any]:
    doc_id = state["doc_id"]
    pages_path = INTERIM_DATA_DIR / doc_id / "pages.jsonl"
    if not pages_path.exists():
        raise FileNotFoundError(
            f"Missing ingested pages: {pages_path}. Run ingestion before pipeline execution."
        )

    pages: list[PageRecord] = []
    with open(pages_path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            payload = json.loads(raw)
            pages.append(
                {
                    "page_num": int(payload["page_num"]),
                    "text": str(payload.get("text", "")),
                    "image_path": str(payload.get("image_path", "")),
                }
            )
    max_pages = state.get("run_config", {}).get("max_pages")
    if max_pages is not None:
        pages = pages[: int(max_pages)]

    logger.info(f"[{doc_id}] Loaded {len(pages)} pages from {pages_path}")
    return {
        "doc_meta": _doc_meta_by_id(doc_id),
        "pages": pages,
        "stats": _merge_stats(state, pages_loaded=len(pages), input_lines=line_no),
    }


def _mock_generate(page: PageRecord, count: int) -> list[dict[str, str]]:
    lines = [
        _normalize_whitespace(line)
        for line in page["text"].splitlines()
        if len(_normalize_whitespace(line)) >= 40
    ]
    if not lines:
        lines = [_shorten(page["text"], max_chars=200)]
    lines = [line for line in lines if line]
    if not lines:
        lines = ["Ingen tekst fundet på siden."]

    items: list[dict[str, str]] = []
    for idx in range(count):
        source = lines[idx % len(lines)]
        anchor = _shorten(source, max_chars=120)
        items.append(
            {
                "question_da": f"Hvad fremgår der om følgende emne: {anchor}?",
                "answer_da": source,
            }
        )
    return items


def _generate_with_llm(
    llm: Any,
    page: PageRecord,
    questions_per_page: int,
) -> list[dict[str, str]]:
    prompt = f"""
Du er en dansk dataannotator til et RAG-benchmark.
Generér {questions_per_page} spørgsmål-svar-par baseret på EN enkelt side.

Krav:
- Spørgsmål og svar skal være på dansk.
- Spørgsmålet må ikke nævne side, figur, tabelplacering, "ovenfor"/"nedenfor" eller dokumentstruktur.
- Spørgsmålet skal være konkret og kunne besvares af kun denne side.
- Svar skal være kort og faktuelt.

Returnér KUN gyldig JSON i formatet:
{{
  "items": [
    {{"question_da": "...", "answer_da": "..."}}
  ]
}}

Sideindhold:
{_shorten(page["text"], max_chars=2200)}
"""
    response = _invoke_with_timeout(llm, prompt)
    parsed = _parse_json_from_llm(_extract_text_content(response))
    if not isinstance(parsed, dict) or "items" not in parsed:
        raise ValueError("Model response did not contain an 'items' array.")
    items = parsed["items"]
    if not isinstance(items, list):
        raise ValueError("Model response field 'items' is not a list.")
    valid_items: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = _normalize_whitespace(str(item.get("question_da", "")))
        a = _normalize_whitespace(str(item.get("answer_da", "")))
        if q and a:
            valid_items.append({"question_da": q, "answer_da": a})
    return valid_items[:questions_per_page]


def generate_questions_node(state: PipelineState) -> dict[str, Any]:
    pages = state.get("pages", [])
    doc_meta = state["doc_meta"]
    run_config = state.get("run_config", {})
    questions_per_page = int(run_config.get("questions_per_page", 2))
    mock_mode = bool(run_config.get("mock_mode", False))

    candidates: list[QuestionCandidate] = []

    for i, page in enumerate(pages, start=1):
        page_text = _normalize_whitespace(page["text"])
        if len(page_text) < 80:
            logger.info(
                f"[{state['doc_id']}] Skipping page {page['page_num']} ({i}/{len(pages)}) — too little text ({len(page_text)} chars)."
            )
            continue
        logger.info(f"[{state['doc_id']}] Generating page {page['page_num']} ({i}/{len(pages)})...")
        llm = None if mock_mode else _build_llm(run_config)
        try:
            page_items = (
                _mock_generate(page, questions_per_page)
                if mock_mode
                else _generate_with_llm(llm, page, questions_per_page)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[{state['doc_id']}] Generation failed on page {page['page_num']}: {exc}"
            )
            continue

        for idx, item in enumerate(page_items, start=1):
            candidate_id = (
                f"{state['doc_id']}_p{page['page_num']:04d}_q{idx:02d}"
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "doc_id": state["doc_id"],
                    "sector": doc_meta["sector"],
                    "title": doc_meta["title"],
                    "source_page": page["page_num"],
                    "source_image_path": page["image_path"],
                    "question_da": item["question_da"],
                    "answer_da": item["answer_da"],
                    "final_question_da": item["question_da"],
                    "metadata": {
                        "stage": "generated",
                        "model": run_config.get("model_name", DEFAULT_MODEL_NAME),
                        "mock_mode": mock_mode,
                    },
                }
            )

    logger.info(f"[{state['doc_id']}] Generated {len(candidates)} question candidates.")
    return {
        "candidates": candidates,
        "stats": _merge_stats(
            state,
            questions_generated=len(candidates),
            pages_attempted=len(pages),
        ),
    }


def _rewrite_with_llm(llm: Any, candidate: QuestionCandidate) -> str:
    prompt = f"""
Omskriv følgende spørgsmål til naturligt dansk uden at ændre betydningen.
Behold spørgsmålet som ét spørgsmål.
Undgå referencer til side/figur/tabelplacering eller dokumentstruktur.

Returnér KUN gyldig JSON:
{{"question_da": "..."}}

Originalt spørgsmål:
{candidate["question_da"]}

Kort svar-reference:
{_shorten(candidate["answer_da"], max_chars=220)}
"""
    response = _invoke_with_timeout(llm, prompt)
    parsed = _parse_json_from_llm(_extract_text_content(response))
    if not isinstance(parsed, dict):
        raise ValueError("Unexpected rewrite response shape.")
    rewritten = _normalize_whitespace(str(parsed.get("question_da", "")))
    return rewritten or candidate["question_da"]


def rewrite_questions_node(state: PipelineState) -> dict[str, Any]:
    candidates = state.get("candidates", [])
    run_config = state.get("run_config", {})
    enable_rewriter = bool(run_config.get("enable_rewriter", True))
    mock_mode = bool(run_config.get("mock_mode", False))

    if not candidates:
        return {"candidates": candidates}

    if not enable_rewriter:
        passthrough = []
        for candidate in candidates:
            updated = dict(candidate)
            updated["final_question_da"] = candidate["question_da"]
            updated["rewritten_question_da"] = None
            passthrough.append(updated)
        return {
            "candidates": passthrough,
            "stats": _merge_stats(state, questions_rewritten=0),
        }

    llm = None if mock_mode else _build_llm(run_config)
    rewritten_count = 0
    rewritten_candidates: list[QuestionCandidate] = []

    for candidate in candidates:
        updated = dict(candidate)
        rewritten = candidate["question_da"]
        if not mock_mode:
            try:
                rewritten = _rewrite_with_llm(llm, candidate)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[{state['doc_id']}] Rewrite failed for {candidate['candidate_id']}: {exc}"
                )
        updated["final_question_da"] = rewritten
        updated["rewritten_question_da"] = (
            rewritten if rewritten != candidate["question_da"] else None
        )
        if updated["rewritten_question_da"]:
            rewritten_count += 1
        rewritten_candidates.append(updated)

    return {
        "candidates": rewritten_candidates,
        "stats": _merge_stats(state, questions_rewritten=rewritten_count),
    }


def _heuristic_critique(candidate: QuestionCandidate) -> tuple[str, str]:
    question = candidate["final_question_da"].lower()
    answer = candidate["answer_da"]
    disallowed_markers = ("side", "figur", "tabellen", "ovenfor", "nedenfor")
    if any(marker in question for marker in disallowed_markers):
        return "discard", "Question references document layout or structure."
    if len(_normalize_whitespace(answer)) < 20:
        return "discard", "Answer is too short to be useful."
    if len(_normalize_whitespace(candidate["final_question_da"])) < 15:
        return "discard", "Question is too short."
    return "keep", "Looks answerable and layout-agnostic."


def _llm_critique(
    llm: Any,
    candidate: QuestionCandidate,
    source_page_text: str,
) -> tuple[str, str]:
    prompt = f"""
Vurder om dette spørgsmål-svar-par skal beholdes til et dansk side-retrieval benchmark.
Kriterier:
1) Kan besvares ud fra én side.
2) Ingen reference til side/figur/tabelplacering eller dokumentstruktur.
3) Spørgsmålet er klart og specifikt.

Returnér KUN gyldig JSON:
{{"decision": "keep" | "discard", "reason": "..."}}

Spørgsmål:
{candidate["final_question_da"]}

Svar:
{candidate["answer_da"]}

Kildeuddrag:
{_shorten(source_page_text, max_chars=1400)}
"""
    response = _invoke_with_timeout(llm, prompt)
    parsed = _parse_json_from_llm(_extract_text_content(response))
    if not isinstance(parsed, dict):
        raise ValueError("Invalid critique response.")
    decision = str(parsed.get("decision", "discard")).strip().lower()
    if decision not in {"keep", "discard"}:
        decision = "discard"
    reason = _normalize_whitespace(str(parsed.get("reason", ""))) or "No reason provided."
    return decision, reason


def critique_candidates_node(state: PipelineState) -> dict[str, Any]:
    candidates = state.get("candidates", [])
    run_config = state.get("run_config", {})
    mock_mode = bool(run_config.get("mock_mode", False))
    page_lookup = {page["page_num"]: page for page in state.get("pages", [])}

    llm = None if mock_mode else _build_llm(run_config)
    reviewed: list[QuestionCandidate] = []
    kept: list[QuestionCandidate] = []

    for candidate in candidates:
        source_page = page_lookup.get(candidate["source_page"], {"text": ""})
        try:
            decision, reason = (
                _heuristic_critique(candidate)
                if mock_mode
                else _llm_critique(llm, candidate, str(source_page.get("text", "")))
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[{state['doc_id']}] Critique failed for {candidate['candidate_id']}: {exc}"
            )
            decision, reason = "discard", "Critique failure."

        updated = dict(candidate)
        updated["critique_decision"] = decision
        updated["critique_reason"] = reason
        reviewed.append(updated)
        if decision == "keep":
            kept.append(updated)

    logger.info(f"[{state['doc_id']}] Critique kept {len(kept)} / {len(candidates)} candidates.")
    return {
        "candidates": reviewed,
        "filtered_candidates": kept,
        "stats": _merge_stats(
            state,
            questions_after_critique=len(kept),
            questions_discarded_critique=max(len(candidates) - len(kept), 0),
        ),
    }


def _select_competing_pages(
    candidate: QuestionCandidate,
    pages: list[PageRecord],
    max_candidates: int = 5,
) -> list[PageRecord]:
    source = candidate["source_page"]
    query_tokens = _tokenize(candidate["final_question_da"] + " " + candidate["answer_da"])
    if not query_tokens:
        return []

    scored: list[tuple[float, PageRecord]] = []
    for page in pages:
        if page["page_num"] == source:
            continue
        page_tokens = _tokenize(page["text"])
        if not page_tokens:
            continue
        overlap = len(query_tokens & page_tokens) / max(len(query_tokens), 1)
        if overlap > 0:
            scored.append((overlap, page))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [page for _, page in scored[:max_candidates]]


def _heuristic_verify_candidate(
    candidate: QuestionCandidate,
    pages: list[PageRecord],
) -> list[int]:
    source_page = candidate["source_page"]
    query_tokens = _tokenize(candidate["final_question_da"] + " " + candidate["answer_da"])
    conflicts: list[int] = []

    if not query_tokens:
        return conflicts

    for page in pages:
        if page["page_num"] == source_page:
            continue
        page_tokens = _tokenize(page["text"])
        if not page_tokens:
            continue
        overlap_count = len(query_tokens & page_tokens)
        overlap_ratio = overlap_count / len(query_tokens)
        if overlap_count >= 6 and overlap_ratio >= 0.55:
            conflicts.append(page["page_num"])
    return sorted(set(conflicts))


def _llm_verify_candidate(
    llm: Any,
    candidate: QuestionCandidate,
    pages: list[PageRecord],
) -> list[int]:
    competing_pages = _select_competing_pages(candidate, pages, max_candidates=5)
    if not competing_pages:
        return []

    snippets = []
    for page in competing_pages:
        snippets.append(
            {
                "page_num": page["page_num"],
                "text_excerpt": _shorten(page["text"], max_chars=700),
            }
        )

    prompt = f"""
Vurder om spørgsmålet sandsynligvis også kan besvares af andre sider i samme dokument.
Returnér KUN gyldig JSON i format:
{{"conflicting_pages": [int, ...]}}

Spørgsmål:
{candidate["final_question_da"]}

Svar:
{candidate["answer_da"]}

Kildeside:
{candidate["source_page"]}

Mulige konkurrerende sider:
{json.dumps(snippets, ensure_ascii=False, indent=2)}
"""
    response = _invoke_with_timeout(llm, prompt)
    parsed = _parse_json_from_llm(_extract_text_content(response))
    if not isinstance(parsed, dict):
        raise ValueError("Invalid verification response.")
    values = parsed.get("conflicting_pages", [])
    if not isinstance(values, list):
        return []
    output: list[int] = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(output))


def false_negative_verification_node(state: PipelineState) -> dict[str, Any]:
    run_config = state.get("run_config", {})
    verification_mode = str(run_config.get("verification_mode", "heuristic")).lower()
    mock_mode = bool(run_config.get("mock_mode", False))
    candidates = state.get("filtered_candidates", [])
    pages = state.get("pages", [])

    if not candidates:
        return {
            "filtered_candidates": [],
            "stats": _merge_stats(state, questions_after_verification=0),
        }

    llm = None
    if verification_mode == "llm" and not mock_mode:
        llm = _build_llm(run_config)

    verified: list[QuestionCandidate] = []
    removed = 0
    for candidate in candidates:
        try:
            if llm is not None:
                conflicts = _llm_verify_candidate(llm, candidate, pages)
            else:
                conflicts = _heuristic_verify_candidate(candidate, pages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[{state['doc_id']}] Verification failed for {candidate['candidate_id']}: {exc}"
            )
            conflicts = []

        updated = dict(candidate)
        updated["conflicting_pages"] = conflicts
        updated["verification_passed"] = len(conflicts) == 0

        if updated["verification_passed"]:
            verified.append(updated)
        else:
            removed += 1

    logger.info(
        f"[{state['doc_id']}] Verification kept {len(verified)} / {len(candidates)} candidates."
    )
    return {
        "filtered_candidates": verified,
        "stats": _merge_stats(
            state,
            questions_after_verification=len(verified),
            questions_discarded_verification=removed,
        ),
    }


def finalize_export_node(state: PipelineState) -> dict[str, Any]:
    doc_id = state["doc_id"]
    doc_meta = state["doc_meta"]
    output_filename = state.get("run_config", {}).get("output_filename", "questions.jsonl")
    output_dir = PROCESSED_DATA_DIR / "qa" / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    records: list[ExportRecord] = []
    with open(output_path, "w", encoding="utf-8") as handle:
        for idx, candidate in enumerate(state.get("filtered_candidates", []), start=1):
            record: ExportRecord = {
                "id": f"{doc_id}_{idx:05d}",
                "doc_id": doc_id,
                "sector": doc_meta["sector"],
                "title": doc_meta["title"],
                "source_page": candidate["source_page"],
                "question_da": candidate["final_question_da"],
                "answer_da": candidate["answer_da"],
                "image_path": candidate["source_image_path"],
                "quality": {
                    "critique_decision": candidate.get("critique_decision", "keep"),
                    "critique_reason": candidate.get("critique_reason", ""),
                    "verification_passed": candidate.get("verification_passed", True),
                    "conflicting_pages": candidate.get("conflicting_pages", []),
                    "candidate_id": candidate["candidate_id"],
                },
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.success(f"[{doc_id}] Exported {len(records)} rows to {output_path}")
    return {
        "final_records": records,
        "output_path": str(output_path),
        "stats": _merge_stats(state, exported_records=len(records)),
    }
