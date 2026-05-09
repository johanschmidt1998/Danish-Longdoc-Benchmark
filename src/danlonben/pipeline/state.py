from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class DocumentMeta(TypedDict):
    doc_id: str
    sector: str
    title: str
    filename: str


class PageRecord(TypedDict):
    page_num: int
    text: str
    image_path: str


class QuestionCandidate(TypedDict):
    candidate_id: str
    doc_id: str
    sector: str
    title: str
    source_page: int
    source_image_path: str
    question_da: str
    answer_da: str
    final_question_da: str
    rewritten_question_da: NotRequired[str | None]
    critique_decision: NotRequired[Literal["keep", "discard"]]
    critique_reason: NotRequired[str]
    verification_passed: NotRequired[bool]
    conflicting_pages: NotRequired[list[int]]
    metadata: dict[str, Any]


class ExportRecord(TypedDict):
    id: str
    doc_id: str
    sector: str
    title: str
    source_page: int
    question_da: str
    answer_da: str
    image_path: str
    quality: dict[str, Any]


class RunConfig(TypedDict, total=False):
    questions_per_page: int
    enable_rewriter: bool
    mock_mode: bool
    model_name: str
    temperature: float
    max_tokens: int
    max_pages: int | None
    verification_mode: Literal["heuristic", "llm"]
    output_filename: str


class PipelineState(TypedDict, total=False):
    doc_id: str
    doc_meta: DocumentMeta
    pages: list[PageRecord]
    candidates: list[QuestionCandidate]
    filtered_candidates: list[QuestionCandidate]
    final_records: list[ExportRecord]
    output_path: str
    run_config: RunConfig
    stats: dict[str, int]
    warnings: list[str]
