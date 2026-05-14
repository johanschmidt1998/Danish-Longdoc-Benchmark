from __future__ import annotations

from typing import Any

import typer
from langgraph.graph import StateGraph
from loguru import logger

from danlonben.config import DOCUMENTS
from danlonben.pipeline.nodes import (
    DEFAULT_MODEL_NAME,
    critique_candidates_node,
    false_negative_verification_node,
    finalize_export_node,
    generate_questions_node,
    load_pages_node,
    rewrite_questions_node,
)
from danlonben.pipeline.state import PipelineState, RunConfig

DEFAULT_RUN_CONFIG: RunConfig = {
    "questions_per_page": 2,
    "enable_rewriter": True,
    "mock_mode": False,
    "model_name": DEFAULT_MODEL_NAME,
    "temperature": 0.2,
    "max_pages": None,
    "verification_mode": "heuristic",
    "output_filename": "questions.jsonl",
}

app = typer.Typer(help="Run the LangGraph benchmark question-generation pipeline.")


def build_pipeline_graph() -> Any:
    workflow = StateGraph(PipelineState)

    workflow.add_node("load_pages", load_pages_node)
    workflow.add_node("generate_questions", generate_questions_node)
    workflow.add_node("rewrite_questions", rewrite_questions_node)
    workflow.add_node("critique_candidates", critique_candidates_node)
    workflow.add_node("verify_false_negatives", false_negative_verification_node)
    workflow.add_node("finalize_export", finalize_export_node)

    workflow.set_entry_point("load_pages")
    workflow.add_edge("load_pages", "generate_questions")
    workflow.add_edge("generate_questions", "rewrite_questions")
    workflow.add_edge("rewrite_questions", "critique_candidates")
    workflow.add_edge("critique_candidates", "verify_false_negatives")
    workflow.add_edge("verify_false_negatives", "finalize_export")
    workflow.set_finish_point("finalize_export")

    return workflow.compile()


def _merge_run_config(overrides: RunConfig | None = None) -> RunConfig:
    merged = dict(DEFAULT_RUN_CONFIG)
    if overrides:
        merged.update(overrides)
    return merged


def run_pipeline(doc_id: str, run_config: RunConfig | None = None) -> PipelineState:
    graph = build_pipeline_graph()
    initial_state: PipelineState = {
        "doc_id": doc_id,
        "run_config": _merge_run_config(run_config),
        "stats": {},
    }
    return graph.invoke(initial_state)


@app.command()
def run(
    doc_id: str = typer.Argument(..., help="Document identifier from danlonben.config.DOCUMENTS"),
    questions_per_page: int = typer.Option(2, min=1),
    enable_rewriter: bool = typer.Option(True, help="Enable optional rewrite stage."),
    mock_mode: bool = typer.Option(
        False,
        help="Use deterministic local generation/critique logic without Together API calls.",
    ),
    verification_mode: str = typer.Option(
        "heuristic",
        help="False-negative verification mode: heuristic or llm.",
    ),
    model_name: str = typer.Option(
        DEFAULT_MODEL_NAME,
        help="Together model for generation/rewrite/critique.",
    ),
    temperature: float = typer.Option(0.2, min=0.0, max=1.0),
    max_pages: int | None = typer.Option(None, min=1, help="Limit processing to first N pages."),
    output_filename: str = typer.Option("questions.jsonl"),
) -> None:
    verification_mode = verification_mode.lower().strip()
    if verification_mode not in {"heuristic", "llm"}:
        raise typer.BadParameter("verification_mode must be one of: heuristic, llm")

    config: RunConfig = {
        "questions_per_page": questions_per_page,
        "enable_rewriter": enable_rewriter,
        "mock_mode": mock_mode,
        "verification_mode": verification_mode,  # type: ignore[typeddict-item]
        "model_name": model_name,
        "temperature": temperature,
        "max_pages": max_pages,
        "output_filename": output_filename,
    }
    final_state = run_pipeline(doc_id=doc_id, run_config=config)
    stats = final_state.get("stats", {})
    typer.echo(
        f"Done: {doc_id}\n"
        f"Output: {final_state.get('output_path', '(not written)')}\n"
        f"Stats: {stats}"
    )


@app.command("run-all")
def run_all(
    mock_mode: bool = typer.Option(False, help="Run all documents in mock mode."),
    questions_per_page: int = typer.Option(2, min=1),
    verification_mode: str = typer.Option("heuristic"),
    max_pages: int | None = typer.Option(None, min=1),
) -> None:
    failed: list[str] = []
    for doc in DOCUMENTS:
        doc_id = doc["doc_id"]
        logger.info(f"Running pipeline for {doc_id}")
        try:
            result = run_pipeline(
                doc_id,
                {
                    "mock_mode": mock_mode,
                    "questions_per_page": questions_per_page,
                    "verification_mode": verification_mode.lower(),  # type: ignore[typeddict-item]
                    "max_pages": max_pages,
                },
            )
            logger.success(
                f"[{doc_id}] exported {result.get('stats', {}).get('exported_records', 0)} records"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[{doc_id}] failed: {exc}")
            failed.append(doc_id)

    if failed:
        typer.echo(f"Completed with failures: {failed}")
        raise typer.Exit(code=1)
    typer.echo("Completed all documents.")


if __name__ == "__main__":
    app()
