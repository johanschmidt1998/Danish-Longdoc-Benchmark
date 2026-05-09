from __future__ import annotations

from pathlib import Path

import typer

from danlonben.retrieval.evaluate import evaluate
from danlonben.retrieval.retrievers import BM25Retriever

RESULTS_DIR = Path("data/results")
VALID_RETRIEVERS = ("bm25", "bge-m3", "colpali", "colqwen2")

app = typer.Typer(help="Run retrieval evaluation for the Danish Longdoc Benchmark.")


@app.command()
def run(
    retriever: str = typer.Option("bm25", help=f"Retriever to use: {' | '.join(VALID_RETRIEVERS)}"),
    k: int = typer.Option(5, help="Cutoff for Recall@k and NDCG@k"),
    doc_ids: list[str] = typer.Option([], help="Subset of doc IDs. Defaults to all."),
    device: str = typer.Option("cuda", help="Device for neural models: cuda | cpu"),
    batch_size: int = typer.Option(4, help="Batch size for neural model indexing."),
) -> None:
    retriever_name = retriever.lower().strip()
    doc_ids_arg = doc_ids if doc_ids else None

    if retriever_name == "bm25":
        r = BM25Retriever()
    elif retriever_name == "bge-m3":
        from danlonben.retrieval.retrievers import BGEM3Retriever
        r = BGEM3Retriever(device=device, batch_size=batch_size)
    elif retriever_name == "colpali":
        from danlonben.retrieval.retrievers import ColPaliRetriever
        r = ColPaliRetriever(device=device, batch_size=batch_size)
    elif retriever_name == "colqwen2":
        from danlonben.retrieval.retrievers import ColQwen2Retriever
        r = ColQwen2Retriever(device=device, batch_size=batch_size)
    else:
        raise typer.BadParameter(f"Unknown retriever: {retriever_name}. Choose one of: {', '.join(VALID_RETRIEVERS)}")

    evaluate(
        retriever=r,
        retriever_name=retriever_name,
        output_dir=RESULTS_DIR,
        doc_ids=doc_ids_arg,
        k=k,
    )


if __name__ == "__main__":
    app()
