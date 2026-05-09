"""
Ingestion pipeline: extract text and render page images for each document.

For each document in the registry, produces:
    data/interim/<doc_id>/
        pages.jsonl          — one JSON object per page: {page_num, text, image_path}
        images/
            page_0001.png
            page_0002.png
            ...

Usage:
    python -m danlonben.ingestion                  # process all documents
    python -m danlonben.ingestion finance_nationalbank_2024  # single doc
"""

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger
from pdf2image import convert_from_path
from tqdm import tqdm

from danlonben.config import DOCUMENTS, INTERIM_DATA_DIR, PNG_DPI, RAW_DATA_DIR


def ingest_document(doc_meta: dict, force: bool = False) -> Path:
    """
    Process one document: extract text + render PNGs, write pages.jsonl.

    Returns the path to the output directory.
    Skips if pages.jsonl already exists unless force=True.
    """
    pdf_path = RAW_DATA_DIR / doc_meta["sector"] / doc_meta["filename"]
    out_dir = INTERIM_DATA_DIR / doc_meta["doc_id"]
    images_dir = out_dir / "images"
    jsonl_path = out_dir / "pages.jsonl"

    if jsonl_path.exists() and not force:
        logger.info(f"[{doc_meta['doc_id']}] Already ingested, skipping (use force=True to redo).")
        return out_dir

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Place the file at data/raw/{doc_meta['sector']}/{doc_meta['filename']}"
        )

    images_dir.mkdir(parents=True, exist_ok=True)

    # --- Text extraction (PyMuPDF) ---
    logger.info(f"[{doc_meta['doc_id']}] Extracting text with PyMuPDF...")
    pdf = fitz.open(pdf_path)
    pages_data = []
    for page in pdf:
        pages_data.append({
            "page_num": page.number + 1,  # 1-indexed
            "text": page.get_text(),
            "image_path": None,  # filled below
        })
    pdf.close()
    n_pages = len(pages_data)
    logger.info(f"[{doc_meta['doc_id']}] {n_pages} pages found.")

    # --- Image rendering (pdf2image / poppler) ---
    logger.info(f"[{doc_meta['doc_id']}] Rendering {n_pages} pages at {PNG_DPI} DPI...")
    images = convert_from_path(pdf_path, dpi=PNG_DPI)

    for i, img in enumerate(tqdm(images, desc=doc_meta["doc_id"], unit="page")):
        img_filename = f"page_{i + 1:04d}.png"
        img_path = images_dir / img_filename
        img.save(img_path, "PNG")
        # Store path relative to project root for portability
        pages_data[i]["image_path"] = str(
            img_path.relative_to(INTERIM_DATA_DIR.parent.parent)
        ).replace("\\", "/")

    # --- Write JSONL ---
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for page in pages_data:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    logger.success(
        f"[{doc_meta['doc_id']}] Done. {n_pages} pages → {jsonl_path}"
    )
    return out_dir


def ingest_all(force: bool = False) -> None:
    for doc_meta in DOCUMENTS:
        try:
            ingest_document(doc_meta, force=force)
        except FileNotFoundError as e:
            logger.warning(str(e))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        doc_id = sys.argv[1]
        matches = [d for d in DOCUMENTS if d["doc_id"] == doc_id]
        if not matches:
            logger.error(f"Unknown doc_id '{doc_id}'. Available: {[d['doc_id'] for d in DOCUMENTS]}")
            sys.exit(1)
        ingest_document(matches[0])
    else:
        ingest_all()
