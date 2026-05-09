from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


# ---------------------------------------------------------------------------
# Shared tokeniser (same logic as pipeline nodes.py)
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9æøåÆØÅ]{2,}")
_DANISH_STOPWORDS = {
    "der", "det", "den", "til", "for", "med", "som", "har", "kan",
    "ikke", "fra", "ved", "på", "af", "og", "en", "et", "de", "i", "at",
    "er", "var", "vil", "men", "om", "så", "han", "hun", "vi", "de",
}


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in _DANISH_STOPWORDS]


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """Common interface for all retrieval models.

    Usage:
        retriever.index(pages)           # build index for one document
        retriever.retrieve(query, k=5)   # returns [(page_num, score), ...]
    """

    @abstractmethod
    def index(self, pages: list[dict[str, Any]]) -> None:
        """Index a list of PageRecords for a single document."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return up to k (page_num, score) pairs ranked by relevance."""


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Retriever(BaseRetriever):
    """Sparse keyword retrieval using BM25 (Robertson et al.)."""

    def __init__(self) -> None:
        self._bm25: Any = None
        self._page_nums: list[int] = []

    def index(self, pages: list[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        self._page_nums = [p["page_num"] for p in pages]
        tokenised_corpus = [_tokenize(p["text"]) for p in pages]
        self._bm25 = BM25Okapi(tokenised_corpus)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve().")

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = self._bm25.get_scores(query_tokens).tolist()
        ranked = sorted(
            zip(self._page_nums, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]


# ---------------------------------------------------------------------------
# BGE-M3
# ---------------------------------------------------------------------------

class BGEM3Retriever(BaseRetriever):
    """Dense text retrieval using BAAI/bge-m3 via sentence-transformers.

    Encodes page texts and queries into dense vectors and ranks by cosine
    similarity. Multilingual model with strong Danish support.
    """

    def __init__(self, device: str = "cuda", batch_size: int = 32) -> None:
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._page_embeddings: Any = None  # numpy array (n_pages, dim)
        self._page_nums: list[int] = []

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-m3", device=self.device)

    def index(self, pages: list[dict[str, Any]]) -> None:
        import numpy as np

        self._load_model()
        self._page_nums = [p["page_num"] for p in pages]
        texts = [p["text"] for p in pages]
        self._page_embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # shape: (n_pages, embedding_dim) as float32 numpy array

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        import numpy as np

        if self._page_embeddings is None:
            raise RuntimeError("Call index() before retrieve().")

        query_embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # shape: (1, dim)

        scores: list[float] = (query_embedding @ self._page_embeddings.T)[0].tolist()
        ranked = sorted(
            zip(self._page_nums, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]


# ---------------------------------------------------------------------------
# ColPali
# ---------------------------------------------------------------------------

class ColPaliRetriever(BaseRetriever):
    """Visual page retrieval using ColPali (vidore/colpali-v1.2).

    Operates entirely on page images — no text extraction required.
    Uses MaxSim scoring over patch-level embeddings.
    """

    MODEL_NAME = "vidore/colpali-v1.2"

    def __init__(self, device: str = "cuda", batch_size: int = 4) -> None:
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None
        self._page_embeddings: list[Any] = []
        self._page_nums: list[int] = []

    def _load_model(self) -> None:
        if self._model is None:
            import torch
            from colpali_engine.models import ColPali, ColPaliProcessor

            self._model = ColPali.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            self._processor = ColPaliProcessor.from_pretrained(self.MODEL_NAME)

    def index(self, pages: list[dict[str, Any]]) -> None:
        import torch
        from pathlib import Path
        from PIL import Image
        from danlonben.config import PROJ_ROOT

        self._load_model()
        self._page_nums = [p["page_num"] for p in pages]
        self._page_embeddings = []

        for i in range(0, len(pages), self.batch_size):
            batch = pages[i : i + self.batch_size]
            images = [
                Image.open(PROJ_ROOT / p["image_path"]).convert("RGB")
                for p in batch
            ]
            inputs = self._processor.process_images(images).to(self.device)
            with torch.no_grad():
                embeddings = self._model(**inputs)
            # store each page embedding on CPU to free GPU memory
            self._page_embeddings.extend([e.cpu() for e in embeddings])

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        import torch

        if not self._page_embeddings:
            raise RuntimeError("Call index() before retrieve().")

        inputs = self._processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embedding = self._model(**inputs)

        doc_embeddings = torch.stack(self._page_embeddings)
        scores = self._processor.score_multi_vector(
            query_embedding.cpu(), doc_embeddings
        )[0].tolist()

        ranked = sorted(
            zip(self._page_nums, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]


# ---------------------------------------------------------------------------
# ColQwen2
# ---------------------------------------------------------------------------

class ColQwen2Retriever(BaseRetriever):
    """Visual page retrieval using ColQwen2 (vidore/colqwen2-v1.0).

    Stronger than ColPali — uses Qwen2-VL as the vision backbone.
    Same MaxSim scoring approach, same interface.
    """

    MODEL_NAME = "vidore/colqwen2-v1.0"

    def __init__(self, device: str = "cuda", batch_size: int = 4) -> None:
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None
        self._page_embeddings: list[Any] = []
        self._page_nums: list[int] = []

    def _load_model(self) -> None:
        if self._model is None:
            import torch
            from colpali_engine.models import ColQwen2, ColQwen2Processor

            self._model = ColQwen2.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            self._processor = ColQwen2Processor.from_pretrained(self.MODEL_NAME)

    def index(self, pages: list[dict[str, Any]]) -> None:
        import torch
        from pathlib import Path
        from PIL import Image
        from danlonben.config import PROJ_ROOT

        self._load_model()
        self._page_nums = [p["page_num"] for p in pages]
        self._page_embeddings = []

        for i in range(0, len(pages), self.batch_size):
            batch = pages[i : i + self.batch_size]
            images = [
                Image.open(PROJ_ROOT / p["image_path"]).convert("RGB")
                for p in batch
            ]
            inputs = self._processor.process_images(images).to(self.device)
            with torch.no_grad():
                embeddings = self._model(**inputs)
            self._page_embeddings.extend([e.cpu() for e in embeddings])

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        import torch

        if not self._page_embeddings:
            raise RuntimeError("Call index() before retrieve().")

        inputs = self._processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embedding = self._model(**inputs)

        doc_embeddings = torch.stack(self._page_embeddings)
        scores = self._processor.score_multi_vector(
            query_embedding.cpu(), doc_embeddings
        )[0].tolist()

        ranked = sorted(
            zip(self._page_nums, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]
