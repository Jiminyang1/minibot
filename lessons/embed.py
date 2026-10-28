"""Local text embeddings for clustering incidents and retrieving lessons.

``fastembed`` (ONNX, no torch) is an optional extra and is imported only when
the first text is embedded, so importing this module never loads a model.
Vectors are unit length, so cosine similarity is a dot product.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib.util
from pathlib import Path
import threading
from typing import Any, Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
# bge-small accepts 512 tokens; lessons and queries are compared on their
# opening, which is where the intent is.
_MAX_TEXT_CHARS = 512


class Embedder(Protocol):
    """Anything that maps texts to unit vectors (one row per text)."""

    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def fastembed_available() -> bool:
    return importlib.util.find_spec("fastembed") is not None


def default_model_dir() -> Path:
    """Model cache under the user's state home, shared by every runtime.

    Deliberately the real home even when a runtime runs against a temp state
    home (evals): the model is a cache, not state, and is ~100 MB.
    """
    from ..config import resolve_state_home

    return resolve_state_home() / "models"


class FastEmbedEmbedder:
    """``fastembed.TextEmbedding`` behind the ``Embedder`` protocol."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Any = None
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        clipped = [text[:_MAX_TEXT_CHARS] for text in texts]
        rows = np.asarray(list(model.embed(clipped)), dtype=np.float32)
        return normalize_rows(rows)

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise RuntimeError(
                        "经验库需要 fastembed：请运行 uv sync --extra lessons。"
                    ) from exc
                cache_dir = self.cache_dir or default_model_dir()
                cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(
                    model_name=self.model_name, cache_dir=str(cache_dir)
                )
            return self._model


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine of unit-row matrices: ``a @ b.T``."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return a @ b.T


def best_match(vector: np.ndarray, matrix: np.ndarray) -> tuple[int, float]:
    """Index and cosine of the closest row; ``(-1, -1.0)`` when empty."""
    if matrix.size == 0:
        return -1, -1.0
    scores = matrix @ vector
    index = int(np.argmax(scores))
    return index, float(scores[index])
