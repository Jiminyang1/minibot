"""Retrieve approved lessons relevant to a user message.

A lesson's score is the best cosine between the query and its *keys*: the
trigger (the "when" half) and the user requests in its evidence. Matching a
new request against past requests compares like with like; the trigger
alone is a description ("用户要求修改配置文件时") and scores poorly against a
concrete request ("把端口改成 8080"). Key vectors are cached per (model,
text) and rebuilt only when the store file or the embedding model changes;
query vectors are memoised because ``ContextBuilder.build`` runs several
times per turn on the same latest message.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading

import numpy as np

from .embed import Embedder
from .store import Lesson, LessonStore

# bge-small-zh-v1.5, 5 lessons with 2 evidence requests each, measured on
# disjoint queries: trigger-only ranking got 7/10 top-1 and its correct hits
# (min 0.41) overlapped unrelated tasks (max 0.46). Ranking by trigger and
# evidence got 10/10, correct hits min 0.58 vs unrelated tasks max 0.51 and
# chit-chat max 0.49. 0.55 sits in that gap. A lesson approved without
# evidence falls back to its trigger and needs a closer match. The set is
# small and hand-written (benchmark_zh.json); re-measure with
# python -m minibot.lessons.benchmark
DEFAULT_MIN_SCORE = 0.55
_MAX_EVIDENCE_KEYS = 8
DEFAULT_TOP_K = 3
_QUERY_CACHE_SIZE = 64


@dataclass(frozen=True)
class ScoredLesson:
    lesson: Lesson
    score: float


class LessonRetriever:
    """Top-k approved lessons for a query, above a cosine floor."""

    def __init__(
        self,
        store: LessonStore,
        embedder: Embedder,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        fail_open: bool = False,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        # fail_open: an embedding failure disables retrieval for the rest of
        # the process instead of failing the turn (MINIBOT_LESSONS=auto).
        self.fail_open = fail_open
        self._disabled = False
        self._lock = threading.Lock()
        self._signature: tuple[object, ...] | None = None
        self._lessons: list[Lesson] = []
        # Row i of the key matrix belongs to lesson self._owners[i].
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._owners: list[int] = []
        self._key_vectors: dict[tuple[str, str], np.ndarray] = {}
        self._queries: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def retrieve(self, query: str) -> list[ScoredLesson]:
        query = query.strip()
        if not query or self.top_k <= 0 or self._disabled:
            return []
        try:
            with self._lock:
                return self._retrieve(query)
        except Exception:
            if not self.fail_open:
                raise
            self._disabled = True
            return []

    def _retrieve(self, query: str) -> list[ScoredLesson]:
        self._refresh()
        if not self._lessons:
            return []
        key_scores = self._matrix @ self._query_vector(query)
        best = [float("-inf")] * len(self._lessons)
        for row, owner in enumerate(self._owners):
            best[owner] = max(best[owner], float(key_scores[row]))
        order = sorted(range(len(best)), key=lambda i: (-best[i], i))
        return [
            ScoredLesson(self._lessons[i], best[i])
            for i in order[: self.top_k]
            if best[i] >= self.min_score
        ]

    def _refresh(self) -> None:
        model = self.embedder.model_name
        signature = (model, self.store.signature())
        if signature == self._signature:
            return
        lessons = self.store.list(status="approved")
        keys_per_lesson = [_lesson_keys(lesson) for lesson in lessons]
        texts = {text for keys in keys_per_lesson for text in keys}
        # Reuse vectors only for this model and for texts still present, so
        # edits, retractions and a model switch all invalidate.
        cached = {
            text: self._key_vectors[(model, text)]
            for text in texts
            if (model, text) in self._key_vectors
        }
        missing = sorted(texts - cached.keys())
        if missing:
            cached.update(zip(missing, self.embedder.embed(missing), strict=True))
        self._key_vectors = {(model, t): v for t, v in cached.items()}
        self._lessons = lessons
        self._owners = [i for i, keys in enumerate(keys_per_lesson) for _ in keys]
        rows = [cached[text] for keys in keys_per_lesson for text in keys]
        self._matrix = np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        self._signature = signature

    def _query_vector(self, query: str) -> np.ndarray:
        key = (self.embedder.model_name, query)
        vector = self._queries.get(key)
        if vector is not None:
            self._queries.move_to_end(key)
            return vector
        vector = self.embedder.embed([query])[0]
        self._queries[key] = vector
        if len(self._queries) > _QUERY_CACHE_SIZE:
            self._queries.popitem(last=False)
        return vector


def _lesson_keys(lesson: Lesson) -> list[str]:
    """The trigger, then distinct evidence requests (bounded per lesson)."""
    keys = [lesson.trigger]
    for evidence in lesson.evidence:
        request = evidence.request.strip()
        if request and request not in keys:
            keys.append(request)
        if len(keys) > _MAX_EVIDENCE_KEYS:
            break
    return keys
