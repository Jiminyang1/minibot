"""Deterministic threshold clustering of unit vectors.

Leader clustering: walk the vectors in input order; each joins the cluster
whose centroid it is closest to if that cosine reaches *threshold*, else it
starts a new cluster. Centroids are re-normalised means, so a cluster does
not drift by chaining the way single linkage does. Same input order, same
clusters — no random init, no sklearn.
"""

from __future__ import annotations

import numpy as np

from .embed import normalize_rows

# Measured with bge-small-zh-v1.5 on ``Incident.embed_text`` for the small
# hand-written set in benchmark_zh.json: same-cause incidents score >= 0.82,
# different causes <= 0.77. 0.8 sits in that narrow gap; a reviewer would
# rather merge two drafts than untangle one that mixes causes.
# Re-measure: python -m minibot.lessons.benchmark
DEFAULT_CLUSTER_THRESHOLD = 0.8


def threshold_clusters(
    vectors: np.ndarray,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> list[list[int]]:
    """Group row indices; clusters ordered by size desc, then first member."""
    vectors = normalize_rows(vectors) if len(vectors) else vectors
    members: list[list[int]] = []
    sums: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    for index, vector in enumerate(vectors):
        if centroids:
            scores = np.stack(centroids) @ vector
            best = int(np.argmax(scores))
            if float(scores[best]) >= threshold:
                members[best].append(index)
                sums[best] = sums[best] + vector
                centroids[best] = normalize_rows(sums[best])[0]
                continue
        members.append([index])
        sums.append(vector.copy())
        centroids.append(vector.copy())
    return sorted(members, key=lambda group: (-len(group), group[0]))
