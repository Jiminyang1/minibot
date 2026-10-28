"""``mine`` end to end: incidents → embeddings → clusters → drafts.

Incremental: incidents already handled by an earlier run are skipped. An
incident counts as handled once its cluster was drafted, attached to an
existing lesson, or judged rule-less; incidents in clusters that were too
small, over the per-run limit, or whose draft failed stay unmined and get
clustered again (with newer incidents) next time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..llm import LLMClient
from .cluster import DEFAULT_CLUSTER_THRESHOLD, threshold_clusters
from .draft import DEFAULT_DEDUPE_THRESHOLD, DraftReport, draft_clusters
from .embed import Embedder
from .mine import Incident, mine_incidents
from .store import LessonStore

DEFAULT_MIN_CLUSTER_SIZE = 1
# Bounds LLM calls per run; the rest waits for the next run.
DEFAULT_MAX_CLUSTERS = 20


@dataclass(frozen=True)
class MinePlan:
    total_incidents: int
    new_incidents: list[Incident]
    # Clusters that will be drafted, largest first.
    clusters: list[list[Incident]]
    # Clusters left for later (below min size or over the limit).
    deferred: list[list[Incident]]


def plan_mining(
    state_home: Path,
    store: LessonStore,
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
) -> MinePlan:
    """Read-only: everything ``mine --dry-run`` shows."""
    incidents = mine_incidents(state_home)
    mined = store.mined_incident_ids()
    new = [item for item in incidents if item.id not in mined]
    groups = cluster_incidents(new, embedder, threshold=threshold)
    eligible = [group for group in groups if len(group) >= min_cluster_size]
    deferred = [group for group in groups if len(group) < min_cluster_size]
    deferred.extend(eligible[max_clusters:])
    return MinePlan(
        total_incidents=len(incidents),
        new_incidents=new,
        clusters=eligible[:max_clusters],
        deferred=deferred,
    )


def cluster_incidents(
    incidents: Sequence[Incident],
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> list[list[Incident]]:
    if not incidents:
        return []
    vectors = embedder.embed([item.embed_text() for item in incidents])
    return [
        [incidents[index] for index in group]
        for group in threshold_clusters(vectors, threshold)
    ]


def run_mining(
    plan: MinePlan,
    *,
    store: LessonStore,
    llm: LLMClient,
    embedder: Embedder,
    dedupe_threshold: float = DEFAULT_DEDUPE_THRESHOLD,
) -> DraftReport:
    report = draft_clusters(
        plan.clusters,
        store=store,
        llm=llm,
        embedder=embedder,
        dedupe_threshold=dedupe_threshold,
    )
    store.mark_mined(
        incident_id
        for outcome in report.outcomes
        if outcome.done
        for incident_id in outcome.incident_ids
    )
    return report
