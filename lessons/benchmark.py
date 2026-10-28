"""Re-measure the three lessons thresholds on a small labelled set.

    uv run python -m minibot.lessons.benchmark [--data PATH]

The data (``benchmark_zh.json``) is hand-written and fabricated: 5 lessons
with 2 evidence requests each, 10 relevant and 13 unrelated queries, 4
incident groups, and 6 trigger pairs. It is small on purpose and is a
starting point, not validation; extend it from real incidents when there
are some. The script reports, for each threshold, the scores on both sides
of it, so a change of model or of ``Incident.embed_text`` can be re-checked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cluster import DEFAULT_CLUSTER_THRESHOLD, threshold_clusters
from .draft import DEFAULT_DEDUPE_THRESHOLD
from .retrieve import DEFAULT_MIN_SCORE

DATA_PATH = Path(__file__).with_name("benchmark_zh.json")


def measure(data: dict[str, Any], embed) -> dict[str, Any]:  # noqa: ANN001 - any Embedder.embed
    """All numbers the report prints; ``embed`` maps texts to unit vectors."""
    return {
        "retrieval": _retrieval(data["retrieval"], embed),
        "clustering": _clustering(data["clustering"], embed),
        "dedupe": _dedupe(data["dedupe"], embed),
    }


def _vectors(embed, texts: list[str]) -> np.ndarray:  # noqa: ANN001
    return np.asarray(embed(texts), dtype=np.float32)


def _retrieval(data: dict[str, Any], embed) -> dict[str, Any]:  # noqa: ANN001
    lessons = data["lessons"]
    triggers = _vectors(embed, [lesson["trigger"] for lesson in lessons])
    evidence = [_vectors(embed, lesson["evidence"]) for lesson in lessons]

    def scores(queries: list[str], *, with_evidence: bool) -> np.ndarray:
        vectors = _vectors(embed, queries)
        columns = []
        for index in range(len(lessons)):
            best = vectors @ triggers[index]
            if with_evidence:
                best = np.maximum(best, (vectors @ evidence[index].T).max(axis=1))
            columns.append(best)
        return np.stack(columns, axis=1)

    targets = [lesson for lesson, _ in data["relevant"]]
    queries = [query for _, query in data["relevant"]]
    result = {}
    for name, with_evidence in (("trigger_only", False), ("trigger_and_evidence", True)):
        relevant = scores(queries, with_evidence=with_evidence)
        correct = [float(relevant[i][t]) for i, t in enumerate(targets)]
        noise = np.concatenate(
            [
                scores(data["unrelated"], with_evidence=with_evidence).max(axis=1),
                scores(data["chitchat"], with_evidence=with_evidence).max(axis=1),
            ]
        )
        result[name] = {
            "top1": f"{sum(int(np.argmax(relevant[i]) == t) for i, t in enumerate(targets))}/{len(targets)}",
            "correct_min": round(min(correct), 3),
            "noise_max": round(float(noise.max()), 3),
            "gap": round(min(correct) - float(noise.max()), 3),
        }
    return result


def _clustering(groups: list[list[str]], embed) -> dict[str, Any]:  # noqa: ANN001
    texts = [text for group in groups for text in group]
    labels = [index for index, group in enumerate(groups) for _ in group]
    vectors = _vectors(embed, texts)
    sims = vectors @ vectors.T
    within = [float(sims[i, j]) for i in range(len(texts)) for j in range(i) if labels[i] == labels[j]]
    between = [float(sims[i, j]) for i in range(len(texts)) for j in range(i) if labels[i] != labels[j]]
    clusters = threshold_clusters(vectors, DEFAULT_CLUSTER_THRESHOLD)
    pure = all(len({labels[i] for i in cluster}) == 1 for cluster in clusters)
    return {
        "within_min": round(min(within), 3),
        "between_max": round(max(between), 3),
        "clusters_at_threshold": len(clusters),
        "groups": len(groups),
        "pure": pure,
    }


def _dedupe(data: dict[str, Any], embed) -> dict[str, Any]:  # noqa: ANN001
    def pair_scores(pairs: list[list[str]]) -> list[float]:
        left = _vectors(embed, [a for a, _ in pairs])
        right = _vectors(embed, [b for _, b in pairs])
        return [round(float(score), 3) for score in (left * right).sum(axis=1)]

    return {"same": pair_scores(data["same"]), "different": pair_scores(data["different"])}


def format_report(numbers: dict[str, Any]) -> str:
    retrieval = numbers["retrieval"]
    clustering = numbers["clustering"]
    dedupe = numbers["dedupe"]
    lines = [f"检索（下限 {DEFAULT_MIN_SCORE}）:"]
    for name, row in retrieval.items():
        lines.append(
            f"  {name:<21} top1 {row['top1']}  正确命中最低 {row['correct_min']}  "
            f"无关/闲聊最高 {row['noise_max']}  间隔 {row['gap']:+}"
        )
    lines += [
        f"聚类（阈值 {DEFAULT_CLUSTER_THRESHOLD}）:",
        f"  同组最低 {clustering['within_min']}  异组最高 {clustering['between_max']}  "
        f"聚成 {clustering['clusters_at_threshold']} 类 / 应为 {clustering['groups']} 类  "
        f"{'每类都纯' if clustering['pure'] else '有混类'}",
        f"去重（阈值 {DEFAULT_DEDUPE_THRESHOLD}）:",
        f"  同义 trigger {dedupe['same']}  不同情境 {dedupe['different']}",
        "",
        "数据是手写的小样本，只能说明阈值落在哪个区间，不能当作验证。",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m minibot.lessons.benchmark")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="标注数据 JSON")
    args = parser.parse_args(argv)

    from .embed import FastEmbedEmbedder

    data = json.loads(args.data.read_text(encoding="utf-8"))
    embedder = FastEmbedEmbedder()
    print(f"模型 {embedder.model_name}，数据 {args.data}")
    print(format_report(measure(data, embedder.embed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
