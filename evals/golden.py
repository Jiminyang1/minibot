"""Per-stage golden assertions shared by the pytest suite (``test_golden.py``).

``GoldenSession`` runs each case's sandboxed turn at most once and scores
stages on demand: the deterministic stages from the cached output, the
answer stage with one judge call, ``end_to_end`` from all of them. Running
only ``-k tool_args`` therefore costs one model turn per case and no judge
call. With Langfuse configured, each run is wrapped in its own span so its
trace URL can go into the assertion message.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from ..config import Config
from ..llm import LLMClient
from .runner import applicable_stages, trace_url
from .sandbox import run_case
from .scoring import StageScore, end_to_end, judge_answer, score_deterministic, stage_passed


@dataclass
class CaseRun:
    output: dict[str, Any]
    trace_id: str | None
    trace_url: str | None
    deterministic: list[StageScore]
    answer: StageScore | None = None


@dataclass(frozen=True)
class StageResult:
    score: StageScore
    passed: bool
    message: str


class GoldenSession:
    def __init__(
        self,
        config: Config,
        *,
        judge: LLMClient | None,
        langfuse: Any = None,
        llm: LLMClient | None = None,
        run: Callable[..., dict[str, Any]] = run_case,
        today: date | None = None,
    ) -> None:
        self.config = config
        self.judge = judge
        self.langfuse = langfuse
        self.llm = llm
        self._run = run
        self._today = today
        self._runs: dict[str, CaseRun] = {}

    def check(self, case: dict[str, Any], stage: str) -> StageResult | None:
        """Score one stage; ``None`` when the case does not specify it."""
        expected = case.get("expected") or {}
        if stage not in applicable_stages(expected):
            return None
        run = self.run(case)
        if stage == "answer":
            score = self._answer(case, run)
        elif stage == "end_to_end":
            scores = list(run.deterministic)
            if expected.get("answer_rubric"):
                scores.append(self._answer(case, run))
            score = end_to_end(scores, error=run.output.get("error"))
        else:
            score = next(s for s in run.deterministic if s.name == stage)
        passed = stage_passed(score.name, score.value)
        return StageResult(score, passed, self._message(case, score, run))

    def run(self, case: dict[str, Any]) -> CaseRun:
        """The case's sandboxed turn, run once and cached for the session."""
        cached = self._runs.get(case["id"])
        if cached is not None:
            return cached
        trace_id = None
        if self.langfuse is not None:
            with self.langfuse.start_as_current_observation(
                name=f"golden-pytest:{case['id']}", input=case["input"]
            ) as span:
                output = self._run(case["input"], config=self.config, llm=self.llm)
                span.update(output=output.get("reply"))
                trace_id = span.trace_id
        else:
            output = self._run(case["input"], config=self.config, llm=self.llm)
        run = CaseRun(
            output=output,
            trace_id=trace_id,
            trace_url=trace_url(self.langfuse, trace_id),
            deterministic=score_deterministic(
                output, case.get("expected") or {}, today=self._today or date.today()
            ),
        )
        self._runs[case["id"]] = run
        return run

    def close(self) -> None:
        if self.langfuse is not None:
            self.langfuse.flush()

    def _answer(self, case: dict[str, Any], run: CaseRun) -> StageScore:
        if run.answer is None:
            if self.judge is None:
                return StageScore("answer", 0.0, "没有配置评测员模型")
            run.answer = judge_answer(
                self.judge,
                prompt=str(case["input"].get("prompt")),
                reply=run.output.get("reply"),
                rubric=case["expected"]["answer_rubric"],
                tool_calls=run.output.get("tool_calls") or [],
            )
        return run.answer

    def _message(self, case: dict[str, Any], score: StageScore, run: CaseRun) -> str:
        lines = [f"{case['id']} · {score.name} = {score.value:g}: {score.comment}"]
        error = run.output.get("error")
        if error and score.name != "end_to_end":
            lines.append(f"运行出错: {error}")
        called = [str(call.get("name")) for call in run.output.get("tool_calls") or []]
        lines.append(f"调用的工具: {called or '无'}")
        lines.append(f"trace: {run.trace_url or '无（未配置 Langfuse）'}")
        return "\n".join(lines)


__all__ = ["CaseRun", "GoldenSession", "StageResult"]
