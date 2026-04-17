"""Local skill loading and progressive matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..tools.registry import ToolRegistry


_EXPLICIT_APP_TERMS = {
    "calendar",
    "日历",
    "reminders",
    "提醒",
    "提醒事项",
    "notes",
    "note",
    "笔记",
    "备忘录",
}


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: tuple[str, ...]
    tools: tuple[str, ...]
    summary: str
    body: str
    path: Path


@dataclass(frozen=True)
class SkillMatch:
    skill: Skill
    score: int
    explicit_mention: bool


class SkillMatcher:
    """Deterministic, trigger-based skill matching."""

    def match(
        self,
        user_input: str,
        *,
        skills: list[Skill],
        tool_registry: ToolRegistry,
    ) -> list[SkillMatch]:
        normalized = user_input.strip().lower()
        if not normalized:
            return []

        matches: list[SkillMatch] = []
        for skill in skills:
            if not self._skill_available(skill, tool_registry):
                continue
            score = sum(1 for trigger in skill.triggers if trigger.lower() in normalized)
            if score <= 0:
                continue
            explicit_mention = any(
                term in normalized and term in {trigger.lower() for trigger in skill.triggers}
                for term in _EXPLICIT_APP_TERMS
            )
            matches.append(
                SkillMatch(
                    skill=skill,
                    score=score,
                    explicit_mention=explicit_mention,
                )
            )

        return sorted(
            matches,
            key=lambda item: (-item.score, item.skill.name),
        )

    @staticmethod
    def _skill_available(skill: Skill, tool_registry: ToolRegistry) -> bool:
        return all(tool_registry.get(name) is not None for name in skill.tools)


class SkillRegistry:
    """Load skill markdown files and expose lightweight matching."""

    def __init__(
        self,
        skills: list[Skill],
        *,
        matcher: SkillMatcher | None = None,
    ) -> None:
        self._skills = tuple(skills)
        self.matcher = matcher or SkillMatcher()

    @classmethod
    def from_directory(cls, directory: Path) -> "SkillRegistry":
        if not directory.exists():
            return cls([])
        skills = [
            _load_skill(path)
            for path in sorted(directory.glob("*.md"))
        ]
        return cls(skills)

    def list(self) -> list[Skill]:
        return list(self._skills)

    def match(self, user_input: str, *, tool_registry: ToolRegistry) -> list[SkillMatch]:
        return self.matcher.match(
            user_input,
            skills=self.list(),
            tool_registry=tool_registry,
        )


def _load_skill(path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"Skill 文件缺少 frontmatter: {path}")
    _, rest = raw.split("---\n", 1)
    frontmatter, body = rest.split("\n---\n", 1)
    meta = _parse_frontmatter(frontmatter)
    return Skill(
        name=str(meta["name"]),
        description=str(meta["description"]),
        triggers=tuple(str(item) for item in meta["triggers"]),
        tools=tuple(str(item) for item in meta["tools"]),
        summary=str(meta["summary"]),
        body=body.strip(),
        path=path,
    )


def _parse_frontmatter(frontmatter: str) -> dict[str, object]:
    data: dict[str, object] = {}
    current_list_key: str | None = None

    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError("frontmatter list item 缺少键。")
            current_value = data.setdefault(current_list_key, [])
            if not isinstance(current_value, list):
                raise ValueError(f"{current_list_key} 不是列表。")
            current_value.append(_strip_quotes(stripped[2:].strip()))
            continue

        if ":" not in line:
            raise ValueError(f"frontmatter 行无法解析: {line}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not value:
            data[key] = []
            current_list_key = key
            continue
        data[key] = _strip_quotes(value)
        current_list_key = None

    required_keys = {"name", "description", "triggers", "tools", "summary"}
    missing = required_keys - set(data)
    if missing:
        raise ValueError(f"frontmatter 缺少字段: {sorted(missing)}")
    if not isinstance(data["triggers"], list) or not isinstance(data["tools"], list):
        raise ValueError("triggers 和 tools 必须是列表。")
    return data


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


__all__ = [
    "Skill",
    "SkillMatch",
    "SkillMatcher",
    "SkillRegistry",
]
