"""Local skill loading.

Skills follow Anthropic's progressive-disclosure shape:

- L1 metadata (name + description + tools) is injected into the system
  prompt every turn. It is always cheap and always visible.
- L2 body is pulled on demand by the model via the ``read_skill`` tool.
  The framework does not match or inject skill bodies itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    tools: tuple[str, ...]
    body: str
    path: Path


class SkillRegistry:
    """Load skill markdown files from disk and expose them by name."""

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = tuple(skills)
        self._by_name = {skill.name: skill for skill in self._skills}

    @classmethod
    def from_directory(cls, directory: Path) -> "SkillRegistry":
        if not directory.exists():
            return cls([])
        skills: list[Skill] = []
        for path in sorted(directory.glob("*.md")):
            try:
                skills.append(_load_skill(path))
            except ValueError as exc:
                warnings.warn(f"跳过无效 skill 文件 {path.name}: {exc}", stacklevel=2)
        return cls(skills)

    def list(self) -> list[Skill]:
        return list(self._skills)

    def get_by_name(self, name: str) -> Skill | None:
        return self._by_name.get(name.strip())


def _load_skill(path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"Skill 文件缺少 frontmatter: {path}")
    frontmatter, body = _split_frontmatter(raw)
    meta = _parse_frontmatter(frontmatter)
    return Skill(
        name=str(meta["name"]),
        description=str(meta["description"]),
        tools=tuple(str(item) for item in meta["tools"]),
        body=body.strip(),
        path=path,
    )


def _split_frontmatter(raw: str) -> tuple[str, str]:
    rest = raw[len("---\n"):]
    if "\n---\n" in rest:
        return rest.split("\n---\n", 1)

    frontmatter_lines: list[str] = []
    body_start = 0
    current_list_key: str | None = None
    lines = rest.splitlines(keepends=True)
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            frontmatter_lines.append(raw_line)
            continue
        candidate = raw_line.lstrip()
        if candidate.startswith("- "):
            if current_list_key is None:
                body_start = index
                break
            frontmatter_lines.append(raw_line)
            continue
        if ":" in raw_line:
            key, raw_value = raw_line.split(":", 1)
            normalized_key = key.strip()
            if normalized_key.startswith("#"):
                normalized_key = normalized_key.lstrip("#").strip()
            if normalized_key:
                current_list_key = normalized_key if not raw_value.strip() else None
                frontmatter_lines.append(raw_line)
                continue
        body_start = index
        break
    else:
        body_start = len(lines)

    frontmatter = "".join(frontmatter_lines).strip("\n")
    body = "".join(lines[body_start:])
    if not frontmatter.strip():
        raise ValueError("frontmatter 缺少结束分隔符。")
    return frontmatter, body


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
        if key.startswith("#"):
            key = key.lstrip("#").strip()
        value = raw_value.strip()
        if not value:
            data[key] = []
            current_list_key = key
            continue
        data[key] = _strip_quotes(value)
        current_list_key = None

    required_keys = {"name", "description", "tools"}
    missing = required_keys - set(data)
    if missing:
        raise ValueError(f"frontmatter 缺少字段: {sorted(missing)}")
    if not isinstance(data["tools"], list):
        raise ValueError("tools 必须是列表。")
    return data


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


__all__ = [
    "Skill",
    "SkillRegistry",
]
