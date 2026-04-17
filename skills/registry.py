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
        skills = [
            _load_skill(path)
            for path in sorted(directory.glob("*.md"))
        ]
        return cls(skills)

    def list(self) -> list[Skill]:
        return list(self._skills)

    def get_by_name(self, name: str) -> Skill | None:
        return self._by_name.get(name.strip())


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
        tools=tuple(str(item) for item in meta["tools"]),
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
