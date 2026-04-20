"""Bundled read-only SQLite MCP server."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

app = FastMCP("SQLite Reader")

_ALLOWED_QUERY_PREFIXES = ("select", "with", "pragma", "explain")
_MAX_LIMIT = 1000


def _default_database_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "mcp" / "demo.sqlite3"


def _database_path() -> Path:
    raw_path = os.environ.get("SQLITE_PATH", "").strip()
    path = Path(raw_path).expanduser().resolve() if raw_path else _default_database_path()
    if not path.exists():
        raise RuntimeError(
            f"SQLite 文件不存在: {path}。"
            "如需使用其他数据库，请设置环境变量 SQLITE_PATH。"
        )
    return path


def _connect() -> sqlite3.Connection:
    database_path = _database_path()
    encoded_path = quote(str(database_path), safe="/")
    connection = sqlite3.connect(
        f"file:{encoded_path}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _normalize_query(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def _validate_table_name(table: str) -> str:
    candidate = table.strip()
    if not candidate:
        raise ValueError("`table` 不能为空。")
    return candidate


def _validate_query(sql: str) -> str:
    candidate = sql.strip()
    if not candidate:
        raise ValueError("`sql` 不能为空。")
    normalized = _normalize_query(candidate)
    if not normalized.startswith(_ALLOWED_QUERY_PREFIXES):
        raise ValueError("只允许只读查询: SELECT / WITH / PRAGMA / EXPLAIN。")
    return candidate


def _validate_limit(limit: int) -> int:
    if limit <= 0:
        raise ValueError("`limit` 必须大于 0。")
    if limit > _MAX_LIMIT:
        raise ValueError(f"`limit` 不能超过 {_MAX_LIMIT}。")
    return limit


def _serialize_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    return value


def _serialize_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: _serialize_value(row[key]) for key in row.keys()}


@app.tool()
def list_tables() -> list[str]:
    """Return non-system table names from the configured SQLite database."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    return [str(row["name"]) for row in rows]


@app.tool()
def describe_table(table: str) -> dict[str, Any]:
    """Describe one table's columns."""
    table_name = _validate_table_name(table)
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM pragma_table_info(?)
            ORDER BY cid
            """,
            (table_name,),
        ).fetchall()
    if not rows:
        raise ValueError(f"表不存在或没有列: {table_name}")
    return {
        "table": table_name,
        "columns": [
            {
                "cid": int(row["cid"]),
                "name": str(row["name"]),
                "type": str(row["type"] or ""),
                "notnull": bool(row["notnull"]),
                "default_value": row["dflt_value"],
                "primary_key": bool(row["pk"]),
            }
            for row in rows
        ],
    }


@app.tool()
def query(sql: str, limit: int = 100) -> dict[str, Any]:
    """Run one read-only SQL query and return up to ``limit`` rows."""
    statement = _validate_query(sql)
    row_limit = _validate_limit(limit)
    with _connect() as connection:
        cursor = connection.execute(statement)
        columns = [column[0] for column in cursor.description or ()]
        fetched = cursor.fetchmany(row_limit + 1)
    rows = [_serialize_row(row) for row in fetched[:row_limit]]
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(fetched) > row_limit,
        "limit": row_limit,
    }


if __name__ == "__main__":
    app.run(transport="stdio")
