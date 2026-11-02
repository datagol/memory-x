"""Export a scope subtree as an Open Knowledge Format bundle: one markdown file per fact with YAML frontmatter.

This is for portability (git, other tools). memx never reads it back for retrieval.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

OKF_VERSION = "0.2"


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(round(value, 4)) if isinstance(value, float) else str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return json.dumps(str(value), ensure_ascii=False)  # a JSON string is a valid YAML scalar


def _yaml(data: dict[str, Any], indent: int = 0) -> list[str]:
    pad = " " * indent
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines += _yaml(value, indent + 2)
        elif isinstance(value, list):
            lines.append(f"{pad}{key}: [{', '.join(_scalar(v) for v in value)}]")
        else:
            lines.append(f"{pad}{key}: {_scalar(value)}")
    return lines


def _title(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def fact_path(fact: dict[str, Any]) -> str:
    return f"{fact['scope'].replace('.', '/')}/{fact['category']}/{fact['id']}.md"


def build_bundle(scope: str, facts: list[dict[str, Any]], links: list[dict[str, Any]]) -> bytes:
    related: dict[Any, list[str]] = defaultdict(list)
    shadows: dict[Any, list[str]] = defaultdict(list)
    for link in links:
        target = related if link["kind"] == "related" else shadows
        target[link["a"]].append(str(link["b"]))
        if link["kind"] == "related":
            related[link["b"]].append(str(link["a"]))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        counts = Counter((f["scope"], f["category"]) for f in facts if f["status"] == "active")
        index = [
            "---", f'okf_version: "{OKF_VERSION}"', "type: index", f"title: {_scalar('memx export: ' + scope)}",
            f"timestamp: {_scalar(datetime.now(UTC))}", "---", "", f"# Memory export: {scope}", "",
        ]
        for (s, c), n in sorted(counts.items()):
            index.append(f"- [{s} / {c}]({s.replace('.', '/')}/{c}/) — {n} active facts")
        zf.writestr("index.md", "\n".join(index) + "\n")

        for fact in facts:
            front = {
                "type": fact["category"],
                "title": _title(fact["text"]),
                "description": fact["text"],
                "tags": [*fact["scope"].split("."), fact.get("speaker") or fact.get("source") or "unknown", "memx"],
                "timestamp": fact["observed_at"],
                "status": fact["status"],
                "superseded-by": str(fact["superseded_by"]) if fact.get("superseded_by") else None,
                "x-memx": {
                    "id": str(fact["id"]),
                    "scope": fact["scope"],
                    "event_at": fact.get("event_at"),
                    "stale_after": fact.get("stale_after"),
                    "trust": fact.get("trust"),
                    "confidence": fact.get("sort_confidence"),
                    "related": sorted(set(related.get(fact["id"], []))),
                    "shadows": sorted(shadows.get(fact["id"], [])),
                    "ingest_id": str(fact["ingest_id"]) if fact.get("ingest_id") else None,
                },
            }
            body = "\n".join(["---", *_yaml(front), "---", "", fact["text"], ""])
            zf.writestr(fact_path(fact), body)
    return buf.getvalue()
