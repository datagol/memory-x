"""Render facts as the markdown memory block an agent reads.

Output is deterministic for a given set of facts: section order is fixed, related facts are grouped, and
ordering inside a section depends on stored attributes only, never on query scores.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from memx.config import MemxConfig
from memx.recall.decay import priority
from memx.types import RENDER_ORDER, SECTION_TITLES, Category, Fact


def _groups(facts: list[Fact], related: list[tuple[UUID, UUID]]) -> list[list[Fact]]:
    parent: dict[UUID, UUID] = {f.id: f.id for f in facts}

    def find(x: UUID) -> UUID:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in related:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb, key=str)] = min(ra, rb, key=str)
    groups: dict[UUID, list[Fact]] = {}
    for f in facts:
        groups.setdefault(find(f.id), []).append(f)
    return list(groups.values())


def _line(fact: Fact, show_scope: bool, leaf: str) -> str:
    text = fact.text
    if fact.category == Category.event:
        when = fact.event_at or fact.observed_at
        text = f"({when.strftime('%Y-%m-%d')}) {text}"
    if show_scope and fact.scope != leaf:
        text = f"{text} [{fact.scope}]"
    return text


def render(scope: str, facts: list[Fact], related: list[tuple[UUID, UUID]], *, now: datetime, cfg: MemxConfig,
           show_scope: bool = False) -> str:
    if not facts:
        return ""
    prio = {f.id: priority(f, now, cfg) for f in facts}
    lines = [f"# Memory: {scope}"]
    for category in RENDER_ORDER:
        section = [f for f in facts if f.category == category and not f.flags.get("injection_suspect")]
        if not section:
            continue
        member_key = lambda f: (-prio[f.id], -f.observed_at.timestamp(), str(f.id))  # noqa: E731
        groups = [sorted(g, key=member_key) for g in _groups(section, related)]
        groups.sort(key=lambda g: (-max(prio[f.id] for f in g), min(str(f.id) for f in g)))
        lines.append("")
        lines.append(f"## {SECTION_TITLES[category]}")
        for group in groups:
            lines.append(f"- {_line(group[0], show_scope, scope)}")
            for member in group[1:]:
                lines.append(f"  - {_line(member, show_scope, scope)}")
    return "\n".join(lines) + "\n"
