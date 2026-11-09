"""Render facts as the markdown memory block an agent reads.

Output is deterministic for a given input. Without scores (full memory, no question) sections come in a fixed
order and facts are ordered by priority (trust and freshness). With scores (ranked recall for a question) the most
relevant come first: sections ordered by their best-scoring fact, facts inside a section by score. Facts related
to a line directly (up to three) nest under it.
"""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from memx.config import MemxConfig
from memx.recall.decay import priority
from memx.types import RENDER_ORDER, SECTION_TITLES, Category, Fact

_DATED = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(19|20)\d{2}\b")
MAX_GROUP = 4  # a line and at most three facts nested under it


def _groups(facts: list[Fact], related: list[tuple[UUID, UUID]], key) -> list[list[Fact]]:
    """Group related facts for display: each group is a head and up to three facts linked to that head directly.

    Facts are taken in ``key`` order. Links are not followed transitively: "related" verdicts chain across a whole
    conversation (game -> team -> endorsement -> wedding), and a chain nests unrelated facts under one line, which
    makes the block harder to read and answer from.
    """
    linked: dict[UUID, set[UUID]] = {f.id: set() for f in facts}
    for a, b in related:
        if a in linked and b in linked and a != b:
            linked[a].add(b)
            linked[b].add(a)
    ordered = sorted(facts, key=key)
    placed: set[UUID] = set()
    groups: list[list[Fact]] = []
    for head in ordered:
        if head.id in placed:
            continue
        placed.add(head.id)
        group = [head]
        for f in ordered:
            if len(group) >= MAX_GROUP:
                break
            if f.id not in placed and f.id in linked[head.id]:
                group.append(f)
                placed.add(f.id)
        groups.append(group)
    return groups


def _line(fact: Fact, show_scope: bool, leaf: str, dated: bool = False) -> str:
    text = fact.text
    if fact.category == Category.event and not _DATED.search(text):
        # Most event facts already name their date ("on 2023-05-20", "in June 2023"); prefix only the ones that don't.
        when = fact.event_at or fact.observed_at
        text = f"({when.strftime('%Y-%m-%d')}) {text}"
    elif dated:
        # Related facts about one subject can disagree; the date they were said lets the reader pick the newer.
        text = f"{text} (said {fact.observed_at.strftime('%Y-%m-%d')})"
    if show_scope and fact.scope != leaf:
        text = f"{text} [{fact.scope}]"
    return text


def history_line(old: Fact) -> str:
    return f"earlier, later replaced: {old.text} (said {old.observed_at.strftime('%Y-%m-%d')})"


def render(scope: str, facts: list[Fact], related: list[tuple[UUID, UUID]], *, now: datetime, cfg: MemxConfig,
           show_scope: bool = False, history: dict[UUID, list[Fact]] | None = None,
           scores: dict[UUID, float] | None = None) -> str:
    """``history`` maps a fact to the earlier versions it replaced; they are listed under it, dated.
    ``scores`` (ranked recall) orders sections and facts by relevance instead of the fixed order."""
    if not facts:
        return ""
    # The ordering key: a query score when there is one, else priority (trust and freshness).
    rank = scores if scores is not None else {f.id: priority(f, now, cfg) for f in facts}
    visible = [f for f in facts if not f.flags.get("injection_suspect")]
    categories = [c for c in RENDER_ORDER if any(f.category == c for f in visible)]
    if scores is not None:
        best = {c: max(rank.get(f.id, 0.0) for f in visible if f.category == c) for c in categories}
        categories.sort(key=lambda c: (-best[c], RENDER_ORDER.index(c)))
    lines = [f"# Memory: {scope}"]
    for category in categories:
        section = [f for f in visible if f.category == category]
        member_key = lambda f: (-rank.get(f.id, 0.0), -f.observed_at.timestamp(), str(f.id))  # noqa: E731
        groups = _groups(section, related, member_key)  # heads in rank order, members in rank order
        lines.append("")
        lines.append(f"## {SECTION_TITLES[category]}")
        history = history or {}
        for group in groups:
            dated = cfg.render_related_dates and len(group) > 1
            lines.append(f"- {_line(group[0], show_scope, scope, dated)}")
            lines += [f"  - {history_line(old)}" for old in history.get(group[0].id, [])]
            for member in group[1:]:
                lines.append(f"  - {_line(member, show_scope, scope, dated)}")
                lines += [f"    - {history_line(old)}" for old in history.get(member.id, [])]
    return "\n".join(lines) + "\n"
