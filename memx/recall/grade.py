"""Rank candidate facts for a query with Jev Score questions, one request for the whole set.

Facts the walk added carry their path in the judge context: which fact they were reached from and through
which entity. Judged alone, a bridge fact ("Marcus Oyelaran is CTO") looks like background for a question about
ARCHIVE-02 and falls under the floor; judged with its bridge in view it can be rated for what it adds.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from memx.providers.jev import UNTRUSTED_NOTE, Judge, JudgeResult, Score, clip, envelope
from memx.timewords import age_phrase, event_phrase
from memx.types import Category, Fact, PathStep, Turn

GRADE_VERSION = "grade.v3"  # v2 added the link context for walked facts; v3 also for linked search hits

LEVELS = [
    "irrelevant: no bearing on the query or the conversation",
    "background: same general topic, unlikely to change the response",
    "useful: would plausibly improve or personalize the response",
    "important: directly answers or constrains the response",
    "critical: the response would be wrong, or would break a stated rule, without it",
]


def grade_state(query: str, recent_turns: list[Turn] | None, facts: list[Fact], scope: str,
                now: datetime, paths: dict[int, list[PathStep]] | None = None) -> dict[str, Any]:
    index = {f.id: i for i, f in enumerate(facts)}
    context = []
    for i, f in enumerate(facts):
        entry: dict[str, Any] = {
            "i": i,
            "category": f.category.value,
            "age": age_phrase(f.observed_at, now),
            "speaker": f.speaker or f.source or "unknown",
            "scope_relation": "same scope" if f.scope == scope else "broader (ancestor) scope",
        }
        if f.category == Category.event and f.event_at:
            entry["event_timing"] = event_phrase(f.event_at, now)
        path = (paths or {}).get(i)
        if path:
            last = path[-1]
            entry["found_by"] = "topic link" if last.kind == "topic" else "entity link"
            entry["via"] = last.via
            entry["from_fact"] = index.get(last.from_fact)
            entry["hops"] = len(path)
        else:
            entry["found_by"] = "search"
        context.append(entry)
    turns = [{"role": t.role, "text": clip(t.content, 500)} for t in (recent_turns or [])[-6:]]
    return envelope(
        {"facts": context},
        {"query": clip(query, 1000), "recent_turns": turns, "facts": [clip(f.text, 600) for f in facts]},
    )


def grade_question(i: int, walked: bool) -> Score:
    if walked:
        instructions = (
            f"How useful is `untrusted.facts[{i}]` for responding to `untrusted.query`, given "
            f"`untrusted.recent_turns`? It is linked to the fact at index `judge_context.facts[{i}].from_fact` "
            f"because both mention `judge_context.facts[{i}].via`. Judge what it adds to answering the query "
            f"together with that fact. Judge only that one fact. "
            + UNTRUSTED_NOTE
        )
    else:
        instructions = (
            f"How useful is `untrusted.facts[{i}]` for responding to `untrusted.query`, given "
            f"`untrusted.recent_turns`? Judge only that one fact. " + UNTRUSTED_NOTE
        )
    return Score(instructions=instructions, criteria=LEVELS)


async def grade(judge: Judge, query: str, recent_turns: list[Turn] | None, facts: list[Fact], scope: str,
                now: datetime, paths: dict[int, list[PathStep]] | None = None,
                both: set[int] | None = None) -> tuple[list[float], JudgeResult]:
    """Relevance in [0, 1] per fact, plus the raw judge result for the decision log.

    A fact with a path is judged together with the fact it links to. Facts in ``both`` (search hits that also
    have a link) are judged alone and linked, and keep the higher grade: the link may add relevance but must
    never take away the relevance the fact has on its own. ``res.answers["f{i}"]`` is the winning answer."""
    if not facts:
        return [], JudgeResult(answers={}, latency_ms=0)
    paths = paths or {}
    both = {i for i in (both or set()) if paths.get(i)}
    questions = {f"f{i}": grade_question(i, bool(paths.get(i)) and i not in both) for i in range(len(facts))}
    questions |= {f"f{i}_l": grade_question(i, True) for i in both}
    res = await judge.ask(grade_state(query, recent_turns, facts, scope, now, paths), questions)
    top = len(LEVELS) - 1
    out = []
    for i in range(len(facts)):
        ans = res.answers[f"f{i}"]
        if i in both and float(res.answers[f"f{i}_l"]["score"]) > float(ans["score"]):
            ans = res.answers[f"f{i}"] = {**res.answers[f"f{i}_l"], "via_link": True}
        out.append(max(0.0, min(1.0, float(ans["score"]) / top)))
    return out, res
