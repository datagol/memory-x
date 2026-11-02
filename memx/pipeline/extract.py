"""Step 1: turn an ingest's raw input into deduplicated candidates."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

from pydantic import TypeAdapter

from memx.context import Context
from memx.db import repo
from memx.pipeline.similarity import cosine, lexical_guard, text_hash
from memx.pipeline.windowing import Window, make_windows, split_window
from memx.providers.extractor import MAX_FACTS
from memx.types import Candidate, EntityMention, IngestInput

_INPUT = TypeAdapter(IngestInput)
MAX_SPLIT_DEPTH = 2


def parse_input(payload: dict[str, Any]):
    return _INPUT.validate_python(payload)


def windows_for(ctx: Context, payload: dict[str, Any]) -> list[Window]:
    cfg = ctx.config
    return make_windows(
        parse_input(payload),
        window_tokens=cfg.window_tokens,
        context_turns=cfg.window_context_turns,
        chars_per_token=cfg.chars_per_token,
    )


async def _extract_window(ctx: Context, window: Window, observed_at: datetime, total: int, depth: int = 0):
    result = await ctx.extractor.extract(window, observed_at=observed_at, total_windows=total)
    cap = min(ctx.config.window_max_candidates, MAX_FACTS)
    if len(result.facts) >= cap and len(window.units) > 1 and depth < MAX_SPLIT_DEPTH:
        first, second = split_window(window, ctx.config.window_context_turns)
        a, b = await asyncio.gather(
            _extract_window(ctx, first, observed_at, total, depth + 1),
            _extract_window(ctx, second, observed_at, total, depth + 1),
        )
        return a[0] + b[0], result.tokens_in + a[1] + b[1], result.tokens_out + a[2] + b[2]
    return result.facts, result.tokens_in, result.tokens_out


async def extract_windows(ctx: Context, ingest: dict[str, Any]) -> list[dict[str, Any]]:
    """Run extraction for every window not already persisted. Returns raw facts in window order."""
    windows = windows_for(ctx, ingest["payload"])
    done: dict[str, list] = dict(ingest.get("windows") or {})
    sem = asyncio.Semaphore(ctx.config.extract_concurrency)
    observed_at: datetime = ingest["observed_at"]

    async def run(window: Window) -> None:
        async with sem:
            facts, tin, tout = await _extract_window(ctx, window, observed_at, len(windows))
        async with ctx.pool.connection() as conn:
            await repo.save_window(conn, ingest["id"], window.index, facts, tin, tout, ctx.extractor.model)
        done[str(window.index)] = facts

    await asyncio.gather(*(run(w) for w in windows if str(w.index) not in done))
    out = []
    for w in windows:
        for fact in done.get(str(w.index), []):
            out.append({**fact, "window": w.index})
    return out


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def to_candidates(raw: list[dict[str, Any]]) -> list[Candidate]:
    out = []
    for i, fact in enumerate(raw):
        out.append(Candidate(
            index=i,
            window=fact.get("window", 0),
            text=fact["text"],
            speaker=fact.get("speaker") or "user",
            category_hint=fact.get("category_hint"),
            is_event=bool(fact.get("is_event")),
            time_text=fact.get("time_text"),
            event_date=_parse_date(fact.get("event_date")),
            confidence=float(fact.get("confidence", 1.0)),
            entities=[EntityMention(name=e["name"], type=e.get("type") or "other")
                      for e in (fact.get("entities") or []) if isinstance(e, dict) and e.get("name")],
        ))
    return out


def dedupe_within(candidates: list[Candidate], embeddings: list[list[float]], band_high: float,
                  min_jaccard: float) -> tuple[list[Candidate], list[list[float]]]:
    """Collapse repeats across overlapping windows before anything touches stored memory."""
    kept: list[Candidate] = []
    kept_emb: list[list[float]] = []
    hashes: set[bytes] = set()
    for cand, emb in zip(candidates, embeddings):
        digest = text_hash(cand.text)
        if digest in hashes:
            continue
        twin = next((k for k, other in zip(kept, kept_emb)
                     if cosine(emb, other) >= band_high and lexical_guard(cand.text, k.text, min_jaccard)), None)
        if twin is not None:
            # The survivor keeps every entity either spelling mentioned.
            known = {e.name.lower() for e in twin.entities}
            twin.entities += [e for e in cand.entities if e.name.lower() not in known]
            continue
        hashes.add(digest)
        kept.append(cand)
        kept_emb.append(emb)
    for i, cand in enumerate(kept):
        cand.index = i
    return kept, kept_emb


async def ensure_candidates(ctx: Context, ingest: dict[str, Any]) -> tuple[list[Candidate], dict[int, list[float]]]:
    """Extract (or reload) candidates. Returns them with any embeddings already computed this run."""
    if ingest.get("candidates") is not None:
        return [Candidate.model_validate(c) for c in ingest["candidates"]], {}
    raw = await extract_windows(ctx, ingest)
    candidates = to_candidates(raw)
    embeddings = await ctx.embedder.embed([c.text for c in candidates]) if candidates else []
    candidates, embeddings = dedupe_within(
        candidates, embeddings, ctx.config.band_high, ctx.config.guard_min_jaccard
    )
    async with ctx.pool.connection() as conn:
        await repo.save_candidates(conn, ingest["id"], [c.model_dump(mode="json") for c in candidates])
    return candidates, {c.index: e for c, e in zip(candidates, embeddings)}
