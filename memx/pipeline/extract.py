"""Step 1: turn an ingest's raw input into deduplicated candidates."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from memx import usage as usage_tracker
from memx.context import Context
from memx.db import repo
from memx.jobs.queue import Deferred
from memx.pipeline.gate import GATE_VERSION, gate_window
from memx.pipeline.similarity import lexical_guard, text_hash
from memx.pipeline.windowing import Window, can_split, make_windows, split_window
from memx.providers.extractor import MAX_FACTS
from memx.providers.http import ProviderError
from memx.types import Candidate, EntityMention, IngestInput

_INPUT = TypeAdapter(IngestInput)
MAX_SPLIT_DEPTH = 4
MAX_CONTINUATIONS = 12  # up to 13 × 25 facts from one window before it is split instead


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
    failed = getattr(result, "failed", None)
    if failed and (not can_split(window) or depth >= MAX_SPLIT_DEPTH):
        raise ProviderError(failed, retryable=False)
    if failed:
        result.truncated = True  # retry as two smaller windows
    if result.truncated and (not can_split(window) or depth >= MAX_SPLIT_DEPTH):
        raise ProviderError(f"window {window.index} produced more facts than fit in one extractor response; "
                            "split the input into smaller pieces", retryable=False)
    facts, tin, tout = list(result.facts), result.tokens_in, result.tokens_out
    if not result.truncated and len(facts) >= cap and getattr(ctx.extractor, "supports_continue", False):
        # A capped but complete answer: keep it and ask for the facts after it, instead of discarding it and
        # extracting both halves again. Input is re-read; output, five times the price, is not regenerated.
        complete = cut_short = False
        for _ in range(MAX_CONTINUATIONS):
            more = await ctx.extractor.extract(window, observed_at=observed_at, total_windows=total,
                                               already=[f["text"] for f in facts])
            if (sink := usage_tracker.current()) is not None:
                sink.extract_continuations += 1
            tin, tout = tin + more.tokens_in, tout + more.tokens_out
            if more.truncated or getattr(more, "failed", None):
                cut_short = True  # the follow-up ran out of output tokens
                break
            seen = {f["text"].strip().lower() for f in facts}
            new = [f for f in more.facts if f["text"].strip().lower() not in seen]
            facts += new
            if len(more.facts) < cap:
                complete = True  # the model had fewer than a full answer left: nothing more to get
                break
            if not new:
                break  # a full answer of repeats: the model is stuck, so the rest can't be had this way
        if complete:
            return facts, tin, tout
        # Not known to be complete (cut short, stuck, or still full after every follow-up). Never return a partial
        # answer silently: split the window and extract each half, or say so when it can't be split.
        if not can_split(window) or depth >= MAX_SPLIT_DEPTH:
            if cut_short:
                raise ProviderError(f"window {window.index} produced more facts than fit in one extractor response; "
                                    "split the input into smaller pieces", retryable=False)
            if (sink := usage_tracker.current()) is not None:
                sink.extract_incomplete_windows += 1
            return facts, tin, tout
        result.tokens_in, result.tokens_out = tin, tout  # everything so far is wasted by the split below
        result.truncated = True
    if (result.truncated or len(result.facts) >= cap) and can_split(window) and depth < MAX_SPLIT_DEPTH:
        # This answer is discarded and both halves are extracted again: count what that costs.
        if (sink := usage_tracker.current()) is not None:
            sink.extract_splits += 1
            sink.extract_wasted_tokens_in += result.tokens_in
            sink.extract_wasted_tokens_out += result.tokens_out
        first, second = split_window(window, ctx.config.window_context_turns)
        a, b = await asyncio.gather(
            _extract_window(ctx, first, observed_at, total, depth + 1),
            _extract_window(ctx, second, observed_at, total, depth + 1),
        )
        return a[0] + b[0], result.tokens_in + a[1] + b[1], result.tokens_out + a[2] + b[2]
    return facts, tin, tout


async def _gate(ctx: Context, ingest: dict[str, Any], window: Window) -> tuple[Window | None, dict | None]:
    if not ctx.config.extract_gate:
        return window, None
    gated, record = await gate_window(ctx.judge, window, ctx.config.gate_skip_below)
    if not record:
        return gated, None
    if (sink := usage_tracker.current()) is not None:
        sink.gate_turns_skipped += record["skipped"]
    return gated, {"step": "gate", "question_version": GATE_VERSION, "ingest_id": ingest["id"], "result": record,
                   "latency_ms": record["latency_ms"]}


async def _save(ctx: Context, ingest: dict[str, Any], index: int, facts: list[dict[str, Any]], tin: int, tout: int,
                decision: dict | None) -> None:
    async with ctx.pool.connection() as conn:
        if decision:
            await repo.add_decisions(conn, [decision])
        await repo.save_window(conn, ingest["id"], index, facts, tin, tout, ctx.extractor.model)


async def _defer(ctx: Context, ingest: dict[str, Any], state: dict[str, Any], reason: str) -> None:
    """Save the batch state, with the usage counted so far (gate requests, Jev cost), and wait for the batch."""
    if (sink := usage_tracker.current()) is not None:
        state = {**state, "usage": sink.as_dict()}
    async with ctx.pool.connection() as conn:
        await repo.set_extract_batch(conn, ingest["id"], state)
    raise Deferred(ctx.config.extract_batch_poll_seconds, reason)


async def _extract_batched(ctx: Context, ingest: dict[str, Any], windows: list[Window], done: dict[str, list]) -> None:
    """Queued inserts: submit every pending window as one provider batch, come back when it is done.

    Raises ``Deferred`` while the batch runs. Windows whose batched answer hit the fact cap, ran out of tokens or
    failed, and every window of a batch that failed or expired, are finished directly (continuation or split), so
    the result is the same as the normal path."""
    observed_at: datetime = ingest["observed_at"]
    state = ingest.get("extract_batch") or {}
    if state.get("submitting") and not state.get("id"):
        # A crash between submitting and saving the batch id. The batch may exist, but its id is lost: extract the
        # rest directly instead of paying for a second batch.
        return
    if not state.get("id"):
        bodies: dict[str, dict] = {}
        decisions = []
        for w in windows:
            if str(w.index) in done:
                continue
            gated, decision = await _gate(ctx, ingest, w)
            if decision:
                decisions.append(decision)
            if gated is None:
                await _save(ctx, ingest, w.index, [], 0, 0, None)
                done[str(w.index)] = []
                continue
            bodies[f"w{w.index}"] = ctx.extractor.request_body(gated, observed_at=observed_at,
                                                               total_windows=len(windows), model=False)
        async with ctx.pool.connection() as conn:
            async with conn.transaction():
                await repo.add_decisions(conn, decisions)
                if bodies:  # mark before submitting, so a crash right after the submit can't submit twice
                    await repo.set_extract_batch(conn, ingest["id"], {"submitting": True, "windows": sorted(bodies)})
        if not bodies:
            return
        batch_id = await ctx.extractor.submit_batch(bodies)
        await _defer(ctx, ingest, {"id": batch_id, "windows": sorted(bodies)}, f"extraction batch {batch_id} submitted")
    results = await ctx.extractor.poll_batch(state["id"])
    if results is None:
        await _defer(ctx, ingest, state, f"extraction batch {state['id']} running")
    cap = min(ctx.config.window_max_candidates, MAX_FACTS)
    by_index = {w.index: w for w in windows}
    sink = usage_tracker.current()
    for cid in state["windows"]:
        index = int(cid[1:])
        if str(index) in done:
            continue
        res = results.get(cid)
        if res is not None and sink is not None:
            sink.extract_batched_windows += 1
            sink.extract_batch_tokens_in += res.tokens_in
            sink.extract_batch_tokens_out += res.tokens_out
        if res is None or res.truncated or len(res.facts) >= cap:
            # Finish this window the normal way. The gate ran when the batch was built; run it again only to get
            # the same turns (its decision is not logged twice).
            gated = by_index[index] if not ctx.config.extract_gate else (await _gate(ctx, ingest, by_index[index]))[0]
            facts, tin, tout = ([], 0, 0) if gated is None else await _extract_window(ctx, gated, observed_at,
                                                                                    len(windows))
            if res is not None:
                tin, tout = tin + res.tokens_in, tout + res.tokens_out
        else:
            facts, tin, tout = res.facts, res.tokens_in, res.tokens_out
        await _save(ctx, ingest, index, facts, tin, tout, None)
        done[str(index)] = facts


async def extract_windows(ctx: Context, ingest: dict[str, Any], *, allow_batch: bool = True) -> list[dict[str, Any]]:
    """Run extraction for every window not already persisted. Returns raw facts in window order.

    ``allow_batch`` is False when the caller is waiting for the result (an inline insert): a batch can take hours,
    and only the job runner can wait for one."""
    windows = windows_for(ctx, ingest["payload"])
    done: dict[str, list] = dict(ingest.get("windows") or {})
    observed_at: datetime = ingest["observed_at"]
    if (allow_batch and ctx.config.extract_batch and ingest.get("mode") == "queued"
            and getattr(ctx.extractor, "supports_batch", False)):
        await _extract_batched(ctx, ingest, windows, done)
    sem = asyncio.Semaphore(ctx.config.extract_concurrency)

    async def run(window: Window) -> None:
        gated, decision = await _gate(ctx, ingest, window)
        facts: list[dict[str, Any]] = []
        tin = tout = 0
        if gated is not None:
            async with sem:
                facts, tin, tout = await _extract_window(ctx, gated, observed_at, len(windows))
        await _save(ctx, ingest, window.index, facts, tin, tout, decision)
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
            attribute=(str(fact.get("attribute") or "").strip()[:60] or None),
            is_event=bool(fact.get("is_event")),
            time_text=fact.get("time_text"),
            event_date=_parse_date(fact.get("event_date")),
            confidence=float(fact.get("confidence", 1.0)),
            entities=[EntityMention(name=e["name"], type=e.get("type") or "other", aka=list(e.get("aka") or []))
                      for e in (fact.get("entities") or []) if isinstance(e, dict) and e.get("name")],
        ))
    return out


def dedupe_within(candidates: list[Candidate], embeddings: list[list[float]], band_high: float,
                  min_jaccard: float) -> tuple[list[Candidate], list[list[float]]]:
    """Collapse repeats across overlapping windows before anything touches stored memory."""
    if not candidates:
        return [], []
    kept: list[Candidate] = []
    kept_emb: list[list[float]] = []
    hashes: set[bytes] = set()
    # Unit vectors of the kept candidates, so one matrix product gives a candidate's cosine with all of them. A long
    # document yields a thousand candidates; comparing them pair by pair in Python took minutes on the event loop.
    unit = np.asarray(embeddings, dtype=np.float64).reshape(len(embeddings), -1)
    norms = np.linalg.norm(unit, axis=1, keepdims=True)
    unit = np.divide(unit, norms, out=np.zeros_like(unit), where=norms > 0)
    kept_unit = np.empty_like(unit)
    for i, (cand, emb) in enumerate(zip(candidates, embeddings)):
        digest = text_hash(cand.text)
        if digest in hashes:
            continue
        close = np.flatnonzero(kept_unit[: len(kept)] @ unit[i] >= band_high)
        twin = next((kept[j] for j in close if lexical_guard(cand.text, kept[j].text, min_jaccard)), None)
        if twin is not None:
            # The survivor keeps every entity either spelling mentioned.
            known = {e.name.lower() for e in twin.entities}
            twin.entities += [e for e in cand.entities if e.name.lower() not in known]
            continue
        hashes.add(digest)
        kept_unit[len(kept)] = unit[i]
        kept.append(cand)
        kept_emb.append(emb)
    for i, cand in enumerate(kept):
        cand.index = i
    return kept, kept_emb


async def ensure_candidates(ctx: Context, ingest: dict[str, Any], *,
                            allow_batch: bool = True) -> tuple[list[Candidate], dict[int, list[float]]]:
    """Extract (or reload) candidates. Returns them with any embeddings already computed this run."""
    if ingest.get("candidates") is not None:
        return [Candidate.model_validate(c) for c in ingest["candidates"]], {}
    raw = await extract_windows(ctx, ingest, allow_batch=allow_batch)
    candidates = to_candidates(raw)
    embeddings = await ctx.embedder.embed([c.text for c in candidates]) if candidates else []
    candidates, embeddings = dedupe_within(
        candidates, embeddings, ctx.config.band_high, ctx.config.guard_min_jaccard
    )
    async with ctx.pool.connection() as conn:
        await repo.save_candidates(conn, ingest["id"], [c.model_dump(mode="json") for c in candidates])
    return candidates, {c.index: e for c, e in zip(candidates, embeddings)}
