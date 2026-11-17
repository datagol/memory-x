"""Two-tier recall: return the whole memory when it fits the budget, otherwise rank and trim.

Every recall records an explanation (why this mode, and for each candidate its ranks, scores and why it was kept
or cut). It is stored in ``memx_recall_log.explain`` and returned when ``explain=True``.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from memx import scope_settings
from memx.context import Context
from memx.db import repo
from memx.db.pool import Borrowed
from memx.recall.decay import factors, final_score, priority
from memx.recall.grade import GRADE_VERSION, grade
from memx.recall.render import render
from memx.recall.walk import Reached, walk
from memx.scopes import ScopePath
from memx.tokens import estimate_tokens
from memx.types import Category, Fact, PathStep, Recall, RecallMode, ScoredFact, Turn

log = logging.getLogger("memx.recall")

# Title line plus a header for every possible section.
RENDER_OVERHEAD_TOKENS = 48


def _fit(items: list[ScoredFact], limit: float, k_max: int, why: dict[str, str]) -> list[ScoredFact]:
    """Take items in score order while they fit the budget and k. Records why each one was left out."""
    picked: list[ScoredFact] = []
    used = RENDER_OVERHEAD_TOKENS
    for item in items:
        if len(picked) >= k_max:
            why[str(item.id)] = "over_k"
            continue
        if used + item.token_count > limit:
            why[str(item.id)] = "over_budget"
            continue
        picked.append(item)
        used += item.token_count
        why[str(item.id)] = "picked"
    return picked


def _scored(fact: Fact, score: float, relevance: float | None = None, *, found_by: str = "search",
            path: list[PathStep] | None = None) -> ScoredFact:
    return ScoredFact(**fact.model_dump(), score=score, relevance=relevance, found_by=found_by, path=path or [])


def _path_json(path: list[PathStep]) -> list[dict[str, str]]:
    return [{"from_fact": str(s.from_fact), "via": s.via, "kind": s.kind} for s in path]


def _components(fact: Fact, now: datetime, cfg) -> dict[str, float]:
    return {name: round(value, 4) for name, value in factors(fact, now, cfg).items()}


async def recall(
    ctx: Context, scope: str, query: str | None = None, *, recent_turns: list[Turn] | None = None,
    budget_tokens: int | None = None, categories: set[Category] | None = None, k: int | None = None,
    force_mode: RecallMode | None = None, as_of: datetime | None = None, explain: bool = False,
) -> Recall:
    """``as_of`` evaluates age, decay and event expiry at that moment instead of now."""
    cfg = ctx.config
    started = time.perf_counter()
    path = ScopePath.parse(scope)
    scope = str(path)
    chain = path.chain_str()
    cats = sorted(c.value for c in categories) if categories else None
    now = as_of or datetime.now(UTC)
    recall_id = uuid.uuid4()
    decisions = []
    timings: dict[str, int] = {}
    why: dict[str, str] = {}
    candidates: list[dict[str, Any]] = []
    history_p: float | None = 1.0 if cfg.history_in_recall == "always" else None
    grade_error: str | None = None

    async with Borrowed(ctx.pool) as conn:  # handed back to the pool during the embed and grade calls
        settings = await scope_settings.resolve(conn, cfg, scope)
        budget = budget_tokens or settings["budget_tokens"]
        limit = budget * cfg.budget_safety
        shadowed = await repo.shadowed_ids(conn, chain)
        total, visible = await repo.chain_token_total(conn, chain, as_of)
        fits = total + RENDER_OVERHEAD_TOKENS <= limit
        if force_mode:
            mode: RecallMode = force_mode
            reason = f"forced {force_mode}"
        elif fits:
            mode, reason = "full", "the whole memory fits the budget"
        elif not query:
            mode, reason = "full", "over budget but no query to rank by, so trimmed by priority"
        else:
            mode, reason = "ranked", "the memory is larger than the budget, so candidates are ranked for the query"
        if mode == "ranked" and not query:
            raise ValueError("ranked recall needs a query")

        if mode == "full":
            t = time.perf_counter()
            facts = await repo.chain_facts(conn, chain, cats, as_of)
            for f in facts:
                if f.id in shadowed:
                    why[str(f.id)] = "shadowed"
            facts = [f for f in facts if f.id not in shadowed]
            items = sorted((_scored(f, priority(f, now, cfg)) for f in facts), key=lambda s: (-s.score, str(s.id)))
            picked = _fit(items, limit, k or len(items) or 1, why)
            timings["load_ms"] = int((time.perf_counter() - t) * 1000)
            for item in items:
                candidates.append({
                    "id": str(item.id), "text": item.text, "category": item.category.value, "scope": item.scope,
                    "score": round(item.score, 4), "status": why.get(str(item.id), "picked"), "found_by": "full",
                    **_components(item, now, cfg),
                })
            walk_info: dict[str, Any] | None = None
        else:
            t = time.perf_counter()
            async with conn.released():
                q_emb = (await ctx.embedder.embed([query]))[0]
            timings["embed_ms"] = int((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            rows = await repo.hybrid_candidates(
                conn, chain, q_emb, query, k_vector=cfg.recall_k_vector, k_fts=cfg.recall_k_fts, categories=cats,
                as_of=as_of,
            )
            timings["search_ms"] = int((time.perf_counter() - t) * 1000)
            info = {r["id"]: r for r in rows}
            for r in rows:
                if r["id"] in shadowed:
                    why[str(r["id"])] = "shadowed"
            rows = [r for r in rows if r["id"] not in shadowed]
            for r in rows[cfg.grade_max:]:
                why[str(r["id"])] = "not_graded"
            graded_rows = rows[: cfg.grade_max]
            facts = [repo.to_fact(r) for r in graded_rows]

            # The walk: follow shared entities and topics outward from the top search hits.
            expanded: list[Reached] = []
            walk_info = None
            if settings["walk_enabled"] and graded_rows:
                t = time.perf_counter()
                top_rrf = max(float(r["rrf"]) for r in graded_rows)
                seeds = {r["id"]: float(r["rrf"]) / top_rrf for r in graded_rows[: cfg.walk_seeds]}
                topics_of = {r["id"]: r.get("topic") for r in graded_rows[: cfg.walk_seeds]}
                result = await walk(conn, cfg, chain=chain, seeds=seeds, visible=visible, as_of=as_of,
                                    shadowed=shadowed, topics_of=topics_of)
                expanded = [r for r in result.ranked if r.fact.id not in info][: cfg.walk_cap]
                # Lower-ranked search hits the walk also reached keep their bridge for grading: on a small memory
                # search returns nearly everything, and "Priya Lindqvist leads the Security Team" judged alone
                # looks irrelevant to a question about Snowflake access.
                graded_ids = {r["id"] for r in graded_rows}
                bridged = {r.fact.id: r.path for r in result.ranked if r.fact.id in graded_ids and r.fact.id not in seeds}
                # Seeds can be bridges too: link each graded fact to the strongest higher-ranked hit sharing a
                # non-hub entity with it ("Priya Lindqvist leads the Security Team" ← "Priya said ... ticket").
                rank = {r["id"]: i for i, r in enumerate(graded_rows)}
                by_entity: dict[Any, list[Any]] = {}
                names: dict[Any, str] = {}
                for e in await repo.entity_edges(conn, chain, list(graded_ids), as_of):
                    if int(e["degree"]) <= result.info["hub_threshold"]:
                        by_entity.setdefault(e["entity_id"], []).append(e["fact_id"])
                        names[e["entity_id"]] = e["name"]
                for eid, fids in by_entity.items():
                    top = min(fids, key=lambda f: rank[f])
                    for f in fids:
                        if f == top:
                            continue
                        current = bridged.get(f)
                        if current is None or rank[top] < rank.get(current[-1].from_fact, len(rank)):
                            bridged[f] = [PathStep(from_fact=top, via=names[eid], kind="entity")]
                walk_info = {**result.info, "expanded": len(expanded), "bridged": len(bridged),
                             "over_cap": max(0, sum(1 for r in result.ranked if r.fact.id not in info) - cfg.walk_cap)}
                timings["walk_ms"] = walk_info["ms"]
            else:
                bridged = {}
            paths: dict[int, list[PathStep]] = {}
            for i, f in enumerate(facts):
                if f.id in bridged:
                    paths[i] = bridged[f.id]
            for r in expanded:
                paths[len(facts)] = r.path
                facts.append(r.fact)
            walked = {r.fact.id: r for r in expanded}

            t = time.perf_counter()
            graded = False
            if cfg.grade_enabled and facts:
                try:
                    searched = {i for i, f in enumerate(facts) if f.id in info and i in paths}
                    relevance, res = await grade(conn.judge(ctx.judge), query, recent_turns, facts, scope, now, paths, both=searched,
                                                 ask_history=cfg.history_in_recall == "when_asked")
                    if "history" in res.answers:
                        history_p = float(res.answers["history"].get("noul", 0.0))
                    decisions.append({
                        "step": "grade", "question_version": GRADE_VERSION, "recall_id": recall_id, "scope": scope,
                        "latency_ms": res.latency_ms,
                        "result": {"query": query, "scores": {str(f.id): r for f, r in zip(facts, relevance)},
                                   "walked": [str(i) for i in walked], "asks_history": history_p},
                    })
                    levels = {str(f.id): res.answers[f"f{i}"].get("probabilities") for i, f in enumerate(facts)}
                    link_won = {str(f.id): res.answers[f"f{i}"].get("via_link", False) for i, f in enumerate(facts)}
                    graded = True
                except Exception as exc:  # noqa: BLE001 - the judge is down or answered unreadably
                    # Search order is a worse ranking than the judge's, but a usable one; failing the recall would
                    # leave the agent with no memory at all.
                    grade_error = f"{type(exc).__name__}: {exc}"[:300]
                    log.warning("recall %s: grading failed, ranking by search order: %s", recall_id, grade_error)
            if not graded:
                top = max((r["rrf"] for r in graded_rows), default=1.0)
                relevance = [r["rrf"] / top for r in graded_rows] + [w.score for w in expanded]
                levels = {}
                link_won = {}
            timings["grade_ms"] = int((time.perf_counter() - t) * 1000)
            items = []
            for f, r in zip(facts, relevance):
                w = walked.get(f.id)
                if r < (cfg.walk_grade_floor if w else cfg.grade_floor):
                    why[str(f.id)] = "below_floor"
                    continue
                score = final_score(r, f, now, cfg) * (cfg.walk_score_factor if w else 1.0)
                items.append(_scored(f, score, r, found_by=w.found_by if w else "search", path=w.path if w else None))
            items.sort(key=lambda s: (-s.score, str(s.id)))
            picked = _fit(items, limit, k or cfg.recall_k_max, why)
            scores = {str(i.id): i.score for i in items}
            rel = {str(f.id): r for f, r in zip(facts, relevance)}
            for fid, row in info.items():
                f = repo.to_fact(row)
                candidates.append({
                    "id": str(fid), "text": row["text"], "category": row["category"], "scope": row["scope"],
                    "found_by": "search",
                    "vec_rank": row.get("vec_rank"), "fts_rank": row.get("fts_rank"),
                    "similarity": round(float(row["sim"]), 4), "rrf": round(float(row["rrf"]), 5),
                    "relevance": round(rel[str(fid)], 4) if str(fid) in rel else None,
                    "grade_levels": levels.get(str(fid)),
                    "score": round(scores[str(fid)], 4) if str(fid) in scores else None,
                    "status": why.get(str(fid), "picked"),
                    **({"path": _path_json(bridged[fid]), "hops": len(bridged[fid]),
                        "graded_via_link": bool(link_won.get(str(fid)))} if fid in bridged else {}),
                    **_components(f, now, cfg),
                })
            for w in expanded:
                fid = str(w.fact.id)
                candidates.append({
                    "id": fid, "text": w.fact.text, "category": w.fact.category.value, "scope": w.fact.scope,
                    "found_by": w.found_by, "path": _path_json(w.path), "walk_score": round(w.score, 4),
                    "hops": len(w.path), "vec_rank": None, "fts_rank": None,
                    "relevance": round(rel[fid], 4) if fid in rel else None,
                    "grade_levels": levels.get(fid),
                    "score": round(scores[fid], 4) if fid in scores else None,
                    "status": why.get(fid, "picked"),
                    **_components(w.fact, now, cfg),
                })

        related = await repo.related_pairs(conn, [f.id for f in picked])
        history: dict[Any, list[Fact]] = {}
        added: list[Any] = []  # history entries in the order they were granted room, so the last can go first
        if history_p is not None and history_p >= cfg.history_min_prob:
            versions = await repo.replaced_versions(conn, [f.id for f in picked], chain, as_of)
            for f in picked:  # score order: the most relevant facts get their history first
                for old in versions.get(f.id, []):
                    history.setdefault(f.id, []).append(old)
                    added.append(f.id)
        # Selection counted each fact's own tokens; the render adds more (history lines, "(said …)" dates on
        # related facts). Measure the real block and trim until it fits: history first, then the lowest-scored fact.
        while True:
            # Ranked recall answers a question: most relevant first. Full memory keeps the fixed order.
            markdown = render(scope, picked, related, now=now, cfg=cfg, history=history,
                              scores={f.id: f.score for f in picked} if mode == "ranked" else None)
            token_count = estimate_tokens(markdown, cfg.chars_per_token)
            if token_count <= limit or (not added and len(picked) <= 1):
                break
            if added:
                fid = added.pop()
                history[fid].pop()
                if not history[fid]:
                    del history[fid]
                continue
            dropped = picked.pop()
            why[str(dropped.id)] = "over_budget"
            for c in candidates:
                if c["id"] == str(dropped.id):
                    c["status"] = "over_budget"
            related = [(a, b) for a, b in related if dropped.id not in (a, b)]
        truncated = len(picked) < len(items)
        ids = [f.id for f in picked]
        timings["total_ms"] = int((time.perf_counter() - started) * 1000)
        explanation = {
            "mode": mode, "mode_reason": reason, "budget_tokens": budget, "budget_limit": round(limit),
            "chain": chain, "chain_tokens": total, "chain_facts": visible, "shadowed": len(shadowed),
            "grade_floor": cfg.grade_floor, "k_max": k or cfg.recall_k_max, "query": query,
            "as_of": now.isoformat(), "timings": timings, "related_pairs": [[str(a), str(b)] for a, b in related],
            "scope_settings": settings.overridden(),
            "walk": walk_info, "walk_score_factor": cfg.walk_score_factor, "walk_grade_floor": cfg.walk_grade_floor,
            "asks_history": history_p, "history": {str(k): [str(o.id) for o in v] for k, v in history.items()},
            "candidates": candidates,
            **({"grade_error": grade_error} if grade_error else {}),
        }
        async with conn.transaction():
            await repo.mark_accessed(conn, ids)
            await repo.add_decisions(conn, decisions)
            await repo.log_recall(
                conn, recall_id=recall_id, scope=scope, query=query, mode=mode, fact_ids=ids,
                scores={str(f.id): round(f.score, 4) for f in picked}, token_count=token_count,
                latency_ms=timings["total_ms"], explain=explanation,
            )
    return Recall(recall_id=recall_id, scope=scope, mode=mode, truncated=truncated, markdown=markdown,
                  facts=picked, token_count=token_count, explain=explanation if explain else None)
