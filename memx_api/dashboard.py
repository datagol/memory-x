"""Read endpoints for the web dashboard: scope tree, stats, activity, traces, explanations, graph."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from memx import Memory, NotFound
from memx.db import repo
from memx.jobs import queue
from memx.pipeline.extract import parse_input
from memx.pipeline.windowing import make_windows
from memx.recall.render import render
from memx.scopes import ScopePath
from memx.types import CATEGORY_DEFINITIONS, RENDER_ORDER, TOPIC_DEFINITIONS


def memory(request: Request) -> Memory:
    mem = getattr(request.app.state, "memory", None)
    if mem is None:
        raise HTTPException(status_code=503, detail=getattr(request.app.state, "startup_error", None)
                            or "memx is not connected")
    return mem


Mem = Annotated[Memory, Depends(memory)]
router = APIRouter(prefix="/v1")


def _scope(value: str | None) -> str | None:
    return str(ScopePath.parse(value)) if value else None


def _preview(payload: dict[str, Any], limit: int = 160) -> str:
    if payload.get("kind") == "text":
        text = payload.get("text", "")
    else:
        text = " · ".join(f"{t.get('role')}: {t.get('content', '')}" for t in payload.get("turns", [])[:3])
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@router.get("/config")
async def config(mem: Mem) -> dict[str, Any]:
    c = mem.ctx.config
    return {
        "models": {"extractor": c.extractor_model, "embedding": c.embedding_model, "judge": c.jev_model},
        "insert_mode": c.insert_mode,
        "bands": {"low": c.band_low, "high": c.band_high, "guard_min_jaccard": c.guard_min_jaccard},
        "verdict": {"same_min": c.verdict_same_min, "replace_min": c.verdict_replace_min,
                    "covers_min": c.verdict_covers_min, "min_prob": c.verdict_min_prob,
                    "confirm": c.verdict_confirm, "confirm_band": list(c.verdict_confirm_band)},
        "sort": {"min_prob": c.sort_min_prob, "min_confidence": c.sort_min_confidence,
                 "injection_flag": c.injection_flag, "injection_reject": c.injection_reject},
        "recall": {"budget_tokens": c.budget_tokens, "budget_safety": c.budget_safety, "grade_floor": c.grade_floor,
                   "k_max": c.recall_k_max, "grade_max": c.grade_max, "grade_enabled": c.grade_enabled},
        "decay": {"half_lives": c.half_lives, "access_boost_cap": c.access_boost_cap},
        "trust": {"user": c.user_trust, "assistant": c.assistant_trust, "text_default": c.default_text_trust},
        "window_tokens": c.window_tokens,
        "event_ttl_days": c.event_ttl_days,
        "categories": [{"key": k.value, "definition": CATEGORY_DEFINITIONS[k.value]} for k in RENDER_ORDER],
        "entities": {"enabled": c.entities_enabled, "merge_min": c.entity_merge_min, "topic_min": c.topic_min},
        "walk": {"enabled": c.walk_enabled, "hops": c.walk_hops, "decay": c.walk_decay, "cap": c.walk_cap,
                 "seeds": c.walk_seeds, "hub_share": c.walk_hub_share, "hub_min_degree": c.walk_hub_min_degree,
                 "topic_weight": c.walk_topic_weight, "score_factor": c.walk_score_factor,
                 "grade_floor": c.walk_grade_floor},
        "topics": [{"key": k, "definition": v} for k, v in TOPIC_DEFINITIONS.items()],
    }


@router.get("/entities")
async def entities(mem: Mem, scope: str, view: str = "subtree", limit: Annotated[int, Query(le=2000)] = 200) -> list[dict]:
    rows = await mem.entities(scope, view="chain" if view == "chain" else "subtree", limit=limit)
    return [{**r, "id": str(r["id"])} for r in rows]


@router.get("/scopes")
async def scopes(mem: Mem, prefix: str | None = None, limit: Annotated[int, Query(le=5000)] = 2000) -> list[dict]:
    async with mem.ctx.pool.connection() as conn:
        return await repo.list_scopes(conn, _scope(prefix), limit)


@router.get("/stats")
async def stats(mem: Mem, scope: str | None = None) -> dict[str, Any]:
    async with mem.ctx.pool.connection() as conn:
        data = await repo.stats(conn, _scope(scope))
        data["jobs"] = await queue.stats(conn)
    return data


@router.get("/ingests")
async def ingests(mem: Mem, scope: str | None = None, limit: Annotated[int, Query(le=500)] = 50,
                  offset: int = 0) -> list[dict[str, Any]]:
    async with mem.ctx.pool.connection() as conn:
        rows = await repo.list_ingests(conn, _scope(scope), limit, offset)
    for r in rows:
        r["preview"] = _preview(r.pop("payload"))
    return rows


@router.get("/ingests/{ingest_id}/trace")
async def ingest_trace(ingest_id: UUID, mem: Mem) -> dict[str, Any]:
    cfg = mem.ctx.config
    async with mem.ctx.pool.connection() as conn:
        row = await repo.get_ingest(conn, ingest_id)
        if row is None:
            raise NotFound(f"ingest {ingest_id} not found")
        candidates = row["candidates"] or []
        ids: set[str] = set()
        for c in candidates:
            ids.update(n["id"] for n in c.get("neighbors") or [])
            ids.update(c.get("superseded_ids") or [])
            ids.update(c.get("shadowed_ids") or [])
            ids.update(c.get("related_ids") or [])
            for key in ("fact_id", "same_as", "superseded_by"):
                if c.get(key):
                    ids.add(c[key])
        facts = await repo.facts_by_ids(conn, [UUID(i) for i in ids])
    windows = make_windows(parse_input(row["payload"]), window_tokens=cfg.window_tokens,
                           context_turns=cfg.window_context_turns, chars_per_token=cfg.chars_per_token)
    per_window = {int(k): len(v) for k, v in (row["windows"] or {}).items()}
    return {
        "ingest": {k: row[k] for k in ("id", "scope", "kind", "mode", "status", "source", "trust", "observed_at",
                                       "created_at", "committed_at", "receipt", "error", "extractor_model",
                                       "tokens_in", "tokens_out", "payload")},
        "windows": [
            {"index": w.index, "units": len(w.units), "chars": sum(len(u.text) for u in w.units),
             "context_units": len(w.context), "roles": sorted({u.role for u in w.units}),
             "extracted": per_window.get(w.index)}
            for w in windows
        ],
        "candidates": candidates,
        "facts": {str(f.id): f.model_dump(mode="json") for f in facts},
        "config": {"band_low": cfg.band_low, "band_high": cfg.band_high},
    }


@router.get("/recalls")
async def recalls(mem: Mem, scope: str | None = None, limit: Annotated[int, Query(le=500)] = 50,
                  offset: int = 0) -> list[dict[str, Any]]:
    async with mem.ctx.pool.connection() as conn:
        return await repo.list_recalls(conn, _scope(scope), limit, offset)


@router.get("/recalls/{recall_id}")
async def recall_detail(recall_id: UUID, mem: Mem) -> dict[str, Any]:
    async with mem.ctx.pool.connection() as conn:
        row = await repo.get_recall(conn, recall_id)
        if row is None:
            raise NotFound(f"recall {recall_id} not found")
        facts = await repo.facts_by_ids(conn, list(row["fact_ids"]))
        related = await repo.related_pairs(conn, list(row["fact_ids"]))
    as_of = datetime.fromisoformat(row["explain"]["as_of"]) if row.get("explain") else datetime.now(UTC)
    row["markdown"] = render(row["scope"], facts, related, now=as_of, cfg=mem.ctx.config)
    row["facts"] = [f.model_dump(mode="json") for f in facts]
    return row


@router.get("/graph")
async def graph(mem: Mem, scope: str, view: str = "chain", include_superseded: bool = True,
                include_expired: bool = False, limit: Annotated[int, Query(le=2000)] = 400) -> dict[str, Any]:
    path = str(ScopePath.parse(scope))
    statuses = ["active"] + (["superseded"] if include_superseded else []) + (["expired"] if include_expired else [])
    async with mem.ctx.pool.connection() as conn:
        data = await repo.graph(conn, path, view="subtree" if view == "subtree" else "chain", statuses=statuses,
                                limit=limit)
        shadowed = await repo.shadowed_ids(conn, ScopePath.parse(path).chain_str())
    facts = data["facts"]
    scopes: set[str] = set()
    for f in facts:
        scopes.update(ScopePath.parse(f.scope).chain_str())
    scopes.add(path)
    edges = [{"source": str(r["a"]), "target": str(r["b"]), "kind": r["kind"], "similarity": r["similarity"]}
             for r in data["links"]]
    ids = {f.id for f in facts}
    edges += [{"source": str(f.superseded_by), "target": str(f.id), "kind": "supersedes", "similarity": None}
              for f in facts if f.superseded_by and f.superseded_by in ids]
    ents: dict[str, dict[str, Any]] = {}
    mentions = []
    for m in data["mentions"]:
        eid = str(m["entity_id"])
        e = ents.setdefault(eid, {"id": eid, "name": m["name"], "type": m["type"], "aliases": m["aliases"] or [],
                                  "facts": 0, "active": 0})
        e["facts"] += 1
        mentions.append({"fact": str(m["fact_id"]), "entity": eid})
    status_of = {str(f.id): f.status for f in facts}
    for m in mentions:
        if status_of.get(m["fact"]) == "active":
            ents[m["entity"]]["active"] += 1
    return {
        "scope": path, "view": view, "total": data["total"], "truncated": data["total"] > len(facts),
        "scopes": sorted(scopes, key=lambda s: (s.count("."), s)),
        "facts": [{**f.model_dump(mode="json"), "shadowed_here": f.id in shadowed} for f in facts],
        "edges": edges,
        "entities": sorted(ents.values(), key=lambda e: (-e["active"], -e["facts"], e["name"])),
        "mentions": mentions,
    }


@router.get("/jobs")
async def jobs(mem: Mem, status: str | None = None, limit: Annotated[int, Query(le=500)] = 50) -> list[dict]:
    async with mem.ctx.pool.connection() as conn:
        return await repo.list_jobs(conn, status, limit)
