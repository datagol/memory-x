"""Public entry point: ``Memory``."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, BinaryIO
from uuid import UUID

from pydantic import TypeAdapter

from memx import scope_settings
from memx.config import InsertMode, MemxConfig
from memx.context import Context
from memx.db import repo
from memx.db.migrate import migrate
from memx.db.pool import open_pool
from memx.export.okf import build_bundle
from memx.jobs import queue
from memx.jobs.runner import JobRunner
from memx.logs import event, exception, log_context
from memx.pipeline.commit import verdict_decisions
from memx.pipeline.ingest_job import run_ingest
from memx.pipeline.reconcile import judge_neighbors, plan_from
from memx.providers.embedder import Embedder, OpenRouterEmbedder
from memx.providers.extractor import Extractor, OpenRouterExtractor
from memx.providers.http import OpenRouterHTTP
from memx.providers.jev import JevClient, Judge
from memx.recall.engine import recall as _recall
from memx.recall.render import render as _render
from memx.scopes import ScopePath

log = logging.getLogger("memx.memory")
from memx.types import (
    Candidate,
    Category,
    ExportReport,
    Fact,
    FactHistory,
    ForgetReport,
    HistoryEntry,
    IngestInput,
    InsertReceipt,
    Messages,
    Recall,
    RecallMode,
    SweepReport,
    Text,
    Turn,
)

_INPUT = TypeAdapter(IngestInput)


class InputTooLarge(ValueError):
    pass


class NotFound(LookupError):
    pass


def _input_chars(inp: Messages | Text) -> int:
    if isinstance(inp, Messages):
        return sum(len(t.content) for t in inp.turns)
    return len(inp.text)


def _default_key(scope: str, payload: dict[str, Any], observed_at: datetime | None) -> str:
    blob = json.dumps({"scope": scope, "payload": payload,
                       "observed_at": observed_at.isoformat() if observed_at else None},
                      sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


class Memory:
    def __init__(self, ctx: Context, http: OpenRouterHTTP | None):
        self.ctx = ctx
        self._http = http
        self.runner: JobRunner | None = None

    # ---- lifecycle --------------------------------------------------------

    @classmethod
    async def connect(
        cls, config: MemxConfig | None = None, *, extractor: Extractor | None = None,
        embedder: Embedder | None = None, judge: Judge | None = None, run_migrations: bool = True,
        start_runner: bool | None = None,
    ) -> Memory:
        cfg = config or MemxConfig()
        if run_migrations:
            await migrate(cfg.dsn, cfg.embedding_dim)
        pool = await open_pool(cfg.dsn, cfg.pool_min, cfg.pool_max)
        http = None
        if extractor is None or embedder is None or judge is None:
            http = OpenRouterHTTP(cfg.api_key(), cfg.openrouter_base_url, cfg.provider_timeout_s)
        ctx = Context(
            config=cfg,
            pool=pool,
            extractor=extractor or OpenRouterExtractor(http, cfg.extractor_model),
            embedder=embedder or OpenRouterEmbedder(http, cfg.embedding_model, cfg.embedding_dim),
            judge=judge or JevClient(http, cfg.jev_model),
        )
        mem = cls(ctx, http)
        if start_runner if start_runner is not None else cfg.insert_mode == "background":
            await mem.start_runner()
        return mem

    def handlers(self) -> dict[str, Any]:
        async def ingest(payload: dict[str, Any]) -> None:
            await run_ingest(self.ctx, UUID(payload["ingest_id"]))

        async def sweep(_: dict[str, Any]) -> None:
            await self.sweep()

        return {"ingest": ingest, "sweep": sweep}

    async def start_runner(self, concurrency: int | None = None, **kwargs: Any) -> JobRunner:
        if self.runner is None:
            self.runner = JobRunner(self.ctx, self.handlers(), concurrency=concurrency, **kwargs)
            await self.runner.start()
        return self.runner

    async def close(self) -> None:
        if self.runner:
            await self.runner.stop()
            self.runner = None
        await self.ctx.pool.close()
        if self._http:
            await self._http.aclose()

    async def __aenter__(self) -> Memory:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ---- insert -----------------------------------------------------------

    async def insert(
        self, scope: str, input: Messages | Text | dict[str, Any], *, observed_at: datetime | None = None,
        idempotency_key: str | None = None, mode: InsertMode | None = None,
    ) -> InsertReceipt:
        cfg = self.ctx.config
        scope = str(ScopePath.parse(scope))
        inp = input if isinstance(input, (Messages, Text)) else _INPUT.validate_python(input)
        if _input_chars(inp) > cfg.max_input_chars:
            raise InputTooLarge(
                f"input is {_input_chars(inp)} characters; the limit is {cfg.max_input_chars}. Split it by session."
            )
        mode = mode or cfg.insert_mode
        payload = inp.model_dump(mode="json")
        key = idempotency_key or _default_key(scope, payload, observed_at)
        when = observed_at or datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        async with self.ctx.pool.connection() as conn:
            if isinstance(inp, Text):
                default_trust = (await scope_settings.resolve(conn, cfg, scope))["text_trust"]
                source, trust = inp.source, inp.trust if inp.trust is not None else default_trust
            else:
                source, trust = None, 1.0
            async with conn.transaction():
                await repo.ensure_scope(conn, scope)
                ingest_id, created = await repo.create_ingest(
                    conn, scope=scope, kind=inp.kind, payload=payload, source=source, trust=trust,
                    observed_at=when, idempotency_key=key, mode=mode,
                )
                if created:
                    await queue.enqueue(
                        conn, "ingest", {"ingest_id": str(ingest_id)}, f"ingest:{ingest_id}",
                        max_attempts=cfg.job_max_attempts, lease_seconds=cfg.job_lease_seconds,
                        status="running" if mode == "inline" else "queued",
                    )
                elif await repo.reopen_failed_ingest(conn, ingest_id, inline=mode == "inline"):
                    created = True  # resending a failed ingest retries it, resuming from saved progress
        if not created:
            event(log, "ingest.duplicate", ingest_id=str(ingest_id), scope=scope, mode=mode)
            return await self.receipt(ingest_id, with_facts=mode == "inline")
        event(log, "ingest.received", ingest_id=str(ingest_id), scope=scope, kind=inp.kind, mode=mode,
              chars=_input_chars(inp))
        if mode == "inline":
            try:
                await run_ingest(self.ctx, ingest_id, allow_batch=False)
            except Exception as exc:
                with log_context(ingest_id=str(ingest_id), scope=scope):
                    exception(log, "ingest.failed", mode=mode, error=f"{type(exc).__name__}: {str(exc)[:300]}")
                async with self.ctx.pool.connection() as conn:
                    await conn.execute(
                        "UPDATE memx_job SET status = 'dead', finished_at = now(), last_error = %s "
                        "WHERE idempotency_key = %s", (f"{type(exc).__name__}: {exc}", f"ingest:{ingest_id}"),
                    )
                    await repo.fail_ingest(conn, ingest_id, f"{type(exc).__name__}: {exc}", final=True)
                raise
            async with self.ctx.pool.connection() as conn:
                await conn.execute(
                    "UPDATE memx_job SET status = 'done', finished_at = now() WHERE idempotency_key = %s",
                    (f"ingest:{ingest_id}",),
                )
            return await self.receipt(ingest_id, with_facts=True)
        if self.runner:
            self.runner.wake()
        return await self.receipt(ingest_id)

    async def receipt(self, ingest_id: UUID, *, with_facts: bool = False) -> InsertReceipt:
        async with self.ctx.pool.connection() as conn:
            row = await repo.get_ingest(conn, ingest_id)
            if row is None:
                raise NotFound(f"ingest {ingest_id} not found")
            counts = row["receipt"] or {}
            facts = None
            if with_facts and counts.get("fact_ids"):
                facts = await repo.facts_by_ids(conn, [UUID(i) for i in counts["fact_ids"]])
        return InsertReceipt(
            ingest_id=ingest_id, scope=str(row["scope"]), status=row["status"], mode=row["mode"], error=row["error"],
            facts=facts,
            **{k: counts.get(k, 0) for k in ("added", "dropped", "historical", "superseded", "shadowed", "rejected",
                                             "held_back")},
        )

    async def wait(self, ingest_id: UUID, timeout: float = 30.0, *, with_facts: bool = True) -> InsertReceipt:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            receipt = await self.receipt(ingest_id, with_facts=with_facts)
            if receipt.status in ("committed", "failed"):
                return receipt
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"ingest {ingest_id} still {receipt.status} after {timeout}s")
            await asyncio.sleep(0.2)

    # ---- read -------------------------------------------------------------

    async def recall(
        self, scope: str, query: str | None = None, *, recent_turns: list[Turn] | None = None,
        budget_tokens: int | None = None, categories: set[Category] | None = None, k: int | None = None,
        force_mode: RecallMode | None = None, as_of: datetime | None = None, explain: bool = False,
    ) -> Recall:
        return await _recall(
            self.ctx, scope, query, recent_turns=recent_turns, budget_tokens=budget_tokens,
            categories=categories, k=k, force_mode=force_mode, as_of=as_of, explain=explain,
        )

    async def reapply_event_expiry(self, scope: str | None = None, *, dry_run: bool = False) -> dict[str, Any]:
        """Recompute when upcoming events expire under the current settings (server default and scope overrides).

        Facts stored under an earlier policy keep the expiry they were given; after the policy changes (event expiry
        was on by default before 2026-11-03) this brings them in line. Expired facts that would not have expired come
        back unless an identical active fact exists."""
        cfg = self.ctx.config
        now = datetime.now(UTC)
        report: dict[str, Any] = {"checked": 0, "changed": 0, "revived": 0, "skipped_duplicate": 0}
        async with self.ctx.pool.connection() as conn:
            rows = await repo.event_facts(conn, str(ScopePath.parse(scope)) if scope else None)
            ttl_by_scope: dict[str, Any] = {}
            for row in rows:
                report["checked"] += 1
                if row["scope"] not in ttl_by_scope:
                    ttl_by_scope[row["scope"]] = (await scope_settings.resolve(conn, cfg, row["scope"]))["event_ttl_days"]
                ttl = ttl_by_scope[row["scope"]]
                upcoming = row["event_at"].date() >= row["observed_at"].date()
                new = row["event_at"] + timedelta(days=ttl) if ttl and upcoming else None
                revive = row["status"] == "expired" and (new is None or new > now)
                if new == row["stale_after"] and not revive:
                    continue
                report["changed"] += 1
                if dry_run:
                    report["revived"] += revive
                    continue
                async with conn.transaction():
                    await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (row["scope"],))
                    ok = await repo.set_event_expiry(conn, row["id"], new, revive)
                if revive and ok:
                    report["revived"] += 1
                elif revive:
                    report["skipped_duplicate"] += 1
        return report

    # ---- per-scope settings ---------------------------------------------

    async def scope_settings(self, scope: str) -> dict[str, Any]:
        """Effective settings at a scope: each value with the scope it comes from ("default" = MEMX_*)."""
        async with self.ctx.pool.connection() as conn:
            resolved = await scope_settings.resolve(conn, self.ctx.config, scope)
        return resolved.as_dict(self.ctx.config)

    async def set_scope_settings(self, scope: str, **changes: Any) -> dict[str, Any]:
        """Override settings for a scope and everything below it. Pass ``None`` to inherit again.

        Keys: budget_tokens, event_ttl_days, text_trust, walk_enabled.
        """
        async with self.ctx.pool.connection() as conn:
            await scope_settings.update(conn, scope, changes)
        event(log, "scope_settings.changed", scope=scope, changes=changes)
        return await self.scope_settings(scope)

    async def render(self, scope: str, *, include_ancestors: bool = True, show_scope: bool = False) -> str:
        path = ScopePath.parse(scope)
        chain = path.chain_str() if include_ancestors else [str(path)]
        async with self.ctx.pool.connection() as conn:
            shadowed = await repo.shadowed_ids(conn, path.chain_str())
            facts = [f for f in await repo.chain_facts(conn, chain) if f.id not in shadowed]
            related = await repo.related_pairs(conn, [f.id for f in facts])
        return _render(str(path), facts, related, now=datetime.now(UTC), cfg=self.ctx.config, show_scope=show_scope)

    async def facts(
        self, scope: str, *, status: str | None = "active", category: Category | str | None = None,
        include_ancestors: bool = False, limit: int = 100, offset: int = 0,
    ) -> list[Fact]:
        path = ScopePath.parse(scope)
        scopes = path.chain_str() if include_ancestors else [str(path)]
        async with self.ctx.pool.connection() as conn:
            facts = await repo.list_facts(
                conn, scopes, status=status, category=str(category) if category else None, limit=limit, offset=offset,
            )
            return await repo.attach_entities(conn, facts)

    async def get_fact(self, fact_id: UUID) -> Fact:
        async with self.ctx.pool.connection() as conn:
            fact = await repo.get_fact(conn, fact_id)
            if fact is None:
                raise NotFound(f"fact {fact_id} not found")
            await repo.attach_entities(conn, [fact])
        return fact

    async def reresolve_entities(self, tenant: str | None = None) -> list[dict[str, Any]]:
        """Re-ask Jev about short names that may be the same as a longer known entity, and merge the agreed ones.
        Runs over every tenant unless one is given."""
        from memx.pipeline.entities import reresolve

        async with self.ctx.pool.connection() as conn:
            if tenant is None:
                cur = await conn.execute("SELECT DISTINCT tenant FROM memx_entity ORDER BY tenant")
                tenants = [r["tenant"] for r in await cur.fetchall()]
            else:
                tenants = [tenant]
            report: list[dict[str, Any]] = []
            for t in tenants:
                async with conn.transaction():
                    report += [{"tenant": t, **row} for row in await reresolve(
                        conn, self.ctx.judge, self.ctx.config, t, embed=self.ctx.embedder.embed)]
            return report

    async def rereconcile(self, scope: str | None = None, *, dry_run: bool = False):
        """Undo replacements and shadows that the current reconcile policy would not make, from the logged
        verdicts (no Jev calls, no re-extraction). Covers every scope, or ``scope`` and everything under it."""
        from memx.pipeline.rereconcile import rereconcile

        path = str(ScopePath.parse(scope)) if scope else None
        async with self.ctx.pool.connection() as conn:
            return await rereconcile(conn, self.ctx.config, scope=path, dry_run=dry_run)

    async def entities(self, scope: str, *, view: str = "subtree", limit: int = 200) -> list[dict[str, Any]]:
        """Entities mentioned by facts at a scope and its children (or its chain), with fact counts."""
        path = str(ScopePath.parse(scope))
        async with self.ctx.pool.connection() as conn:
            return await repo.list_entities(conn, path, view=view, limit=limit)

    async def history(self, fact_id: UUID) -> FactHistory:
        async with self.ctx.pool.connection() as conn:
            ids = await repo.version_chain(conn, fact_id)
            if not ids:
                raise NotFound(f"fact {fact_id} not found")
            facts = await repo.attach_entities(conn, await repo.facts_by_ids(conn, ids))
            decisions = await repo.decisions_for(conn, ids)
            links = await repo.links_for(conn, ids)
        entries = []
        for fact in facts:
            mine = [
                {**d, "role": "subject" if d["subject_fact_id"] == fact.id else "object"}
                for d in decisions if fact.id in (d["subject_fact_id"], d["object_fact_id"])
            ]
            entries.append(HistoryEntry(fact=fact, decisions=mine))
        return FactHistory(fact_id=fact_id, versions=entries, links=links)

    # ---- maintenance ------------------------------------------------------

    async def forget(self, *, scope: str | None = None, fact_id: UUID | None = None, descendants: bool = True,
                     requested_by: str | None = None) -> ForgetReport:
        if (scope is None) == (fact_id is None):
            raise ValueError("pass exactly one of scope or fact_id")
        async with self.ctx.pool.connection() as conn:
            async with conn.transaction():
                if scope is not None:
                    scope = str(ScopePath.parse(scope))
                    facts, ingests, scopes = await repo.forget_scope(conn, scope, descendants)
                else:
                    facts, ingests, scopes = await repo.forget_fact(conn, fact_id), 0, 0
                await repo.log_forget(conn, scope=scope, fact_id=fact_id, descendants=descendants, facts=facts,
                                      ingests=ingests, requested_by=requested_by)
        event(log, "forget", scope=scope, fact_id=str(fact_id) if fact_id else None,
              descendants=descendants if scope else None, facts_removed=facts, ingests_removed=ingests,
              scopes_removed=scopes, requested_by=requested_by)
        return ForgetReport(facts_removed=facts, ingests_removed=ingests, scopes_removed=scopes)

    async def promote(self, fact_id: UUID, to_scope: str) -> Fact:
        """Copy a fact to a broader scope, reconciling it there; the original is superseded by the copy."""
        cfg = self.ctx.config
        target = str(ScopePath.parse(to_scope))
        chain = ScopePath.parse(target).chain_str()
        async with self.ctx.pool.connection() as conn:
            fact = await repo.get_fact(conn, fact_id)
            if fact is None:
                raise NotFound(f"fact {fact_id} not found")
            if fact.status != "active":
                raise ValueError("only active facts can be promoted")
            if fact.scope == target:
                return fact
            embedding = await repo.get_fact_embedding(conn, fact_id)
            cur = await conn.execute("SELECT attrs->>'attribute' AS attribute FROM memx_fact WHERE id = %s", (fact_id,))
            attribute = (await cur.fetchone())["attribute"]
            # Same guards as an insert: the attribute label counts; the window rule does not (window -1), since the
            # copy is compared with facts at the broader scope, not with facts split from the same statement.
            candidate = Candidate(text=fact.text, speaker=fact.speaker or "user", category_hint=fact.category,
                                  attribute=attribute, window=-1)
            await repo.ensure_scope(conn, target)
            verdicts = await judge_neighbors(conn, self.ctx.judge, cfg, scope=target, chain=chain,
                                             candidate=candidate, category=fact.category, embedding=embedding,
                                             observed_at=fact.observed_at)
            plan = plan_from(verdicts, cfg, target, candidate)
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (target,))
                if plan.action == "drop":
                    new_id = plan.reinforce
                    await repo.reinforce(conn, new_id, fact.observed_at)
                else:
                    cur = await conn.execute(
                        """
                        INSERT INTO memx_fact (scope, category, text, text_hash, token_count, embedding,
                            embedding_model, status, superseded_by, ingest_id, speaker, source, trust,
                            extract_confidence, sort_confidence, sort_probs, flags, observed_at, event_at,
                            stale_after, attrs)
                        SELECT %s::ltree, category, text, text_hash, token_count, embedding, embedding_model,
                               %s, %s, ingest_id, speaker, source, trust, extract_confidence, sort_confidence,
                               sort_probs, flags, observed_at, event_at, stale_after,
                               (attrs - 'window') || jsonb_build_object('promoted_from', id::text)
                          FROM memx_fact WHERE id = %s
                        RETURNING id
                        """,
                        (target, plan.status, plan.superseded_by, fact_id),
                    )
                    new_id = (await cur.fetchone())["id"]
                    await conn.execute(
                        "INSERT INTO memx_fact_entity (fact_id, entity_id, surface) "
                        "SELECT %s, entity_id, surface FROM memx_fact_entity WHERE fact_id = %s ON CONFLICT DO NOTHING",
                        (new_id, fact_id),
                    )
                    active = await repo.still_active(
                        conn, plan.supersede + plan.supersede_below + plan.shadow + [r for r, _ in plan.related])
                    await repo.supersede(conn, [i for i in plan.supersede if i in active], new_id, target)
                    await repo.supersede_below(
                        conn, [i for i in plan.supersede_below if i in active and i != fact_id], new_id, target)
                    await repo.add_links(conn, [(new_id, b, "shadows", None) for b in plan.shadow if b in active]
                                         + [(new_id, b, "related", s) for b, s in plan.related if b in active])
                await conn.execute(
                    """
                    UPDATE memx_fact SET status = 'superseded', superseded_by = %s, superseded_at = now()
                     WHERE id = %s AND status = 'active'
                    """,
                    (new_id, fact_id),
                )
                await repo.add_decisions(conn, verdict_decisions(
                    {"id": fact.ingest_id}, candidate, plan.verdicts, target, new_id))
            promoted = await repo.get_fact(conn, new_id)
        event(log, "fact.promoted", fact_id=str(fact_id), new_fact_id=str(new_id), from_scope=fact.scope,
              scope=target)
        return promoted

    async def export_okf(self, scope: str, dest: BinaryIO | None = None) -> tuple[bytes, ExportReport]:
        path = str(ScopePath.parse(scope))
        async with self.ctx.pool.connection() as conn:
            rows = await repo.subtree_facts(conn, path)
            links = await repo.links_for(conn, [r["id"] for r in rows])
        data = build_bundle(path, rows, links)
        if dest is not None:
            dest.write(data)
        return data, ExportReport(files=len(rows) + 1, scope=path)

    async def sweep(self) -> SweepReport:
        cfg = self.ctx.config
        async with self.ctx.pool.connection() as conn:
            async with conn.transaction():
                counts = await repo.sweep(
                    conn, expired_retention_days=cfg.expired_retention_days,
                    decision_retention_days=cfg.decision_retention_days,
                    done_job_retention_days=cfg.done_job_retention_days,
                )
        return SweepReport(**counts)

    async def health(self) -> dict[str, Any]:
        async with self.ctx.pool.connection() as conn:
            await conn.execute("SELECT 1")
            jobs = await queue.stats(conn)
        return {"database": "ok", "jobs": jobs}


__all__ = ["Memory", "InputTooLarge", "NotFound"]
