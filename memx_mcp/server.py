"""The memx MCP server: tools over ``memx.Memory``.

The tools are deliberately few and map one-to-one onto the library: recall before answering, remember what the user
says, inspect or forget individual facts. Everything else (scope settings, exports, entity maintenance) stays in the
dashboard and the REST API.

The memx connection is opened on the first tool call rather than at start-up, so the server starts (and its health
check passes) without an OpenRouter key; tools then answer with the same instructions the dashboard shows.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from memx import InputTooLarge, InvalidScope, Memory, MemxConfig, NotFound
from memx.db import repo
from memx.providers.http import ProviderError
from memx.types import Category, Fact, InsertReceipt, Messages, Text, Turn

DEFAULT_SCOPE = os.environ.get("MEMX_MCP_SCOPE", "memory.default")
NO_KEY = ("OPENROUTER_API_KEY is not set on the memx server. Add it to .env next to docker-compose.yml, then run "
          "`docker compose up -d`. One OpenRouter key covers extraction, embeddings and Jev.")

INSTRUCTIONS = f"""memx is long-term memory for this conversation's user, shared across sessions and agents.

- Call `recall` with the user's question before answering anything that may depend on what they told you earlier
  (preferences, people, projects, past decisions, dates). It returns a ready-to-read markdown block.
- Call `remember` when the user states something worth keeping: a preference, a fact about themselves or their work,
  a decision, a plan. Pass their words; memx extracts atomic facts, deduplicates them and resolves contradictions.
  Use `remember_conversation` for a whole exchange and `remember_document` for pasted documents.
- Memory is organised in scopes such as `memory.acme.alex`: dot-separated, most general first. A recall at a scope
  also sees its ancestors (`memory.acme`, `memory`). Omit `scope` to use `{DEFAULT_SCOPE}`.
- Inserts are processed in the background. `remember` waits a little and reports what was stored; if it is still
  running, check later with `ingest_status`.
"""

mcp = MCPServer(name="memx", title="memx memory", instructions=INSTRUCTIONS, version="0.1.0")


# ---- connection ----------------------------------------------------------------------------------------------------

class _State:
    memory: Memory | None = None
    lock = asyncio.Lock()


async def memory() -> Memory:
    if _State.memory is not None:
        return _State.memory
    async with _State.lock:
        if _State.memory is None:
            cfg = MemxConfig()
            if cfg.providers == "demo":
                from memx.testing import demo_providers

                extractor, embedder, judge = demo_providers(cfg.embedding_dim)
                _State.memory = await Memory.connect(cfg, extractor=extractor, embedder=embedder, judge=judge)
            elif not cfg.has_api_key():
                raise ToolError(NO_KEY)
            else:
                _State.memory = await Memory.connect(cfg)
    return _State.memory


async def close() -> None:
    if _State.memory is not None:
        await _State.memory.close()
        _State.memory = None


def _scope(scope: str | None) -> str:
    return scope or DEFAULT_SCOPE


async def _call(coro_fn, *args, **kwargs):
    """Run a memx call and turn its errors into tool errors the model can act on."""
    try:
        return await coro_fn(*args, **kwargs)
    except ToolError:
        raise
    except (InvalidScope, NotFound, InputTooLarge, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    except ProviderError as exc:
        raise ToolError(f"a model provider failed: {exc}") from exc


# ---- shapes returned to the model ------------------------------------------------------------------------------------

def _fact(f: Fact, **extra: Any) -> dict[str, Any]:
    out = {"id": str(f.id), "text": f.text, "category": f.category.value, "scope": f.scope, "status": f.status,
           "observed_at": f.observed_at.date().isoformat()}
    if f.event_at:
        out["event_date"] = f.event_at.date().isoformat()
    if f.speaker:
        out["speaker"] = f.speaker
    return {**out, **extra}


def _receipt(r: InsertReceipt) -> dict[str, Any]:
    out: dict[str, Any] = {"ingest_id": str(r.ingest_id), "status": r.status}
    if r.status == "committed":
        out.update({k: getattr(r, k) for k in ("added", "dropped", "superseded", "historical", "held_back")
                    if getattr(r, k)})
        if r.facts is not None:
            out["facts"] = [f.text for f in r.facts]
        if not r.added:
            out["note"] = "nothing new: memory already held this, or there was nothing durable to keep"
    elif r.status == "failed":
        out["error"] = r.error
    else:
        out["note"] = "still processing; call ingest_status with this ingest_id to see the result"
    return out


async def _insert_and_wait(scope: str, payload: Messages | Text, observed_at: datetime | None,
                           wait_seconds: float) -> dict[str, Any]:
    mem = await memory()
    receipt = await _call(mem.insert, _scope(scope), payload, observed_at=observed_at)
    if receipt.status in ("committed", "failed") or wait_seconds <= 0:
        return _receipt(receipt)
    try:
        return _receipt(await mem.wait(receipt.ingest_id, timeout=min(wait_seconds, 120)))
    except TimeoutError:
        return _receipt(await mem.receipt(receipt.ingest_id))


ScopeArg = Annotated[str | None, Field(description=f"Memory scope, e.g. memory.acme.alex. Default: {DEFAULT_SCOPE}")]
Wait = Annotated[float, Field(ge=0, le=120, description="Seconds to wait for processing before returning")]


class TurnIn(BaseModel):
    role: str = Field(description="user, assistant, or a participant's name")
    content: str
    at: datetime | None = Field(default=None, description="When it was said (ISO 8601), if known")


# ---- tools ---------------------------------------------------------------------------------------------------------

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False, openWorldHint=False))
async def recall(
    query: Annotated[str, Field(description="The user's question or the topic to remember about")],
    scope: ScopeArg = None,
    budget_tokens: Annotated[int | None, Field(ge=200, le=50_000, description="Max size of the memory block")] = None,
) -> dict[str, Any]:
    """Recall what memory holds that is relevant to a question. Returns a markdown block to read before answering,
    plus the facts it contains with their ids. When the whole memory fits the budget, all of it is returned."""
    mem = await memory()
    r = await _call(mem.recall, _scope(scope), query, budget_tokens=budget_tokens)
    return {
        "scope": r.scope, "mode": r.mode, "token_count": r.token_count, "truncated": r.truncated,
        "memory": r.markdown or "(nothing relevant in memory)",
        "facts": [_fact(f, score=round(f.score, 3)) for f in r.facts],
    }


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                      openWorldHint=False))
async def remember(
    text: Annotated[str, Field(description="What was said, in the speaker's words")],
    scope: ScopeArg = None,
    speaker: Annotated[str, Field(description="Who said it: user, assistant, or a person's name")] = "user",
    observed_at: Annotated[datetime | None, Field(description="When it was said (ISO 8601); default now")] = None,
    wait_seconds: Wait = 20,
) -> dict[str, Any]:
    """Store something a person said. memx extracts atomic facts, merges duplicates and supersedes outdated facts
    (a new job title replaces the old one, the old one stays in history). Returns the facts that were stored."""
    payload = Messages(turns=[Turn(role=speaker, content=text)])
    return await _insert_and_wait(scope, payload, observed_at, wait_seconds)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                      openWorldHint=False))
async def remember_conversation(
    turns: Annotated[list[TurnIn], Field(min_length=1, description="The exchange, oldest turn first")],
    scope: ScopeArg = None,
    observed_at: Annotated[datetime | None, Field(description="When the conversation happened; default now")] = None,
    wait_seconds: Wait = 30,
) -> dict[str, Any]:
    """Store a whole conversation. Facts are attributed to their speakers; the assistant's turns only contribute
    what was given to the user (recommendations, answers, plans)."""
    payload = Messages(turns=[Turn(role=t.role, content=t.content, at=t.at) for t in turns])
    return await _insert_and_wait(scope, payload, observed_at, wait_seconds)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                      openWorldHint=False))
async def remember_document(
    text: Annotated[str, Field(description="The document's text")],
    source: Annotated[str, Field(description="Where it came from: a file name, URL or title")],
    scope: ScopeArg = None,
    wait_seconds: Wait = 60,
) -> dict[str, Any]:
    """Store the facts in a document (notes, a spec, a directory, an email). Long documents are split and processed
    in parallel; document facts carry the scope's document trust, below what a person states directly."""
    return await _insert_and_wait(scope, Text(text=text, source=source), None, wait_seconds)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
async def ingest_status(ingest_id: Annotated[str, Field(description="The ingest_id a remember call returned")]
                        ) -> dict[str, Any]:
    """Check on an insert that was still processing, and see the facts it stored."""
    mem = await memory()
    return _receipt(await _call(mem.receipt, _uuid(ingest_id), with_facts=True))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
async def list_facts(
    scope: ScopeArg = None,
    category: Annotated[Category | None, Field(description="Only this category")] = None,
    include_ancestors: Annotated[bool, Field(description="Also list facts inherited from parent scopes")] = False,
    limit: Annotated[int, Field(ge=1, le=500)] = 50,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict[str, Any]:
    """List the active facts stored at a scope, newest first. Use recall to answer questions; use this to browse
    or to find a fact's id."""
    mem = await memory()
    facts = await _call(mem.facts, _scope(scope), category=category, include_ancestors=include_ancestors,
                        limit=limit, offset=offset)
    return {"scope": _scope(scope), "count": len(facts), "facts": [_fact(f) for f in facts]}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
async def fact_history(fact_id: Annotated[str, Field(description="A fact id from recall or list_facts")]
                       ) -> dict[str, Any]:
    """Show how a fact changed over time: every version, oldest first, and why each was replaced."""
    mem = await memory()
    h = await _call(mem.history, _uuid(fact_id))
    versions = []
    for entry in h.versions:
        why = [{"step": d.get("step"), "result": (d.get("result") or {}).get("label") or d.get("result")}
               for d in entry.decisions if d.get("step") == "verdict"]
        versions.append(_fact(entry.fact, **({"decisions": why} if why else {})))
    return {"fact_id": fact_id, "versions": versions}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
async def list_scopes(prefix: Annotated[str | None, Field(description="Only scopes under this one")] = None,
                      limit: Annotated[int, Field(ge=1, le=1000)] = 200) -> dict[str, Any]:
    """List memory scopes with their number of active facts."""
    mem = await memory()

    async def run() -> list[dict[str, Any]]:
        async with mem.ctx.pool.connection() as conn:
            return await repo.list_scopes(conn, prefix, limit)

    rows = await _call(run)
    return {"scopes": [{"scope": r["path"], "active_facts": r["active"]} for r in rows]}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True,
                                      openWorldHint=False))
async def forget_fact(fact_id: Annotated[str, Field(description="The id of the fact to delete")]) -> dict[str, Any]:
    """Permanently delete one fact (for example when the user asks you to forget something). The deletion is logged
    without its content. Use only when the user asked; to correct a fact, remember the new value instead."""
    mem = await memory()
    report = await _call(mem.forget, fact_id=_uuid(fact_id), requested_by="mcp")
    return {"facts_removed": report.facts_removed}


def _uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise ToolError(f"not a valid id: {value!r}") from exc
