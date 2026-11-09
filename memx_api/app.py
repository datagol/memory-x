"""Thin FastAPI wrapper around ``memx.Memory``."""

from __future__ import annotations

import hmac
import re
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError

from memx import InputTooLarge, InvalidScope, Memory, MemxConfig, NotFound
from memx.config import InsertMode
from memx.documents import DocumentError, read_document
from memx.providers.http import ProviderError
from memx.types import Category, Fact, FactHistory, ForgetReport, IngestInput, InsertReceipt, Recall, RecallMode, Turn
from memx_api import dashboard

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = MemxConfig()
    app.state.config, app.state.startup_error, app.state.memory = cfg, None, None
    if cfg.providers != "demo" and not cfg.has_api_key():
        # Start anyway, so the dashboard can say what is missing instead of the container restarting in a loop.
        app.state.startup_error = (
            "OPENROUTER_API_KEY is not set. Add it to .env (copy .env.example), then run `docker compose up -d`. "
            "One OpenRouter key covers extraction, embeddings and Jev. To try memx without a key, run the demo: "
            "docker compose -f compose.demo.yml up -d")
        yield
        return
    if cfg.providers == "demo":
        from memx.testing import demo_providers

        extractor, embedder, judge = demo_providers(cfg.embedding_dim)
        app.state.memory = await Memory.connect(cfg, extractor=extractor, embedder=embedder, judge=judge)
    else:
        app.state.memory = await Memory.connect(cfg)
    yield
    await app.state.memory.close()


app = FastAPI(title="memx", version="0.1.0", lifespan=lifespan)


def memory(request: Request) -> Memory:
    mem = getattr(request.app.state, "memory", None)
    if mem is None:
        raise HTTPException(status_code=503, detail=getattr(request.app.state, "startup_error", None)
                            or "memx is not connected")
    return mem


def require_token(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    mem = getattr(request.app.state, "memory", None)
    token = (mem.ctx.config if mem is not None else request.app.state.config).api_token
    if token is None or not token.get_secret_value():
        return
    expected = f"Bearer {token.get_secret_value()}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


Mem = Annotated[Memory, Depends(memory)]
v1 = [Depends(require_token)]


@app.exception_handler(NotFound)
async def _not_found(_: Request, exc: NotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(InvalidScope)
async def _bad_scope(_: Request, exc: InvalidScope) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ProviderError)
async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
    """A model provider refused or failed. Say which, in words, instead of a bare 500."""
    raw = str(exc)
    found = re.search(r'"message"\s*:\s*"([^"]+)"', raw)
    reason = found.group(1) if found else raw[:300]
    if exc.status in (401, 402, 403) or "limit" in reason.lower():
        detail = f"The model provider refused the request: {reason}"
    else:
        detail = f"The model provider failed ({reason}). Resend the same request to retry; progress is kept."
    return JSONResponse(status_code=503 if exc.retryable else 502, content={"detail": detail})


@app.exception_handler(InputTooLarge)
async def _too_large(_: Request, exc: InputTooLarge) -> JSONResponse:
    return JSONResponse(status_code=413, content={"detail": str(exc)})


class IngestBody(BaseModel):
    input: IngestInput
    observed_at: datetime | None = None
    idempotency_key: str | None = None
    mode: InsertMode | None = None


class RecallBody(BaseModel):
    query: str | None = None
    recent_turns: list[Turn] | None = None
    budget_tokens: int | None = Field(default=None, gt=0)
    categories: set[Category] | None = None
    k: int | None = Field(default=None, gt=0)
    force_mode: RecallMode | None = None
    as_of: datetime | None = None
    explain: bool = False


@app.post("/v1/scopes/{scope}/ingests", response_model=InsertReceipt, dependencies=v1)
async def create_ingest(scope: str, body: IngestBody, mem: Mem, response: Response,
                        idempotency_key: Annotated[str | None, Header()] = None) -> InsertReceipt:
    receipt = await mem.insert(scope, body.input, observed_at=body.observed_at,
                               idempotency_key=body.idempotency_key or idempotency_key, mode=body.mode)
    if receipt.status not in ("committed", "failed"):
        response.status_code = 202
    return receipt


MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@app.post("/v1/documents/extract", dependencies=v1)
async def extract_document(request: Request, mem: Mem, filename: str | None = None) -> dict[str, Any]:
    """Turn an uploaded PDF or text file (sent as the raw request body) into the text a ``Text`` ingest takes.

    Nothing is stored: the caller reviews the text and then posts it to ``/v1/scopes/{scope}/ingests``.
    """
    data = await request.body()
    if not data:
        raise HTTPException(status_code=422, detail="the request body is empty; send the file's bytes")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"the file is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        doc = read_document(data, filename)
    except DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    limit = mem.ctx.config.max_input_chars
    return {"text": doc.text, "kind": doc.kind, "filename": doc.filename, "pages": doc.pages,
            "empty_pages": doc.empty_pages, "chars": len(doc.text), "max_chars": limit,
            "too_large": len(doc.text) > limit}


@app.get("/v1/ingests/{ingest_id}", response_model=InsertReceipt, dependencies=v1)
async def get_ingest(ingest_id: UUID, mem: Mem) -> InsertReceipt:
    return await mem.receipt(ingest_id, with_facts=True)


@app.post("/v1/scopes/{scope}/recall", response_model=Recall, dependencies=v1)
async def recall(scope: str, body: RecallBody, mem: Mem) -> Recall:
    try:
        return await mem.recall(scope, body.query, recent_turns=body.recent_turns, budget_tokens=body.budget_tokens,
                                categories=body.categories, k=body.k, force_mode=body.force_mode,
                                as_of=body.as_of, explain=body.explain)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/scopes/{scope}/settings", dependencies=v1)
async def get_scope_settings(scope: str, mem: Mem) -> dict[str, Any]:
    return await mem.scope_settings(scope)


@app.patch("/v1/scopes/{scope}/settings", dependencies=v1)
async def patch_scope_settings(scope: str, body: dict[str, Any], mem: Mem) -> dict[str, Any]:
    """Set overrides for this scope and below. A key with ``null`` removes the override (inherit again)."""
    try:
        return await mem.set_scope_settings(scope, **body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="; ".join(
            f"{'.'.join(str(x) for x in e['loc']) or 'body'}: {e['msg']}" for e in exc.errors())) from exc


@app.get("/v1/scopes/{scope}/render", response_class=PlainTextResponse, dependencies=v1)
async def render(scope: str, mem: Mem, show_scope: bool = False, include_ancestors: bool = True) -> str:
    return await mem.render(scope, include_ancestors=include_ancestors, show_scope=show_scope)


@app.get("/v1/scopes/{scope}/facts", response_model=list[Fact], dependencies=v1)
async def list_facts(scope: str, mem: Mem, status: str | None = "active", category: Category | None = None,
                     include_ancestors: bool = False, limit: Annotated[int, Query(le=1000)] = 100,
                     offset: int = 0) -> list[Fact]:
    return await mem.facts(scope, status=status or None, category=category, include_ancestors=include_ancestors,
                           limit=limit, offset=offset)


@app.get("/v1/facts/{fact_id}", response_model=Fact, dependencies=v1)
async def get_fact(fact_id: UUID, mem: Mem) -> Fact:
    return await mem.get_fact(fact_id)


@app.get("/v1/facts/{fact_id}/history", response_model=FactHistory, dependencies=v1)
async def history(fact_id: UUID, mem: Mem) -> FactHistory:
    return await mem.history(fact_id)


class PromoteBody(BaseModel):
    to_scope: str


@app.post("/v1/facts/{fact_id}/promote", response_model=Fact, dependencies=v1)
async def promote(fact_id: UUID, body: PromoteBody, mem: Mem) -> Fact:
    try:
        return await mem.promote(fact_id, body.to_scope)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/v1/scopes/{scope}", response_model=ForgetReport, dependencies=v1)
async def forget_scope(scope: str, mem: Mem, descendants: bool = True,
                       x_requested_by: Annotated[str | None, Header()] = None) -> ForgetReport:
    return await mem.forget(scope=scope, descendants=descendants, requested_by=x_requested_by)


@app.delete("/v1/facts/{fact_id}", response_model=ForgetReport, dependencies=v1)
async def forget_fact(fact_id: UUID, mem: Mem, x_requested_by: Annotated[str | None, Header()] = None) -> ForgetReport:
    report = await mem.forget(fact_id=fact_id, requested_by=x_requested_by)
    if report.facts_removed == 0:
        raise NotFound(f"fact {fact_id} not found")
    return report


@app.get("/v1/scopes/{scope}/export.okf", dependencies=v1)
async def export(scope: str, mem: Mem) -> Response:
    data, _ = await mem.export_okf(scope)
    filename = f"memx-{scope}.okf.zip"
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(mem: Mem) -> dict[str, Any]:
    return await mem.health()


app.include_router(dashboard.router, dependencies=v1)


# The web dashboard (built from web/ into memx_api/static). Registered last so API routes win.
@app.get("/{path:path}", include_in_schema=False)
async def ui(path: str) -> Response:
    if path.startswith("v1/"):
        raise HTTPException(status_code=404, detail="not found")
    target = (STATIC / path).resolve()
    if path and target.is_file() and STATIC.resolve() in target.parents:
        return FileResponse(target)
    index = STATIC / "index.html"
    if index.exists():
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    return PlainTextResponse("The dashboard is not built. Run: cd web && pnpm install && pnpm build", status_code=404)
