"""Operational logs: one line per event, carrying the ids that tie it to a request, an ingest or a job.

Library code only calls ``logging.getLogger("memx...")`` and :func:`event`; nothing is printed unless the host
application configures logging. The memx entry points (API, worker, MCP server, CLI) call :func:`configure`, which
reads ``MEMX_LOG_LEVEL`` (default INFO) and ``MEMX_LOG_FORMAT`` (``text`` or ``json``).

What reaches INFO: ids, scopes, counts, sizes, timings, costs and outcomes. What never does: memory text, queries,
document content, API keys or tokens. Fact text and queries appear only at DEBUG, for local troubleshooting.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

_context: ContextVar[dict[str, Any]] = ContextVar("memx_log_context", default={})
_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}
_FIRST = ("event", "request_id", "job_id", "ingest_id", "recall_id", "scope")
_NOISY = ("httpx", "httpcore", "hpack", "openai", "mcp", "sse_starlette")
_configured = False

logging.getLogger("memx").addHandler(logging.NullHandler())


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach fields (ingest_id, job_id, request_id, scope …) to every memx log line inside this block,
    including lines from tasks started inside it."""
    token = bind(**fields)
    try:
        yield
    finally:
        _context.reset(token)


def bind(**fields: Any) -> Token:
    return _context.set({**_context.get(), **{k: v for k, v in fields.items() if v is not None}})


def unbind(token: Token) -> None:
    _context.reset(token)


def current() -> dict[str, Any]:
    return dict(_context.get())


def _extra(name: str, fields: dict[str, Any]) -> dict[str, Any]:
    # The context goes on the record itself, so any handler sees it (the host app's, pytest's, ours).
    return {**_context.get(), "event": name, **{k: v for k, v in fields.items() if v is not None}}


def event(logger: logging.Logger, name: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log a named event: the message is the event name, the details are fields (never free text)."""
    if logger.isEnabledFor(level):
        logger.log(level, name, extra=_extra(name, fields))


def exception(logger: logging.Logger, name: str, **fields: Any) -> None:
    """An ERROR event with the current exception's traceback."""
    logger.exception(name, extra=_extra(name, fields))


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    merged = {**getattr(record, "_memx_context", {}),
              **{k: v for k, v in record.__dict__.items() if k not in _STANDARD and not k.startswith("_")}}
    ordered = {k: merged.pop(k) for k in _FIRST if k in merged}
    return {**ordered, **merged}


class ContextFilter(logging.Filter):
    """Copies the current log context onto each record (at emit time, in the task that logged it)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record._memx_context = _context.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in _fields(record).items():
            out.setdefault(key, value)
        if record.exc_info:
            out["error"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s  %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        head, sep, tail = line.partition("\n")  # keep a traceback below the key=value pairs
        pairs = " ".join(f"{k}={_text(v)}" for k, v in _fields(record).items() if k != "event")
        return f"{head}  {pairs}{sep}{tail}" if pairs else line


def _text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return json.dumps(text) if (not text or " " in text or "=" in text) and isinstance(value, str) else text


def configure(level: str | None = None, fmt: str | None = None, *, force: bool = False) -> None:
    """Send memx and server logs to stdout in one format. Safe to call more than once."""
    global _configured
    if _configured and not force:
        return
    level_name = (level or os.environ.get("MEMX_LOG_LEVEL") or "INFO").upper()
    numeric = logging.getLevelName(level_name)
    numeric = numeric if isinstance(numeric, int) else logging.INFO
    use_json = (fmt or os.environ.get("MEMX_LOG_FORMAT") or "text").lower() == "json"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if use_json else TextFormatter())
    handler.addFilter(ContextFilter())
    handler._memx = True  # type: ignore[attr-defined]  # ours to replace; other handlers are left alone
    root = logging.getLogger()
    for old in [h for h in root.handlers if getattr(h, "_memx", False)]:
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(numeric)
    # Server loggers go through the same handler, so a JSON deployment gets JSON everywhere.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    # memx logs each HTTP request itself (with its request id and duration), so uvicorn's own access line is off.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Per-call lines from HTTP and SDK libraries only at DEBUG: one OpenRouter call per line drowns everything else.
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.DEBUG if numeric <= logging.DEBUG else logging.WARNING)
    _configured = True


# ---- HTTP request logging (pure ASGI, used by the API and the MCP server) ------------------------------------------

_SAFE_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:")
QUIET_PATHS = frozenset({"/healthz", "/readyz", "/health"})


def _request_id(headers: list[tuple[bytes, bytes]]) -> str:
    """The caller's X-Request-ID when it is a plain id (so a gateway's id flows through), otherwise a new one."""
    import uuid

    for key, value in headers:
        if key.lower() == b"x-request-id":
            given = value.decode("latin-1")[:64]
            if given and set(given) <= _SAFE_ID:
                return given
    return uuid.uuid4().hex[:16]


class RequestLogMiddleware:
    """One `http.request` line per API call (method, path, status, duration, request id), the request id bound to
    every memx line the call causes, and `X-Request-ID` on the response. Health checks are not logged; other
    non-API paths (dashboard files) only at DEBUG."""

    def __init__(self, app: Any, *, service: str, logged_prefixes: tuple[str, ...] = ("/v1", "/mcp")):
        self.app, self.service, self.prefixes = app, service, logged_prefixes
        self.log = logging.getLogger(f"memx.{service}")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        import time

        request_id = _request_id(scope.get("headers") or [])
        status = {"code": 500}

        async def send_with_id(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message = {**message, "headers": [*message.get("headers", []),
                                                   (b"x-request-id", request_id.encode())]}
            await send(message)

        path = scope.get("path", "")
        started = time.perf_counter()
        token = bind(request_id=request_id)
        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            exception(self.log, "http.error", method=scope.get("method"), path=path)
            raise
        finally:
            if path not in QUIET_PATHS:
                code = status["code"]
                level = (logging.ERROR if code >= 500 else logging.INFO) if path.startswith(self.prefixes) \
                    else logging.DEBUG
                event(self.log, "http.request", level, method=scope.get("method"), path=path, status=code,
                      ms=int((time.perf_counter() - started) * 1000))
            unbind(token)
