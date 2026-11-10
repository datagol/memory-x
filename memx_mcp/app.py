"""HTTP entry point: the MCP server over streamable HTTP at /mcp, for `uvicorn memx_mcp.app:app`.

Auth uses the same bearer token as the REST API (MEMX_API_TOKEN): when it is set, every /mcp request must carry
``Authorization: Bearer <token>``. /healthz is always open and says whether the OpenRouter key is configured.
"""

from __future__ import annotations

import hmac
import json
import logging
import os

from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from memx import MemxConfig, logs
from memx.banner import quiet_health_checks
from memx_mcp.server import close, mcp

quiet_health_checks()


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_: Request) -> JSONResponse:
    cfg = MemxConfig()
    return JSONResponse({"status": "ok", "openrouter_key": cfg.providers == "demo" or cfg.has_api_key()})


def _security() -> TransportSecuritySettings:
    """Host/Origin checks against DNS rebinding. Defaults allow this machine and other compose services; set
    MEMX_MCP_ALLOWED_HOSTS to a comma list (host or host:*) when clients reach it by another name, or '*' to turn
    the check off (only behind a proxy that does its own)."""
    hosts = os.environ.get("MEMX_MCP_ALLOWED_HOSTS", "localhost,localhost:*,127.0.0.1,127.0.0.1:*,[::1]:*,mcp,mcp:*")
    if hosts.strip() == "*":
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    allowed = [h.strip() for h in hosts.split(",") if h.strip()]
    origins = [f"{scheme}://{h}" for h in allowed for scheme in ("http", "https")]
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=allowed,
                                     allowed_origins=origins)


class BearerAuth:
    """Require the API token on the MCP endpoint when one is configured."""

    def __init__(self, app: ASGIApp, token: str | None):
        self.app, self.expected = app, f"Bearer {token}".encode() if token else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.expected and scope["path"].startswith("/mcp"):
            given = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(given, self.expected):
                logs.event(logging.getLogger("memx.mcp"), "mcp.auth_rejected", logging.WARNING,
                           reason="missing token" if not given else "wrong token")
                body = json.dumps({"error": "missing or invalid bearer token (MEMX_API_TOKEN)"}).encode()
                await send({"type": "http.response.start", "status": 401, "headers": [
                    (b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": body})
                return
        if scope["type"] == "lifespan":
            await self._lifespan(scope, receive, send)
            return
        await self.app(scope, receive, send)

    async def _lifespan(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_closing(message):  # close the memx pool after the MCP session manager stops
            if message["type"] == "lifespan.shutdown.complete":
                await close()
            await send(message)

        await self.app(scope, receive, send_closing)


def build() -> ASGIApp:
    logs.configure()
    cfg = MemxConfig()
    logs.event(logging.getLogger("memx.mcp"), "mcp.started", auth="on" if cfg.api_token and
               cfg.api_token.get_secret_value() else "off", openrouter_key="set" if cfg.has_api_key() else "missing",
               default_scope=os.environ.get("MEMX_MCP_SCOPE", "memory.default"))
    token = cfg.api_token.get_secret_value() if cfg.api_token and cfg.api_token.get_secret_value() else None
    inner = mcp.streamable_http_app(stateless_http=True, json_response=True, host="0.0.0.0",
                                    transport_security=_security())
    return logs.RequestLogMiddleware(BearerAuth(inner, token), service="mcp")


app = build()
