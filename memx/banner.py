"""The "memory-x is up" message printed by the compose `info` service once the API and MCP server are healthy.

Containers do not know which host address their ports are published on, so compose passes the same MEMX_PUBLISH and
MEMX_MCP_PUBLISH values it uses for the port mappings, and this module turns them into URLs.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence

LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}
QUIET_PATHS = ("/healthz", "/readyz")


class HealthCheckFilter:
    """Drop uvicorn access-log lines for health checks: compose probes every 10 seconds per service, which would
    bury the start-up message and real requests in `docker compose up` output."""

    def filter(self, record) -> bool:  # logging.Filter protocol
        args = record.args if isinstance(record.args, tuple) else ()
        return not (len(args) >= 3 and str(args[2]).split("?")[0] in QUIET_PATHS)


def quiet_health_checks() -> None:
    import logging

    logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, HealthCheckFilter) for f in logger.filters):
        logger.addFilter(HealthCheckFilter())


def url_for(publish: str, default_port: int) -> tuple[str, bool]:
    """``127.0.0.1:8000`` -> (``http://localhost:8000``, local only); ``8000`` or ``0.0.0.0:8000`` -> all interfaces."""
    value = (publish or "").strip() or f"127.0.0.1:{default_port}"  # compose's default: this machine only
    host, _, port = value.rpartition(":") if ":" in value else ("", "", value)
    port = port or str(default_port)
    if host in LOOPBACK:
        return f"http://localhost:{port}", True
    if host in ("", "0.0.0.0", "[::]", "::"):
        return f"http://localhost:{port}", False
    return f"http://{host}:{port}", False


def banner(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    ui, ui_local = url_for(env.get("MEMX_PUBLISH", ""), 8000)
    mcp, mcp_local = url_for(env.get("MEMX_MCP_PUBLISH", ""), 8100)
    token = bool(env.get("MEMX_API_TOKEN", "").strip())
    demo = env.get("MEMX_PROVIDERS", "").strip() == "demo"
    key = bool((env.get("OPENROUTER_API_KEY") or env.get("MEMX_OPENROUTER_API_KEY") or "").strip())
    header = ' --header "Authorization: Bearer <MEMX_API_TOKEN>"' if token else ""

    lines = [
        "",
        "  memory-x demo is up (stand-in models: no key, no cost, results only illustrative)" if demo
        else "  memory-x is up",
        "",
        f"  Dashboard    {ui}",
        f"  REST API     {ui}/v1        (reference: {ui}/docs)",
        f"  MCP server   {mcp}/mcp",
        "",
        "  Connect an MCP client",
        f"    Claude Code:  claude mcp add --transport http memx {mcp}/mcp{header}",
        f"    Others:       add an MCP server of type \"http\" with the URL {mcp}/mcp",
        "    Tools: recall, remember, remember_conversation, remember_document, list_facts, fact_history,",
        "           list_scopes, ingest_status, forget_fact. Default scope: "
        + (env.get("MEMX_MCP_SCOPE") or "memory.default"),
        "",
    ]
    if demo:
        pass
    elif key:
        lines.append("  OpenRouter key: set")
    else:
        lines += ["  OpenRouter key: NOT SET. Everything is running, but memory cannot be added or recalled yet.",
                  "    Put OPENROUTER_API_KEY=sk-or-... in .env next to docker-compose.yml, then run: docker compose up -d"]
    if token:
        lines.append("  Auth: on. Send Authorization: Bearer <MEMX_API_TOKEN>; paste it in the dashboard under Settings.")
    elif ui_local and mcp_local:
        lines.append("  Auth: off. The ports are reachable from this machine only.")
    else:
        lines.append("  Auth: OFF and a port is open to the network. Set MEMX_API_TOKEN in .env and run docker compose up -d.")
    if not mcp_local:
        lines.append("  MCP clients on other machines: add the name they use for this host to MEMX_MCP_ALLOWED_HOSTS.")
    if demo:
        lines += ["", "  Stop: docker compose -f compose.demo.yml down -v", ""]
    else:
        lines += ["", "  Logs: docker compose logs -f api worker mcp    Stop: docker compose down (data is kept)",
                  "  Show this again: docker compose exec api memx banner", ""]
    width = max(len(line) for line in lines) + 2
    rule = "  " + "=" * (width - 2)
    return "\n".join([rule, *lines, rule])


async def announce_when_ready(urls: Sequence[str], timeout: float = 120.0) -> None:
    """Print the banner once every URL answers 200 (the API itself, then the MCP server), so it appears after
    everything is reachable. Started by the API when MEMX_BANNER is set, which compose does."""
    import httpx

    loop = asyncio.get_running_loop()
    deadline, pending = loop.time() + timeout, list(urls)
    async with httpx.AsyncClient(timeout=2.0) as client:
        while pending and loop.time() < deadline:
            for url in list(pending):
                try:
                    if (await client.get(url)).status_code == 200:
                        pending.remove(url)
                except httpx.HTTPError:
                    pass
            if pending:
                await asyncio.sleep(1)
    text = banner()
    if pending:
        text += "\n  Not answering yet: " + ", ".join(pending) + " (see docker compose logs)"
    print(text, flush=True)
