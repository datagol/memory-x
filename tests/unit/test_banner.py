import logging

from memx.banner import HealthCheckFilter, banner, url_for


def test_url_for_publish_values():
    assert url_for("127.0.0.1:8000", 8000) == ("http://localhost:8000", True)
    assert url_for("", 8100) == ("http://localhost:8100", True)  # compose default
    assert url_for("127.0.0.1:8080", 8000) == ("http://localhost:8080", True)
    assert url_for("8000", 8000) == ("http://localhost:8000", False)  # bare port: all interfaces
    assert url_for("0.0.0.0:9000", 8000) == ("http://localhost:9000", False)
    assert url_for("10.0.0.5:8000", 8000) == ("http://10.0.0.5:8000", False)


def test_banner_defaults_without_key():
    text = banner({})
    assert "Dashboard    http://localhost:8000" in text
    assert "MCP server   http://localhost:8100/mcp" in text
    assert "claude mcp add --transport http memx http://localhost:8100/mcp\n" in text
    assert "OpenRouter key: NOT SET" in text and "docker compose up -d" in text
    assert "Auth: off. The ports are reachable from this machine only." in text


def test_banner_with_key_token_and_custom_ports():
    text = banner({"MEMX_PUBLISH": "127.0.0.1:8080", "MEMX_MCP_PUBLISH": "8200", "OPENROUTER_API_KEY": "sk-or-x",
                   "MEMX_API_TOKEN": "secret", "MEMX_MCP_SCOPE": "memory.acme"})
    assert "Dashboard    http://localhost:8080" in text
    assert '--header "Authorization: Bearer <MEMX_API_TOKEN>"' in text
    assert "secret" not in text and "sk-or-x" not in text
    assert "OpenRouter key: set" in text and "Auth: on." in text
    assert "MEMX_MCP_ALLOWED_HOSTS" in text  # MCP published on all interfaces
    assert "Default scope: memory.acme" in text


def test_health_checks_are_dropped_from_access_logs():
    f = HealthCheckFilter()
    record = lambda path: logging.LogRecord("uvicorn.access", logging.INFO, "", 0,  # noqa: E731
                                            '%s - "%s %s HTTP/%s" %d', ("1.2.3.4:5", "GET", path, "1.1", 200), None)
    assert not f.filter(record("/healthz"))
    assert not f.filter(record("/readyz"))
    assert f.filter(record("/v1/scopes"))
    assert f.filter(record("/mcp"))


def test_demo_banner_needs_no_key():
    text = banner({"MEMX_PROVIDERS": "demo", "MEMX_PUBLISH": "127.0.0.1:8001", "MEMX_MCP_PUBLISH": "127.0.0.1:8101"})
    assert "demo is up" in text and "OpenRouter key" not in text
    assert "http://localhost:8101/mcp" in text and "compose.demo.yml down" in text


async def test_announce_prints_after_urls_answer(capsys):
    from memx.banner import announce_when_ready

    await announce_when_ready(["http://127.0.0.1:9/never"], timeout=0.5)
    out = capsys.readouterr().out
    assert "memory-x is up" in out and "Not answering yet: http://127.0.0.1:9/never" in out
