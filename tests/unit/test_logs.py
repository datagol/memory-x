"""memx.logs: formats, context ids, levels and request ids."""

import io
import json
import logging

import pytest

from memx import logs


@pytest.fixture
def capture():
    """Configure memx logging into a buffer; restore the root logger afterwards."""
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)

    def setup(fmt: str, level: str = "INFO") -> io.StringIO:
        logs.configure(level, fmt, force=True)
        buf = io.StringIO()
        next(h for h in root.handlers if getattr(h, "_memx", False)).stream = buf
        return buf

    yield setup
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])


def test_json_lines_carry_context_and_fields(capture):
    buf = capture("json")
    log = logging.getLogger("memx.test")
    with logs.log_context(request_id="req-1", scope="memory.acme"):
        with logs.log_context(ingest_id="ing-1"):
            logs.event(log, "ingest.committed", added=2, ms=40)
        logs.event(log, "recall", facts=3)
    first, second = (json.loads(line) for line in buf.getvalue().splitlines())
    assert first["event"] == "ingest.committed" and first["level"] == "info"
    assert (first["request_id"], first["ingest_id"], first["scope"], first["added"]) == ("req-1", "ing-1",
                                                                                       "memory.acme", 2)
    assert "ingest_id" not in second and second["request_id"] == "req-1"
    assert list(first)[:5] == ["ts", "level", "logger", "msg", "event"]


def test_text_lines_are_key_value_and_keep_tracebacks(capture):
    buf = capture("text")
    log = logging.getLogger("memx.test")
    with logs.log_context(job_id=7):
        logs.event(log, "job.failed", logging.WARNING, error="Provider: HTTP 502", retryable=True)
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("job.crashed", extra={"event": "job.crashed"})
    lines = buf.getvalue().splitlines()
    assert lines[0].endswith('job.failed  job_id=7 error="Provider: HTTP 502" retryable=true')
    assert "job.crashed  job_id=7" in lines[1] and "ValueError: boom" in buf.getvalue()


def test_debug_content_stays_out_at_info(capture):
    buf = capture("json", "INFO")
    logs.event(logging.getLogger("memx.recall"), "recall.query", logging.DEBUG, query="what is my salary?")
    assert buf.getvalue() == ""
    buf = capture("json", "DEBUG")
    logs.event(logging.getLogger("memx.recall"), "recall.query", logging.DEBUG, query="what is my salary?")
    assert json.loads(buf.getvalue())["query"] == "what is my salary?"


def test_configure_quiets_http_libraries_and_uvicorn_access(capture):
    capture("text", "INFO")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("uvicorn.access").level == logging.WARNING
    capture("text", "DEBUG")
    assert logging.getLogger("httpx").level == logging.DEBUG


def test_request_ids():
    assert logs._request_id([(b"x-request-id", b"gw-123_abc")]) == "gw-123_abc"
    generated = logs._request_id([(b"x-request-id", b"bad id with spaces\n")])
    assert len(generated) == 16 and generated != "bad id with spaces"
    assert len(logs._request_id([])) == 16


def test_configure_keeps_handlers_it_did_not_install(capture):
    root = logging.getLogger()
    theirs = logging.NullHandler()
    root.addHandler(theirs)
    try:
        capture("text")
        capture("json")
        assert theirs in root.handlers
        assert sum(getattr(h, "_memx", False) for h in root.handlers) == 1
    finally:
        root.removeHandler(theirs)
