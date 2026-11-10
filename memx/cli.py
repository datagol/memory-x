"""``memx`` command line: migrate, worker, sweep, insert, recall, render, export, serve."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

from memx.config import MemxConfig
from memx.db.migrate import migrate
from memx.logs import event
from memx.memory import Memory
from memx.types import Messages, Text, Turn


async def _worker(args: argparse.Namespace) -> None:
    cfg = MemxConfig()
    if cfg.providers != "demo" and not cfg.has_api_key():
        # Idle, not crash: a restart loop would hide the message among restarts. Set the key and restart.
        event(logging.getLogger("memx.worker"), "worker.no_key", logging.ERROR,
              detail="OPENROUTER_API_KEY is not set; the worker is idle. Add it to .env and run `docker compose up -d`.")
        await asyncio.Event().wait()
    mem = await Memory.connect(start_runner=False)
    runner = await mem.start_runner(concurrency=args.concurrency)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    cfg = mem.ctx.config
    event(logging.getLogger("memx.worker"), "worker.started", worker_id=runner.worker_id, slots=runner.concurrency,
          extractor_model=cfg.extractor_model, jev_model=cfg.jev_model,
          openrouter_key="set" if cfg.has_api_key() else "missing")
    await stop.wait()
    await mem.close()


async def _run(args: argparse.Namespace) -> None:
    cfg = MemxConfig()
    if args.cmd == "migrate":
        applied = await migrate(cfg.dsn, cfg.embedding_dim)
        print("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
        return
    if args.cmd == "worker":
        await _worker(args)
        return
    async with await Memory.connect(cfg, start_runner=False) as mem:
        if args.cmd == "sweep":
            print((await mem.sweep()).model_dump_json(indent=2))
        elif args.cmd == "insert":
            raw = Path(args.file).read_text() if args.file else args.text
            if raw is None:
                raw = sys.stdin.read()
            inp = Text(text=raw, source=args.source) if args.as_text else Messages(turns=[Turn(role="user", content=raw)])
            receipt = await mem.insert(args.scope, inp, mode="inline")
            print(receipt.model_dump_json(indent=2))
        elif args.cmd == "recall":
            result = await mem.recall(args.scope, args.query, budget_tokens=args.budget)
            if args.json:
                print(result.model_dump_json(indent=2))
            else:
                print(f"<!-- mode={result.mode} tokens={result.token_count} facts={len(result.facts)} -->")
                print(result.markdown)
        elif args.cmd == "render":
            print(await mem.render(args.scope, show_scope=args.show_scope))
        elif args.cmd == "history":
            from uuid import UUID
            print(json.dumps((await mem.history(UUID(args.fact_id))).model_dump(mode="json"), indent=2))
        elif args.cmd == "entities-reresolve":
            for row in await mem.reresolve_entities(args.tenant):
                verb = "merged into" if row["merged"] else "kept apart from"
                print(f"[{row['tenant']}] {row['short']!r} {verb} {row['into']!r} "
                      f"({row['how']}, Jev {row['p_same']:.0%} same)")
        elif args.cmd == "rereconcile":
            report = await mem.rereconcile(args.scope, dry_run=args.dry_run)
            for c in report.changes[: args.show]:
                print(f"{c['action']:<9} {c['reason']:<20} p={c['probability']:.2f}  {c['fact']!r}  (by {c['by']!r})")
            verb = "would restore" if args.dry_run else "restored"
            print(f"checked {report.checked}: {verb} {report.restored}, unshadowed {report.unshadowed}, "
                  f"kept {report.kept}; reasons {report.reasons}; left alone {report.skipped}")
        elif args.cmd == "expiry-reapply":
            report = await mem.reapply_event_expiry(args.scope, dry_run=args.dry_run)
            verb = "would change" if args.dry_run else "changed"
            print(f"checked {report['checked']} dated events: {verb} {report['changed']}, revived {report['revived']}"
                  f"{', skipped (duplicate) ' + str(report['skipped_duplicate']) if report['skipped_duplicate'] else ''}")
        elif args.cmd == "export":
            data, report = await mem.export_okf(args.scope)
            Path(args.out).write_bytes(data)
            print(f"wrote {report.files} files to {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="memx")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="apply database migrations")
    w = sub.add_parser("worker", help="process queued jobs")
    w.add_argument("--concurrency", type=int, default=None)
    sub.add_parser("sweep", help="expire stale events and prune old logs")
    i = sub.add_parser("insert", help="ingest text inline and print the receipt")
    i.add_argument("scope")
    i.add_argument("--text")
    i.add_argument("--file")
    i.add_argument("--as-text", action="store_true", help="treat input as a document instead of a user turn")
    i.add_argument("--source", default="cli")
    r = sub.add_parser("recall", help="recall memory for a query")
    r.add_argument("scope")
    r.add_argument("query", nargs="?")
    r.add_argument("--budget", type=int, default=None)
    r.add_argument("--json", action="store_true")
    rd = sub.add_parser("render", help="print the full memory block for a scope")
    rd.add_argument("scope")
    rd.add_argument("--show-scope", action="store_true")
    h = sub.add_parser("history", help="print a fact's version history")
    h.add_argument("fact_id")
    e = sub.add_parser("export", help="write an OKF bundle (zip) for a scope subtree")
    e.add_argument("scope")
    e.add_argument("out")
    er = sub.add_parser("entities-reresolve", help="re-ask Jev about short entity names and merge agreed ones")
    er.add_argument("--tenant", default=None, help="one tenant (the label under memory); default: all")
    rr = sub.add_parser("rereconcile", help="undo logged replacements the current policy would not make")
    rr.add_argument("--scope", default=None, help="this scope and everything under it; default: all")
    rr.add_argument("--dry-run", action="store_true", help="report only")
    rr.add_argument("--show", type=int, default=20, help="print this many changes")
    ex = sub.add_parser("expiry-reapply", help="recompute event expiry under the current settings")
    ex.add_argument("--scope", default=None, help="this scope and everything under it; default: all")
    ex.add_argument("--dry-run", action="store_true", help="report only")
    sub.add_parser("banner", help="print where the dashboard, API and MCP server are reachable")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    from memx import logs

    logs.configure()
    if args.cmd == "banner":
        from memx.banner import banner

        print(banner(), flush=True)
        return
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("memx_api.app:app", host=args.host, port=args.port)
        return
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
