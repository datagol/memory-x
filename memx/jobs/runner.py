"""Job runner shared by background mode (inside the app process) and the ``memx worker`` process."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psycopg
from pydantic import ValidationError

from memx.context import Context
from memx.logs import event, exception, log_context
from memx.db import repo
from memx.jobs import queue
from memx.providers.http import ProviderError

log = logging.getLogger("memx.jobs")


def loop_time() -> float:
    return asyncio.get_running_loop().time()

Handler = Callable[[dict[str, Any]], Awaitable[None]]


class JobRunner:
    def __init__(self, ctx: Context, handlers: dict[str, Handler], *, concurrency: int | None = None,
                 worker_id: str | None = None, maintenance_seconds: float = 60.0, schedule_sweep: bool = True):
        self.ctx = ctx
        self.handlers = handlers
        self.concurrency = concurrency or ctx.config.worker_concurrency
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.maintenance_seconds = maintenance_seconds
        self.schedule_sweep = schedule_sweep
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._stop.clear()
        self._tasks = [asyncio.create_task(self._worker(i), name=f"memx-worker-{i}") for i in range(self.concurrency)]
        self._tasks.append(asyncio.create_task(self._listen(), name="memx-listen"))
        self._tasks.append(asyncio.create_task(self._maintain(), name="memx-maintain"))

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks = []

    async def run_forever(self) -> None:
        await self.start()
        await self._stop.wait()

    def wake(self) -> None:
        self._wake.set()

    async def _sleep(self, seconds: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        self._wake.clear()

    async def _worker(self, n: int) -> None:
        while not self._stop.is_set():
            try:
                async with self.ctx.pool.connection() as conn:
                    job = await queue.claim(conn, f"{self.worker_id}/{n}")
            except Exception:  # database blip: back off and retry
                exception(log, "job.claim_failed")
                await self._sleep(5)
                continue
            if job is None:
                await self._sleep(self.ctx.config.job_poll_seconds)
                continue
            await self.run_job(job)

    async def run_job(self, job: dict[str, Any]) -> None:
        payload = job.get("payload") or {}
        with log_context(job_id=job["id"], job_kind=job["kind"], ingest_id=payload.get("ingest_id")):
            await self._run_job(job)

    async def _run_job(self, job: dict[str, Any]) -> None:
        handler = self.handlers.get(job["kind"])
        started = loop_time()
        event(log, "job.started", logging.DEBUG, attempt=job.get("attempts"))
        try:
            if handler is None:
                raise ValueError(f"no handler for job kind {job['kind']!r}")
            await handler(job["payload"])
        except asyncio.CancelledError:
            raise  # lease expiry will requeue it
        except queue.Deferred as wait:
            async with self.ctx.pool.connection() as conn:
                await queue.defer(conn, job["id"], wait.seconds)
            event(log, "job.deferred", reason=str(wait),
                  retry_in_s=wait.seconds)
            return
        except Exception as exc:  # noqa: BLE001
            retryable = exc.retryable if isinstance(exc, ProviderError) else not isinstance(
                exc, (ValidationError, ValueError, TypeError, KeyError))
            message = f"{type(exc).__name__}: {exc}"
            async with self.ctx.pool.connection() as conn:
                dead = await queue.fail(conn, job, message, retryable=retryable)
                if job["kind"] == "ingest" and job["payload"].get("ingest_id"):
                    await repo.fail_ingest(conn, UUID(job["payload"]["ingest_id"]), message, final=dead)
            # A job that will be retried is a warning; one that gave up (dead letter) needs a person.
            event(log, "job.dead" if dead else "job.failed", logging.ERROR if dead else logging.WARNING,
                  error=message[:500], retryable=retryable, attempt=job.get("attempts"),
                  ms=int((loop_time() - started) * 1000))
            return
        async with self.ctx.pool.connection() as conn:
            await queue.complete(conn, job["id"])
        event(log, "job.done", logging.DEBUG, ms=int((loop_time() - started) * 1000))

    async def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                async with await psycopg.AsyncConnection.connect(self.ctx.config.dsn, autocommit=True) as conn:
                    await conn.execute(f"LISTEN {queue.CHANNEL}")
                    async for _ in conn.notifies():
                        self._wake.set()
                        if self._stop.is_set():
                            return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - polling still works without notifications
                log.debug("listener disconnected; retrying", exc_info=True)
                await asyncio.sleep(5)

    async def _maintain(self) -> None:
        while not self._stop.is_set():
            try:
                async with self.ctx.pool.connection() as conn:
                    requeued = await queue.reap(conn)
                    if requeued:
                        event(log, "job.requeued", logging.WARNING, jobs=requeued, reason="lease expired")
                        self._wake.set()
                    if self.schedule_sweep and "sweep" in self.handlers:
                        bucket = datetime.now(UTC).strftime("%Y%m%d%H")
                        await queue.enqueue(conn, "sweep", {}, f"sweep:{bucket}", max_attempts=3)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                exception(log, "job.maintenance_failed")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self.maintenance_seconds)
