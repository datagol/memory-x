from __future__ import annotations

import asyncio
import random
import ssl
from typing import Any

import httpx


class ProviderError(Exception):
    """A model provider call failed. ``retryable`` tells the job runner whether to try again later."""

    def __init__(self, message: str, *, retryable: bool, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class OpenRouterHTTP:
    """Thin async client for OpenRouter's chat, embeddings and System One endpoints."""

    def __init__(self, api_key: str, base_url: str, timeout_s: float = 60.0, attempts: int = 3):
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-Title": "memx"}
        self._timeout = timeout_s
        self._client = self._new_client()
        self._attempts = attempts

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self._base_url, headers=self._headers, timeout=self._timeout)

    async def _reset(self) -> None:
        """Drop pooled connections; after a network break (or the host sleeping) they are usually all dead."""
        old, self._client = self._client, self._new_client()
        try:
            await old.aclose()
        except Exception:  # noqa: BLE001 - closing a broken pool can itself fail
            pass

    @property
    def timeout(self) -> float:
        return self._timeout

    async def _pause(self, attempt: int, resp: httpx.Response | None = None) -> None:
        """Wait before the next attempt: as long as the provider asked for (Retry-After, capped), else backing off."""
        wait = 0.5 * 2**attempt + random.random() * 0.25
        asked = resp.headers.get("retry-after") if resp is not None else None
        if asked:
            try:
                wait = max(wait, min(float(asked), 30.0))
            except ValueError:  # an HTTP date; keep the backoff
                pass
        await asyncio.sleep(wait)

    async def post(self, path: str, body: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """``timeout`` overrides the client's for calls known to run long (a full extraction)."""
        last: ProviderError | None = None
        for attempt in range(self._attempts):
            resp = None
            try:
                resp = await self._client.post(path, json=body, timeout=timeout or self._timeout)
            except (httpx.HTTPError, ssl.SSLError) as exc:  # TLS record errors can escape httpx under load
                last = ProviderError(f"{path}: {type(exc).__name__}: {exc}", retryable=True)
                if isinstance(exc, (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError, httpx.ConnectError,
                                    ssl.SSLError)):
                    await self._reset()
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = ProviderError(
                        f"{path}: HTTP {resp.status_code}: {resp.text[:300]}",
                        retryable=True, status=resp.status_code,
                    )
                elif resp.status_code >= 400:
                    raise ProviderError(
                        f"{path}: HTTP {resp.status_code}: {resp.text[:500]}",
                        retryable=False, status=resp.status_code,
                    )
                else:
                    try:
                        data = resp.json()
                    except ValueError:  # truncated or keep-alive-only body
                        last = ProviderError(f"{path}: malformed JSON body: {resp.text[:200]!r}", retryable=True)
                        if attempt < self._attempts - 1:
                            await asyncio.sleep(0.5 * 2**attempt + random.random() * 0.25)
                        continue
                    err = data.get("error") if isinstance(data, dict) else None
                    if err:
                        code = err.get("code") if isinstance(err, dict) else None
                        retryable = isinstance(code, int) and (code == 429 or code >= 500)
                        last = ProviderError(f"{path}: {err}", retryable=retryable, status=code)
                        if not retryable:
                            raise last
                    else:
                        return data
            if attempt < self._attempts - 1:
                await self._pause(attempt, resp)
        assert last is not None
        raise last

    async def get(self, path: str) -> dict[str, Any]:
        last: ProviderError | None = None
        for attempt in range(self._attempts):
            try:
                resp = await self._client.get(path)
            except (httpx.HTTPError, ssl.SSLError) as exc:
                last = ProviderError(f"{path}: {type(exc).__name__}: {exc}", retryable=True)
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = ProviderError(f"{path}: HTTP {resp.status_code}: {resp.text[:300]}", retryable=True,
                                         status=resp.status_code)
                elif resp.status_code >= 400:
                    raise ProviderError(f"{path}: HTTP {resp.status_code}: {resp.text[:500]}", retryable=False,
                                        status=resp.status_code)
                else:
                    try:
                        return resp.json()
                    except ValueError:
                        last = ProviderError(f"{path}: malformed JSON body: {resp.text[:200]!r}", retryable=True)
            if attempt < self._attempts - 1:
                await asyncio.sleep(0.5 * 2**attempt + random.random() * 0.25)
        assert last is not None
        raise last

    async def aclose(self) -> None:
        await self._client.aclose()
