from __future__ import annotations

from typing import Protocol

from memx.providers.http import OpenRouterHTTP, ProviderError


class Embedder(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenRouterEmbedder:
    """OpenAI text-embedding-3-* through OpenRouter's embeddings endpoint."""

    def __init__(self, http: OpenRouterHTTP, model: str, dim: int, batch_size: int = 128):
        self._http = http
        self.model = model
        self.dim = dim
        self._batch = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            chunk = texts[start : start + self._batch]
            body: dict = {"model": self.model, "input": chunk}
            if self.dim != 1536 or "3-large" in self.model:
                body["dimensions"] = self.dim
            data = await self._http.post("/embeddings", body)
            rows = sorted(data["data"], key=lambda r: r["index"])
            if len(rows) != len(chunk):
                raise ProviderError("embedding count mismatch", retryable=True)
            for row in rows:
                vec = row["embedding"]
                if len(vec) != self.dim:
                    raise ProviderError(
                        f"embedding dim {len(vec)} != configured {self.dim}", retryable=False
                    )
                out.append([float(x) for x in vec])
        return out
