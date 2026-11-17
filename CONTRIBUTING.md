# Contributing to memx

## Set up

```bash
docker compose up -d postgres                 # Postgres 16 with pgvector on localhost:5433
uv venv --python 3.12 && uv pip install -e '.[dev]'
.venv/bin/pytest                              # stand-in models: no API key, no network
.venv/bin/ruff check .
```

The tests create a `memx_test` database on that Postgres and never touch `memx`. The dashboard lives in `web/`:
`pnpm install && pnpm dev` (proxies to the API on port 8000), `pnpm typecheck`, `pnpm build`.

## Making a change

- **A fix comes with a test that fails without it.** Integration tests are in `tests/integration` and use the fakes
  in `memx/testing.py` (`SentenceExtractor`, `HashEmbedder`, `FakeJudge`), so they are deterministic.
- **Schema changes are a new numbered file** in `memx/db/migrations`. Migrations already applied are never edited.
  Run yours against a database that has data in it, not only an empty one.
- **Anything that reads or writes facts must respect scopes.** A fact is visible at its own scope and below. Code
  that follows links between facts (history, neighbors, entities) has to check the other end is visible too.
- **A database connection is not held across a model call.** Use `memx.db.pool.Borrowed` where a function
  alternates queries with calls to the extractor, embedder or Jev.
- **Benchmark runs go to their own database** (`MEMX_BENCH_DSN`), never the one your application uses.
- Changing an extraction or judging prompt changes results. Say in the pull request which benchmark you ran before
  and after, on which questions.

## Pull requests

Keep one change per pull request, describe what a user would notice, and add a line to `CHANGELOG.md`.
By contributing you agree your work is licensed under the Apache License 2.0, as the rest of the project is.
