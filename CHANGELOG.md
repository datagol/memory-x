# Changelog

## Unreleased

### Changed, and worth knowing before upgrading
- An idempotency key now identifies an input within its own scope. The same key at two scopes is two inputs.
- With no `observed_at`, the same input sent on a later day is a new ingest and reinforces the fact; the same day
  it is still ignored as a resend.
- `promote` only accepts a broader scope of the fact's own. Other targets are refused.
- With no `MEMX_API_TOKEN`, the API answers only requests addressed to this machine (`MEMX_ALLOWED_HOSTS`).
- Entities are keyed by root and tenant label (`memory.acme`), so `memx reresolve --tenant` takes that form.
- Recall logs are deleted after 90 days (`MEMX_RECALL_LOG_RETENTION_DAYS`, `0` keeps them).
- The HTTP API, the MCP server and PDF reading are install extras: `pip install 'memx[server]'` for all three.

### Fixed
- A fact's history no longer shows facts from sibling or child scopes.
- Forgetting one fact also removes its wording from the saved extraction and from recall logs.
- A job that runs longer than its lease is no longer started a second time; leases are renewed while a job runs.
- A job whose worker dies on every attempt is given up on instead of being requeued for ever.
- Recall falls back to search order when the judge is unavailable instead of failing.
- Database connections are returned to the pool during model calls.
- Removing repeats inside one large document no longer blocks the server (about 40 s down to 0.04 s for 1,000
  candidates).
- Fact text can no longer add headings to the recalled block, and input can no longer close the extractor prompt's
  sections.
- Concurrent ingests in sibling scopes no longer race on creating or renaming the same entity.

## 0.1.0
First version: scoped memory on Postgres and pgvector, the HTTP API, the MCP server and the dashboard.
