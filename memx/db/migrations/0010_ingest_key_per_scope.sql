-- An idempotency key was unique across every scope, so a key reused at another scope returned the first scope's
-- receipt and dropped the new input. A key now identifies an input within its own scope only.
ALTER TABLE memx_ingest DROP CONSTRAINT memx_ingest_idempotency_key_key;
ALTER TABLE memx_ingest ADD CONSTRAINT memx_ingest_scope_key UNIQUE (scope, idempotency_key);
