-- Store the full ranking explanation for each recall so it can be replayed in the UI.
ALTER TABLE memx_recall_log ADD COLUMN explain jsonb;
CREATE INDEX memx_recall_log_time ON memx_recall_log (created_at DESC);
CREATE INDEX memx_ingest_time ON memx_ingest (created_at DESC);
