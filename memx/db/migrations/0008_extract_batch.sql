-- Queued inserts may extract through the provider's batch API (half price, results within 24 h).
-- The submitted batch id and the window it holds live here until the results are read.
ALTER TABLE memx_ingest ADD COLUMN extract_batch jsonb;
