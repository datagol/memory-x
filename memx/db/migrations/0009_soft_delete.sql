-- Deleting a fact keeps its row as 'deleted', with who deleted it and when. Recall and listings show active facts only.
ALTER TABLE memx_fact DROP CONSTRAINT IF EXISTS memx_fact_status_check;
ALTER TABLE memx_fact ADD CONSTRAINT memx_fact_status_check
  CHECK (status IN ('active', 'superseded', 'expired', 'deleted'));
ALTER TABLE memx_fact ADD COLUMN deleted_at timestamptz;
ALTER TABLE memx_fact ADD COLUMN deleted_by text;
