-- memx initial schema. {{EMBED_DIM}} is substituted from MemxConfig.embedding_dim at migrate time.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS ltree;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- One row per scope that has ever been written to. Holds per-scope overrides.
CREATE TABLE memx_scope (
  path        ltree PRIMARY KEY,
  settings    jsonb NOT NULL DEFAULT '{}',
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX memx_scope_gist ON memx_scope USING gist (path);

-- Raw input and extraction results. Retries resume from here.
CREATE TABLE memx_ingest (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  scope            ltree NOT NULL,
  kind             text  NOT NULL CHECK (kind IN ('messages', 'text')),
  payload          jsonb NOT NULL,
  source           text,
  trust            real  NOT NULL DEFAULT 1.0 CHECK (trust BETWEEN 0 AND 1),
  observed_at      timestamptz NOT NULL,
  idempotency_key  text  NOT NULL UNIQUE,
  mode             text  NOT NULL,
  status           text  NOT NULL DEFAULT 'received'
                   CHECK (status IN ('received', 'extracted', 'committed', 'failed')),
  windows          jsonb NOT NULL DEFAULT '{}',   -- {"<window index>": [candidate, ...]}
  candidates       jsonb,                         -- deduped candidates with per-candidate outcome
  receipt          jsonb,
  extractor_model  text,
  tokens_in        int NOT NULL DEFAULT 0,
  tokens_out       int NOT NULL DEFAULT 0,
  error            text,
  created_at       timestamptz NOT NULL DEFAULT now(),
  committed_at     timestamptz
);
CREATE INDEX memx_ingest_scope ON memx_ingest (scope, created_at DESC);
CREATE INDEX memx_ingest_scope_gist ON memx_ingest USING gist (scope);

-- Atomic facts. Never edited in place: replaced facts keep their row as 'superseded'.
CREATE TABLE memx_fact (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  scope               ltree NOT NULL REFERENCES memx_scope (path) ON DELETE CASCADE,
  category            text  NOT NULL CHECK (category IN ('identity', 'preference', 'goal', 'rule',
                        'relationship', 'work_context', 'event', 'procedure', 'other')),
  text                text  NOT NULL CHECK (length(text) <= 600),
  text_hash           bytea NOT NULL,
  token_count         int   NOT NULL,
  embedding           vector({{EMBED_DIM}}) NOT NULL,
  embedding_model     text  NOT NULL,
  tsv                 tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  status              text  NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'expired')),
  superseded_by       uuid REFERENCES memx_fact (id) ON DELETE SET NULL,
  superseded_at       timestamptz,
  ingest_id           uuid REFERENCES memx_ingest (id) ON DELETE CASCADE,
  candidate_index     int,
  speaker             text,
  source              text,
  trust               real NOT NULL,
  extract_confidence  real NOT NULL DEFAULT 1.0,
  sort_confidence     real NOT NULL DEFAULT 1.0,
  sort_probs          jsonb NOT NULL DEFAULT '{}',
  flags               jsonb NOT NULL DEFAULT '{}',
  observed_at         timestamptz NOT NULL,
  event_at            timestamptz,
  stale_after         timestamptz,
  reinforced_count    int NOT NULL DEFAULT 0,
  last_reinforced_at  timestamptz,
  access_count        int NOT NULL DEFAULT 0,
  last_accessed_at    timestamptz,
  attrs               jsonb NOT NULL DEFAULT '{}',
  created_at          timestamptz NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (ingest_id, candidate_index)
);
CREATE INDEX memx_fact_scope_status ON memx_fact (scope, status) INCLUDE (token_count);
CREATE INDEX memx_fact_scope_gist ON memx_fact USING gist (scope);
CREATE UNIQUE INDEX memx_fact_exact_dup ON memx_fact (scope, text_hash) WHERE status = 'active';
CREATE INDEX memx_fact_hnsw ON memx_fact USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX memx_fact_tsv ON memx_fact USING gin (tsv);
CREATE INDEX memx_fact_stale ON memx_fact (stale_after) WHERE stale_after IS NOT NULL;
CREATE INDEX memx_fact_superseded_by ON memx_fact (superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX memx_fact_ingest ON memx_fact (ingest_id);

-- related: both true, grouped at render.  shadows: a (child scope) hides b (ancestor scope).
CREATE TABLE memx_fact_link (
  a           uuid NOT NULL REFERENCES memx_fact (id) ON DELETE CASCADE,
  b           uuid NOT NULL REFERENCES memx_fact (id) ON DELETE CASCADE,
  kind        text NOT NULL CHECK (kind IN ('related', 'shadows')),
  similarity  real,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (a, b, kind)
);
CREATE INDEX memx_fact_link_b ON memx_fact_link (b);

-- Every judge call. Doubles as the calibration dataset.
CREATE TABLE memx_decision (
  id                bigserial PRIMARY KEY,
  step              text NOT NULL CHECK (step IN ('sort', 'verdict', 'grade', 'auto')),
  question_version  text NOT NULL,
  ingest_id         uuid,
  candidate_index   int,
  subject_fact_id   uuid,
  object_fact_id    uuid,
  recall_id         uuid,
  scope             ltree,
  similarity        real,
  result            jsonb NOT NULL,
  latency_ms        int,
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX memx_decision_step_time ON memx_decision (step, created_at);
CREATE INDEX memx_decision_subject ON memx_decision (subject_fact_id) WHERE subject_fact_id IS NOT NULL;
CREATE INDEX memx_decision_object ON memx_decision (object_fact_id) WHERE object_fact_id IS NOT NULL;
CREATE INDEX memx_decision_ingest ON memx_decision (ingest_id) WHERE ingest_id IS NOT NULL;
CREATE INDEX memx_decision_scope_gist ON memx_decision USING gist (scope);

CREATE TABLE memx_recall_log (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  scope        ltree NOT NULL,
  query        text,
  mode         text NOT NULL CHECK (mode IN ('full', 'ranked')),
  fact_ids     uuid[] NOT NULL,
  scores       jsonb,
  token_count  int,
  latency_ms   int,
  created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX memx_recall_log_scope_gist ON memx_recall_log USING gist (scope);

CREATE TABLE memx_job (
  id               bigserial PRIMARY KEY,
  kind             text NOT NULL,
  payload          jsonb NOT NULL,
  idempotency_key  text NOT NULL UNIQUE,
  status           text NOT NULL DEFAULT 'queued'
                   CHECK (status IN ('queued', 'running', 'done', 'dead')),
  attempts         int NOT NULL DEFAULT 0,
  max_attempts     int NOT NULL DEFAULT 5,
  run_after        timestamptz NOT NULL DEFAULT now(),
  locked_at        timestamptz,
  locked_by        text,
  lease_seconds    int NOT NULL DEFAULT 300,
  last_error       text,
  created_at       timestamptz NOT NULL DEFAULT now(),
  finished_at      timestamptz
);
CREATE INDEX memx_job_dequeue ON memx_job (run_after, id) WHERE status = 'queued';
CREATE INDEX memx_job_running ON memx_job (locked_at) WHERE status = 'running';

-- Hard deletes leave a trace without keeping the deleted content.
CREATE TABLE memx_forget_log (
  id               bigserial PRIMARY KEY,
  scope            ltree,
  fact_id          uuid,
  descendants      boolean NOT NULL DEFAULT true,
  facts_removed    int NOT NULL,
  ingests_removed  int NOT NULL,
  requested_by     text,
  created_at       timestamptz NOT NULL DEFAULT now()
);
