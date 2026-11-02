-- Entities: the named people, organizations, teams, systems, projects, places and products facts mention.
-- Facts stay the memory; entities are annotations that connect them. Keyed by tenant (first scope label) so
-- company-level and user-level facts in one tenant share nodes, and two tenants never do.
CREATE TABLE memx_entity (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant      text NOT NULL,
  name        text NOT NULL,                      -- canonical display name
  norm        text NOT NULL,                      -- normalized key (lowercase, titles and punctuation stripped)
  type        text NOT NULL DEFAULT 'other'
              CHECK (type IN ('person', 'organization', 'team', 'system', 'project', 'place', 'product', 'other')),
  aliases     text[] NOT NULL DEFAULT '{}',       -- normalized spellings merged into this entity
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tenant, norm)
);
CREATE INDEX memx_entity_aliases ON memx_entity USING gin (aliases);
CREATE INDEX memx_entity_tokens ON memx_entity USING gin (string_to_array(norm, ' '));

-- "fact mentions entity". surface is the name as written in that fact.
CREATE TABLE memx_fact_entity (
  fact_id    uuid NOT NULL REFERENCES memx_fact (id) ON DELETE CASCADE,
  entity_id  uuid NOT NULL REFERENCES memx_entity (id) ON DELETE CASCADE,
  surface    text,
  PRIMARY KEY (fact_id, entity_id)
);
CREATE INDEX memx_fact_entity_entity ON memx_fact_entity (entity_id);

-- One topic per fact, chosen by Jev from a fixed list in the sort request.
ALTER TABLE memx_fact ADD COLUMN topic text;
ALTER TABLE memx_fact ADD COLUMN topic_confidence real;
CREATE INDEX memx_fact_topic ON memx_fact (topic) WHERE topic IS NOT NULL;

-- Entity alias merges are judge calls too.
ALTER TABLE memx_decision DROP CONSTRAINT memx_decision_step_check;
ALTER TABLE memx_decision ADD CONSTRAINT memx_decision_step_check
  CHECK (step IN ('sort', 'verdict', 'grade', 'auto', 'entity'));
