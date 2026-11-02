-- Entities were keyed by the root label, which is "memory" for every scope, so all tenants shared one entity
-- namespace. The tenant is the label under the root (company_2 in memory.company_2.user_3). Re-key every mention
-- to an entity in its own fact's tenant, then drop entities nothing names any more.
CREATE TEMP TABLE memx_rekey ON COMMIT DROP AS
SELECT fe.fact_id, fe.entity_id, e.name, e.norm, e.type, e.aliases,
       CASE WHEN nlevel(f.scope) > 1 THEN subpath(f.scope, 1, 1)::text ELSE f.scope::text END AS tenant
  FROM memx_fact_entity fe
  JOIN memx_fact f ON f.id = fe.fact_id
  JOIN memx_entity e ON e.id = fe.entity_id;

INSERT INTO memx_entity (tenant, name, norm, type, aliases)
SELECT DISTINCT ON (tenant, norm) tenant, name, norm, type, aliases FROM memx_rekey
 ORDER BY tenant, norm, name
ON CONFLICT (tenant, norm) DO NOTHING;

UPDATE memx_fact_entity fe SET entity_id = n.id
  FROM memx_rekey r JOIN memx_entity n ON n.tenant = r.tenant AND n.norm = r.norm
 WHERE fe.fact_id = r.fact_id AND fe.entity_id = r.entity_id AND fe.entity_id <> n.id;

DELETE FROM memx_entity e WHERE NOT EXISTS (SELECT 1 FROM memx_fact_entity fe WHERE fe.entity_id = e.id);
