-- Entities were keyed by the label under the root alone (acme in memory.acme.alex), so two roots with the same
-- label under them (memory.acme and other.acme) shared one entity namespace. The key is now the root and that
-- label together (memory.acme). Each entity takes the root of the facts that name it; one named under several roots
-- (only possible through the old sharing) goes to the first, and entities nothing names are dropped.
DELETE FROM memx_entity e WHERE NOT EXISTS (SELECT 1 FROM memx_fact_entity fe WHERE fe.entity_id = e.id);

UPDATE memx_entity e SET tenant = k.tenant
  FROM (SELECT fe.entity_id,
               min(CASE WHEN nlevel(f.scope) > 1 THEN subpath(f.scope, 0, 2)::text ELSE f.scope::text END) AS tenant
          FROM memx_fact_entity fe JOIN memx_fact f ON f.id = fe.fact_id
         GROUP BY fe.entity_id) k
 WHERE k.entity_id = e.id AND e.tenant <> k.tenant;
