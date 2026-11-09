-- Better name matching. An embedding of each entity's name finds variants that share no word ("Blore" and
-- "Bangalore"); known_as keeps the alternative names the extractor suggested, so a later "Bangalore" can be paired
-- with a stored "Blore" for Jev to judge. Neither ever merges anything on its own.
ALTER TABLE memx_entity ADD COLUMN embedding vector({{EMBED_DIM}});
ALTER TABLE memx_entity ADD COLUMN known_as text[] NOT NULL DEFAULT '{}';
CREATE INDEX memx_entity_known_as ON memx_entity USING gin (known_as);
