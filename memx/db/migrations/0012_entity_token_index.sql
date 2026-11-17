-- The word index on entity names was built on string_to_array(norm, ' '), but the lookup strips the namesake
-- suffix first (split_part(norm, '~', 1)), so the planner could never use it. Index the expression the query uses.
DROP INDEX memx_entity_tokens;
CREATE INDEX memx_entity_tokens ON memx_entity USING gin (string_to_array(split_part(norm, '~', 1), ' '));
