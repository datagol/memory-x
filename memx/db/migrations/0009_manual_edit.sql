-- A person correcting a fact (for now, its category) logs one decision, so the fact's history shows who changed what.
ALTER TABLE memx_decision DROP CONSTRAINT memx_decision_step_check;
ALTER TABLE memx_decision ADD CONSTRAINT memx_decision_step_check
  CHECK (step IN ('sort', 'verdict', 'grade', 'auto', 'entity', 'rereconcile', 'gate', 'manual'));
