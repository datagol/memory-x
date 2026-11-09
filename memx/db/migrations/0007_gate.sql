-- The chit-chat gate before extraction logs one decision per window.
ALTER TABLE memx_decision DROP CONSTRAINT memx_decision_step_check;
ALTER TABLE memx_decision ADD CONSTRAINT memx_decision_step_check
  CHECK (step IN ('sort', 'verdict', 'grade', 'auto', 'entity', 'rereconcile', 'gate'));
