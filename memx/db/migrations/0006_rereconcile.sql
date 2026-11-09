-- Re-reconcile: restoring a fact that an earlier, weaker policy retired is a logged decision of its own.
ALTER TABLE memx_decision DROP CONSTRAINT memx_decision_step_check;
ALTER TABLE memx_decision ADD CONSTRAINT memx_decision_step_check
  CHECK (step IN ('sort', 'verdict', 'grade', 'auto', 'entity', 'rereconcile'));
