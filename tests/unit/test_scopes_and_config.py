from __future__ import annotations
import pytest
from memx.config import MemxConfig
from memx.scopes import InvalidScope, ScopePath


def test_scope_chain_and_validation():
    s = ScopePath.parse("memory.acme.alex.agent-1")
    assert s.chain_str() == ["memory", "memory.acme", "memory.acme.alex", "memory.acme.alex.agent-1"]
    assert s.tenant == "memory.acme" and ScopePath.parse("memory").tenant == "memory"
    assert ScopePath.parse("other.acme.sam").tenant == "other.acme"
    assert ScopePath.parse("memory.acme").is_ancestor_of(s)
    assert not s.is_ancestor_of(s)
    for bad in ("", "memory..a", "memory.a b", "memory.ä", "x" * 65):
        with pytest.raises(InvalidScope):
            ScopePath.parse(bad)


def test_recall_defaults_are_wide():
    """Benchmarked defaults (LoCoMo 1,540 q / LongMemEval 82 q): the floor at 0.1 and the 80-fact cap recovered most
    answers that were in memory but not shown; budget_tokens stays the hard cap on the rendered block."""
    cfg = MemxConfig(_env_file=None)
    assert (cfg.grade_floor, cfg.recall_k_max) == (0.1, 80)
    assert cfg.grade_max >= cfg.recall_k_max, "everything that can be returned must have been graded"
    assert cfg.recall_k_vector + cfg.recall_k_fts >= cfg.grade_max, "the grader must be offered enough candidates"
    assert cfg.budget_tokens == 4_000


def test_empty_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MEMX_INSERT_MODE", "")
    monkeypatch.setenv("MEMX_POOL_MAX", "")
    cfg = MemxConfig(_env_file=None)
    assert cfg.insert_mode == "background"
    assert cfg.pool_max == 10
