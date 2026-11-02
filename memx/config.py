from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

InsertMode = Literal["inline", "background", "queued"]

DEFAULT_HALF_LIVES: dict[str, float | None] = {
    "identity": None,
    "rule": None,
    "preference": 365,
    "goal": 180,
    "relationship": 365,
    "work_context": 120,
    "procedure": 365,
    "event": 30,
    "other": 90,
}


class MemxConfig(BaseSettings):
    """All tunables. Read from MEMX_* env vars and .env."""

    model_config = SettingsConfigDict(env_prefix="MEMX_", env_file=".env", extra="ignore")

    dsn: str = "postgresql://memx:memx@localhost:5433/memx"
    pool_min: int = 1
    pool_max: int = 10

    # Providers (all routed through OpenRouter by default). "demo" uses deterministic stand-ins from
    # memx.testing: no API key, no cost; useful for UI work. Use a separate database for it.
    providers: Literal["openrouter", "demo"] = "openrouter"
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MEMX_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    extractor_model: str = "anthropic/claude-haiku-4.5"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    jev_model: str = "jev-1.14"
    provider_timeout_s: float = 60.0

    # Insert
    insert_mode: InsertMode = "background"
    max_input_chars: int = 400_000
    window_tokens: int = 2_500
    window_context_turns: int = 2
    window_max_candidates: int = 25
    extract_concurrency: int = 4
    jev_concurrency: int = 8
    max_fact_chars: int = 600
    user_trust: float = 1.0
    assistant_trust: float = 0.6
    default_text_trust: float = 0.8

    # Sort
    sort_use_hint: bool = True
    sort_min_prob: float = 0.45
    sort_min_confidence: float = 0.5
    injection_flag: float = 0.5
    injection_reject: float = 0.85
    topic_min: float = 0.45  # below this top probability a fact gets no topic

    # Entities (links between facts)
    entities_enabled: bool = True
    entity_merge_min: float = 0.7  # Jev probability needed to merge two spellings into one entity
    entity_max_per_fact: int = 6

    # Walk: expand ranked recall through shared entities and topics
    walk_enabled: bool = True
    walk_hops: int = 2
    walk_decay: float = 0.5  # per hop
    walk_cap: int = 8  # most facts the walk may add to the grade request
    walk_seeds: int = 10  # search hits the walk starts from
    walk_hub_share: float = 0.2  # entities mentioned by more than this share of visible facts are not walked
    walk_hub_min_degree: int = 6  # ...but never cut an entity with fewer facts than this
    walk_topic_weight: float = 0.3  # extra multiplier on hops through a topic
    walk_score_factor: float = 0.85  # final-score multiplier for facts reached by a hop
    walk_grade_floor: float = 0.5  # a walked fact must be graded at least "useful"; background is not enough

    # Reconcile
    band_low: float = 0.45
    band_high: float = 0.95
    guard_min_jaccard: float = 0.8
    neighbor_k_vector: int = 10
    neighbor_k_fts: int = 5
    neighbor_max: int = 5
    verdict_same_min: float = 0.5
    verdict_min_prob: float = 0.5

    # Recall
    budget_tokens: int = 4_000
    budget_safety: float = 0.9
    recall_k_vector: int = 40
    recall_k_fts: int = 20
    grade_max: int = 40
    grade_enabled: bool = True
    grade_floor: float = 0.3
    recall_k_max: int = 30
    chars_per_token: float = 3.6

    # Decay / retention
    half_lives: dict[str, float | None] = Field(default_factory=lambda: dict(DEFAULT_HALF_LIVES))
    access_boost_cap: float = 1.3
    event_ttl_days: int = 30  # upcoming events expire this long after their date
    expired_retention_days: int = 30
    decision_retention_days: int = 180
    done_job_retention_days: int = 7

    # Jobs
    worker_concurrency: int = 4
    job_lease_seconds: int = 300
    job_max_attempts: int = 5
    job_poll_seconds: float = 1.0

    # API
    api_token: SecretStr | None = None

    def api_key(self) -> str:
        if self.openrouter_api_key is None:
            raise RuntimeError("OPENROUTER_API_KEY (or MEMX_OPENROUTER_API_KEY) is not set")
        return self.openrouter_api_key.get_secret_value()
