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
    extractor_model: str = "anthropic/claude-haiku-5.5"
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
    # One Jev request per window drops turns that say nothing worth remembering before extraction.
    extract_gate: bool = True
    gate_skip_below: float = 0.1
    # Queued inserts extract through the provider's batch API: half the token price, results within 24 hours.
    extract_batch: bool = False
    extract_batch_poll_seconds: float = 60.0

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
    entity_name_sim_min: float = 0.5  # name-embedding similarity that makes a known entity a candidate for Jev
    entity_candidates_max: int = 3  # known entities Jev compares a new name with
    entity_split_min: float = 0.7  # an identical name becomes a second entity only when Jev is this sure it differs

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
    # Destructive verdicts need more confidence than linking. Benchmark data: 87% of acted-on replace/covers verdicts
    # sat in [0.5, 0.9), where a larger reference model agreed only about half the time; a 0.9 floor lifted
    # LongMemEval full-memory accuracy from 57 to 65 of 82. Below its floor a verdict falls back to "related".
    verdict_same_min: float = 0.8
    verdict_replace_min: float = 0.9
    verdict_covers_min: float = 0.9
    verdict_min_prob: float | None = None  # deprecated: when set, overrides both replace and covers floors
    # Code rules that hold back replace/covers: different attribute labels, same extraction window, past-only statement.
    verdict_guards: bool = True
    # Optional second, narrower question for destructive verdicts in this band: act only if Jev says both
    # statements cannot be true at once (P(coexist) <= verdict_confirm_max_coexist). Measured and left off: on 67
    # Sonnet-labelled verdicts in the band Jev said "can coexist" (median 0.96) whether or not the reference agreed
    # (AUC 0.36), so it would act on almost nothing (eval/calibration/coexist_probe.py).
    verdict_confirm: bool = False
    verdict_confirm_band: tuple[float, float] = (0.6, 0.9)
    verdict_confirm_max_coexist: float = 0.2
    # Attribute labels that share no word ("location" / "city of residence") block a replace unless Jev says they
    # name the same attribute with at least this probability.
    same_attribute_min: float = 0.5

    # Recall
    budget_tokens: int = 4_000
    budget_safety: float = 0.9
    # Candidate width. Measured on LoCoMo (1,540 questions) and LongMemEval (82): the 40/20/40/0.3/30 settings lost
    # 76 answers that were in memory (not proposed, not graded, graded under 0.3, or cut by the 30-fact cap); the
    # values below recover 52 of them for about 2.5x the tokens, still bounded by budget_tokens.
    recall_k_vector: int = 80
    recall_k_fts: int = 30
    grade_max: int = 90
    grade_enabled: bool = True
    grade_floor: float = 0.1
    recall_k_max: int = 80
    chars_per_token: float = 3.6
    # Earlier versions of a picked fact are shown under it when the query asks about a past or changed state
    # ("when_asked": one extra yes/no in the ranked grade request), always, or never. Full mode calls no model,
    # so it shows history only with "always".
    history_in_recall: Literal["off", "when_asked", "always"] = "when_asked"
    history_min_prob: float = 0.5
    render_related_dates: bool = True  # date related facts so disagreeing statements show which is newer

    # Decay / retention
    half_lives: dict[str, float | None] = Field(default_factory=lambda: dict(DEFAULT_HALF_LIVES))
    access_boost_cap: float = 1.3
    # Upcoming events can be hidden this many days after their date. None (default): never hidden; they fade with the
    # event half-life instead. Expiry hid facts that stay true after the date ("the draft reached 4,700 words by May 13",
    # "the call was moved to April 22") and cost BEAM's knowledge-update questions 16 points.
    event_ttl_days: int | None = None
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

    def has_api_key(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.get_secret_value().strip())

    def api_key(self) -> str:
        if not self.has_api_key():
            raise RuntimeError("OPENROUTER_API_KEY (or MEMX_OPENROUTER_API_KEY) is not set")
        return self.openrouter_api_key.get_secret_value()
