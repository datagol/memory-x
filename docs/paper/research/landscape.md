# Landscape: Long-Term Memory Systems for LLM Agents (2023 – Nov 2026)

*Survey for the Memory-X (memx) paper. Focus: **how each system decides** what to write, merge, supersede, or delete (write time), and **how it ranks** what to return (read time).*

Compiled 2 Nov 2026. Every claim links to its source (arXiv id, official docs, or repo). Where a source doesn't give a value, the text says **not reported**. Vendor benchmark numbers are labeled as self-reported. They are **not comparable across rows** unless the text says they come from one harness (see §4).

---

## 0. Reading guide / key definitions

- **Write-time decision**: what happens when a new observation arrives and may duplicate, refine, contradict, or obsolete stored memory.
- **Read-time decision**: candidate generation (vector / BM25 / graph), fusion, reranking, and any LLM grading or filtering before the answer model sees the context.
- **LoCoMo "J"**: binary LLM-as-judge accuracy on LoCoMo categories 1–4, with the adversarial category 5 excluded. The protocol was popularized by Mem0's paper ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)).
- **LongMemEval-S**: 500 questions over ~115k-token histories ([arXiv:2410.10813](https://arxiv.org/abs/2410.10813); history length per [Zep blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory)).

---

## 1. Summary comparison table

Columns: (1) unit of memory · (2) write-time dedup/update/contradiction/delete · (3) entity resolution · (4) retrieval · (5) time & decay · (6) scoping / cross-scope correction · (7) audit / provenance · (8) reported LoCoMo / LongMemEval (self-reported unless noted).

| System | (1) Unit | (2) Write-time decision | (3) Entity resolution | (4) Retrieval | (5) Time / decay | (6) Scoping | (7) Audit / provenance | (8) Benchmarks (source) |
|---|---|---|---|---|---|---|---|---|
| **Mem0 (paper, 2025)** | Atomic NL fact | Top-s similar memories, then one LLM tool call picks ADD/UPDATE/DELETE/NOOP ([2504.19413](https://arxiv.org/abs/2504.19413)) | None in base. Mem0g: embedding threshold + LLM | Vector top-k | Timestamps in text. No decay in paper | user/agent/run ids | SQLite history (OSS) | LoCoMo J 66.88 (Mem0), 68.44 (Mem0g) ([2504.19413](https://arxiv.org/abs/2504.19413)) |
| **Mem0g** | Entity–relation triple + NL | LLM "update resolver" marks conflicting edges **invalid** (soft delete) ([2504.19413](https://arxiv.org/abs/2504.19413)) | Embedding similarity to existing nodes above threshold *t* | Entity-anchored subgraph expansion + triplet-embedding match | Invalidated edges kept for temporal reasoning | same as Mem0 | Edge invalidation history | LoCoMo J 68.44 |
| **Mem0 v3 algorithm (Apr 2026; OSS + Platform)** | Atomic fact + entity layer + temporal metadata | **ADD-only** single LLM pass. Exact-hash (MD5) dedup. No UPDATE/DELETE ([docs](https://docs.mem0.ai/core-concepts/memory-evaluation), [migration](https://docs.mem0.ai/migration/oss-v2-to-v3), [main.py](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py)) | Entities (proper nouns, quoted text, compound NPs) embedded and linked | Semantic + BM25 (lemmatized) + entity boost + temporal score, rank-fused | Separate temporal pass writes event time, ongoing/completed, precision, memory type. "Memory decay" recency ranking ([mem0.ai/research](https://mem0.ai/research)) | user_id / agent_id / run_id / actor_id **filters** (conjunctive). No inheritance | SQL history log of ADD events. `attributed_to` | Platform: LoCoMo 92.5, LongMemEval 94.4. OSS: 91.6 / 93.4 ([README](https://github.com/mem0ai/mem0), [migration](https://docs.mem0.ai/migration/oss-v2-to-v3)) |
| **Zep / Graphiti** | Episode + entity node + fact edge (+ community) | LLM compares new edge with related edges between the same entity pair: `is_duplicate`. Contradictions with temporal overlap → set old edge's `t_invalid` ([2501.13956](https://arxiv.org/abs/2501.13956)) | Cosine over name embeddings + full-text candidates, then LLM `is_duplicate` prompt | Cosine + BM25 + BFS. RRF, MMR, episode-mentions, node-distance, cross-encoder rerankers | **Bi-temporal**: valid time (t_valid/t_invalid) + transaction time (t'_created/t'_expired). Invalidate, never delete | `group_id` namespaces. App must authorize ([docs](https://help.getzep.com/graphiti/core-concepts/graph-namespacing)) | Every fact traces to source episodes ([Graphiti README](https://github.com/getzep/graphiti)) | DMR 94.8% (gpt-4-turbo). LongMemEval-S 71.2% (gpt-4o), 63.8% (4o-mini) ([2501.13956](https://arxiv.org/abs/2501.13956)). LoCoMo 75.14 (Zep's own re-run) ([blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory)) |
| **MemGPT / Letta** | In-context memory blocks + recall (message DB) + archival (text passages) | The agent LLM edits its own memory via function calls. Memory-pressure warning at ~70% of context, flush at 100% ([2310.08560](https://arxiv.org/abs/2310.08560)). Sleep-time agent rewrites blocks asynchronously ([letta blog](https://www.letta.com/blog/sleep-time-compute)) | None built in | Agent-issued search tools (vector archival, conversation search). Filesystem grep/semantic | Recursive summary of evicted messages. No decay | Blocks shareable across agents; read-only blocks ([docs](https://docs.letta.com/guides/agents/memory-blocks)) | Full message log in recall storage | DMR 93.4% (gpt-4-turbo) per [2501.13956](https://arxiv.org/abs/2501.13956). LoCoMo 74.0% (Letta Filesystem, gpt-4o-mini) ([blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)) |
| **A-MEM** | Zettelkasten note (content, time, keywords, tags, context, embedding, links) | No delete. LLM "evolution" over top-k neighbours: `strengthen` / `update_neighbor` (rewrite neighbour context/tags) ([2502.12110](https://arxiv.org/abs/2502.12110)) | None (links via LLM) | Cosine top-k + linked notes in same "box" | Timestamp attribute only | not reported | Links only | LoCoMo J 48.38 per Mem0 paper's harness ([2504.19413](https://arxiv.org/abs/2504.19413)). Own paper reports F1/BLEU |
| **MemoryOS (EMNLP'25)** | Dialogue page → topic segment → persona/KB | STM→MTM FIFO. MTM segments scored by **Heat** (retrieval count, page count, recency). Lowest heat evicted. Heat > 5 promoted to LPM. User KB a FIFO queue of 100 ([2506.06326](https://arxiv.org/abs/2506.06326)) | none | Two-stage: segment match, then page semantic top-k. LPM top-10 | Exponential recency factor in Heat | per-user persona | not reported | LoCoMo: +49.11% F1, +46.18% BLEU-1 avg over baselines (GPT-4o-mini) ([2506.06326](https://arxiv.org/abs/2506.06326)) |
| **MemOS (MemTensor)** | **MemCube** (plaintext / activation-KV / parametric) with metadata | Paper claims conflict detection, dedup, versioning, forgetting; Version Chain logs lineage ([2507.03724](https://arxiv.org/abs/2507.03724)). Algorithm details not given | not specified | "Hybrid semantic retrieval" + MemScheduler | Lifecycle states, TTL/decay policy per MemCube. "Time Machine" rollback, Frozen state | Ternary permission model (user × memory × calling context): private/shared/read-only | Version chain, logs, watermarks (MemGovernance) | LoCoMo LLM-judge 75.80. LongMemEval 77.8 (GPT-4o-mini, own harness) ([2507.03724](https://arxiv.org/abs/2507.03724)) |
| **LangMem** | Collection of memories or a profile schema (Pydantic) | LLM rewrites memory state. `enable_inserts`/`enable_updates` default True, `enable_deletes` default **False** ([ref](https://langchain-ai.github.io/langmem/reference/memory/)) | none | LangGraph BaseStore search. Docs recommend combining similarity with importance + "strength" ([guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)) | Strength = f(recency, frequency) suggested, not enforced | Hierarchical **namespaces** e.g. (org, user_id, app) | not reported | LoCoMo J 58.10 per Mem0 harness ([2504.19413](https://arxiv.org/abs/2504.19413)) |
| **Cognee** | KG (entities, relations) + chunks + embeddings | `improve()` "conflict review": marks entity facts as **current / superseded / disputed** using source docs + dates. Feedback weights from rated answers ([docs](https://docs.cognee.ai/core-concepts/main-operations/improve)) | LLM or GLiNER extraction ([README](https://github.com/topoteretes/cognee)) | Graph completion + vector. Feedback-weighted | Dates of source documents | datasets / sessions | Source documents per fact | Multi-hop QA tuning only (HotPotQA, 2Wiki, MuSiQue) ([2505.24478](https://arxiv.org/abs/2505.24478)). LoCoMo/LME not reported |
| **HippoRAG / HippoRAG 2** | Passages + OpenIE triples (schemaless KG) | Append-only indexing. Synonymy edges when entity cosine > threshold ([2405.14831](https://arxiv.org/abs/2405.14831)) | Embedding-threshold synonymy edges | Personalized PageRank from query-linked seeds. HR2 adds LLM "recognition memory" triple filtering ([2502.14802](https://arxiv.org/abs/2502.14802)) | none | none | Passage nodes retained | +7% associative-memory vs NV-Embed-v2 ([2502.14802](https://arxiv.org/abs/2502.14802)). Not evaluated on LoCoMo/LME there |
| **Memory-R1** | Atomic fact entries | **RL-trained** Memory Manager (PPO/GRPO) emits ADD/UPDATE/DELETE/NOOP. Reward = downstream answer correctness ([2508.19828](https://arxiv.org/abs/2508.19828)) | none | RAG retrieve ~60 → RL Answer Agent "memory distillation" filter | not specific | not reported | not reported | LoCoMo (LLaMA-3.1-8B): +28% F1, +34% BLEU-1, +30% J relative to Mem0. 152 training QA pairs. Zero-shot to MSC/LME ([2508.19828](https://arxiv.org/abs/2508.19828)) |
| **MIRIX** | 6 typed stores: Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault | Meta Memory Manager routes input to per-type managers, which "avoid redundant information" (LLM agents) ([2507.07957](https://arxiv.org/abs/2507.07957)) | none | Active Retrieval (generate topic first) + embedding_match / bm25_match / string_match tools | Episodic timestamps | per user | not reported | LoCoMo 85.4% (abstract). ScreenshotVQA +35% vs RAG ([2507.07957](https://arxiv.org/abs/2507.07957)) |
| **MemoryBank** | Turn + daily event summary + user portrait | No merge. Ebbinghaus retention R = e^(−t/S). Recall ⇒ S += 1, t reset ([2305.10250](https://arxiv.org/abs/2305.10250)) | none | DPR-style dense retrieval | **Forgetting curve** | per user | none | Qualitative / SiliconFriend. Not on LoCoMo/LME |
| **Generative Agents** | Observation / reflection / plan in memory stream | Append-only. LLM rates importance 1–10. Reflection when importance sum > 150 ([2304.03442](https://arxiv.org/abs/2304.03442)) | none | Score = recency + importance + relevance (min-max normalized, weights = 1) | Exponential recency decay since last access | single agent | Reflections cite source memories | not applicable |
| **LightMem (ICLR'26)** | Topic-segmented, compressed entries | Online: **soft insert** only. Offline "sleep-time" update: top-k similar entries, only **later timestamps may update earlier** ones, executed in parallel ([2510.18866](https://arxiv.org/abs/2510.18866)) | none | Embedding retrieval | Timestamp ordering constraint | not reported | not reported | LongMemEval: +2.09–6.40% accuracy (GPT), up to +7.67% (Qwen) over strongest baseline. Tokens cut up to 38× ([2510.18866](https://arxiv.org/abs/2510.18866)) |
| **Mem-α** | Core (≤512 tok) + semantic facts + episodic timestamped events | **RL-trained** tool policy: memory_insert / update / delete (core: update only). Reward = QA correctness + format + compression + content ([2509.25911](https://arxiv.org/abs/2509.25911)) | none | Fixed BM25 retriever over semantic/episodic pools during training reward | Episodic timestamps | not reported | not reported | MemoryAgentBench. Trained ≤30k tokens, generalizes to >400k ([2509.25911](https://arxiv.org/abs/2509.25911)) |
| **SeCom (ICLR'25)** | Topical segment | No update logic. Segmentation model + LLMLingua-2 denoising ([2502.05589](https://arxiv.org/abs/2502.05589)) | none | BM25 / MPNet over segments | none | none | Raw segment kept | LoCoMo GPT4Score 69.33 ([2502.05589](https://arxiv.org/abs/2502.05589)) |
| **Nemori** | Narrative episode + semantic facts | Event-segmentation boundary detection. **Predict-Calibrate**: predict episode content from current memory, then store only the prediction gap as new semantic knowledge ([2508.03341](https://arxiv.org/abs/2508.03341)) | none | Retrieval over episodic + semantic memories (method details: see paper) | Pre-resolves relative dates at write time | not reported | Episodes keep source | LoCoMo LLM 0.744 (4o-mini), 0.794 (4.1-mini) ([2508.03341](https://arxiv.org/abs/2508.03341)) |
| **Supermemory** | Atomic memory + source chunk + profile | Learning model links memories as **updates / extends / derives**. Updates make the old value non-current without deleting it ([LME report](https://supermemory.ai/research/longmembench), [docs](https://supermemory.ai/docs/concepts/how-it-works)) | not documented | Semantic match, then source-chunk injection. "Vector, FTS and graph built in" | `documentDate` + `eventDate` | Hard namespaces + metadata + scoped API keys | Chunks kept as evidence | LongMemEval-S **Recall@20** 97% (not QA accuracy) ([research](https://supermemory.ai/research/)). QA 81.6% (GPT-4o) per [Hindsight blog](https://vectorize.io/blog/introducing-hindsight-agent-memory-that-works-like-human-memory) |
| **Memobase** | User profile slots (topic/sub-topic) + event timeline | Per-user buffer batch-processes chats. LLM updates profile slots (schema in config.yaml) ([README](https://github.com/memodb-io/memobase), [docs](https://docs.memobase.io/features/profile/profile)) | n/a (slot schema) | Profile always in context + event-gist search | Timestamped events | per user | Raw blobs deleted after processing by default | LoCoMo J 75.78 (v0.0.37, own run) ([benchmark](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)) |
| **ReadAgent** | Page "gist" | LLM decides pagination breakpoints, then gists each page. No updates ([2402.09727](https://arxiv.org/abs/2402.09727)) | none | LLM looks at all gists and chooses pages to re-read | none | none | Raw pages retained | Up to 20× effective context. QuALITY/NarrativeQA/QMSum ([2402.09727](https://arxiv.org/abs/2402.09727)) |
| **MemoRAG** | KV-compressed global memory of corpus | None (static corpus) | none | Memory model generates draft "clues", which drive the retriever ([2409.05591](https://arxiv.org/abs/2409.05591)) | none | none | none | Long-context QA, not LoCoMo/LME |
| **Hindsight (Dec 2025)** | Narrative facts in 4 networks: world, experience, opinion, observation | LLM extraction. **Opinions carry confidence scores** reinforced or weakened by evidence ([2512.12818](https://arxiv.org/abs/2512.12818)) | Weighted string (Levenshtein) + co-occurrence + temporal-proximity score | Semantic + BM25 + spreading-activation graph + temporal → RRF → cross-encoder → token budget | Temporal ranges on facts | memory "banks" | Facts vs beliefs separated | LongMemEval 91.4%. LoCoMo 89.61% (Gemini-3 backbone) ([2512.12818](https://arxiv.org/abs/2512.12818)) |
| **EverMemOS (Jan 2026)** | MemCell (episode + atomic facts + foresight) → MemScene | Incremental semantic clustering. Scene-driven profile evolution ([2601.02163](https://arxiv.org/abs/2601.02163)) | not specified | Dense + BM25 (RRF) + rerank + agentic verification / query rewrite | Time-bounded "foresight" | per user | not reported | LoCoMo 93.05 (4.1-mini). LongMemEval 83.00 ([2601.02163](https://arxiv.org/abs/2601.02163)) |

---

## 2. Per-system sections

### 2.1 Mem0 (paper v1, Apr 2025) and Mem0g

**Unit.** Natural-language atomic "facts" extracted from each new message pair. Extraction uses a rolling conversation summary and the last *m* messages as context ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)).

**Write-time decision.** For each candidate fact, Mem0 retrieves the top-*s* semantically similar memories. The fact and these memories go to the LLM through a function-calling interface. The LLM picks one of four operations: **ADD** (no semantically equivalent memory exists), **UPDATE** (adds complementary information), **DELETE** (memory contradicted by new information), or **NOOP**. The paper says this is done "rather than using a separate classifier", by leveraging "the LLM's reasoning capabilities to directly select the appropriate operation" ([2504.19413](https://arxiv.org/abs/2504.19413)). One generative call therefore detects duplicates and contradictions and chooses the action. There is no threshold, no confidence, and no separate judge.

**Mem0g (graph variant).** A two-stage LLM pipeline extracts entities, then relation triplets. A new entity is matched to existing nodes by embedding similarity above "a defined threshold *t*". A "conflict detection mechanism" plus an **LLM-based update resolver** marks conflicting relationships **invalid rather than physically removing them, to enable temporal reasoning** ([2504.19413](https://arxiv.org/abs/2504.19413)). Retrieval has two parts. An entity-centric pass finds query entities, locates their nodes, and expands incoming and outgoing edges. A "semantic triplet" pass matches the query embedding against encoded triplets ([2504.19413](https://arxiv.org/abs/2504.19413)).

**Benchmarks (LoCoMo J, GPT-4o-mini, self-run harness).** Mem0 66.88. Mem0g 68.44. Zep 65.99. LangMem 58.10. OpenAI memory 52.90. A-Mem 48.38. Full-context 72.90. Best RAG ≈ 61 ([2504.19413](https://arxiv.org/abs/2504.19413), Table 2). The paper also claims 91% lower p95 latency and >90% token savings versus full context ([2504.19413](https://arxiv.org/abs/2504.19413)).

### 2.2 Mem0 v3 "token-efficient algorithm" (April 2026; OSS v2 → v3 migration)

Mem0 **abandoned the ADD/UPDATE/DELETE reconciliation step** in this release. Their blog says: "Overwrites sometimes erased key information… Deletes sometimes removed information that would be relevant later". The new algorithm "collapses extraction into a single LLM call that only adds" ([mem0 blog](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm)).

- **Write path.** The pipeline is: context lookup → single-pass ADD-only distillation → **hash-based dedup** → embed → entity linking → separate temporal-reasoning pass. The temporal pass labels each memory with event time, ongoing/completed, timing precision, and type (event, state, plan, preference, relationship, absence) ([docs](https://docs.mem0.ai/core-concepts/memory-evaluation)). In the OSS code, dedup is an exact **MD5 of the memory text** checked against existing and in-batch hashes ([mem0/memory/main.py](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py)). Paraphrased duplicates are therefore not deduplicated at write time.
- **Agent-generated facts** (e.g., "I've booked your flight") are now stored as first-class memories ([blog](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm)).
- **Read path.** Semantic + BM25 (with verb lemmatization) + entity-match boost + additive temporal score, fused by rank. Mem0 says the temporal score "nudges ranking… without filtering candidates out" ([docs](https://docs.mem0.ai/core-concepts/memory-evaluation)). OSS defaults changed: `top_k` 100→20, `threshold` None→0.1, `rerank` True→False. Graph memory was **removed from OSS** and is now Platform-only ([migration](https://docs.mem0.ai/migration/oss-v2-to-v3)).
- **Scoping.** `user_id`, `agent_id`, `run_id` scope storage, and `actor_id` narrows queries. Freeform metadata "cannot place a memory into an unrequested scope" ([main.py](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py)). Scopes act as filters. No inheritance or shadowing between, say, org-level and user-level memories is documented.
- **Audit.** A SQL history log records ADD events plus a rolling message window ([docs](https://docs.mem0.ai/core-concepts/memory-evaluation)).
- **Benchmarks (self-reported).**
  - Platform at a top_200 retrieval budget: LoCoMo 92.5, LongMemEval 94.4, BEAM-1M 64.1, BEAM-10M 48.6. Old algorithm: LoCoMo 71.4, LongMemEval 67.8. Scores carry "±1 point… due to judge inconsistency" ([docs](https://docs.mem0.ai/core-concepts/memory-evaluation), [README](https://github.com/mem0ai/mem0)).
  - OSS: 91.6 / 93.4 ([migration](https://docs.mem0.ai/migration/oss-v2-to-v3)).
  - Mem0 itself notes that knowledge-update (93.6) "remains the hardest category for an additive, ADD-only architecture", because older facts still surface. BEAM-1M contradiction_resolution is 35.7 ([docs](https://docs.mem0.ai/core-concepts/memory-evaluation)).
  - The per-category LoCoMo numbers differ between the docs page (single-hop 91.2, multi-hop 91.3) and the research page (94.6, 95.4) ([docs](https://docs.mem0.ai/core-concepts/memory-evaluation), [research](https://mem0.ai/research)).

> **Why this matters for memx.** The most widely deployed OSS memory layer concluded that a generative LLM emitting UPDATE/DELETE was destroying information. It replaced that step with append-only storage plus read-time temporal ranking. Contradiction is now resolved implicitly by ranking, not explicitly by a decision.

### 2.3 Zep / Graphiti (arXiv:2501.13956)

**Unit.** A three-tier graph ([2501.13956](https://arxiv.org/abs/2501.13956)):
- an episode subgraph (raw messages, text, or JSON);
- a semantic entity subgraph (entity nodes plus fact edges, with hyper-edges for multi-entity facts);
- a community subgraph built by label propagation and extended incrementally. A new node joins the plurality community of its neighbours, and periodic refreshes are needed.

**Entity resolution.** Entity names are embedded (1024-d). Candidates come from cosine search plus full-text search on names and summaries. An LLM prompt returns `is_duplicate` and the uuid, and writes "the most complete full name" ([2501.13956](https://arxiv.org/abs/2501.13956)). Graph writes use predefined Cypher rather than LLM-generated queries.

**Fact dedup and contradiction.** Edge dedup searches only edges **between the same entity pair** and asks an LLM whether the new edge "represents the same factual information". For contradiction, "an LLM [compares] new edges against semantically related existing edges". When the system finds "temporally overlapping contradictions, it invalidates the affected edges by setting their t_invalid to the t_valid of the invalidating edge". Graphiti "consistently prioritizes new information" on the transaction timeline ([2501.13956](https://arxiv.org/abs/2501.13956)).

**Time.** **Bi-temporal**: T (event/valid timeline) and T′ (transactional ingestion timeline), stored as four timestamps per edge. Relative dates ("two weeks ago") are resolved against the episode's reference timestamp ([2501.13956](https://arxiv.org/abs/2501.13956)). Old facts are "invalidated — not deleted", which supports point-in-time queries ([Graphiti README](https://github.com/getzep/graphiti)).

**Retrieval.** Cosine + Okapi BM25 (Lucene/Neo4j) + breadth-first search seeded by nodes or recent episodes. Rerankers: RRF, MMR, episode-mentions frequency, node distance, and cross-encoder ([2501.13956](https://arxiv.org/abs/2501.13956)).

**Scoping.** `group_id` namespaces isolate graphs. The docs warn that the application "must authorize each `group_id`" and that "a namespace filter does not replace application authorization" ([docs](https://help.getzep.com/graphiti/core-concepts/graph-namespacing)). The README says the reference servers "do not authenticate callers" ([README](https://github.com/getzep/graphiti)). No cross-namespace correction mechanism is documented.

**Provenance.** "Every entity and relationship traces back to the episodes… that produced it" ([README](https://github.com/getzep/graphiti)). Custom entity and edge types can be defined via Pydantic.

**Benchmarks (self-reported).**
- DMR: Zep 94.8% vs MemGPT 93.4% (gpt-4-turbo). 98.2% with gpt-4o-mini ([2501.13956](https://arxiv.org/abs/2501.13956)).
- LongMemEval-S: 71.2% vs full-context 60.2% (gpt-4o). 63.8% vs 55.4% (gpt-4o-mini). Average context 1.6k vs 115k tokens. Gains of 18.5% / 15.2% ([2501.13956](https://arxiv.org/abs/2501.13956)).
- LoCoMo: see the dispute in §3.

### 2.4 MemGPT → Letta

**Mechanism.** Virtual context management in the style of an operating system. Main context is system instructions + working context + FIFO queue. External context is recall storage (all messages) and archival storage (arbitrary text). At the "warning token count" (e.g., 70% of the window) the queue manager inserts a "memory pressure" system message so the LLM can save important items. At the flush count (100%) it evicts ~50% of messages and generates a recursive summary ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)).

**Write-time decision.** Entirely **agent-initiated**: the LLM decides via function calls what to write or replace. No dedup, contradiction, or delete rule is built in. Letta's sleep-time agents move memory editing off the primary agent. The primary agent "is not provided with tools to edit its core memory". A background sleep-time agent manages and rewrites the memory blocks ([Letta blog](https://www.letta.com/blog/sleep-time-compute), [arXiv:2504.13171](https://arxiv.org/abs/2504.13171)).

**Memory blocks.** Each block has a label, description, value, and character limit, and is always in context. Blocks are shareable across agents ("update once, visible everywhere") and can be read-only policy blocks ([docs](https://docs.letta.com/guides/agents/memory-blocks)). This is the closest any system comes to hierarchical scope (a shared org block plus per-agent blocks). Precedence between conflicting blocks is left to the LLM.

**Benchmarks.** Letta put LoCoMo conversations into a file with grep/search/open tools and reports 74.0% with gpt-4o-mini. Letta argues that "agent capabilities matter more than the tools" ([Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)). LongMemEval: not reported by Letta.

### 2.5 A-MEM (NeurIPS 2025; arXiv:2502.12110)

**Unit.** An atomic Zettelkasten note: content, timestamp, LLM-generated keywords, tags, and context description, plus an embedding and links.

**Write path.**
1. Note construction (LLM).
2. **Link generation**: top-k cosine neighbours, then the LLM decides which connections to make.
3. **Memory evolution**: an LLM prompt over the new note and its nearest neighbours returns JSON with `should_evolve` and actions, e.g. `strengthen` (connect and update tags) or `update_neighbor` (rewrite the neighbours' context and tags). The JSON schema also lists "merge" and "prune" ([2502.12110](https://arxiv.org/abs/2502.12110), App. B.3).

The design has **no explicit contradiction or delete path**. Old notes can be rewritten in place by neighbour updates, which leaves no version history.

**Read.** Cosine top-k plus notes linked in the same "box". **Benchmarks.** The paper reports LoCoMo F1/BLEU and an 85–93% token reduction versus LoCoMo/MemGPT baselines ([2502.12110](https://arxiv.org/abs/2502.12110)). Mem0's harness measured LoCoMo J at 48.38 ([2504.19413](https://arxiv.org/abs/2504.19413)).

### 2.6 MemoryOS (BAI-LAB, EMNLP 2025 oral; arXiv:2506.06326)

**Structure.** Three tiers: STM (dialogue pages), MTM (topic segments of pages), and LPM (user profile, user KB, 90-dimension user traits, agent traits).

**Write-time decisions are deterministic heuristics** ([2506.06326](https://arxiv.org/abs/2506.06326)):
- STM→MTM: FIFO.
- MTM eviction: lowest **Heat** = α·N_visit + β·L_interaction + γ·R_recency, where R_recency is an exponential decay with time constant 1e7.
- Promotion: segments with heat above 5 are promoted to LPM.
- User KB and agent traits are FIFO queues of size 100.

No contradiction detection is described. **Read.** Segment-level matching selects the top-m segments, then semantic top-k pages within them. LPM retrieves the top-10 KB entries. **Benchmark.** LoCoMo average improvement of 49.11% F1 and 46.18% BLEU-1 over baselines with GPT-4o-mini ([2506.06326](https://arxiv.org/abs/2506.06326)). In EverMemOS's harness MemoryOS scores 54.70 LoCoMo accuracy with 4o-mini ([2601.02163](https://arxiv.org/abs/2601.02163)).

### 2.7 MemOS (MemTensor; arXiv:2507.03724)

**Unit.** The **MemCube**, which wraps plaintext, activation (KV-cache), or parameter (LoRA-style) memory together with metadata. Metadata comes in three groups ([2507.03724](https://arxiv.org/abs/2507.03724)):
- descriptive identifiers;
- governance attributes: access control (read/write/share scope), lifespan policy (TTL or decay), priority, compliance tags, watermarks, logs;
- behavioral usage indicators (frequency/recency "hot/cold").

**Write/lifecycle.** Plaintext memory "supports conflict detection, deduplication, versioning, and forgetting policies". A **Version Chain** "logs each memory's modification history and derivation lineage, enabling version control, conflict resolution, and rollback". The **Time Machine** snapshots and rolls back states, and a **Frozen** state disables updates while retaining history ([2507.03724](https://arxiv.org/abs/2507.03724)). The paper does not describe the concrete algorithm, i.e. what judge decides a conflict, or with what threshold.

**Scoping.** MemGovernance uses a "ternary permission model involving the user identity, the memory object, and the calling context" (private, shared, or read-only) ([2507.03724](https://arxiv.org/abs/2507.03724)). This is the most explicit governance design in the literature. It concerns access control, not correction precedence.

**Benchmarks (own harness, GPT-4o-mini for all systems).**

| System | LoCoMo LLM-judge | LongMemEval |
|---|---|---|
| MemOS-1031 | 75.80 | 77.8 |
| Memobase | 72.01 | 72.4 |
| Mem0 | 64.57 | 66.4 |
| MIRIX | 64.33 | 43.49 |
| Zep | 59.22 | 63.8 |
| MemU | 56.55 | 38.4 |
| Supermemory | 55.34 | 58.4 |

Source: [2507.03724](https://arxiv.org/abs/2507.03724), Tables 3–4.

### 2.8 LangMem (LangChain)

Every operation is the same loop: "Accept conversation(s) and current memory state → Prompt an LLM to determine how to expand or consolidate the memory state → Respond with the updated memory state" ([conceptual guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)). `create_memory_manager` takes flags `enable_inserts=True`, `enable_updates=True` ("outdated or contradicted"), and `enable_deletes=False` by default ([API ref](https://langchain-ai.github.io/langmem/reference/memory/)). Memory can be "subconscious" (background) or "conscious" (hot-path tools). **Scoping** uses multi-level namespaces such as `("acme_corp", "{user_id}", "code_assistant")` resolved at runtime ([guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)). The guide recommends that recall combine similarity with "importance" and "strength" (recency/frequency), but that is design advice, not a built-in scorer. **Benchmarks.** LoCoMo J 58.10 with p95 search latency 59.82 s in Mem0's harness ([2504.19413](https://arxiv.org/abs/2504.19413)). Nemori's harness gives 0.513 ([2508.03341](https://arxiv.org/abs/2508.03341)).

### 2.9 Cognee

Cognee is an open-source KG and vector memory with `remember`, `recall`, `improve`, and `forget` operations. Extraction can use an LLM or a local GLiNER model ([README](https://github.com/topoteretes/cognee)). The **`improve()`** operation runs nine fixed stages, including `feedback_weights`, `update_user_preferences`, and `review_conflicts`. Conflict review "reviews entity facts that disagree, using the documents and dates they came from, and marks which values are **current, superseded, or still disputed**" ([docs](https://docs.cognee.ai/core-concepts/main-operations/improve)). This is the only system surveyed with an explicit **"disputed"** state rather than a forced binary choice. Feedback on rated answers raises or lowers the importance of the graph elements used in retrieval ([docs](https://docs.cognee.ai/core-concepts/main-operations/improve)). **Benchmarks.** The Cognee paper studies hyperparameter tuning on HotPotQA, TwoWikiMultiHop, and MuSiQue ([arXiv:2505.24478](https://arxiv.org/abs/2505.24478)). LoCoMo and LongMemEval: not reported.

### 2.10 HippoRAG (NeurIPS'24) / HippoRAG 2 (ICML'25)

- **Indexing.** OpenIE produces schemaless triples, and synonymy edges are added "when the cosine similarity between two entity representations… is above a threshold" ([arXiv:2405.14831](https://arxiv.org/abs/2405.14831)). This is the only entity-resolution step, and it is soft: an edge is added rather than nodes merged.
- **Read.** Query entities seed **Personalized PageRank** over the KG ([2405.14831](https://arxiv.org/abs/2405.14831)). HippoRAG 2 adds passage nodes ("dense-sparse integration") and query-to-triple linking. Its **recognition memory** step "filters the top triples using an LLM" before PPR ([arXiv:2502.14802](https://arxiv.org/abs/2502.14802)).
- **Results.** HippoRAG 2 reports +7% on associative-memory tasks over NV-Embed-v2 ([2502.14802](https://arxiv.org/abs/2502.14802)).
- **Limits.** No update, contradiction, or time handling: the corpus is append-only.

### 2.11 Memory-R1 (arXiv:2508.19828; ACL 2026)

- **Memory Manager.** An LLM fine-tuned with **PPO or GRPO** selects ADD, UPDATE, DELETE, or NOOP for each extracted fact against retrieved memories. Its reward is the correctness of a *frozen* Answer Agent's answer after the edit is applied ([2508.19828](https://arxiv.org/abs/2508.19828)). The paper gives an example where a vanilla manager issues DELETE+ADD, "fragmenting memory", while the RL manager "issues a single UPDATE".
- **Answer Agent.** Trained with an exact-match reward, it "distills" ~60 retrieved memories down to the relevant ones.
- **Data.** 152/81/1307 LoCoMo train/validation/test questions, with category 5 excluded.
- **Results.** Relative to Mem0 with LLaMA-3.1-8B: +28% F1, +34% BLEU-1, +30% J. Evaluated zero-shot on MSC and LongMemEval ([2508.19828](https://arxiv.org/abs/2508.19828)).

This is the clearest example of a **learned write policy**. The policy is still a generator that emits a discrete op, and it outputs no probability or calibration.

### 2.12 MIRIX (arXiv:2507.07957)

- **Memory types.** Six: Core, Episodic, Semantic, Procedural, Resource, and Knowledge Vault (sensitive data).
- **Write path.** The system first searches the memory base. A **Meta Memory Manager** then routes the input to the relevant type-specific Memory Managers. These update in parallel "while ensuring that redundant information is avoided within each memory type" ([2507.07957](https://arxiv.org/abs/2507.07957)). The dedup criterion is not specified beyond the LLM agent's judgment.
- **Read.** "Active Retrieval": the agent first generates a topic, then retrieves with `embedding_match`, `bm25_match`, or `string_match` tools.
- **Results.** LoCoMo 85.4% (abstract) with gpt-4.1-mini. ScreenshotVQA: +35% over RAG with 99.9% less storage ([2507.07957](https://arxiv.org/abs/2507.07957)). MemOS's harness measured 64.33 LoCoMo with 4o-mini ([2507.03724](https://arxiv.org/abs/2507.03724)).

### 2.13 MemoryBank (AAAI 2024; arXiv:2305.10250)

Stores conversation turns, hierarchical daily and global event summaries, and a user personality profile, all retrieved by DPR-style dense retrieval. Forgetting follows the **Ebbinghaus curve** R = e^(−t/S). *S* is a discrete strength: when a memory is recalled, "we increase S by 1 and reset t" ([2305.10250](https://arxiv.org/abs/2305.10250)). This is the canonical **access-reinforced decay** design. There are no contradiction semantics: a stale fact fades only if it is not recalled.

### 2.14 Generative Agents (UIST 2023; arXiv:2304.03442)

An append-only memory stream. Retrieval score = recency + importance + relevance, each min-max normalized, with all weights set to 1. Recency is exponential decay over sandbox hours since the memory was last retrieved (the paper uses decay factor 0.995). The LLM directly outputs an importance integer from 1 to 10. **Reflections** fire when the summed importance of recent events exceeds 150, about 2–3 times a day ([2304.03442](https://arxiv.org/abs/2304.03442)). This is the origin of the "recency × importance × relevance" read-time scorer that LangMem and others echo.

### 2.15 LightMem (ICLR 2026; arXiv:2510.18866)

The design follows the Atkinson–Shiffrin model:
1. **Sensory memory.** LLMLingua-2 token-retention pre-compression (binary keep/drop with a dynamic percentile threshold), followed by hybrid topic segmentation.
2. **Short-term memory.** Topic-grouped summarization.
3. **Long-term memory with sleep-time update.** At test time new entries are **soft-inserted** without reconciliation. Offline, each entry gets an update queue of its top-k most similar entries, "only entries with later timestamps are allowed to update earlier ones", and the queues execute in parallel ([2510.18866](https://arxiv.org/abs/2510.18866)).

**Results.** LongMemEval +2.09–6.40% accuracy (GPT) and up to +7.67% (Qwen) over the strongest baseline. Up to 38× fewer tokens and 30× fewer API calls (GPT) ([2510.18866](https://arxiv.org/abs/2510.18866)). Like Mem0 v3 and Letta sleep-time, it **moves reconciliation off the hot path**.

### 2.16 Mem-α (arXiv:2509.25911)

The memory has three parts ([2509.25911](https://arxiv.org/abs/2509.25911)):
- Core: a ≤512-token summary that only supports `memory_update`.
- Semantic: atomic facts.
- Episodic: timestamped events.

Semantic and episodic support `memory_insert`, `memory_update`, and `memory_delete`. An RL-trained policy chooses tool calls. The reward combines QA correctness, tool-call format, compression, and memory content. Trained on ≤30k-token instances, it generalizes to >400k tokens (13×). Evaluation uses MemoryAgentBench.

### 2.17 SeCom (ICLR 2025; arXiv:2502.05589)

Studies memory **granularity**: turn-level, session-level, or summary. It proposes **segment-level** units produced by a conversation-segmentation model, plus LLMLingua-2 "compression-based denoising" before retrieval. LoCoMo GPT4Score 69.33, dropping to 59.87 without denoising ([2502.05589](https://arxiv.org/abs/2502.05589)). There is no update or contradiction logic. Its relevance for memx is that the choice of unit changes retrieval quality materially.

### 2.18 Nemori (arXiv:2508.03341)

- **Two-Step Alignment.** Event-segmentation-theory boundary detection splits the stream into episodes, which are written as narrative episodic memories.
- **Predict-Calibrate.** Inspired by the free-energy principle: the system "learn[s] from prediction gaps", distilling into semantic memory only what current memory failed to predict ([2508.03341](https://arxiv.org/abs/2508.03341)). The novelty test is generative prediction error, not an embedding threshold.
- **Time.** Relative dates are resolved at write time (e.g., "Jon was mentored on June 15, 2023").
- **Results.** LoCoMo LLM score 0.744 (gpt-4o-mini) vs full-context 0.723, Mem0 0.613, Zep 0.585, LangMem 0.513 in its own harness ([2508.03341](https://arxiv.org/abs/2508.03341)).

### 2.19 Supermemory

- **Architecture.** A "custom learning model" decides "what and how to learn, what is important, when to forget, creating relations". Storage is a "fact-based temporal graph that has Vector, FTS, and graph built in" ([docs](https://supermemory.ai/docs/concepts/how-it-works)).
- **Relations between memories.** **updates** ("records a contradiction or correction as a state change"), **extends** (adds detail without contradiction), and **derives** (an inference from several memories). Each memory has a `documentDate` and an `eventDate`. A retrieved memory comes back with its source chunk ([LongMemEval report](https://supermemory.ai/research/longmembench)). Updates "make an earlier value no longer current; that is different from permanent deletion" ([blog](https://supermemory.ai/blog/ais-next-big-thing-personalization-and-super-memory/)).
- **Scoping.** "namespace — hard isolation boundary (user, tenant, project)", metadata for soft filtering, and scoped API keys ([docs](https://supermemory.ai/docs/concepts/how-it-works)).
- **Benchmark caveat.** The headline "97%" is **Recall@20 with aggregation**. In the same table it is placed next to Zep's 71.2% and full-context's 60.2%, which are QA accuracies from the Zep paper ([research](https://supermemory.ai/research/), [2501.13956](https://arxiv.org/abs/2501.13956)). Earlier QA accuracy claims are 81.6% (GPT-4o) and 84.6% (GPT-5), as listed by Vectorize ([Hindsight blog](https://vectorize.io/blog/introducing-hindsight-agent-memory-that-works-like-human-memory)). Mixing retrieval recall with QA accuracy in vendor comparisons is a known problem ([lhl/agentic-memory](https://github.com/lhl/agentic-memory/blob/main/benchmarks/README.md)).

### 2.20 Memobase

A user-**profile**-based memory: a profile of topic/sub-topic slots defined in `config.yaml`, plus a timestamped event timeline ([docs](https://docs.memobase.io/features/profile/profile)). Each user has a buffer (`max_chat_blob_buffer_token_size`, `buffer_flush_interval`) that batch-processes chats. Raw blobs are deleted after processing by default ([README](https://github.com/memodb-io/memobase)). In v0.0.40 the workflow was fixed at 3 LLM calls per flush, down from 3–10 ([README](https://github.com/memodb-io/memobase)). Write-time conflict handling is LLM rewriting of the slot value, and no versioning is documented. **Benchmark (self-run, mixed with numbers copied from Mem0's paper).** LoCoMo J 75.78 for v0.0.37 and 70.91 for v0.0.32 ([benchmark](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)). MemOS's harness gives 72.01 LoCoMo and 72.4 LongMemEval ([2507.03724](https://arxiv.org/abs/2507.03724)).

### 2.21 ReadAgent (ICML 2024; arXiv:2402.09727)

The LLM (1) decides episode **pagination** breakpoints, (2) compresses each page into a **gist**, and (3) at query time sees all gists and chooses which raw pages to look up. It reports an effective context increase "up to 20x" ([2402.09727](https://arxiv.org/abs/2402.09727)). Read-time selection is fully LLM-driven, with no index.

### 2.22 MemoRAG (arXiv:2409.05591)

Uses a dual system. A light long-range model builds a **KV-compressed global memory** of the corpus and generates **draft answers / clues**, which a retriever uses to locate evidence. A stronger model then answers. The memory is trained with generation-quality feedback ([2409.05591](https://arxiv.org/abs/2409.05591)). It targets static long-context corpora and has no update semantics.

### 2.23 New in 2025–2026 (beyond the required list)

- **Hindsight** (Vectorize; [arXiv:2512.12818](https://arxiv.org/abs/2512.12818); ACL 2026 demo).
  - Four networks: *world*, *experience*, *opinion*, *observation*. The **opinion** network stores "subjective beliefs with confidence scores that can be updated as new evidence arrives".
  - Entity resolution is a weighted sum of string similarity (Levenshtein), co-occurrence, and temporal proximity.
  - Recall runs four retrievers in parallel (semantic, BM25, spreading-activation graph, temporal), then RRF, then a cross-encoder, then a token-budget filter.
  - Results: LongMemEval 91.4% and LoCoMo 89.61% with large backbones. With an OSS-20B backbone, LongMemEval goes from 39.0% (full context) to 83.6% ([2512.12818](https://arxiv.org/abs/2512.12818)). The confidence scores are LLM-maintained and not calibrated against outcomes.
- **EverMemOS** ([arXiv:2601.02163](https://arxiv.org/abs/2601.02163); ACL 2026). MemCells (episode + atomic facts + time-bounded "foresight") are clustered into MemScenes, which drive profile evolution. Retrieval is dense+BM25 RRF, reranking, and agentic verification. Results: LoCoMo 93.05 (4.1-mini) and LongMemEval 83.00.
- **MemForest** ([arXiv:2605.23986](https://arxiv.org/abs/2605.23986)). "Canonical facts serve as durable write units", and summaries and indexes are refreshed locally. MemTree is a hierarchical temporal index. Results: 81.8% LongMemEval-S and 84.09% LoCoMo cat. 1–4 (Qwen3-30B).
- **Mem++** ([arXiv:2610.02002](https://arxiv.org/abs/2610.02002)). "Non-destructive": it stores whole documents with date and author, "calls no generative model at write time", and selects at read time by as-of date with lexical+semantic fusion. Reported gains: +8.0–13.1 on OrgMemBench, and the best average on LoCoMo. This is the extreme of the anti-write-time-distillation trend.
- **Engram** ([arXiv:2606.09900](https://arxiv.org/abs/2606.09900)). Lossless episode append on the hot path. Asynchronous SPO fact extraction into a bi-temporal KG. Contradiction resolution "without an LLM call per fact — invalidating, never deleting", with provenance and supersession chains. LongMemEval-S 83.6% vs 73.2% full context.
- **Graph-native bitemporal store** ([arXiv:2607.26520](https://arxiv.org/abs/2607.26520)). Immutable identity nodes link to versioned content nodes with valid-time and transaction-time intervals. Semantic edges are added at write time by cosine similarity. Result: 46.7% R@10 on a 60-question LongMemEval sample.
- **TOKI** ([arXiv:2606.06240](https://arxiv.org/abs/2606.06240)). Recasts contradiction resolution as **write-time concurrency control**. It types four production heuristics (last-writer-wins, evidence-weighted merge, await-confirmation, per-rule policy) as bitemporal operators, each with an isolation precondition and an audit row that preserves the losing fact. It proves that **keyed logging of the adjudicating judge is necessary for replay consistency, "which every audited baseline omits"**. Its eight-system audit finds every baseline with an LLM judge on the write path admits at least one anomaly: replay inconsistency, belief-drift skew, or audit erasure.
- **PreCoMem** ([arXiv:2610.29467](https://arxiv.org/abs/2610.29467)). Represents memory as **confidence-weighted beliefs**. "Effective surprisal" combines semantic deviation and contradiction evidence, and is downweighted when retrieval is diffuse. A **two-threshold gate** maps each turn to MAINTAIN, PROFILE (a low-confidence "waiting room"), or CORRECT (soft decay of contradicted facts).
- **Nous / belief-based memory** ([arXiv:2606.22030](https://arxiv.org/abs/2606.22030)). Each entity–attribute pair is a categorical distribution updated by closed-form Bayesian inference. On LoCoMo it gives "little benefit over naive last-write-wins because existing… benchmarks rarely contain contradictory or differently reliable evidence". It also proposes provenance-capped trust and reports a **27.5-point gap between token-F1 and LLM-judge** on identical outputs.
- **MemIR** ([arXiv:2605.25869](https://arxiv.org/abs/2605.25869)). A typed memory IR separating raw evidence, retrieval cues, and truth-bearing claims, to fix "provenance-role collapse".
- **Governed Shared Memory / MemClaw** ([arXiv:2606.24535](https://arxiv.org/abs/2606.24535)). A production multi-tenant service with scoped retrieval, temporal supersession, and provenance. It reports that sub-tenant scope was "initially bypassed on direct GET-by-id requests", and that a synchronous near-duplicate gate can prematurely reject writes that conflict with supersession.
- **Memory security.** Two papers show that content- and lineage-based trust can be laundered through the agent's own summarization. TMA-NM ([arXiv:2606.24322](https://arxiv.org/abs/2606.24322)) proves write-time origin binding is necessary. PPMF ([arXiv:2607.29167](https://arxiv.org/abs/2607.29167)) finds that consolidated memories can reach 1.000 attack success rate without platform-maintained provenance.
- **Read-side papers.**
  - AgentMemRec ([arXiv:2610.34673](https://arxiv.org/abs/2610.34673)) frames memory selection as budgeted slate recommendation with no LLM calls.
  - "Reliable Post-Retrieval Assembly" ([arXiv:2606.01435](https://arxiv.org/abs/2606.01435)) shows that separating evidence extraction from policy execution (e.g., freshness) raises MemoryAgentBench FactConsolidation single-hop from ≤54% to 82–93%.
  - EMBER ([arXiv:2606.05894](https://arxiv.org/abs/2606.05894)) learns a budgeted pre-query retention policy over verbatim "evidence capsules".
- **Benchmarks.** LongMemEval-V2 ([arXiv:2605.12493](https://arxiv.org/abs/2605.12493)) has 451 questions over web-agent trajectories of up to 115M tokens, testing state recall, dynamic state tracking, workflows, "gotchas", and premise awareness. BEAM (1M/10M) is used by Mem0 ([docs](https://docs.mem0.ai/core-concepts/memory-evaluation)). A 2026 survey reports that "models that score near-perfectly on LoCoMo plummet to 40–60% in MemoryArena" (quoted in [letta issue #3115](https://github.com/letta-ai/letta/issues/3115), citing [arXiv:2603.07670](https://arxiv.org/abs/2603.07670)).

---

## 3. The Mem0 vs Zep LoCoMo dispute

1. **Mem0 paper (Apr 2025).** Zep scored LoCoMo J **65.99**, below Mem0 (66.88) and Mem0g (68.44) ([2504.19413](https://arxiv.org/abs/2504.19413)).
2. **Zep's first response.** Zep reported **84%** (cited in [zep-papers#5](https://github.com/getzep/zep-papers/issues/5) and [a40-labs/memory](https://github.com/a40-labs/memory)).
3. **Mem0's rebuttal (GitHub issue filed by Mem0's CTO, per a40-labs).** Zep's calculation counted adversarial category-5 correct answers in the numerator but excluded them from the denominator. Corrected, the figure is **58.44%**. The issue also accuses Zep of changing the system prompt and retrieval template, e.g. "If the memories contain contradictory information, prioritize the most recent memory" ([zep-papers#5](https://github.com/getzep/zep-papers/issues/5)).
4. **Zep blog "Lies, Damn Lies, & Statistics".** Zep reports **75.14% ± 0.17** with a corrected implementation and a p95 search latency of 0.632 s. It argues that Mem0's harness:
   - assigned the user role to *both* speakers;
   - appended timestamps to message text instead of using `created_at`;
   - ran searches sequentially.

   It also criticizes LoCoMo itself: it is short (16–26k tokens), has no knowledge-update questions, an unusable category 5, and speaker-attribution errors ([Zep blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory)).
5. **Independent audits.** About 99 golden answers are wrong (6.4% of 1,540), which gives a ~93.57% ceiling. The LLM judge accepted "vague-but-topical wrong answers" 62.8% of the time ([lhl/agentic-memory](https://github.com/lhl/agentic-memory/blob/main/benchmarks/README.md), citing locomo-audit). a40-labs re-scored Zep's released contexts at 0.7461, "within a point of their published 75.14" ([a40-labs/memory](https://github.com/a40-labs/memory)).

**Zep's LoCoMo score varies by harness, from about 58 to about 85:**

| Source | Zep LoCoMo | Backbone |
|---|---|---|
| Mem0 paper | 65.99 | 4o-mini |
| Mem0's correction | 58.44 | — |
| Zep re-run | 75.14 | 4o-mini |
| MemOS harness ([2507.03724](https://arxiv.org/abs/2507.03724)) | 59.22 | 4o-mini |
| Nemori harness ([2508.03341](https://arxiv.org/abs/2508.03341)) | 0.585 | 4o-mini |
| EverMemOS harness ([2601.02163](https://arxiv.org/abs/2601.02163)) | 81.06 | 4o-mini |
| EverMemOS harness ([2601.02163](https://arxiv.org/abs/2601.02163)) | 85.22 | 4.1-mini |

Mem0 itself appears at 66.88 (paper), 64.57 (MemOS), 0.613 (Nemori), 61.00 (EverMemOS, 4o-mini), and 92.5 (own 2026 Platform run at top_200).

---

## 4. Cross-cutting observations (write-time vs read-time)

1. **The field has moved away from destructive write-time decisions.**
   - Mem0 dropped UPDATE/DELETE for ADD-only ([migration](https://docs.mem0.ai/migration/oss-v2-to-v3)).
   - Zep/Graphiti, Engram, Supermemory, Cognee, and Mem0g invalidate or supersede instead of deleting ([2501.13956](https://arxiv.org/abs/2501.13956), [2606.09900](https://arxiv.org/abs/2606.09900), [supermemory](https://supermemory.ai/research/longmembench), [cognee](https://docs.cognee.ai/core-concepts/main-operations/improve)).
   - Mem++ removes write-time LLMs entirely ([2610.02002](https://arxiv.org/abs/2610.02002)).
2. **Reconciliation is being moved off the hot path**: Letta sleep-time agents, LightMem offline updates, Mem0 async extraction and temporal pass, Engram's async path, and Memobase buffers ([letta](https://www.letta.com/blog/sleep-time-compute), [2510.18866](https://arxiv.org/abs/2510.18866), [docs](https://docs.mem0.ai/core-concepts/memory-evaluation), [2606.09900](https://arxiv.org/abs/2606.09900), [memobase](https://github.com/memodb-io/memobase)).
3. **Write decisions are still almost always one generative LLM call** that emits a discrete label: Mem0 v1, Zep's `is_duplicate` and invalidation, A-MEM evolution, LangMem, and MIRIX. Exceptions:
   - learned RL policies (Memory-R1, Mem-α);
   - deterministic scores (MemoryOS Heat, MemoryBank, LightMem's timestamp rule);
   - 2026 threshold or Bayesian gates (PreCoMem, Nous).
4. **Hybrid retrieval is now standard**: vector + BM25 + graph/entity, fused with RRF and optionally a cross-encoder. Examples: Zep, Mem0 v3, Hindsight, EverMemOS, MIRIX tools.
5. **Benchmarks no longer rank systems reliably.** LoCoMo is near or above its label-noise ceiling, harness choices move a single system by more than 25 points (§3), recall and QA accuracy get mixed, and judges are inconsistent (±1 pt per Mem0; 27.5-pt F1-vs-judge gap per [2606.22030](https://arxiv.org/abs/2606.22030)).

---

## 5. Gaps: what no surveyed system does (as of Nov 2026)

"No system" here means none of the ~30 systems above, per their papers and docs. Partial 2026 exceptions are noted so the claims are not overstated.

1. **Calibrated probabilities on memory decisions.** No production system attaches a *calibrated* probability (e.g., ECE-evaluated) to ADD / UPDATE / SUPERSEDE / DELETE / duplicate / same-entity decisions.
   - Mem0 v1, Zep, A-MEM, LangMem, and MIRIX use a single LLM label.
   - Hindsight's opinion confidences are LLM-assigned and not calibrated ([2512.12818](https://arxiv.org/abs/2512.12818)).
   - PreCoMem's gate and Nous' Bayesian beliefs are the closest. PreCoMem uses fixed thresholds on a surprisal heuristic ([2610.29467](https://arxiv.org/abs/2610.29467)). Nous models beliefs over attribute values, not the decision ([2606.22030](https://arxiv.org/abs/2606.22030)). Neither reports calibration of the write decision itself.
2. **A typed judge separate from the generator.** In every surveyed system the model that extracts or generates memory content also decides its relation to existing memory: Mem0's single tool call, Zep's resolution prompts, A-MEM's evolution prompt. Memory-R1 trains a separate manager, but it is still a generator emitting ops ([2508.19828](https://arxiv.org/abs/2508.19828)). TOKI keeps an LLM judge and argues its decisions must be keyed and logged for replay consistency. Its audit says every baseline omits this ([2606.06240](https://arxiv.org/abs/2606.06240)). No system ships a *typed* adjudicator with a fixed output schema (relation type, confidence, evidence span) that is evaluated independently of extraction quality.
3. **Scope-aware shadowing and cross-scope correction.** Scoping is isolation by filter or namespace: Mem0 user/agent/run filters, Graphiti `group_id`, LangMem namespace tuples, Supermemory namespaces, MemOS private/shared/read-only permissions ([main.py](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py), [graphiti](https://help.getzep.com/graphiti/core-concepts/graph-namespacing), [langmem](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/), [supermemory](https://supermemory.ai/docs/concepts/how-it-works), [2507.03724](https://arxiv.org/abs/2507.03724)). None defines **precedence semantics**, for example a user-scope correction that shadows an org-scope fact for that user only while leaving the org fact intact for others. Letta's shared blocks leave such conflicts to the LLM ([docs](https://docs.letta.com/guides/agents/memory-blocks)). MemClaw reports real sub-tenant scope bypasses ([2606.24535](https://arxiv.org/abs/2606.24535)).
4. **Decision-level audit and replay.** Zep, Graphiti, Supermemory, and Engram keep *data* provenance (fact → source episode/chunk). Mem0 logs ADD events. MemOS proposes version chains. None logs the *decision record*: which candidates were compared, which judge version, and which rationale. That record is needed to replay or contest a supersession. TOKI formalizes this requirement and finds it missing across eight audited systems ([2606.06240](https://arxiv.org/abs/2606.06240)).
5. **An explicit "undecided/disputed" state with later resolution.** Only Cognee ("still disputed") and PreCoMem (PROFILE waiting room) avoid forcing a binary choice ([cognee](https://docs.cognee.ai/core-concepts/main-operations/improve), [2610.29467](https://arxiv.org/abs/2610.29467)). Neither ties the state to a calibrated threshold or an abstention policy at read time.
6. **Source-authority-aware writes.** Except for research prototypes (TMA-NM, PPMF, Nous provenance caps), no production system weights or gates a memory write by the trust level of its source (user statement vs tool output vs web content). Consolidation can launder low-trust sources into "user history" ([2607.29167](https://arxiv.org/abs/2607.29167), [2606.24322](https://arxiv.org/abs/2606.24322)).
7. **Paraphrase-robust dedup without an LLM on the write path.** Mem0 v3 dedups by exact MD5 only ([main.py](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py)). Zep and Graphiti use an LLM per candidate. A cheap calibrated "same-fact" classifier is not documented anywhere.
8. **Evaluation of the write decisions themselves.** All headline numbers are end-to-end QA (LoCoMo, LongMemEval) or recall@k. No benchmark scores per-decision accuracy (duplicate / refine / contradict / unrelated) with gold labels. Nous shows that standard benchmarks barely exercise contradictions ([2606.22030](https://arxiv.org/abs/2606.22030)). Mem0's lowest BEAM category is contradiction_resolution at 35.7 ([docs](https://docs.mem0.ai/core-concepts/memory-evaluation)).

---

## 6. Source index (primary)

arXiv:
- 2504.19413 (Mem0), 2501.13956 (Zep), 2310.08560 (MemGPT), 2504.13171 (Sleep-time compute)
- 2502.12110 (A-MEM), 2506.06326 (MemoryOS), 2507.03724 (MemOS), 2508.19828 (Memory-R1), 2507.07957 (MIRIX)
- 2305.10250 (MemoryBank), 2304.03442 (Generative Agents), 2510.18866 (LightMem), 2509.25911 (Mem-α)
- 2502.05589 (SeCom), 2508.03341 (Nemori), 2405.14831 (HippoRAG), 2502.14802 (HippoRAG 2)
- 2402.09727 (ReadAgent), 2409.05591 (MemoRAG), 2505.24478 (Cognee), 2410.10813 (LongMemEval)
- 2512.12818 (Hindsight), 2601.02163 (EverMemOS), 2605.23986 (MemForest), 2610.02002 (Mem++), 2606.09900 (Engram)
- 2607.26520 (bitemporal store), 2606.06240 (TOKI), 2610.29467 (PreCoMem), 2606.22030 (Nous), 2605.25869 (MemIR)
- 2606.24535 (MemClaw), 2606.24322 (TMA-NM), 2607.29167 (PPMF), 2610.34673 (AgentMemRec), 2606.01435 (post-retrieval assembly)
- 2606.05894 (EMBER), 2605.12493 (LongMemEval-V2), 2603.07670 (survey)

Docs / repos / blogs:
- Mem0: [docs](https://docs.mem0.ai/core-concepts/memory-evaluation), [migration](https://docs.mem0.ai/migration/oss-v2-to-v3), [README](https://github.com/mem0ai/mem0), [main.py](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py), [blog](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm), [research](https://mem0.ai/research)
- Zep / Graphiti: [Zep blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory), [zep-papers#5](https://github.com/getzep/zep-papers/issues/5), [Graphiti README](https://github.com/getzep/graphiti), [Graphiti namespacing](https://help.getzep.com/graphiti/core-concepts/graph-namespacing)
- Letta: [memory blocks](https://docs.letta.com/guides/agents/memory-blocks), [sleep-time blog](https://www.letta.com/blog/sleep-time-compute), [LoCoMo blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- LangMem: [guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/), [API ref](https://langchain-ai.github.io/langmem/reference/memory/)
- Cognee: [improve](https://docs.cognee.ai/core-concepts/main-operations/improve), [README](https://github.com/topoteretes/cognee)
- Supermemory: [how it works](https://supermemory.ai/docs/concepts/how-it-works), [LongMemEval report](https://supermemory.ai/research/longmembench), [research](https://supermemory.ai/research/)
- Memobase: [README](https://github.com/memodb-io/memobase), [LoCoMo results](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md), [profile docs](https://docs.memobase.io/features/profile/profile)
- Benchmark audits: [lhl/agentic-memory](https://github.com/lhl/agentic-memory/blob/main/benchmarks/README.md), [a40-labs/memory](https://github.com/a40-labs/memory)
- Other: [Vectorize Hindsight blog](https://vectorize.io/blog/introducing-hindsight-agent-memory-that-works-like-human-memory), [MemoryOS README](https://github.com/BAI-LAB/MemoryOS)

*Not verified / caveats.* MemOS's conflict-detection algorithm is not described in the paper. Supermemory's "learning model" is proprietary. MIRIX's dedup criterion is unspecified. LangMem's LoCoMo number exists only from third-party harnesses. Mem0 Platform numbers include "proprietary optimizations not available in the open-source SDK" ([README](https://github.com/mem0ai/mem0)).
