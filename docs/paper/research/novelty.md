# Novelty and prior art: agent memory whose decisions are made by Jev (or another calibrated typed-decision model)

Researched 2 November 2026, about seven weeks after Jev's public release on 15 September 2026. I read every source below through web search or web extraction during this session. A **(snippet)** tag means I saw only a search-result snippet, not the full page. Nothing here comes from memory without a source. Where I couldn't find something, the document says so.

---

## TL;DR verdict

- **Using Jev or a System One model to make the decisions inside an agent memory system is not new.** It's the most crowded early use case for Jev. The closest prior work is **Jev-Mem** (UT Dallas, arXiv 2609.23986, 21 Sep 2026). Jev-Mem is a peer-paper-style system in which a System-One controller makes the memory's write-path decisions (typing, redundancy, relation and entity edges, consolidation) and its read-path decisions (routing, candidate scoring, stopping). Probability thresholds gate those decisions, and an LLM is used only for synthesis. **Mnemon** (arXiv 2609.36059) and roughly 15 open-source projects (invalidate, jevmem, jev-memory, dsh-memory-jev, jcode, PerfectRecall, Space, Oracle's notebook, Hindsight, AtMem, and others) were all public before this document. Confidence: **high**.
- **Novelty is in specific mechanisms and in how memx combines them:**
  - asymmetric, string-identity-conditioned priors for entity merge vs. split;
  - a five-way reconcile verdict gated by cosine bands;
  - "grade alone vs. grade with the fact it was reached from, take the higher" for linked facts;
  - correction shadowing scoped to a scope tree (a child scope shadows only at and below itself).

  I found no exact public precedent for any of these four. Each has a close neighbour, listed in §4. Confidence: **medium**. The ecosystem is very large and fast-moving, and much of it is GitHub READMEs and blogs that search engines index unevenly.
- **Recommendation for the paper:** frame memx as a systems contribution and an empirical study of *policy design over calibrated probabilities*, not as "the first Jev memory". Cite Jev-Mem and Mnemon as the nearest prior work. Support the four mechanisms with ablations. Address the contested calibration evidence in §1.4 directly, because memx's thresholds act on raw Jev probabilities.

---

## 1. Jev / System One: facts with sources

### 1.1 Vendor claims (TypeSafe)

| Fact | Value | Source |
|---|---|---|
| Release | "Sep 15, 2026", author Diogo Almeida, founder (says he helped build the research behind ChatGPT at OpenAI) | https://typesafe.ai/blog/introducing-system-one-models-and-jev |
| What it is | "unstructured state in, typed probabilistic decisions out"; gives up string generation; "can't hallucinate" (vendor claim) | same |
| Training | "Reinforcement Learning for Calibrated Decisions (RLCD)", optimising for "answers with epistemically honest probabilities on System One tasks" | same |
| Sampling | "Parallel. Generates all outputs in a single query." | same |
| Price | Input "$0.042 / MTok ($42 per billion tokens)"; output tokens "FREE" | same; also https://docs.typesafe.ai/models ("$42 / $0.042" per Btok/Mtok for both jev-1.14.0 and jev-1.13.0) |
| Latency | "End-to-end response time is 70ms-500ms … 40x-200x faster" than frontier LLMs (quoted at 3–329 s) | blog |
| Headline speed/cost | "193.6x faster, 444.6x cheaper" came from TypeSafe's workflow evals. TypeSafe says "we expect that these are on the higher end of real world gains"; the reference answers are an average of GPT-6 Astra and Fable 5.1 | blog |
| Choice cardinality | "Jev supports a cardinality up to 255"; above that, a two-stage process scores independently and then chooses | blog |
| Primitives | Choice (one of the options, plus a probability for each), Score (ordered rubric), Noul (probability that a yes/no proposition holds) | https://www.langchain.com/blog/jev-agent-evals-langsmith ; https://openrouter.ai/docs/guides/community/jev |
| Endpoint and models | `POST /v1/systemone`; `jev-1.14.0` (available from 14 Oct 2026) and `jev-1.13.0`; aliases `jev-latest` and `jev-preview` both point to 1.14.0 | https://docs.typesafe.ai/models |
| Context | 64k tokens per request, 32k per question; 262k with the `TypeSafe-Danger-Zone-Long-Context: true` header | https://docs.typesafe.ai/models |
| Rate limits | 250,000 tokens/s and 1,200 requests/min **(snippet; third-party mirror)** | https://raw.githubusercontent.com/api-evangelist/typesafe-ai/refs/heads/main/rate-limits/typesafe-ai-rate-limits.yml ; https://opentweet.io/jev/limits |
| Customisation | No fine-tuning or LoRA with customer data; "the same weights serve every account" | https://docs.typesafe.ai/models |
| Languages | English is primary; other languages "handled but not equally well" | https://docs.typesafe.ai/models |
| Confidence statistic | Confidence is computed from the returned distribution. Score confidence = 1 − (probability-weighted distance from the peak) / (the same distance for a uniform spread); worked example (0, 0.57, 0.43) → ≈0.35 | https://docs.typesafe.ai/confidence |
| OpenRouter access | `typesafe/jev-1.14` and `~typesafe/jev-latest`; a Decisions API at `POST /api/alpha/decisions`; TypeSafe SDKs work through a base-URL change | https://openrouter.ai/docs/guides/community/jev ; https://openrouter.ai/docs/guides/community/typesafe-sdk |
| Funding (context only) | HN thread title: "Typesafe AI raises $870M at $7.5B" **(snippet)** | https://news.ycombinator.com/item?id=50023450 |

> Note: the memx brief says OpenRouter `POST /api/v1/systemone`. OpenRouter's own Jev index documents `POST /api/alpha/decisions`, and its SDK guide documents a TypeSafe-compatible "SystemOne" endpoint (https://openrouter.ai/docs/client-sdks/python/sdks/systemone/README, snippet). Check the exact path memx calls before the paper states it.

### 1.2 Documented jaggedness (jev-1.13, last reviewed 2026-10-02)

Source: https://docs.typesafe.ai/model-jaggedness/jev-1.13. TypeSafe lists nine failure modes:

1. literal reading
2. math and numbers ("does not count reliably")
3. date and time comparison ("reads dates as text, not as ordered quantities")
4. indirection
5. "Accuracy falls as the state grows with content unrelated to the decision"
6. adversarial content ("State is data, and `jev-1.13` does not treat it as hostile by default")
7. contradictory instructions and criteria
8. Choice option order
9. generation

It also warns that Score levels are "weak in numerical calibration", so they shouldn't be used to interpolate magnitudes.

*Relevance to memx:*

- Failure mode 5 argues for small neighbour sets in reconcile and recall states.
- Failure mode 6 means the injection check is itself steerable.
- Failure mode 8 means the 5-way reconcile Choice should be checked for option-order sensitivity.
- Failure mode 3 means "replaces" verdicts on time-dependent facts shouldn't depend on Jev comparing dates.

### 1.3 Independent explainers and evaluator studies

- **LangChain, "Can Jev Be a Better Agent Evaluator?"** (Jev-as-a-Judge in LangSmith): Jev's "quality-score variance was 92–913x lower than GPT-5.6 Luna, Terra, and Claude Sonnet 4.6". It "averaged 0.44s and $0.00035/call ($0.34 total vs. $28.17 for Claude)". The test was small: five weather-agent traces with 100 repetitions each, which LangChain calls "promising, but early". https://www.langchain.com/blog/jev-agent-evals-langsmith
- **Victor Dibia, "How Jev works: calibrated decision models"** (newsletter issue #69):
  - Reproduces the "score every option" mechanism on Qwen2.5. Scoring took 0.23 s versus 4.57 s for generating 10-option probabilities.
  - Base models were close to calibrated (ECE 0.05 / 0.02) and instruct models were not (0.18 / 0.10). At 77 options, average confidence was 95% while accuracy was 64%.
  - Generated-JSON probabilities were the worst method measured.
  - Cites an independent Jev ECE of **0.0588** on 662 prompt-injection messages, with 96.5% accuracy when the task is described.
  - Notes that RLCD "has not been published".
  https://newsletter.victordibia.com/p/how-jev-works-calibrated-decision ; https://victordibia.com/explainers/jev/
- **Survey: Chen, Zhang, Yang, Wan, Hong, Wu (Nokia), "A Survey on Jev: Prior Research, Adaptation Costs, and Evidence of System One Models"** (accepted to an ACM venue according to the landing page). Findings:
  - "Jev has no technical report yet."
  - Documented limits: literal reading, arithmetic, dates, multi-hop, distractors, steerable state. "A Boolean question and its negation need not receive complementary probabilities."
  - Calibration: one study found a pooled ECE of **0.047 over 31 binary benchmarks but a median per-benchmark ECE of 0.168**, against 0.074 expected for a perfectly calibrated model at those sample sizes.
  - Agent security: "9.20% of the inputs that Jev scored below 0.1 were labeled unsafe, against a mean prediction of 0.205%."
  - JevBench hard tier: Jev 74.1% vs. 97.3% for GPT-6 Luna (low reasoning effort).
  - Next-action accuracy: 0.763 for Jev vs. 0.884 for GPT-6.1 at about 70× Jev's cost.
  - Vendor example: 13 questions in one call took 0.27 s, against 2.71 s for 13 sequential calls.
  - Four deployment risks, including "a silent model update may change the probability scale on which thresholds were tuned".
  https://evan-py-chen.github.io/jev-survey/jev-survey.pdf (landing: https://evan-py-chen.github.io/jev-survey/)

### 1.4 Calibration studies (the brief's named authors, plus others found)

| Study | Key numbers / finding | Source |
|---|---|---|
| **Deußer, Ramamurthy, Sparrenberg, Sifa** (Bonn / Lamarr), "Evaluating and Benchmarking the System One Model Jev", arXiv 2609.37647 (v2 29 Oct 2026) | jev-1.13.0 zero-shot on 37 datasets: 346,009 requests "for under US$10". 95–99% on IMDB, SST-2, HellaSwag and ARC. Beats Qwen3.8-27B on 27/37 and Gemma-4-E4B on 37/37. "choice probabilities are well calibrated and support selective prediction", **but** "binary probabilities need tuned thresholds" (UNFAIR-ToS micro-F1 0.50 → 0.75). Weak on low-resource languages, fine-grained labels and rubric judgments. | https://arxiv.org/abs/2609.37647 ; https://arxiv.org/html/2609.37647v2 |
| **Yixuan Liu** (TraceStone / NTU), "Evaluating System One Models for Agent Security Decisions: Reliability, Calibration, and Selective Automation", arXiv 2609.33401 | Good average calibration can hide confidently wrong "safe" calls on particular attack groups. Under strict miss budgets few inputs are auto-allowed. Separate allow and block thresholds raise automation "mainly by blocking more". Thresholds that pass validation can fail on unseen inputs. The survey reports R-Judge coverage at a 1% miss budget going from 7.63% to 47.03% with separate thresholds. | https://arxiv.org/html/2609.33401v1 ; survey above |
| **"Guo et al."** | Ambiguous. Both Dibia and Liu cite **Guo, Pleiss, Sun & Weinberger (2017)**, *On Calibration of Modern Neural Networks* (temperature scaling). Liu's related-work section also mentions a "Guo" on alignment-failure detection, but the extract was truncated. The closest Jev paper there is "Just Ask Jev: RLCD as a Zero-Shot Detector of AI Alignment Failures" (arXiv 2609.29429; 44 benchmarks, 10 failure types; submitted by Yi Liu). I couldn't confirm a Jev-specific "Guo et al." calibration paper. | https://arxiv.org/abs/2609.29429 ; https://arxiv.org/html/2609.33401v1 |
| **Ibrahim & Zaki** | A replication of the Ziems et al. social-science annotation suite. Per the evidence audit, they find Jev "better calibrated than most LLMs' verbalized confidence but worse than the best three". Per the author's LinkedIn post, Jev "trails the per-task best LLM on 14 of 15 evaluation tasks, with a median deficit of 11.6 macro-F1 points" **(snippets)**. | https://arxiv.org/html/2609.32160v1 (snippet); https://www.linkedin.com/posts/hazem-ibrahim-817997120_github-hazemibrahim97decision-models-css-activity-7507995554707243008-UteW (snippet); https://github.com/xbill9/gemma4-dev/blob/main/jev/reports/Jev%20independent%20evidence%20review.md (snippet) |
| **"Iqbal et al."** | **Not found.** I searched four Iqbal-specific queries. The only Iqbal hits were LinkedIn posts by Mohsin Iqbal, Temoor Iqbal and Asad Iqbal, none of them calibration studies. | see search log |
| Lingsen Meng (UCLA), calibration audit of Jev 1.13.0 | 3,244 evaluations of 2,372 texts. Top-label calibration "differs between tasks and question forms": calibration error "changes by 0.071 between two ways of posing the same question". | https://escholarship.org/uc/item/4t4449jv |
| Lan, Kaul, Pandurangan, Sanyal, "From Detection to Decision: Calibration and Deferral for Prompt-Injection Detection", arXiv 2610.20907 | 47,611 held-out examples. AUROC: an LLM judge highest (0.897), then Jev (0.871), Prompt Guard 2 86M (0.757) and 22M (0.714). "Jev's native probabilities are not c[alibrated]…" (snippet, cut off). | https://arxiv.org/abs/2610.20907 ; https://arxiv.org/pdf/2610.20907 (snippet) |
| Porcedda, "Jev thinks 'I don't know', but doesn't say it: Sys1Cal-v1", arXiv 2609.35342 | Argues the calibration claim "is not backed by any public test". Introduces a calibration dataset. | https://arxiv.org/abs/2609.35342 |
| Typed Decision Models under Prevalence Shift, arXiv 2610.26325 | Calibration measured at one class prevalence doesn't carry over when prevalence shifts. | https://arxiv.org/html/2610.26325 |
| "Confidence Is Not a Promise", arXiv 2610.36311 | "Calibrated in English, Jev keeps a promise of 95% in 8 of 51 locales"; typing noise breaks calibration **(snippet)** | https://arxiv.org/pdf/2610.36311v1 |
| SlashID, identity resolution | Jev F1 0.95 on large organisations at $0.62 per 1,000 accounts, 0.3 s median latency. **Underconfident on matches:** "right about 95% of the time … yet genuine matches typically score around 0.75 and almost none score 0.95 or above" (snippet). | https://www.slashid.dev/blog/jev-identity-resolution/ |
| Steiner & Bizer (Mannheim), "Entity Matching using Decision Models", arXiv 2610.34626 | Six decision models across eight benchmarks. The best slightly beats GPT-6 Luna and beats Ditto on 6 of 8. "For three of the four strongest decision models, the predicted match probabilities are well calibrated". A cascade that sends the least-confident pairs to LLMs adds +1.9–4.0 F1. | https://arxiv.org/html/2610.34626v1 |
| Vernesystems, Banking77 | "Jev's 1.0 answers were 97.5% right and its 0.6 to 0.7 answers 44%" (snippet) | https://blog.vernesystems.com/jev-calibration-test |

**Implication for memx.** memx's thresholds (0.7 merge/split, cosine bands, recall grade cut-offs) act on raw Jev probabilities. That's defensible only if the paper (a) pins the model version, (b) reports per-decision reliability diagrams on memx's own labelled data, and (c) acknowledges the SlashID underconfidence finding. If true matches cluster near 0.75, a 0.7 merge bar sits right at the edge of that cluster.

---

## 2. Search log

I ran 67 web queries in six batches, plus about 40 page extractions. Hits are summarised; "→" marks the relevance to memx.

| # | Query | What it found |
|---|---|---|
| 1 | TypeSafe AI Jev System One model | TypeSafe docs (System One, Models, llms.txt), Jev Hub explainer, ReDoc API |
| 2 | Jev calibrated decision model RLCD | "Just Ask Jev" arXiv 2609.29429; Dibia explainer; Medium and GoPenAI RLCD posts |
| 3 | typesafe.ai systemone OpenRouter | OpenRouter TypeSafe SDK guide, SystemOne SDK README, Jev index; LangSmith TypeSafe-compatible provider; community repos (typesafe-jev-examples, infinitylogesh/systemone proxy) |
| 4 | Can Jev Be a Better Agent Evaluator LangChain | LangChain blog, plus news coverage ($0.34 vs $28) |
| 5 | victordibia How Jev works | Newsletter #69 and explainer |
| 6 | Jev survey Evan Chen calibration | Survey landing page; UCLA audit; Sys1Cal arXiv; calibration repos (nxtg-ai, sudo-sagar, AnthusAI) |
| 7 | **Jev memory agent** | **Jev-Mem (arXiv 2609.23986 + GitHub + HF + YouTube + Flowtivity)**; Reddit r/LLMDevs "How I use Jev for agentic memory" (403, not read); Medium "Jev Is Changing How I Build Memory" (Space); AI SDK Agents "Jev Memory" pattern → **direct prior art** |
| 8 | **Jev entity resolution** | Exasol blog; jev-ai.pro entity matching; Neo4j/Medium ER test; Southbridge campaign-finance ER; LakeSail; metatext jev-graphrag → prior art for the entity-merge decision |
| 9 | System One model memory agent typesafe | Jev-Mem HTML; LangChain harness blog; DataCamp; Reddit r/PiCodingAgent |
| 10 | **Jev RAG rerank** | askthisguy; Spring AI JevDocumentReranker; Thoughtworks; Jev Gallery; Cheney Zhang (search stopping); jev-rag-benchmark → prior art for recall grading |
| 11 | **Jev deduplication** | ChatPRD dedup workflow; LinkedIn support-ticket dedup; metatext data-cleaning (27 projects) |
| 12 | **typesafe jev mem0** | **mem0-vk commit using Jev for extraction gating** (aurapunk-desktop); mem0's X post on Jev-Mem; Jev Radar; awesome-jev |
| 13 | Jev jaggedness typesafe | Jaggedness page 1.13; Pydantic AI TypeSafeModel; Flavio Copes deep dive |
| 14 | Jev pricing latency typesafe | Launch blog; Flowtivity; Crescent pricing; Requesty; MindStudio |
| 15 | jev-memory npm package remember select compact | **NicolasMontone/jev-memory**; treg jev-memory Claude Code mod; hermes-jev-skills compaction; **jevmem**; jev-memory-selector; pi-observational-memory-jev |
| 16 | Jev knowledge graph memory consolidation contradiction | **lyonwj typed judgments for KG (supersede/alternatives Noul)**; redreamality review of Jev-Mem; herrington unified memory plane; Neo4j navigation; **supermemory blog**; shipwithjev "Dreaming" KG |
| 17 | Jev memory github site:github.com | **perfectrecall**; **Grivn/mnemon-memory-agent**; **jcode MEMORY_ARCHITECTURE**; treg; **jev-lucid-memory**; must-lioncho/jev-mem |
| 18 | System One model knowledge base deduplication typed decisions | Kai Waehner; awesome-system-one; Cloudflare "Clairic" extraction models; zbrain |
| 19 | Jev Hacker News memory | Jev-Mem PDF; Jevmem HN story (49 points per promptzone); shipwithjev Jev-Mem; **Hippo Memory Jev reranking** |
| 20 | news.ycombinator.com Jev TypeSafe | Launch HN (1989 points); Jev 1.14 HN; funding HN; Show HN Jevpipe; Unsiloed document review |
| 21 | Jev Graphiti Zep edge invalidation | Graphiti issues #1728 ("Edge invalidation searches the whole graph, so unrelated facts retire each other"), #1841, #1956; Zep Facts docs. No Jev+Graphiti integration found |
| 22 | Jev supersede contradiction fact update memory | **Oracle blog: Jev + Oracle AI DB to govern agent memory**; **chopratejas/invalidate**; Supersede arXiv 2606.27472; Hindsight consolidation; **AtMem–Jev HF study** |
| 23 | calibrated classifier entity merge threshold asymmetric knowledge graph LLM | **TypeSafe entity-alignment cookbook (3-level Score)**; graph-engineering refs; node-canon; entity-alignment papers |
| 24 | Deußer Jev calibration | arXiv 2609.37647; masturbyte; befailproof; AnthusAI; jujumilk3 audit; alexmolas "Jev can't be calibrated"; prevalence-shift paper; Klosterman DIY Jev |
| 25 | Guo Jev calibration audit | UCLA audit PDF; Lan et al. 2610.20907; "Beyond Answer Confidence" 2610.01006; "Confidence Is Not a Promise" 2610.36311 |
| 26 | Liu Jev agent security prompt injection detection calibration | Liu arXiv 2609.33401; HiddenLayer; Datadog Security Labs (counterfeit policy steers Jev) |
| 27 | Ibrahim Jev calibration | Vernesystems Banking77; anth.us; Sys1Cal; Grazian; Jev in Medicine |
| 28 | Iqbal Jev | Wikipedia "Jev (AI model)"; LinkedIn posts by people named Iqbal (not studies) |
| 29 | Jev prompt injection detection benchmark | jev-sec-bench; jev-injection-bench (11,900 prompts); backnotprop; jev-injection-guard; RLCDAlignBench |
| 30 | Jev retrieval grading relevance score rubric memory recall | Deußer; **Hindsight "We Added Jev as a Reranker"**; LanceDB; Unblocked; AtMem; Opine; LambdaDB |
| 31 | Jev Letta OR Mem0 OR Zep OR Cognee integration | Nothing Jev-specific (generic comparisons only) |
| 32 | Laya System One model NandhaKishorM | Laya open-source System One engine (served over the Jev `/v1/systemone` protocol) |
| 33–40 | Guo et al. / Ibrahim et al. / Iqbal et al. arXiv; ECE benchmark; survey PDF; docs confidence; docs pricing; latency p50/p95 | Ibrahim & Zaki (via 2609.32160); TypedBench 2610.11392; JEV vs LLMs political 2610.06625; System1-Arena 2610.21244; REFLEX 2609.26532; Risk-budgeted decisions 2610.19811; Jeva first principles; JevAgentHarness; LLM-as-Jev 2610.02076; "Releasing the Inner Jev" 2610.17205; jev-xyz latency measurements; Vilinski benchmark |
| 41–44 | `"Jev" "Ibrahim"`, `"Jev" "Iqbal"`, Ibrahim/Iqbal 2610 | SlashID identity resolution; Indonesian transfer 2610.14112; no Iqbal study |
| 45 | mem0 ADD UPDATE DELETE NOOP arXiv 2504.19413 | Mem0 paper (Chhikara et al. 2025) and analyses |
| 46 | Memory-R1 RL memory manager | Memory-R1 repo; ACL 2026 long paper |
| 47 | Zep Graphiti temporal KG edge invalidation | Zep docs and Graphiti issues (as above) |
| 48 | Self-RAG reflection tokens | Self-RAG arXiv 2310.11511 |
| 49 | Corrective RAG CRAG | CRAG arXiv 2401.15884 |
| 50 | RankGPT listwise reranking | RankGPT arXiv 2304.09542 |
| 51 | Ditto entity matching | Ditto arXiv 2004.00584; **Steiner & Bizer, Entity Matching using Decision Models, arXiv 2610.34626** |
| 52 | LLM entity resolution GPT-4 Peeters | Peeters et al., "Entity Matching using LLMs" (EDBT 2025); MatchGPT |
| 53 | selective prediction abstention Geifman | Selective Classification (NeurIPS 2017); SelectiveNet (ICML 2019); conformal selective prediction |
| 54 | generative agents memory scoring Park 2023 | Generative Agents (recency × importance × relevance; LLM-rated importance on 1–10) |
| 55 | NLI contradiction detection KB update | ACL 2023 contradiction QA; knowledgebase_guardian; KG inconsistency repair |
| 56 | A-MEM agentic memory | A-MEM arXiv 2502.12110 (NeurIPS 2025) |
| 57 | calibrated thresholds memory write gate 2026 | **"When to Remember, When to Abstain: Category-Conditioned Retention", arXiv 2610.07100 (Meta)**; PreCoMem |
| 58 | ER asymmetric merge/split cost | ER pattern libraries; Cleon identity ("Over-merge leaks one customer's data to another; over-split fragments one person"); VLDB ER evaluation |
| 59 | Jev multi-hop memory linked facts graded jointly bridge evidence | Jev-Mem; **Mnemon arXiv 2609.36059**; SAG2 (BM25 + Jev multi-hop) arXiv 2610.22588 |
| 60 | Jev memory scope project global override shadowing | **jev-recipes `memory-scope`**; true-mem; "Layered context" (closer wins); jevmem how-it-works |
| 61 | agent memory scope precedence shadowing | Claude Code issue #63430; Codex issue #18343 ("Automatic scope inference … unreliable"); Hermes memory provider docs |
| 62 | Jev entity merge memory graph alias person same name | Hindsight ER in agent memory; HelixDB ("merges two people with the same name … irreversible merge"); Neo4j agent-memory resolution; lyonwj |
| 63 | Jev memory prompt injection gate memory write poisoning | Memory-poisoning study 2606.04329; **Towzai/dsh-memory-jev**; hermes-jev-skills; JevAdvBench 2609.31142; Hidden Risks of Jev 2610.04985 |
| 64 | Jev memory theme category classification facts taxonomy | jev-recipes `memory-value`; jev4k classification; remnic MECE taxonomy |
| 65 | System One decision model memory reconciliation same replaces covers related | **cogmem Reconciliation (ADD/MERGE/REPLACE/REJECT, deliberately biased)**; SurrealDB Spectron reconciliation and supersession; vayl reconciliation report |
| 66 | site:github.com jev memory supersede reconcile neighbour cosine | hippo-memory changelog (Jev rerankers + supersede); MAMA changelog (reconcile with memory scope); gbrain |
| 67 | (Reddit thread fetch) | r/LLMDevs thread blocked (HTTP 403). Not read; listed but not relied on |

---

## 3. Every Jev-in-memory hit, and how close it is to memx

Each row lists the decisions the system hands to Jev and how its gating works. Closeness is rated against memx's specific decisions: category/theme, injection, five-way reconcile with cosine bands, asymmetric entity merge/split, Score-rubric recall with joint bridge grading, and scope-aware shadowing.

| System (date) | Decisions made by Jev / gating | Closeness |
|---|---|---|
| **Jev-Mem**, Jiang, Li, Li (UT Dallas), arXiv 2609.23986, 21 Sep 2026; code: https://github.com/libingzheren/Jev-Mem ; paper: https://arxiv.org/html/2609.23986v1 | System-One control plane over a multi-relational graph, with System Two (an LLM) only for synthesis.<br>**Write path:** 4 overlapping type scores (episodic, semantic, procedural, preference); deterministic top-K candidate discovery (vector + lexical + entity + time), then pairwise Jev for semantic relatedness, causal direction, same-episode and "when necessary, entity equivalence". Exact shared identifiers link deterministically. Edge if P ≥ θ_rel = **0.60**.<br>**Consolidation:** every 20 writes, four Nouls (redundancy, contradiction, obsolescence, link usefulness) plus a Choice (keep_separate / merge / promote / uncertain). Rewriting only if P ≥ **0.85** and contradiction < 0.85.<br>**Read path:** six routing Nouls; candidate scoring **with "currently selected evidence" and the edge the candidate was reached through** (relevance, new_information, relation_usefulness, supports_current_evidence). Stop if evidence_sufficient ≥ **0.95** and missing/contradiction < **0.15**.<br>Results on LoCoMo: LLM judge 0.777 (+11.0%), construction 158 s (6.6× faster), query 0.93 s (−36.7%). | **Very close (nearest prior work).** Same architecture thesis (decision model separate from generator; thresholds on probabilities; write and read both governed). Differences: Jev-Mem extracts no atomic facts (it keeps raw observations); it has no 5-way supersession verdict, no cosine-band gating, no asymmetric entity priors (identical identifiers are always linked), no scope tree or shadowing, and no injection gate. Its traversal scoring conditions candidates on current evidence and the edge, which is close to memx's joint bridge grading. |
| **Mnemon**, arXiv 2609.36059; https://github.com/Grivn/mnemon-memory-agent | Raw dated records, with nothing decided at write time. An LLM plans searches; Jev screens records and then judges them "jointly and with the last six messages" as needed or no longer current. Rules use p ≥ 1/2 and ranking. A background LLM consolidates timelines and value histories. LoCoMo 91.7%, LongMemEval-S 83.8% (gpt-4.1-mini). Reports Jev evidence AUC **0.942** vs 0.900 (DeepSeek) and 0.853 (gpt-4.1-mini). Claims Jev-Mem scores 84.4% under its protocol. | **Close on the recall side** ("no longer current" judgments at read time; a joint state). Opposite on the write side (no write-time reconcile or entities). |
| **chopratejas/invalidate** (created 19 Sep 2026); https://github.com/chopratejas/invalidate | An invalidation layer in front of Mem0, Chroma, Zep and others. "six yes/no votes from Jev, then fixed rules in code" per (memory, event) pair. Checks *all* memories, not the top 10. Questions and instructions "change nothing". Unsure cases go to human review. Eval: 157 cases, 89.2% strict, 0 wrongly retired. | **Close to memx's supersede/replace verdict and injection handling.** Differs: Nouls plus rules rather than a 5-way Choice, no cosine bands, no scopes or entities. |
| **jevmem** (Claude Code/Cursor plugin); https://raw.githubusercontent.com/Avinash-jetwani/jevmem/main/README.md ; https://github.com/Avinash-jetwani/jevmem/blob/main/docs/how-it-works.md | Per turn, Jev Nouls decide the kind (decision, rule, bug…), chit-chat, injection (4 Nouls), duplicate/restatement, and contradiction ("reverses_or_replaces_a_listed_memory"…), plus a Choice "which existing line does it change". Thresholds in config (e.g. contradictionMin 0.7, injectionMax 0.5); `jevmem fit` retunes thresholds from right/wrong labels. Old lines are tagged `[superseded]`. | **Close** (write gate + injection + duplicate + supersede, all thresholds on Jev probabilities). Differs: flat file, no entities, no scope tree, no cosine bands. Threshold refitting from labels goes *further* than memx on calibration practice. |
| **Towzai/dsh-memory-jev**; https://github.com/Towzai/dsh-memory-jev | "every read and every write is a judgement" by Jev. Write gate: `worth_keeping` (Noul) plus `supersedes` (a Choice over candidates + none). Recall gate: a relevance Noul per candidate. Injection gate fails closed. Memory text is never model-generated. | **Close** (whole lifecycle decided by Jev; supersede Choice over neighbours). No entities, scopes or bands. |
| **NicolasMontone/jev-memory** (Vercel AI SDK) and the AI SDK Agents "Jev Memory" pattern; https://github.com/NicolasMontone/jev-memory ; https://www.aisdkagents.com/patterns/jev-memory | Three gates: write (P(durable) ≥ 0.6), retrieve (score every fact in one call, then pack greedily by density into a token budget), evict (P(stale/superseded) ≥ 0.6). Reason strings are deterministic. | Moderate (gates on probabilities; eviction of superseded facts). |
| **jcode** memory architecture; https://github.com/1jehuang/jcode/blob/master/docs/MEMORY_ARCHITECTURE.md | Recall is purely Jev Nouls: no embeddings and no LLM. Threshold must be in 0.8–1.0. Project/global scopes. "Exact duplicate content within the same category/scope reinforces." | Moderate (recall grading; scope partition, but no shadowing). |
| **Space** (Hugo Sequier, Medium, 23 Sep 2026); https://medium.com/@sequierh/jev-is-changing-how-i-build-memory-for-ai-agents-3f204497756f | Up to 25 vector neighbours, then 5 shortlisted. Two Jev judgments per pair (related; directional expands_on), both at **0.9**. Links are invalidated when the claim text changes. Recall follows explicit links one hop, within a slot budget. | **Close on neighbour-relation verdicts** (directional "expands_on" ≈ memx "covers"/"related"). Jev-inferred links are not yet used in recall. |
| **Oracle Developers**, "Using Jev and Oracle AI Database to govern agent memory"; https://blogs.oracle.com/developers/using-jev-and-oracle-ai-database-to-govern-agent-memory | Jev gates query selection, context selection and memory *promotion*. Checks "whether a proposed memory is supported and whether its scope matches the evidence". Advises calibrating thresholds per decision. | Moderate (scope-match check; promotion gate). |
| **William Lyon**, "Typed Judgments for KG Extraction with Jev"; https://lyonwj.com/blog/typesafe-jev-knowledge-graph-extraction | Edge-assertion gate; entity adjudication with a 3-level Score (round(score): unlinked / curator / merge). Finds identical strings split by *type* blocking. An agent-memory layer "supersedes 'I use npm' with 'I've switched to pnpm'" using one Noul ("are a and b alternatives, two options filling the same role?"). | **Close on entity resolution and supersession.** The "identical names in different clusters" finding is the problem memx's split prior addresses, but Lyon solves it with distribution-aware blocking, not an asymmetric prior. |
| **TypeSafe entity-alignment cookbook**; https://docs.typesafe.ai/cookbooks/entity_alignment | One Score with three levels as outcomes (different / related → curator / same → assert sameAs), plus field Nouls. "There is no threshold constant anywhere in this file." | Vendor-recommended ER pattern. It contrasts with memx's probability-threshold design and has no asymmetric priors. |
| **Hindsight** (Vectorize) Jev reranker; https://hindsight.vectorize.io/blog/2026/09/24/adding-jev-reranker-what-we-learned | Listwise Choice over the pool beats per-candidate Nouls (LoCoMo recall@1 0.94 vs 0.87). A cut-off Score ("first 1/2/3/5/10/all") lifts precision 0.051 → 0.850. A "none" option fails. Warns that its scores "are positions, not confidences". | Prior art for recall grading. Directly challenges per-candidate absolute-Score grading such as memx's 5-level rubric. |
| **supermemory**, "Jev changes a lot in memory & context engineering"; https://supermemory.ai/blog/jev-memory-context-engineering/ | Reranking on BEIR: Score-10 is the best Jev method (SciFact 0.751). A Noul-as-delete-gate "failed (it kept nothing)". Sentence-level Noul pre-filter before extraction. | Prior art for recall grading and extraction gating. |
| **AtMem–Jev study** (HF blog); https://huggingface.co/blog/javadtaghia/an-atmem-and-jev-experiment-can-a-judgment-model-h | Jev as a reranker over AtMem's candidates on 1,986 LoCoMo questions. AtMem keeps authority over storage, scope and delivery. | Moderate (recall only; scope governance is in AtMem, not Jev). |
| **PerfectRecall** (Hermes memory provider); https://github.com/arslanr-com/perfectrecall | Jev-powered prefetch and recall over Mnemosyne storage. LongMemEval-S (120 questions) 48.3% → 85.8% "for a frozen predecessor". | Moderate (recall). |
| **jev-lucid-memory**; https://github.com/JayYun98/jev-lucid-memory | Wake–sleep lessons; decision-model gates with 0.8/0.2 thresholds. Merge, revise or archive; uncertain or conflicting candidates stay pending. Says "Decision scores are not calibrated guarantees." | Moderate (two-sided thresholds; merge/revise). |
| **mem0-vk commit** (aurapunk-desktop); https://git.librography.org/datapoint/aurapunk-desktop/commit/475bfacf912a84c79a3b1ea7c222df4510a9f3f9 | Jev keep-Noul at 0.6 refining mem0-style extraction | Low (extraction filter only) |
| Others found but not read in depth: jev-memory-selector; pi-observational-memory-jev; must-lioncho/jev-mem; treg jev-memory mod; hermes-jev-skills; Hippo Memory Jev reranking; shipwithjev "Dreaming" KG; jev-recipes `memory-scope` / `memory-value`; Reddit r/LLMDevs (blocked) | Write filters, recall selection, scope classification (user/project/task/session/unclear) | Low to moderate |

**Count:** at least 15 public systems, two of them arXiv papers, had Jev making memory decisions before 2 Nov 2026.

---

## 4. Adjacent (non-Jev) prior art, and how memx differs

| Line of work | What it does | memx difference |
|---|---|---|
| **Mem0** (Chhikara et al., arXiv 2504.19413) | An LLM extracts facts, then an LLM emits ADD / UPDATE / DELETE / NOOP against retrieved neighbours (https://alphaxiv.org/abs/2504.19413 ; https://github.com/lhl/agentic-memory/blob/main/ANALYSIS-arxiv-2504.19413-mem0.md) | memx replaces a generated op with a calibrated distribution over a 5-way verdict and acts only above thresholds. It also separates "neighbour covers candidate" (drop the new fact) from "candidate replaces neighbour" (supersede). |
| **Memory-R1** (ACL 2026) | RL-trained LLM memory manager choosing ADD/UPDATE/DELETE/NOOP (https://github.com/yansikuan/memory-r1 ; https://aclanthology.org/2026.acl-long.583/) | memx does no RL training of its own. It relies on an off-the-shelf calibrated model and hand-set thresholds. |
| **Zep / Graphiti** | LLM-based edge invalidation with temporal validity windows (https://github.com/getzep/graphiti ; https://help.getzep.com/facts). Open issue: "Edge invalidation searches the whole graph, so unrelated facts retire each other" (#1728). Another: invalidation misses current facts after about 10 superseded ones (#1956). | memx gates invalidation by cosine band and a Jev verdict over a bounded neighbour set, and **scopes** the effect (a child scope shadows ancestors only below itself). That addresses exactly the global-invalidation failure in #1728. |
| **A-MEM** (arXiv 2502.12110) | LLM-generated links and "memory evolution" of neighbours | memx never rewrites a neighbour's text; its links are typed verdicts. |
| **Generative Agents** (Park et al. 2023) | Retrieval = recency + importance (LLM-rated 1–10) + relevance (cosine) (https://3dvar.com/Park2023Generative.pdf) | memx grades relevance with a calibrated 5-level Score instead of a generated integer. |
| **Self-RAG** (arXiv 2310.11511), **CRAG** (arXiv 2401.15884), **RankGPT** (arXiv 2304.09542) | Reflection tokens (ISREL and others); a lightweight retrieval evaluator with Correct/Ambiguous/Incorrect thresholds; listwise generated permutations | memx's recall grade is closest to CRAG's thresholded evaluator, but uses a general calibrated decision model with no task training. Hindsight's evidence suggests listwise beats absolute grading, so memx should justify or ablate its choice. |
| **NLI for knowledge updates** (e.g. ACL 2023 contradictions in QA, https://aclanthology.org/2023.acl-short.72/ ; knowledgebase_guardian, https://github.com/datarootsio/knowledgebase_guardian) | Entail/contradict/neutral classifiers or LLMs flag conflicting KB entries | memx's 5-way verdict is an NLI-like relation extended with *direction* (replaces vs. covers) and *scope*. |
| **Entity matching**: Ditto (arXiv 2004.00584), Peeters et al. (EDBT 2025), Steiner & Bizer decision models (arXiv 2610.34626) | Pairwise match/no-match, symmetric threshold (typically 0.5) | memx's **asymmetric prior** isn't in these: string-different pairs default to split and need P(same) ≥ 0.7; string-identical pairs default to merge and need P(different) ≥ 0.7 to split. Merge/split cost asymmetry *is* known in ER practice (Cleon: "Over-merge leaks one customer's data to another; over-split fragments one person", https://www.wearecleon.com/en/docs/data-cloud/identity ; HelixDB on irreversible same-name merges, https://www.helix-db.com/content/guides/deduplicate-entities-in-an-agent-memory-knowledge-graph). cogmem builds its whole reconcile around a deliberate bias, "When both errors are invisible, choose the one that leaves evidence" (https://cogmem.astracollab.app/docs/reconciliation). Conditioning the *direction* of the burden of proof on string identity is the specific memx twist. |
| **Selective prediction** (Geifman & El-Yaniv 2017; SelectiveNet 2019; conformal selective prediction arXiv 2603.24704) | Abstain below a confidence threshold to control risk vs. coverage | memx's bands, thresholds and "keep both on uncertainty" are a direct application. Not new as a principle. |
| **Category-conditioned retention** (Owolabi et al., Meta, arXiv 2610.07100) | Per-category confidence bars at the memory write boundary. Value/belief assertions are only 77.9% supported vs 96.2% for other categories; a stricter bar on values cuts unsupported retention 6.2% → 4.0%. | Close to memx's (category, threshold) coupling. If memx uses one global bar per decision, this paper argues for making it per-category. |
| **Scope layering** ("Layered context: global → project → task … Closer-to-now wins", https://incontext.info/docs/patterns/layered-context/ ; Codex issue #18343 on scoped memory, https://github.com/openai/codex/issues/18343 ; jev-recipes `memory-scope`, https://jev-recipes.com/recipes/memory-scope/) | Static precedence of nearer scopes; a Jev classifier for a fact's narrowest scope | memx *derives* shadow edges from a Jev "replaces" verdict across scope levels, and they apply only to the subtree. I found no system that does this. |

---

## 5. Honest novelty assessment

### Not new (high confidence)

1. **A calibrated typed-decision model making memory decisions, with a separate LLM for generation or extraction.** This was done first and explicitly by Jev-Mem (21 Sep 2026; "the key novelty of Jev-Mem is the architectural separation of lightweight memory control from deliberative reasoning"). Mnemon and at least 13 OSS systems followed.
2. **Thresholds acting directly on Jev probabilities**, including two-sided and abstain bands. Examples: Jev-Mem (0.60 / 0.85 / 0.95 / 0.15), jevmem (configurable and *refit from labels*), invalidate, jev-memory (0.6), jcode (0.8–1.0), jev-lucid (0.8/0.2). This is also classic selective prediction.
3. **Write-time category/type classification by Jev** (Jev-Mem's 4 type scores, jevmem kinds, jev-recipes memory-value).
4. **A Jev prompt-injection check at the memory write path** (jevmem's 4 injection Nouls, invalidate, jcode, dsh-memory-jev), plus a body of Jev injection-detection benchmarks.
5. **Jev supersession or contradiction against retrieved neighbours** (invalidate, jevmem, dsh-memory-jev's `supersedes` Choice, Lyon's "alternatives" Noul, Jev-Mem's consolidation Choice).
6. **Jev relevance grading at recall**, including Score rubrics (Hindsight, supermemory Score-10, jcode, AtMem, PerfectRecall, Mnemon).
7. **Jev for entity resolution and merging** (TypeSafe cookbook, Lyon, SlashID, Southbridge, Exasol, Steiner & Bizer).

### Plausibly new (medium confidence; no exact precedent found)

1. **Asymmetric, string-identity-conditioned entity priors**: different spellings merge only if P(same) ≥ 0.7; identical names split only if P(different) ≥ 0.7. Neighbours:
   - Jev-Mem always links exact identifiers, so it can't split identical names.
   - Lyon handles identical-string splits through type blocking.
   - The TypeSafe cookbook uses a symmetric 3-level Score.
   - cogmem applies an asymmetric bias, but to statement merges, not entities.

   **Caveat:** the SlashID finding that true matches cluster near 0.75 means the 0.7 bar needs empirical support.
2. **A five-way directional reconcile verdict (same / replaces / covered-by / related / unrelated) gated by cosine bands (0.45–0.95).** Pieces exist: mem0's ops, cogmem's ADD/MERGE/REPLACE/REJECT, Jev-Mem's 4 Nouls + Choice, Space's directional `expands_on`. I didn't find this exact taxonomy, with "neighbour covers candidate" as a distinct outcome, combined with similarity bands that decide whether Jev is asked at all. Novelty here is incremental.
3. **Joint grading of bridge (linked) facts.** A fact reached by following a link is graded both alone and together with the fact it was reached from, and the higher grade is kept (see `tests/integration/test_relationships.py`). Jev-Mem's traversal scoring already conditions candidates on "currently selected evidence" and the traversed edge (relation_usefulness, supports_current_evidence), and Mnemon judges records jointly. **The principle is not new.** The specific max(alone, with-source) rule, and evidence that it helps multi-hop recall, may be. Novelty: low to medium.
4. **Scope-aware shadowing.** A Jev "replaces" verdict between a child-scope fact and an ancestor-scope fact creates a shadow edge that hides the ancestor only at and below the child scope; the ancestor stays visible elsewhere (see `tests/integration/test_pipeline.py::test_child_correction_shadows_ancestor_only_below`). Neighbours: static layered precedence, jev-recipes scope *classification*, jcode/AtMem scope partitions, and Graphiti's global invalidation, whose open issue shows the cost of *not* scoping. **I found no precedent for model-decided, subtree-scoped supersession.** This is the strongest novelty candidate. Medium confidence.
5. **The combination:** a single decision model owning *every* decision across nine-class category, 30-theme topic, injection, banded reconcile, asymmetric entity merge, scoped shadowing and joint recall grading, with a generative LLM used only for atomic-fact extraction. dsh-memory-jev and Jev-Mem come closest to "every decision". Neither has entities with asymmetric priors or scoped shadowing.

### Risks the paper must address

- **Priority:** Jev-Mem (21 Sep 2026) and Mnemon predate memx's public description. Cite them as the closest work and position memx against them, including the LoCoMo / LongMemEval numbers they report: Jev-Mem 0.777 LLM judge on LoCoMo; Mnemon 91.7% LoCoMo and 83.8% LongMemEval-S.
- **Calibration is contested:**
  - median per-benchmark ECE 0.168 (survey);
  - phrasing-dependent calibration error (Meng, Δ0.071);
  - prevalence shift (2610.26325);
  - locale breakage (2610.36311);
  - confident misses on attack groups (Liu);
  - underconfidence on true matches (SlashID).

  "Thresholds act directly on calibrated probabilities" needs memx-specific reliability diagrams, a pinned model version (`jev-1.13.0` or `jev-1.14.0`, not `jev-latest`), and ideally a threshold-refit or ablation study like jevmem's `fit`.
- **Absolute vs. listwise grading:** Hindsight measured listwise Choice beating per-candidate grading (recall@1 0.94 vs 0.87). memx's per-fact 5-level Score should be ablated against a listwise alternative.
- **Choice option order** (jaggedness #8) for the five-way verdict: report robustness to option permutation.

### Bottom line

memx is **not** novel as "a Jev-decided memory". It is one of 15+ such systems, and a published paper already makes that architectural claim. It **may** be novel in four specific decision policies (asymmetric entity priors, banded five-way directional reconcile, max(alone, with-source) bridge grading, and subtree-scoped shadowing) and in applying one decision model to every decision type. I'd rate overall novelty **incremental-to-moderate**, conditional on ablations showing each policy matters. Confidence in "not first": **high**. Confidence in "the four policies lack exact public precedent": **medium**, because this ecosystem is about seven weeks old, very large, and much of it lives in READMEs that search engines index unevenly.

---

## Addendum (2026-11-03): Graphiti ships Jev dedupe

Found while integrating Zep's open-source engine for the benchmark. **graphiti-core added `JevClient` (and an
OpenAI Decisions client) for node dedupe and edge duplicate/contradiction resolution on 2026-10-13** (commit
`ccae88c`, "Add Jev and OpenAI Decisions clients for node and edge dedupe (#1984)"; released in 0.31.x; README section
"Jev dedupe"; example `examples/jev`). Defaults: duplicate threshold 0.7, contradiction threshold 0.8; extraction,
summaries and search stay on a general LLM.

Consequences for the claims above:
- The *most widely used* temporal-graph memory now offers the same split memx is built on (generative model writes,
  decision model decides) for the two decisions that matter most at write time. "A Jev-decided memory" is now a
  mainstream configuration, not a niche one.
- Graphiti's Jev path still makes contradiction decisions **globally per group** (no scope tree), with **one fixed
  threshold** and binary duplicate/contradiction questions (no banded five-way verdict, no asymmetric entity
  priors, no joint link grading). The four memx policies listed above remain without an exact precedent there.
- It gives a clean experiment: the same engine with an LLM judge vs with Jev (run in `eval/longmemeval/run_graphiti.py`
  and `eval/beam/run.py`, configs `zep_llm` / `zep_jev`).

Source: https://github.com/getzep/graphiti (README "Jev dedupe"), `graphiti_core/llm_client/jev_client.py`.

---

## Sources not reachable or not found

- Reddit r/LLMDevs "How I use Jev for agentic memory and retrieval": HTTP 403, not read.
- "Iqbal et al." (Jev calibration): no such study found after four targeted queries.
- "Guo et al." (Jev-specific): unresolved. Only Guo et al. 2017 (temperature scaling) was confirmed as cited.
- The Jev survey PDF extraction returned excerpts, not the full text. Numbers quoted above appear verbatim in the extracted excerpts.
