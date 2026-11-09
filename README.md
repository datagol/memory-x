# memx

Scoped, auditable long-term memory for DataGOL agents.

memx turns conversations and documents into small, atomic facts stored in Postgres. It decides whether each new fact repeats, replaces or adds to what is already known. It links facts through the people, teams and systems they mention. When an agent needs context, memx hands it a markdown memory block: the whole memory when it fits the token budget, or a ranked slice chosen for the question when it does not.

Every decision along the way is recorded, so you can always answer "why does the agent believe this?" and "why did it get this fact and not that one?".

```
insert ─► extract (Haiku 5.5) ─► sort (Jev) ─► reconcile (pgvector + full text, Jev) ─► commit (Postgres)
recall ─► budget gate ─► whole memory
                      └► search ─► walk through entities ─► grade (Jev) ─► decay ─► fit budget ─► render
```

## Contents

- [Principles](#principles)
- [Quick start](#quick-start)
- [How an insert works](#how-an-insert-works)
- [How recall works](#how-recall-works)
- [Concepts](#concepts)
- [Dashboard](#dashboard)
- [Python library](#python-library)
- [HTTP API](#http-api)
- [Command line](#command-line)
- [Configuration](#configuration)
- [Data model](#data-model)
- [Audit, privacy and safety](#audit-privacy-and-safety)
- [Deployment](#deployment)
- [Tests](#tests)
- [Evaluation](#evaluation)
- [Project layout](#project-layout)
- [Lessons from building it](#lessons-from-building-it)
- [Known gaps and next steps](#known-gaps-and-next-steps)

## Principles

- **Facts are atomic and never edited.** "The user prefers dark mode" is one row. A correction adds a new row and marks the old one as replaced, so history is never lost.
- **Scopes are paths.** `memory.company_2.user_3.agent_1334` sees its own facts plus everything stored at `memory.company_2.user_3`, `memory.company_2` and `memory`. A narrower scope can correct a broader one without changing it for anyone else.
- **Models do what they are good at.** A generative model (Haiku 5.5) reads text and writes candidate facts. Jev, a judge model that returns calibrated probabilities instead of text, makes every yes/no and multiple-choice decision: category, duplicate or replacement, entity merges, relevance.
- **Cheap rules first, the judge for the hard middle.** Exact duplicates and near-identical text are settled by hashing and similarity bands. Jev is only asked about the ambiguous cases.
- **Everything is explainable.** Each insert keeps a trace of every candidate, neighbor and verdict. Each recall keeps the rank, grade, score and reason for every fact it considered. The dashboard shows both.
- **Postgres is the only store.** Facts, vectors, full-text indexes, the entity graph, the job queue and the audit logs all live in one Postgres 16 database with pgvector.

## Quick start

You need Docker and an [OpenRouter](https://openrouter.ai/keys) API key. One key covers all three models: Haiku 5.5 for extraction, OpenAI embeddings and Jev.

```bash
git clone https://github.com/datagol/memory-x && cd memory-x
docker compose up
```

That builds the image and starts everything:

| | |
|---|---|
| Dashboard and REST API | http://localhost:8000 |
| MCP server | http://localhost:8100/mcp |

On first start without a key, the dashboard and every tool say what to set. Add the key and restart in the background:

```bash
cp .env.example .env          # paste your key after OPENROUTER_API_KEY=
docker compose up -d
```

Or pass it on the first start: `OPENROUTER_API_KEY=sk-or-... docker compose up`. Then open the dashboard, go to **Add memory**, paste a conversation or upload a PDF, and try **Ask**.

| | |
|---|---|
| What runs | `postgres` (16 + pgvector), `api` (REST API and dashboard), `worker` (runs queued inserts), `mcp` (MCP server) |
| Database setup | Automatic: migrations run when the API or worker starts |
| Data | The `memx-pg` volume. `docker compose down` keeps it; `docker compose down -v` deletes it |
| Logs | `docker compose logs -f api worker` |
| Upgrade | `git pull && docker compose up -d --build` |
| Port taken | Set `MEMX_PUBLISH=127.0.0.1:8080` (dashboard) or `MEMX_MCP_PUBLISH=127.0.0.1:8180` (MCP) in `.env` |
| Auth | Off by default, and the ports are bound to this machine only. Set `MEMX_API_TOKEN` before exposing them (`MEMX_PUBLISH=8000`); it protects the API, the dashboard and the MCP server. Paste it in the dashboard under Settings |

Without a key the stack still starts: the dashboard and every API call say what to set, and the worker waits. `docker compose up -d postgres` alone never needs a key.

### Connect an MCP client

The `mcp` service speaks MCP over streamable HTTP. Claude Code:

```bash
claude mcp add --transport http memx http://localhost:8100/mcp
# with MEMX_API_TOKEN set:
claude mcp add --transport http memx http://localhost:8100/mcp --header "Authorization: Bearer <token>"
```

Other clients take the same URL in their MCP server settings. Tools:

| Tool | What it does |
|---|---|
| `recall` | Memory relevant to a question, as a markdown block plus the facts and their ids |
| `remember` | Store what someone said; returns the facts that were extracted |
| `remember_conversation` | Store a whole exchange, attributed per speaker |
| `remember_document` | Store the facts in a document's text |
| `ingest_status` | Result of an insert that was still processing |
| `list_facts`, `fact_history`, `list_scopes` | Browse memory and see how a fact changed |
| `forget_fact` | Delete one fact (marked destructive, so clients ask first) |

Every tool takes an optional `scope`; without it the server uses `memory.default` (`MEMX_MCP_SCOPE`). If the client reaches the server by a name other than localhost, add that name to `MEMX_MCP_ALLOWED_HOSTS`. For a client that starts servers as local processes, run it over stdio instead: `memx-mcp` (installed with the package; reads the same environment variables).

### Demo mode

Demo mode swaps the models for deterministic stand-ins. It needs no API key and costs nothing; results are only illustrative. It runs as its own project with its own database, so demo facts never mix with real ones:

```bash
docker compose -f compose.demo.yml up -d --build   # http://localhost:8001, MCP on :8101 (MEMX_DEMO_PORT, MEMX_DEMO_MCP_PORT)
docker compose -f compose.demo.yml down -v         # remove it
```

### Library and CLI without Docker for the app

For development, run only Postgres in Docker and memx from source (Python 3.12 and [uv](https://docs.astral.sh/uv/)):

```bash
docker compose up -d postgres                     # localhost:5433
uv venv --python 3.12 && uv pip install -e '.[dev]'
.venv/bin/memx migrate

.venv/bin/memx insert memory.acme.alex --text "I lead the data platform team. We never deploy on Fridays."
.venv/bin/memx render memory.acme.alex
.venv/bin/memx recall memory.acme.alex.agent1 "can I ship this on Friday?"
```

## How an insert works

An insert takes a conversation (`Messages`, a list of turns with a speaker each) or a document (`Text`) and a scope. It runs as one resumable job with four steps. If a step fails, a retry resumes where it stopped.

### 1. Extract

Long input is split into windows of about 2,500 tokens on turn or paragraph boundaries. Each window carries the previous two turns as read-only context, so "he" and "that project" can be resolved. Windows are extracted in parallel by Haiku 5.5, which must call one tool, `record_facts`, so the output is always structured.

The extraction prompt asks for:

- **strictly atomic facts**: one subject and one attribute each, named in an `attribute` field ("role", "email", "spouse", "headquarters"). "Dana Whitfield is CEO of Northwind Care in Brentwood, TN; her email is …" is wrong; it is three facts, and the location belongs to Northwind Care. A list answering one attribute (the states a company operates in) stays one fact;
- a change of value as **one** fact naming both values ("the on-call channel changed from #ops-alerts to #sre-pager"), never a current fact plus a separate "was previously" fact;
- every stated fact recorded, none skipped as unimportant, and every detail kept: numbers, names, places, dates, versions. Nothing the input doesn't state (no computed counts or summaries);
- reasons folded into the fact they explain ("prefers light mode because dark mode hurts their eyes");
- relative dates resolved against a list of precomputed calendar anchors, so "next week" becomes an actual date;
- from the assistant, only what was given to the user (recommendations, answers, plans, specific names and numbers); its general explanations are skipped, and a list it gives is **one** fact naming every item in order;
- the named entities each fact mentions, in their fullest known form, with a type;
- instructions found in the input treated as content, never followed.

A window whose answer hits the 25-fact cap is continued: the model gets the facts already recorded and is asked for the ones after them, so no output is paid for twice. Only an answer that runs out of output tokens makes the window split in half and extract again (a single very long message is split at paragraph or sentence boundaries). Candidates repeated across windows are collapsed before anything touches stored memory.

Three things keep extraction cheap. The tool schema asks only for the text, its attribute, the speaker, an event date when there is one, and named entities. The system prompt and tool schema are sent as a cached prompt prefix, so most input tokens bill at the cache-read price. Before extraction, one Jev request per window asks which turns say anything worth remembering; turns Jev is sure say nothing ("thanks!") are left out (`MEMX_EXTRACT_GATE`). Queued inserts can also extract through the provider's batch API at half the token price, with results within 24 hours (`MEMX_EXTRACT_BATCH`); the job waits for the batch without using up its retries.

### 2. Sort

For each candidate, one Jev request asks three independent questions:

- **Category:** one of nine (see [Categories](#categories)). Below 45% probability the fact goes to "Other".
- **Theme:** one of 30 fixed themes, such as "Food and drink" or "Cloud and infrastructure". Below 45% the fact gets no theme.
- **Injection check:** does the text contain instructions aimed at an AI system? At 50% the fact is stored but flagged and kept out of recall. At 85% it is rejected.

### 3. Reconcile

memx looks for existing facts close to the candidate: vector search and full-text search over the scope chain, plus facts in narrower scopes below it. Each neighbor falls in a band by cosine similarity:

| Similarity | Decision | Jev call? |
|---|---|---|
| Same normalized text | Duplicate | No |
| Above 0.95 and no new content words, numbers or negation | Duplicate | No |
| 0.45 to 0.95, or above 0.95 with new words | Jev decides | Yes, one request for all neighbors |
| Below 0.45 | Unrelated | No |

Jev picks one of five verdicts for each neighbor it is asked about:

| Verdict | Meaning | What happens |
|---|---|---|
| same | Nothing new | The existing fact is reinforced; the candidate is dropped |
| candidate replaces | They conflict and the candidate is newer, or more specific | Same scope: the old fact is marked replaced. Broader scope: the old fact is hidden for this scope only (a *shadow*). Narrower scope: the old fact is retired below this scope |
| neighbor covers | The stored fact already says it, or is the newer of two | The candidate is stored as history, already replaced |
| related | Same subject, both true | A related link is recorded |
| unrelated | Different subjects | Nothing |

Destructive verdicts need more confidence than linking: **same** needs 0.8, **candidate replaces** and **neighbor covers** need 0.9. Below its floor a verdict becomes **related**. Three code rules also hold back a replacement at any probability: the two facts name different attributes ("page count" vs "reading progress"), both came from the same extraction window, or exactly one of the two only describes an earlier state ("the previous channel was …" against "alerts go to #sre-pager"). The receipt counts them as `held_back`, and the insert trace on the dashboard shows each one with its reason. The floors come from measuring Jev against a larger reference model on logged decisions: below 0.9, destructive verdicts agreed only about half the time, and changed label when the options were reordered.

Because every verdict is logged with its probabilities, a changed policy can be applied to memory that already exists: `memx rereconcile` re-reads the verdict behind every replacement and shadow under the current policy and undoes the ones it would not make, without calling Jev or re-extracting. Verdicts behind a replacement that is still in effect are kept past the decision-log retention, so this always works. `memx expiry-reapply` does the same for event expiry: facts stored when upcoming events expired after 30 days by default get the current setting, and ones that expired only because of it come back.

Entity names are resolved in the same step. For each name, cheap code finds up to three known entities it might be, and Jev decides; code never merges two different spellings on its own. Candidates come from four places:

| Pairing reason | Example | Found by |
|---|---|---|
| Identical name that could be someone else | "Priya" at work vs "Priya" from the bakery | Same normalized name, when it is a single-word person name or the types differ |
| One name inside the other | "Priya" vs "Priya Lindqvist" | Shared words |
| Alternative name | "Bangalore" vs "Blore" | Haiku lists other names it is confident of ("Blore" → "Bangalore", "Bengaluru") |
| Similar name | "Kubernetes" vs "K8s" | Name embeddings at 0.5 similarity or more |

Jev gets one of three wordings depending on the reason, all in one request per fact, with up to three facts about each known entity. Full names and codes ("Marcus Oyelaran", "ARCHIVE-02") and names already used earlier in the same insert are linked without asking.

The two kinds of pair start from opposite defaults. Different spellings stay apart unless Jev is at least 70% sure they are the same, because a wrong merge is worse than a missed link. Identical names stay together unless Jev is at least 70% sure they differ; only then is a second entity with the same name created. A merged spelling is kept as an alias, and the longer spelling becomes the display name. Lowercase descriptions such as "the bakery down the street" are not entities, while descriptive team names such as "the data platform team" are.

### 4. Commit

Each candidate commits in its own transaction under a per-scope advisory lock. Inside the lock memx checks again for duplicates written by a concurrent insert, then inserts the fact, applies replacements and shadows, links entities, and writes every Jev decision to the decision log.

The receipt counts what happened: added, dropped as duplicates, stored as history, replaced, shadowed and rejected.

## How recall works

`recall(scope, query)` returns a markdown block for an agent at `scope`.

### Budget gate

memx adds up the tokens of every visible fact in the scope chain. If they fit in 90% of the budget (4,000 tokens by default), the agent gets the whole memory in **full** mode, no model calls. Otherwise recall switches to **ranked** mode.

A fact is visible when it is active, not expired, not flagged as an injection, and not shadowed by a correction in a narrower scope of the chain. `as_of` evaluates all of this at a past moment, which is how replays and benchmarks work.

### Ranked mode

1. **Search.** Vector search (40 hits) and full-text search (20 hits) over the visible facts, merged by reciprocal rank.
2. **Walk.** From the top 10 search hits, memx follows shared entities for up to two hops. A hit's score is split across the other facts that name the same entity and halved per hop. Entities named in more than 20% of visible facts, like the company's own name, are not followed at all. Shared themes count too, at 30% weight. Up to 8 facts the search missed are added, each with the path that reached it, such as "Marcus Oyelaran approved retiring ARCHIVE-02 → Marcus Oyelaran → Marcus leads the data platform team → data platform team → the team owns all database migrations".
3. **Grade.** One Jev request rates every candidate on five levels, from irrelevant to critical. A fact reached through an entity is judged *together with the fact it is linked to*; judged alone, "Priya Lindqvist leads the Security Team" looks irrelevant to a question about Snowflake access. A search hit that also has a link is graded alone and with the link, and keeps the higher grade.
4. **Cut.** Search hits need a relevance of at least 0.3 ("background" is not enough). Walked facts need 0.5 ("useful").
5. **Score.** `relevance × (0.6 + 0.4 × freshness) × (0.5 + 0.5 × trust) × recall-history boost`, times 0.85 for walked facts so a direct hit wins ties.
6. **Fit.** Facts are taken in score order until the budget or the 30-fact limit is reached.
7. **Render.** Picked facts are grouped by category, related facts nested together and dated (so two disagreeing statements show which is newer), events prefixed with their date. In ranked recall the most relevant come first: sections are ordered by their best-scoring fact and facts by score, so the model reads the best match first. A full-memory block has no question to rank by, so it uses a fixed section order and orders facts by trust and freshness. Either way the same input renders the same way.
8. **History, when asked.** The grade request also asks whether the question is about an earlier or changed state ("what was my previous goal?"). If so, the versions each picked fact replaced are listed under it, dated, within the budget.

Facts that are returned get their recall count bumped, which slowly raises their score (up to ×1.3).

### Example memory block

```markdown
# Memory: memory.acme.alex

## Identity
- Sam Rivera is the CTO of Acme.
- Dana Kim leads the DevOps team at Acme.

## Rules
- Technical questions should be directed to Sam Rivera.

## Events
- (2026-11-03) The user is on vacation on 2026-11-03.
```

## Concepts

### Scopes

A scope is a dot path of any depth, such as `memory.{tenant}.{user}.{agent}`. Labels may contain letters, digits, `_` and `-`, up to 64 characters each. The label under `memory` is the **tenant**: entities are shared within a tenant and never across tenants.

- **Reads inherit downward.** An agent at `memory.acme.alex.agent1` sees facts from all four levels. Siblings never see each other.
- **Writes go to the exact scope given.**
- **Corrections respect scope.** A fact at a narrower scope that contradicts a broader one hides the broader fact for that branch only (a shadow). A broader correction retires contradicting facts stored below it, so "I don't like chicken anymore" at the user level retires "likes chicken" at the agent level.
- **Promote** copies a fact to a broader scope, reconciles it there and marks the original as replaced.

### Categories

| Category | Definition given to Jev | Half-life |
|---|---|---|
| Identity | A stable attribute of a person or organization: name, role, title, location, language, background, expertise | never fades |
| Rule | A constraint, policy, requirement or prohibition: must, must not, always, never | never fades |
| Preference | What someone likes, dislikes, or how they want things done or presented | 365 days |
| Goal | An outcome someone wants, a plan, an intention, or a project they are pursuing | 180 days |
| Relationship | A connection between people, teams or organizations | 365 days |
| Work context | Tools, systems, datasets, repositories, environments, or the current state of ongoing work | 120 days |
| Procedure | A method, workflow or sequence of steps | 365 days |
| Event | Something that happened or is scheduled at a particular time | 30 days |
| Other | None of the above, too vague, or flagged | 90 days |

Half-lives only affect ranking. A fact's age counts from when it was observed or last repeated, so saying something again keeps it fresh.

### Events and time

Every fact keeps `observed_at`, the time it was said, not the time it was stored, so backfilled transcripts reconcile correctly. Events also keep the date they refer to.

Events never expire by default; they fade through the 30-day half-life instead, so history stays answerable. A scope can opt in to expiring upcoming events ("on PTO the week of November 9") a set number of days after their date.

### Speakers and trust

Each fact records who said it, and trust feeds the score:

| Speaker | Trust | Score factor |
|---|---|---|
| user | 1.0 | ×1.00 |
| a named participant, such as "Priya" in a meeting transcript | 1.0 | ×1.00 |
| document | 0.8 by default, settable per insert | ×0.90 |
| assistant | 0.6 | ×0.80 |

The speaker also decides who "I" is. Haiku writes facts in the third person from each turn's speaker.

### Entities and themes

Entities are the people, organizations, teams, systems, projects, places and products a fact names. They connect facts without changing them. Entities are keyed by tenant, so company-level and user-level facts in one tenant share them, and two tenants never do. Generic words ("the user", "the manager", "the meeting") are never entities.

Themes come from a fixed list of 30 so that "food", "diet" and "cuisine" become one theme instead of three. Jev chooses the theme in the same request that picks the category.

Supersede and shadow links are never walked during recall; old versions must not come back into an answer. Jev's "related" links are used only to group bullets in the rendered block.

## Dashboard

The API serves a web console at `/`, built from `web/` (React, Vite, React Flow) into `memx_api/static`.

| Page | What it shows |
|---|---|
| Overview | Facts by category, new facts per day, inserts, recalls, Jev decisions, entities and links |
| Memory | The exact block an agent receives at a scope, next to every fact behind it. Filter, open, delete, or forget a whole scope |
| Relationships | Facts grouped into boxes under the person, team or system they are about; facts with no shared entity grouped by theme. Hubs (names in most facts) listed above. Click anything for a side panel with its links; drag entities, facts and boxes to rearrange. "Group by scope" shows one box per scope instead |
| Add memory | Conversation or document input, with a scope field. Documents can be pasted, or uploaded or dropped as PDF or text files; the extracted text fills the box for review before anything is stored. After insert, the **insert trace**: windows, candidates, Jev's category and theme probabilities, entities found and merged, every neighbor on a similarity axis against the bands, and the decision in plain words |
| Ask | The **retrieval path** from question to memory block, including the walk; a funnel of how many facts survived each step; and a table with every candidate's ranks, how it was found, its path, relevance, freshness, trust, score and why it was picked or cut |
| Activity | Every insert, question and job; open any to replay it |
| Settings | API token and every threshold the server runs with |
| Fact drawer | Version history with the decisions behind each change, names, theme, links, promote and delete |

Develop the UI against a running API:

```bash
cd web && pnpm install && pnpm dev     # http://localhost:5173, proxies /v1 to :8000
pnpm build                              # writes memx_api/static
```

### Per-scope settings

Most settings are server-wide and set in `.env` (see [Configuration](#configuration)). Four can be overridden for one scope and every scope below it. A scope uses the value from its nearest ancestor that sets one, otherwise the server default. Set them on the dashboard's Memory page under **Settings for this scope**, or from code or the API:

| Setting | Server default | Effect |
|---|---|---|
| `budget_tokens` | `MEMX_BUDGET_TOKENS` | Tokens an agent here may receive; decides full versus ranked recall |
| `event_ttl_days` | `MEMX_EVENT_TTL_DAYS` | Days after its date that an upcoming event expires (new inserts); unset means never |
| `text_trust` | `MEMX_DEFAULT_TEXT_TRUST` | Trust for documents that don't set one (new inserts) |
| `walk_enabled` | `MEMX_WALK_ENABLED` | Whether ranked recall walks shared entities |

```python
await mem.set_scope_settings("memory.acme", budget_tokens=2000, walk_enabled=False)
await mem.set_scope_settings("memory.acme.alex", budget_tokens=None)      # inherit again
(await mem.scope_settings("memory.acme.alex.agent1"))["effective"]["budget_tokens"]
# {"value": 2000, "from": "memory.acme", "default": 4000, "env": "MEMX_BUDGET_TOKENS"}
```

Everything else stays server-wide on purpose. Thresholds change behavior for every agent at once. The embedding model, categories and themes can't change without reprocessing stored facts. A recall's `explain` lists any scope overrides it used.

## Python library

```python
from memx import Memory, Messages, Text, Turn, load_text

async with await Memory.connect() as mem:
    receipt = await mem.insert(
        "memory.acme.alex.agent1",
        Messages(turns=[Turn(role="user", content="btw I switched to light mode, the dark one hurts my eyes")]),
        mode="inline",                       # or "background" / "queued"
    )
    await mem.insert("memory.acme", Text(text=open("onboarding.md").read(), source="onboarding", trust=0.9))
    await mem.insert("memory.acme", load_text("handbook.pdf", trust=0.9))   # PDF or text file; source = file name

    result = await mem.recall("memory.acme.alex.agent1", "which theme should the dashboard use?", explain=True)
    print(result.mode, result.markdown)
```

| Method | Purpose |
|---|---|
| `insert(scope, Messages \| Text, mode=, observed_at=, idempotency_key=)` | Ingest input. Resending the same key returns the original ingest; resending a failed one retries it from saved progress |
| `wait(ingest_id)` / `receipt(ingest_id)` | Wait for or check a background or queued ingest |
| `recall(scope, query, recent_turns=, budget_tokens=, categories=, k=, force_mode=, as_of=, explain=)` | Memory block for an agent. `explain=True` returns the full decision record |
| `render(scope)` | The whole visible memory as markdown |
| `facts(scope, status=, category=, include_ancestors=)` / `get_fact(id)` | Raw facts, with their entities |
| `entities(scope, view=)` | Entities named around a scope, with fact counts |
| `history(fact_id)` | Version chain plus the decisions behind each change |
| `forget(scope= \| fact_id=, descendants=, requested_by=)` | Hard delete, recorded in the forget log |
| `promote(fact_id, to_scope)` | Move a fact to a broader scope |
| `export_okf(scope)` | Open Knowledge Format zip of a scope subtree |
| `reresolve_entities(tenant=None)` | Repair pass: fill in missing name embeddings, pair stored entities that may be the same (shared words, alternative names, similar names), ask Jev, and merge agreed pairs into the entity with more facts. All tenants unless one is given |
| `scope_settings(scope)` / `set_scope_settings(scope, **values)` | Read the effective settings at a scope (each value with the scope it comes from), or override them; `None` inherits again |
| `sweep()` | Expire stale events and prune old logs. Runs hourly in any job runner |

`load_text(path, source=, trust=)` reads a PDF (via pypdf, page by page) or a plain-text file (.txt, .md, .csv, .json, …) into a `Text` input. `read_document(bytes, filename)` does the same for bytes in memory. A PDF with no text layer (a scanned image) raises `DocumentError`; run OCR first. Long documents are fine: extraction reads them in windows, up to `max_input_chars`.

`memx.testing` provides `HashEmbedder`, `SentenceExtractor` and `FakeJudge` for tests without API calls.

## HTTP API

All `/v1` routes need `Authorization: Bearer $MEMX_API_TOKEN` when a token is set. Errors return readable JSON: 404 for a missing fact or ingest, 413 for oversized input, 422 for a bad scope, 502 when a model provider refuses (for example a spent API limit) and 503 when it is unreachable.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/scopes/{scope}/ingests` | Insert. Body `{"input": {...}, "mode": "inline"}`. Honors `Idempotency-Key`; returns 202 unless inline |
| POST | `/v1/documents/extract?filename=` | Raw PDF or text file as the body (up to 25 MB). Returns `{text, kind, pages, empty_pages, chars, too_large}`. Stores nothing; post the text as a `text` input to ingest it |
| GET / PATCH | `/v1/scopes/{scope}/settings` | Per-scope settings. PATCH body `{"budget_tokens": 2000, "walk_enabled": null}`: a value overrides, `null` inherits again |
| GET | `/v1/ingests/{id}` | Receipt |
| POST | `/v1/scopes/{scope}/recall` | Body `{"query", "budget_tokens", "force_mode", "as_of", "categories", "explain"}` |
| GET | `/v1/scopes/{scope}/render` | Whole memory as markdown |
| GET | `/v1/scopes/{scope}/facts` | Facts, filterable by status and category |
| GET | `/v1/scopes/{scope}/export.okf` | OKF zip |
| GET | `/v1/facts/{id}`, `/v1/facts/{id}/history` | One fact; its version history |
| POST | `/v1/facts/{id}/promote` | Body `{"to_scope"}` |
| DELETE | `/v1/scopes/{scope}?descendants=true`, `/v1/facts/{id}` | Forget. `X-Requested-By` is logged |
| GET | `/healthz`, `/readyz` | Liveness; database and queue readiness |

Read-only routes used by the dashboard: `/v1/config`, `/v1/scopes`, `/v1/stats`, `/v1/ingests`, `/v1/ingests/{id}/trace`, `/v1/recalls`, `/v1/recalls/{id}`, `/v1/graph`, `/v1/entities`, `/v1/jobs`.

Insert body examples:

```json
{"input": {"kind": "messages", "turns": [{"role": "user", "content": "I'm on PTO next week"}]}, "mode": "inline"}
{"input": {"kind": "text", "text": "…onboarding doc…", "source": "onboarding", "trust": 0.9}}
```

## Command line

```bash
memx migrate                                   # apply database migrations (safe to run repeatedly)
memx worker [--concurrency 4]                  # process queued inserts and hourly sweeps
memx insert SCOPE --text "..." [--as-text]     # ingest inline and print the receipt
memx recall SCOPE "question" [--json]
memx render SCOPE [--show-scope]
memx history FACT_ID
memx export SCOPE out.zip
memx entities-reresolve [--tenant company_2]  # repair pass for entity name merges; all tenants by default
memx rereconcile [--scope S] [--dry-run]       # re-apply the current reconcile policy to logged verdicts
memx expiry-reapply [--scope S] [--dry-run]    # recompute event expiry under the current settings
memx sweep
memx serve [--port 8000]
```

## Configuration

Every field of `memx/config.py` can be set as an environment variable `MEMX_<FIELD>` or in `.env`.

### Connection and models

| Variable | Default | Meaning |
|---|---|---|
| `MEMX_DSN` | `postgresql://memx:memx@localhost:5433/memx` | Postgres connection |
| `OPENROUTER_API_KEY` | | Key for all three models |
| `MEMX_PROVIDERS` | `openrouter` | `demo` uses deterministic stand-ins |
| `MEMX_EXTRACTOR_MODEL` | `anthropic/claude-haiku-5.5` | Extraction model |
| `MEMX_EMBEDDING_MODEL` / `MEMX_EMBEDDING_DIM` | `openai/text-embedding-3-small` / `1536` | The dimension is fixed by the first migration; change it only on a fresh database |
| `MEMX_JEV_MODEL` | `jev-1.14` | Judge model |
| `MEMX_API_TOKEN` | | Bearer token for the API; unset means no auth |

### Insert

| Variable | Default | Meaning |
|---|---|---|
| `MEMX_INSERT_MODE` | `background` | `inline` (in the request), `background` (runner in this process) or `queued` (separate `memx worker`) |
| `MEMX_MAX_INPUT_CHARS` | `400000` | Larger inputs return 413; split them by session |
| `MEMX_WINDOW_TOKENS` | `2500` | Extraction window size |
| `MEMX_USER_TRUST` / `MEMX_ASSISTANT_TRUST` / `MEMX_DEFAULT_TEXT_TRUST` | `1.0` / `0.6` / `0.8` | Trust per speaker |
| `MEMX_INJECTION_FLAG` / `MEMX_INJECTION_REJECT` | `0.5` / `0.85` | Injection thresholds |
| `MEMX_TOPIC_MIN` | `0.45` | Minimum probability for a theme |
| `MEMX_EXTRACT_GATE` / `MEMX_GATE_SKIP_BELOW` | `true` / `0.1` | Skip turns Jev is sure say nothing worth remembering |
| `MEMX_EXTRACT_BATCH` / `MEMX_EXTRACT_BATCH_POLL_SECONDS` | `false` / `60` | Queued inserts extract through the batch API (half price, up to 24 h) |

### Reconcile and entities

| Variable | Default | Meaning |
|---|---|---|
| `MEMX_BAND_LOW` / `MEMX_BAND_HIGH` | `0.45` / `0.95` | Similarity bands |
| `MEMX_VERDICT_SAME_MIN` | `0.8` | Probability needed to drop a candidate as a duplicate |
| `MEMX_VERDICT_REPLACE_MIN` / `MEMX_VERDICT_COVERS_MIN` | `0.9` / `0.9` | Below these, replace or covers verdicts are treated as related |
| `MEMX_VERDICT_GUARDS` | `true` | Different attribute, same window and past-only statements never replace |
| `MEMX_VERDICT_CONFIRM` | `false` | Ask "can both be true?" for borderline replacements. Measured and left off: Jev answered "yes" whether or not a reference model agreed |
| `MEMX_ENTITIES_ENABLED` | `true` | Extract and link entities |
| `MEMX_ENTITY_MERGE_MIN` | `0.7` | Probability needed to merge two different spellings |
| `MEMX_ENTITY_SPLIT_MIN` | `0.7` | Probability of "different" needed to split an identical name into a second entity |
| `MEMX_ENTITY_NAME_SIM_MIN` | `0.5` | Name-embedding similarity that makes a known entity a candidate for Jev |
| `MEMX_ENTITY_CANDIDATES_MAX` | `3` | Known entities compared with each new name |

### Recall

| Variable | Default | Meaning |
|---|---|---|
| `MEMX_BUDGET_TOKENS` | `4000` | Token budget; full mode when the visible memory fits 90% of it |
| `MEMX_GRADE_FLOOR` | `0.3` | Minimum relevance for search hits |
| `MEMX_RECALL_K_MAX` | `30` | Most facts per block |
| `MEMX_GRADE_MAX` | `40` | Most search hits sent to Jev |
| `MEMX_WALK_ENABLED` | `true` | Walk through shared entities |
| `MEMX_WALK_HOPS` / `MEMX_WALK_CAP` / `MEMX_WALK_SEEDS` | `2` / `8` / `10` | Hops, facts added, search hits to start from |
| `MEMX_WALK_HUB_SHARE` / `MEMX_WALK_HUB_MIN_DEGREE` | `0.2` / `6` | Entities in more than this share of facts (and at least this many) are not walked |
| `MEMX_WALK_TOPIC_WEIGHT` | `0.3` | Weight of a hop through a shared theme |
| `MEMX_WALK_GRADE_FLOOR` / `MEMX_WALK_SCORE_FACTOR` | `0.5` / `0.85` | Relevance needed, and score multiplier, for walked facts |
| `MEMX_HISTORY_IN_RECALL` | `when_asked` | Show earlier, replaced versions under a fact: `when_asked` (one yes/no in the ranked grade request), `always`, or `off` |
| `MEMX_RENDER_RELATED_DATES` | `true` | Date related facts in the block so disagreeing statements show which is newer |

### Retention and jobs

| Variable | Default | Meaning |
|---|---|---|
| `MEMX_EVENT_TTL_DAYS` | unset | If set, upcoming events expire this long after their date. Unset: events never expire and fade with the event half-life |
| `MEMX_EXPIRED_RETENTION_DAYS` | `30` | Expired facts are deleted after this |
| `MEMX_DECISION_RETENTION_DAYS` | `180` | Decision log retention |
| `MEMX_WORKER_CONCURRENCY` | `4` | Jobs per worker |
| `MEMX_JOB_MAX_ATTEMPTS` / `MEMX_JOB_LEASE_SECONDS` | `5` / `300` | Retries before a job is dead; lease before a stuck job is requeued |

## Data model

Ten tables, all prefixed `memx_`, created by numbered SQL migrations in `memx/db/migrations`.

| Table | Holds |
|---|---|
| `memx_scope` | Every scope ever written to, with per-scope settings |
| `memx_ingest` | Raw input, per-window extraction results and per-candidate outcomes. Retries resume from here |
| `memx_fact` | Facts: text, category, theme, embedding, full-text vector, status, replaced-by, speaker, trust, dates, recall counts |
| `memx_fact_link` | `shadows` and `related` links between facts |
| `memx_entity` | Entities per tenant, with type and alternative spellings |
| `memx_fact_entity` | Which fact names which entity |
| `memx_decision` | Every Jev call and automatic decision, with probabilities. Also the calibration dataset |
| `memx_recall_log` | Every recall: mode, facts returned, scores, and the full explanation |
| `memx_job` | The job queue (`FOR UPDATE SKIP LOCKED`, leases, retries with backoff, dead letters) |
| `memx_forget_log` | Every hard delete: what scope or fact, how many rows, who asked. No content |

Indexes: HNSW on embeddings, GIN on the full-text vector, GiST on scope paths, and a unique index that blocks exact duplicates among active facts in a scope.

## Audit, privacy and safety

- **Nothing is edited in place.** Replaced facts keep their row with `superseded_by`; shadows are links; expiry is a status change before a delayed purge.
- **Every fact traces back** to its raw input and its position in it, and to the decisions that categorized and reconciled it.
- **Forget is complete.** Deleting a scope removes its raw inputs, facts, links, entity mentions, decisions and recall logs in one transaction, then removes entities no longer named anywhere. The forget log keeps counts and who asked, never content.
- **User text never reaches Jev as instructions.** It always sits under an `untrusted` key in the judge's input, with a note to ignore directives inside it, and never appears in question wording or option labels.
- **Tenants never share entities**, and recall only ever considers facts visible to the asking scope, including during the walk.

## Deployment

The `Dockerfile` builds the dashboard with Node 22 and pnpm, then installs the Python package into a slim Python 3.12 image running as a non-root user. The same image runs the API (`uvicorn memx_api.app:app`) and the worker (`memx worker`).

`k8s/memx.yaml` contains a migration Job, an API Deployment (2 replicas, queued inserts) and a worker Deployment (2 replicas). All three read their settings from a secret named `memx-env`.

Postgres needs version 16 or later with pgvector 0.8 or later.

## Tests

```bash
docker compose up -d postgres
.venv/bin/pytest                         # 137 tests with stand-in models; creates and drops a memx_test database
MEMX_LIVE=1 .venv/bin/pytest -m live     # real Haiku, embeddings and Jev via OpenRouter (about $0.01)
```

## Evaluation

`eval/longmemeval` runs the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark: long chat histories followed by questions that need memory, judged by a model.

```bash
.venv/bin/python -m eval.longmemeval.download
.venv/bin/python -m eval.longmemeval.run --config ranked_4k dump_all --per-type 25 --abstention 20
.venv/bin/python -m eval.longmemeval.run_mem0           # same questions through mem0, for comparison
.venv/bin/python -m eval.longmemeval.calibrate          # suggests similarity bands from logged verdicts
```

Benchmark runs write to their own database, `memx_bench` (override with `--dsn` or `MEMX_BENCH_DSN`), so they never mix with real memory. Ingested scopes are reused across configurations, so only the first run pays for extraction, about $0.15 per question. Configurations: `dump_all`, `ranked_2k`, `ranked_4k`, `auto_4k`, and the ablations `no_jev_grade`, `no_decay`, `no_hint`, `no_assistant_facts`. Results go to `eval/longmemeval/results/`. `--resume <run id>` continues an interrupted run.

**Results so far.** The comparison with mem0 stopped at the API key's daily limit. On the 29 questions both systems answered, memx scored 20 and mem0 22. That gap is not statistically significant. memx handed the answering model about a tenth as many memory tokens. These runs predate the entity walk; a walk-on versus walk-off comparison is still to do.

## Project layout

```
memx/                 library
  config.py           every setting
  scopes.py           scope paths and chains
  types.py            facts, inputs, receipts, categories, themes
  memory.py           the Memory facade
  cli.py              the memx command
  testing.py          stand-in models for tests and demo mode
  db/                 migrations, connection pool, all SQL (repo.py)
  providers/          OpenRouter HTTP client, extractor, embedder, Jev client
  pipeline/           windowing, extract, sort, reconcile, entities, commit, ingest job
  recall/             engine, walk, grade, decay, render
  jobs/               Postgres queue and runner
  export/             OKF bundle
memx_api/             FastAPI app and dashboard routes
web/                  React dashboard (builds into memx_api/static)
eval/longmemeval/     benchmark harness, mem0 comparison, band calibration
tests/                unit and integration tests
k8s/                  Kubernetes manifests
```

## Lessons from building it

Each of these came from a real failure found while testing against the live models.

- **Jev's score scale is 0 to levels−1.** With five grade levels, relevance is `score / 4`.
- **The low band moved from 0.60 to 0.45.** With `text-embedding-3-small`, a genuine conflict ("prefers light mode because dark mode hurts their eyes" against "wants dashboards in dark mode") scored 0.595 and would have skipped the judge.
- **The duplicate check is directional.** "Lives in Pune, Maharashtra" scored 0.965 against "lives in Pune". A candidate with any content word the stored fact lacks now always goes to Jev.
- **Verdict wording matters.** Naming the conflict explicitly ("both cannot be true now") moved a real preference change from 38% to 76% probability.
- **Entity-merge wording matters more.** "Do these name the same thing?" put "Priya" against "Priya Lindqvist" at 9%. Asking whether the short name refers to the known entity, with a few facts about it, gives 90%, and still keeps an unrelated Priya at 3%.
- **Embeddings can find name variants but cannot judge them.** Every true variant in a 20-pair probe scored at least 0.53 ("MSFT" vs "Microsoft"), but so did look-alikes: "Azure" vs "AWS" 0.71, "Priya Lindqvist" vs "Priya Sharma" 0.61. So embeddings only propose candidates and Jev decides; on that probe set Jev got 18 of 20 right, and both misses kept two names apart rather than merging them.
- **One wording does not fit every pair.** A broader "are these the same thing?" question caught "Bangalore" vs "Blore" (0.94) but weakened "Priya" vs "Priya Lindqvist" from 0.90 to 0.65. Each pairing reason now has its own wording.
- **Identical names need the opposite default.** Requiring Jev to say "same" for identical names split one Mahesh into two at 0.3 on a live run. Identical names now stay together unless Jev is sure they differ, which it was for "Priya from the bakery" (0.93).
- **Haiku needs date anchors.** It resolved "next week", said on a Friday, to the wrong week. The prompt now lists the actual dates and the model copies them.
- **Events should not expire by default.** The first benchmark run hid every past event because all events had an expiry. Later, expiring only upcoming events still hid facts that stay true after their date ("the draft reached 4,700 words by May 13", "the call was moved to April 22") and cost BEAM's knowledge-update questions 16 points. Expiry is now opt-in per scope.
- **Extraction must keep stated details.** Haiku turned "saw 10 amateur comedians perform" into "found the open mic night inspiring" until the prompt required numbers and names to survive.
- **A bridge fact judged alone looks irrelevant.** The walk found the right facts and the grader cut them, until each linked fact was graded together with the fact it was reached from.
- **A link must never lower a grade.** Grading a strong search hit only in the context of its link pushed the key fact out of an answer. Search hits are now graded both ways and keep the higher grade.

## Known gaps and next steps

- **"The user" has no name.** Facts from a user's own conversations say "The user…", and nothing tells memx that the user at `memory.acme.alex` is Alex. A per-scope subject would let those facts link to the person entity. Proposed, not built.
- **Named participants get full trust.** A colleague's claim about the user counts as much as the user's own statement. A separate, lower trust for named participants is under consideration.
- **Direct provider clients.** Everything goes through OpenRouter today. Direct Anthropic, OpenAI or TypeSafe clients fit behind the same `Extractor`, `Embedder` and `Judge` interfaces.
