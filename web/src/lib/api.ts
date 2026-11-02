// Typed client for the memx API. The token (if the server requires one) lives in localStorage.

export type Category =
  | "identity" | "rule" | "preference" | "goal" | "relationship" | "work_context" | "procedure" | "event" | "other";
export type FactStatus = "active" | "superseded" | "expired";

export interface Fact {
  id: string;
  scope: string;
  category: Category;
  text: string;
  status: FactStatus;
  superseded_by: string | null;
  superseded_at: string | null;
  speaker: string | null;
  source: string | null;
  trust: number;
  sort_confidence: number;
  observed_at: string;
  event_at: string | null;
  stale_after: string | null;
  reinforced_count: number;
  last_reinforced_at: string | null;
  access_count: number;
  last_accessed_at: string | null;
  flags: Record<string, unknown>;
  ingest_id: string | null;
  created_at: string;
  token_count: number;
  topic: string | null;
  topic_confidence: number | null;
  entities: EntityRef[];
}

export type EntityType = "person" | "organization" | "team" | "system" | "project" | "place" | "product" | "other";

export interface EntityRef {
  id: string;
  name: string;
  type: EntityType;
  surface: string | null;
}

export interface EntityRow {
  id: string;
  name: string;
  type: EntityType;
  aliases: string[];
  active: number;
  facts: number;
}

export interface ScopeRow {
  path: string;
  depth: number;
  active: number;
  superseded: number;
  expired: number;
  created_at: string;
  last_activity: string;
}

export interface Stats {
  facts: { category: Category; status: FactStatus; n: number }[];
  ingests: Record<string, number>;
  recalls: { n: number; avg_ms: number; avg_tokens: number };
  decisions: Record<string, number>;
  links: Record<string, number>;
  facts_per_day: { day: string; n: number }[];
  scopes: number;
  entities: number;
  jobs: Record<string, number>;
}

export interface Receipt {
  ingest_id: string;
  status: "received" | "extracted" | "committed" | "failed";
  mode: string;
  added: number;
  dropped: number;
  historical: number;
  superseded: number;
  shadowed: number;
  rejected: number;
  facts: Fact[] | null;
  error: string | null;
}

export interface IngestRow {
  id: string;
  scope: string;
  kind: "messages" | "text";
  mode: string;
  status: Receipt["status"];
  source: string | null;
  trust: number;
  observed_at: string;
  created_at: string;
  committed_at: string | null;
  receipt: (Record<string, number> & { timings?: Record<string, number> }) | null;
  error: string | null;
  candidates: number;
  preview: string;
}

export type Band = "exact" | "auto" | "judge" | "below" | "over_cap" | "not_needed";

export interface Neighbor {
  id: string;
  text: string;
  scope: string;
  category: Category;
  sim: number;
  via: string;
  band: Band;
}

export interface VerdictRow {
  neighbor_id: string;
  label: "same" | "candidate_replaces" | "neighbor_covers" | "related" | "unrelated";
  probability: number;
  probabilities: Record<string, number>;
  source: "hash" | "auto" | "jev";
  sim: number;
}

export interface TraceCandidate {
  index: number;
  window: number;
  text: string;
  speaker: string;
  category_hint: Category | null;
  is_event: boolean;
  time_text: string | null;
  event_date: string | null;
  confidence: number;
  outcome: "added" | "dropped" | "rejected" | "superseded_on_arrival" | null;
  reason?: string;
  fact_id: string | null;
  same_as?: string;
  superseded: number;
  shadowed: number;
  superseded_ids?: string[];
  shadowed_ids?: string[];
  related_ids?: string[];
  superseded_by?: string | null;
  category?: Category;
  topic?: string | null;
  entities?: EntityPlanRow[];
  entity_ids?: string[];
  sort?: { category: Category; confidence: number; probabilities: Record<string, number>; injection: number; ms: number;
           topic?: string | null; topic_confidence?: number; topic_probabilities?: Record<string, number> } | null;
  neighbors?: Neighbor[];
  verdicts?: VerdictRow[];
  reconcile_ms?: number;
  note?: string;
}

export interface IngestTrace {
  ingest: {
    id: string;
    scope: string;
    kind: "messages" | "text";
    mode: string;
    status: Receipt["status"];
    source: string | null;
    trust: number;
    observed_at: string;
    created_at: string;
    committed_at: string | null;
    receipt: (Record<string, number> & { timings?: Record<string, number> }) | null;
    error: string | null;
    extractor_model: string | null;
    tokens_in: number;
    tokens_out: number;
    payload: { kind: "messages"; turns: { role: string; content: string }[] } | { kind: "text"; text: string; source: string };
  };
  windows: { index: number; units: number; chars: number; context_units: number; roles: string[]; extracted: number | null }[];
  candidates: TraceCandidate[];
  facts: Record<string, Fact>;
  config: { band_low: number; band_high: number };
}

export interface EntityPlanRow {
  name: string;
  type: EntityType;
  action: "link" | "alias" | "create" | "skip";
  canonical?: string;
  entity_id?: string;
  merge?: { with: string; entity_id: string; p_same: number; probabilities: Record<string, number>; latency_ms: number };
}

export type CandidateStatus = "picked" | "below_floor" | "over_budget" | "over_k" | "shadowed" | "not_graded";
export type FoundBy = "search" | "entity_hop" | "topic_hop" | "full";

export interface PathStep {
  from_fact: string;
  via: string;
  kind: "entity" | "topic";
}

export interface WalkInfo {
  enabled: boolean;
  seeds: number;
  hops: number;
  reached: number;
  expanded: number;
  hub_threshold: number;
  hubs_cut: { name: string; type: string; degree: number }[];
  over_cap: number;
  ms: number;
}

export interface ExplainCandidate {
  id: string;
  text: string;
  category: Category;
  scope: string;
  vec_rank?: number | null;
  fts_rank?: number | null;
  similarity?: number;
  rrf?: number;
  relevance?: number | null;
  grade_levels?: Record<string, number> | null;
  score: number | null;
  status: CandidateStatus;
  found_by: FoundBy;
  path?: PathStep[];
  walk_score?: number;
  hops?: number;
  freshness: number;
  freshness_factor: number;
  trust_factor: number;
  access_factor: number;
}

export interface Explain {
  mode: "full" | "ranked";
  mode_reason: string;
  budget_tokens: number;
  budget_limit: number;
  chain: string[];
  chain_tokens: number;
  chain_facts: number;
  shadowed: number;
  grade_floor: number;
  k_max: number;
  query: string | null;
  as_of: string;
  timings: Record<string, number>;
  related_pairs: [string, string][];
  walk: WalkInfo | null;
  walk_score_factor?: number;
  walk_grade_floor?: number;
  candidates: ExplainCandidate[];
}

export interface RecallResult {
  recall_id: string;
  scope: string;
  mode: "full" | "ranked";
  truncated: boolean;
  markdown: string;
  facts: (Fact & { relevance: number | null; score: number })[];
  token_count: number;
  explain: Explain | null;
}

export interface RecallRow {
  id: string;
  scope: string;
  query: string | null;
  mode: "full" | "ranked";
  facts: number;
  token_count: number;
  latency_ms: number;
  created_at: string;
}

export interface RecallDetail extends RecallRow {
  fact_ids: string[];
  scores: Record<string, number>;
  explain: Explain | null;
  markdown: string;
  facts_detail?: Fact[];
}

export interface GraphData {
  scope: string;
  view: "chain" | "subtree";
  total: number;
  truncated: boolean;
  scopes: string[];
  facts: (Fact & { shadowed_here: boolean })[];
  edges: { source: string; target: string; kind: "related" | "shadows" | "supersedes"; similarity: number | null }[];
  entities: { id: string; name: string; type: EntityType; aliases?: string[]; facts: number; active: number }[];
  mentions: { fact: string; entity: string }[];
}

export interface Decision {
  id: number;
  step: "sort" | "verdict" | "grade" | "auto" | "entity";
  question_version: string;
  subject_fact_id: string | null;
  object_fact_id: string | null;
  similarity: number | null;
  result: Record<string, any>;
  latency_ms: number | null;
  created_at: string;
  role: "subject" | "object";
}

export interface FactHistory {
  fact_id: string;
  versions: { fact: Fact; decisions: Decision[] }[];
  links: { a: string; b: string; kind: string; similarity: number | null; created_at: string }[];
}

export interface JobRow {
  id: number;
  kind: string;
  payload: Record<string, unknown>;
  status: "queued" | "running" | "done" | "dead";
  attempts: number;
  max_attempts: number;
  run_after: string;
  locked_at: string | null;
  locked_by: string | null;
  last_error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface Config {
  models: { extractor: string; embedding: string; judge: string };
  insert_mode: string;
  bands: { low: number; high: number; guard_min_jaccard: number };
  verdict: { same_min: number; min_prob: number };
  sort: { min_prob: number; min_confidence: number; injection_flag: number; injection_reject: number };
  recall: { budget_tokens: number; budget_safety: number; grade_floor: number; k_max: number; grade_max: number; grade_enabled: boolean };
  decay: { half_lives: Record<string, number | null>; access_boost_cap: number };
  trust: { user: number; assistant: number; text_default: number };
  window_tokens: number;
  categories: { key: Category; definition: string }[];
  entities: { enabled: boolean; merge_min: number; topic_min: number };
  walk: { enabled: boolean; hops: number; decay: number; cap: number; seeds: number; hub_share: number; hub_min_degree: number; topic_weight: number; score_factor: number; grade_floor: number };
  topics: { key: string; definition: string }[];
}

const TOKEN_KEY = "memx.token";

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setToken(value: string) {
  try {
    if (value) localStorage.setItem(TOKEN_KEY, value);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable */
  }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

const q = (params: Record<string, string | number | boolean | null | undefined>) => {
  const s = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") s.set(k, String(v));
  const out = s.toString();
  return out ? `?${out}` : "";
};

const enc = encodeURIComponent;

export const api = {
  config: () => request<Config>("GET", "/v1/config"),
  scopes: () => request<ScopeRow[]>("GET", "/v1/scopes"),
  stats: (scope?: string) => request<Stats>("GET", `/v1/stats${q({ scope })}`),
  facts: (scope: string, opts: { status?: string; category?: string; include_ancestors?: boolean; limit?: number }) =>
    request<Fact[]>("GET", `/v1/scopes/${enc(scope)}/facts${q({ status: opts.status ?? "", category: opts.category, include_ancestors: opts.include_ancestors, limit: opts.limit ?? 500 })}`),
  fact: (id: string) => request<Fact>("GET", `/v1/facts/${id}`),
  history: (id: string) => request<FactHistory>("GET", `/v1/facts/${id}/history`),
  render: (scope: string) => request<string>("GET", `/v1/scopes/${enc(scope)}/render?show_scope=true`),
  graph: (scope: string, view: "chain" | "subtree", includeSuperseded: boolean, limit = 400) =>
    request<GraphData>("GET", `/v1/graph${q({ scope, view, include_superseded: includeSuperseded, limit })}`),
  entities: (scope: string, view: "chain" | "subtree" = "subtree", limit = 200) =>
    request<EntityRow[]>("GET", `/v1/entities${q({ scope, view, limit })}`),
  ingests: (scope?: string, limit = 50) => request<IngestRow[]>("GET", `/v1/ingests${q({ scope, limit })}`),
  trace: (id: string) => request<IngestTrace>("GET", `/v1/ingests/${id}/trace`),
  recalls: (scope?: string, limit = 50) => request<RecallRow[]>("GET", `/v1/recalls${q({ scope, limit })}`),
  recallDetail: (id: string) => request<RecallDetail>("GET", `/v1/recalls/${id}`),
  jobs: (status?: string) => request<JobRow[]>("GET", `/v1/jobs${q({ status, limit: 100 })}`),
  ready: () => request<{ database: string; jobs: Record<string, number> }>("GET", "/readyz"),
  insert: (scope: string, input: unknown, observedAt?: string) =>
    request<Receipt>("POST", `/v1/scopes/${enc(scope)}/ingests`, {
      input, mode: "inline", observed_at: observedAt || undefined,
    }),
  recall: (scope: string, body: { query?: string; budget_tokens?: number; force_mode?: "full" | "ranked"; as_of?: string; categories?: Category[] }) =>
    request<RecallResult>("POST", `/v1/scopes/${enc(scope)}/recall`, { ...body, explain: true }),
  deleteFact: (id: string) => request<unknown>("DELETE", `/v1/facts/${id}`),
  deleteScope: (scope: string) => request<{ facts_removed: number; ingests_removed: number }>("DELETE", `/v1/scopes/${enc(scope)}`),
  promote: (id: string, toScope: string) => request<Fact>("POST", `/v1/facts/${id}/promote`, { to_scope: toScope }),
};
