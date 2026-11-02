import type { Band, CandidateStatus, Category, EntityType, FoundBy, TraceCandidate, VerdictRow } from "./api";

// Fixed categorical order (validated reference palette); "other" is neutral. Colors live in CSS as --cat-<key>.
export const CATEGORIES: Category[] = [
  "identity", "rule", "preference", "goal", "relationship", "work_context", "procedure", "event", "other",
];

export const CATEGORY_LABEL: Record<Category, string> = {
  identity: "Identity",
  rule: "Rule",
  preference: "Preference",
  goal: "Goal",
  relationship: "Relationship",
  work_context: "Work context",
  procedure: "Procedure",
  event: "Event",
  other: "Other",
};

export const catVar = (c: Category) => `var(--cat-${c})`;

export const BAND_META: Record<Band, { label: string; tone: string; help: string }> = {
  exact: { label: "Exact text", tone: "ok", help: "Same normalized text as an existing fact. No judge needed." },
  auto: { label: "Auto duplicate", tone: "ok", help: "Similarity above the high band and the lexical guard passed. No judge needed." },
  judge: { label: "Sent to Jev", tone: "accent", help: "In the ambiguous band, so Jev decided how the two relate." },
  below: { label: "Below band", tone: "muted", help: "Similarity under the low band: treated as unrelated without asking." },
  over_cap: { label: "Over cap", tone: "muted", help: "Above the band but beyond the 5 closest neighbors." },
  not_needed: { label: "Not needed", tone: "muted", help: "An auto duplicate was found first, so this neighbor was not judged." },
};

export const VERDICT_LABEL: Record<VerdictRow["label"], string> = {
  same: "Same",
  candidate_replaces: "Replaces neighbor",
  neighbor_covers: "Neighbor covers it",
  related: "Related",
  unrelated: "Unrelated",
};

export const OUTCOME_META: Record<NonNullable<TraceCandidate["outcome"]>, { label: string; tone: string; help: string }> = {
  added: { label: "Added", tone: "ok", help: "Stored as a new active fact." },
  dropped: { label: "Duplicate", tone: "muted", help: "Already known. The existing fact was reinforced instead." },
  rejected: { label: "Rejected", tone: "danger", help: "Not stored." },
  superseded_on_arrival: { label: "Historical", tone: "warn", help: "An existing fact already covers it; stored as superseded history." },
};

export const STATUS_META: Record<CandidateStatus, { label: string; tone: string; help: string }> = {
  picked: { label: "Picked", tone: "ok", help: "Included in the memory block." },
  below_floor: { label: "Below cut-off", tone: "muted", help: "Jev rated it under the relevance cut-off." },
  over_budget: { label: "Over budget", tone: "warn", help: "Relevant, but the token budget was already spent." },
  over_k: { label: "Over limit", tone: "warn", help: "Relevant, but the maximum number of facts was reached." },
  shadowed: { label: "Shadowed", tone: "muted", help: "Hidden here by a correction in a narrower scope." },
  not_graded: { label: "Not graded", tone: "muted", help: "Ranked too low in search to be sent to Jev." },
};

export const GRADE_LEVELS = ["Irrelevant", "Background", "Useful", "Important", "Critical"];

export function fmtDate(iso: string | null | undefined, withTime = false): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, withTime
    ? { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
    : { year: "numeric", month: "short", day: "numeric" });
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  if (s < 86400 * 30) return `${Math.round(s / 86400)} d ago`;
  return fmtDate(iso);
}

export const pct = (x: number | null | undefined, digits = 0) =>
  x === null || x === undefined ? "—" : `${(x * 100).toFixed(digits)}%`;

export const num = (x: number | null | undefined, digits = 2) =>
  x === null || x === undefined ? "—" : x.toFixed(digits);

export const ms = (x: number | null | undefined) =>
  x === null || x === undefined ? "—" : x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${Math.round(x)} ms`;

export const short = (id: string) => id.slice(0, 8);

export function truncate(text: string, n: number) {
  return text.length <= n ? text : `${text.slice(0, n - 1)}…`;
}

export const FOUND_BY_META: Record<FoundBy, { label: string; help: string }> = {
  search: { label: "Search", help: "Found by vector or keyword search." },
  entity_hop: { label: "Entity hop", help: "Not found by search: reached from a search hit through a shared entity." },
  topic_hop: { label: "Topic hop", help: "Not found by search: reached from a search hit through a shared theme." },
  full: { label: "Whole memory", help: "The whole memory fit the budget, so nothing was searched." },
};

export const ENTITY_TYPE_LABEL: Record<EntityType, string> = {
  person: "Person", organization: "Organization", team: "Team", system: "System", project: "Project",
  place: "Place", product: "Product", other: "Entity",
};

/** "food_and_drink" → "Food and drink". */
export function topicLabel(key: string | null | undefined): string {
  if (!key) return "—";
  const words = key.split("_");
  return words.map((w, i) => (i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w)).join(" ");
}
