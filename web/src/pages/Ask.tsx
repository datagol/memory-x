import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Background, Controls, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";
import { Copy, FileStack, Footprints, Search, Sparkles, Text as TextIcon, Waypoints } from "lucide-react";
import { api, type Explain, type ExplainCandidate, type PathStep } from "../lib/api";
import { useApp } from "../lib/app";
import { FOUND_BY_META, GRADE_LEVELS, STATUS_META, fmtDate, ms, num, pct, short, topicLabel, truncate } from "../lib/meta";
import { nodeTypes } from "../components/FlowNodes";
import { Card, CategoryChip, Empty, ErrorBox, Loading, MemoryMarkdown, Meter, Pill, Seg, useTip, useToast, rowKeys } from "../components/ui";

const walked = (c: ExplainCandidate) => c.found_by === "entity_hop" || c.found_by === "topic_hop";

/** "F5 → Marcus Oyelaran → F4 → Data Platform Team → here", with fact texts resolved from the explanation. */
export function PathLine({ path, ex, onOpen }: { path: PathStep[]; ex: Explain; onOpen?: (id: string) => void }) {
  const byId = new Map(ex.candidates.map((c) => [c.id, c]));
  return (
    <span className="path">
      {path.map((s, i) => (
        <span key={i} className="path" style={{ gap: 4 }}>
          <a href="#" className="frm" title={byId.get(s.from_fact)?.text} onClick={(e) => { e.preventDefault(); onOpen?.(s.from_fact); }}>
            “{truncate(byId.get(s.from_fact)?.text ?? short(s.from_fact), 42)}”
          </a>
          <span className="arr">→</span>
          <span className="via" title={s.kind === "topic" ? "shared theme" : "shared entity"}>{s.kind === "topic" ? topicLabel(s.via) : s.via}</span>
          <span className="arr">→</span>
        </span>
      ))}
      <span className="frm">this fact</span>
    </span>
  );
}

function pathText(c: ExplainCandidate, ex: Explain): string {
  const byId = new Map(ex.candidates.map((x) => [x.id, x]));
  const last = c.path?.at(-1);
  if (!last) return "";
  const via = last.kind === "topic" ? `the theme “${topicLabel(last.via)}”` : `“${last.via}”`;
  const from = truncate(byId.get(last.from_fact)?.text ?? "a search hit", 60);
  return `Not found by search: reached in ${c.hops ?? c.path!.length} hop${(c.hops ?? 1) === 1 ? "" : "s"} from “${from}” because both mention ${via}. Jev graded it with that bridge in view.`;
}

function why(c: ExplainCandidate, ex: Explain, rank: number | null): string {
  const fac = `freshness ×${num(c.freshness_factor)}, trust ×${num(c.trust_factor)}, recall history ×${num(c.access_factor)}`;
  const linkNote = !walked(c) && c.path?.length
    ? `Also linked to “${truncate(ex.candidates.find((x) => x.id === c.path!.at(-1)!.from_fact)?.text ?? "a stronger hit", 60)}” through ${c.path!.at(-1)!.kind === "topic" ? `the theme “${topicLabel(c.path!.at(-1)!.via)}”` : `“${c.path!.at(-1)!.via}”`}; Jev graded it alone and with that link${(c as { graded_via_link?: boolean }).graded_via_link ? ", and the link raised its grade" : ", and the link did not raise it"}. `
    : "";
  const hop = walked(c) ? `${pathText(c, ex)} ` : linkNote;
  const factor = walked(c) && ex.walk_score_factor ? `, walked ×${num(ex.walk_score_factor)}` : "";
  switch (c.status) {
    case "picked":
      return ex.mode === "ranked"
        ? `${hop}Jev rated it ${pct(c.relevance)} relevant. With ${fac}${factor}, it scored ${num(c.score, 3)}, ranked #${rank}, and fit the budget.`
        : `The whole memory fits the budget, so every visible fact is included. Ordered by trust and freshness (${num(c.score, 3)}).`;
    case "below_floor":
      return walked(c) && ex.walk_grade_floor
        ? `${hop}Jev rated it ${pct(c.relevance)} relevant. A fact search did not find must reach ${pct(ex.walk_grade_floor)} (“useful”), so it was left out.`
        : `${hop}Jev rated it ${pct(c.relevance)} relevant, under the ${pct(ex.grade_floor)} cut-off, so it was left out.`;
    case "over_budget":
      return `Relevant (score ${num(c.score, 3)}), but higher-scoring facts had already used the ${ex.budget_limit}-token budget.`;
    case "over_k":
      return `Relevant (score ${num(c.score, 3)}), but the limit of ${ex.k_max} facts was reached.`;
    case "shadowed":
      return "A narrower scope corrected this fact, so this agent does not see it.";
    case "not_graded":
      return "Ranked too low in search to be sent to Jev for grading.";
  }
}

function Funnel({ ex }: { ex: Explain }) {
  const c = ex.candidates;
  const picked = c.filter((x) => x.status === "picked").length;
  const searched = c.filter((x) => !walked(x));
  const steps = ex.mode === "ranked"
    ? [
        ["Visible facts in scope chain", ex.chain_facts],
        ["Found by vector or keyword search", searched.length],
        ["Not shadowed", searched.filter((x) => x.status !== "shadowed").length],
        ...(ex.walk ? [[`Added by the walk (${ex.walk.hops} hop${ex.walk.hops === 1 ? "" : "s"})`, c.filter(walked).length] as const] : []),
        ["Graded by Jev", c.filter((x) => x.relevance !== null && x.relevance !== undefined).length],
        ["Above relevance cut-off", c.filter((x) => ["picked", "over_budget", "over_k"].includes(x.status)).length],
        ["Fit the token budget → picked", picked],
      ] as const
    : [
        ["Visible facts in scope chain", ex.chain_facts],
        ["Not shadowed", c.filter((x) => x.status !== "shadowed").length],
        ["Fit the token budget → picked", picked],
      ] as const;
  const max = Math.max(1, ...steps.map((s) => s[1]));
  return (
    <div className="funnel">
      {steps.map(([label, n]) => (
        <div key={label} className="step">
          <span className="small">{label}</span>
          <div className="b" style={{ width: `${Math.max(0.5, (n / max) * 100)}%` }} />
          <span className="num small" style={{ textAlign: "right" }}>{n.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}

function RetrievalPath({ ex, markdownTokens }: { ex: Explain; markdownTokens: number }) {
  const { openFact } = useApp();
  const { nodes, edges, height } = useMemo(() => {
    const ranked = ex.mode === "ranked";
    const list = [...ex.candidates]
      .sort((a, b) => (b.status === "picked" ? 1 : 0) - (a.status === "picked" ? 1 : 0) || (b.score ?? b.relevance ?? 0) - (a.score ?? a.relevance ?? 0));
    const picked = list.filter((c) => c.status === "picked");
    const cut = list.filter((c) => c.status !== "picked").slice(0, Math.max(3, 8 - picked.length));
    const shown = [...picked.slice(0, 14), ...cut];
    const GAP = 96;
    const h = Math.max(shown.length * GAP, 300);
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const hasWalk = ranked && !!ex.walk && ex.candidates.some(walked);
    const X = hasWalk ? { q: 0, src: 270, walk: 560, cand: 850, block: 1220 } : { q: 0, src: 270, walk: 0, cand: 560, block: 930 };
    const arrow = (color: string) => ({ type: MarkerType.ArrowClosed, color, width: 14, height: 14 });
    nodes.push({ id: "q", type: "stage", position: { x: X.q, y: h / 2 - 70 }, data: {
      kicker: <><Search /> Question</>, title: truncate(ex.query ?? "(no query)", 90),
      body: <>at {ex.chain.at(-1)}<br />{ex.chain.length} scopes in chain</>, tone: "query", width: 220,
    } });
    if (ranked) {
      nodes.push({ id: "vec", type: "stage", position: { x: X.src, y: h / 2 - 150 }, data: {
        kicker: <><Waypoints /> Vector search</>, title: `${ex.candidates.filter((c) => c.vec_rank).length} by meaning`, body: `embed ${ms(ex.timings.embed_ms)}`, width: 200,
      } });
      nodes.push({ id: "kw", type: "stage", position: { x: X.src, y: h / 2 + 10 }, data: {
        kicker: <><TextIcon /> Keyword search</>, title: `${ex.candidates.filter((c) => c.fts_rank).length} by shared words`, body: `search ${ms(ex.timings.search_ms)} · graded ${ms(ex.timings.grade_ms)}`, width: 200,
      } });
      edges.push({ id: "q-vec", source: "q", target: "vec", type: "smoothstep", style: { stroke: "var(--accent)" }, markerEnd: arrow("var(--accent)") });
      edges.push({ id: "q-kw", source: "q", target: "kw", type: "smoothstep", style: { stroke: "var(--accent)" }, markerEnd: arrow("var(--accent)") });
      if (hasWalk && ex.walk) {
        const cut = ex.walk.hubs_cut.length ? `hub cut: ${ex.walk.hubs_cut.slice(0, 2).map((h) => h.name).join(", ")}${ex.walk.hubs_cut.length > 2 ? "…" : ""}` : "no hubs cut";
        nodes.push({ id: "walk", type: "stage", position: { x: X.walk, y: h / 2 + 150 }, data: {
          kicker: <><Footprints /> Walk</>, title: `${ex.walk.expanded} reached through shared entities`,
          body: <>{ex.walk.hops} hop{ex.walk.hops === 1 ? "" : "s"} from {ex.walk.seeds} seeds · {cut} · {ms(ex.walk.ms)}</>, width: 230,
        } });
        edges.push({ id: "vec-walk", source: "vec", target: "walk", type: "smoothstep", style: { stroke: "var(--entity)" }, markerEnd: arrow("var(--entity)") });
        edges.push({ id: "kw-walk", source: "kw", target: "walk", type: "smoothstep", style: { stroke: "var(--entity)" }, markerEnd: arrow("var(--entity)") });
      }
    } else {
      nodes.push({ id: "all", type: "stage", position: { x: X.src, y: h / 2 - 60 }, data: {
        kicker: <><FileStack /> Whole memory</>, title: `${ex.chain_facts} facts · ${ex.chain_tokens} tokens`, body: ex.mode_reason, width: 210,
      } });
      edges.push({ id: "q-all", source: "q", target: "all", type: "smoothstep", style: { stroke: "var(--accent)" }, markerEnd: arrow("var(--accent)") });
    }
    nodes.push({ id: "block", type: "stage", position: { x: X.block, y: h / 2 - 60 }, data: {
      kicker: <><Sparkles /> Memory block</>, title: `${picked.length} facts · ${markdownTokens} tokens`, body: `budget ${ex.budget_limit} tokens`, tone: "block", width: 210,
    } });
    shown.forEach((c, i) => {
      const isPicked = c.status === "picked";
      const viaHop = walked(c);
      const last = c.path?.at(-1);
      nodes.push({ id: c.id, type: "fact", position: { x: X.cand, y: i * GAP }, data: {
        text: c.text, category: c.category, status: "active", dim: !isPicked, width: 280,
        badges: <span>{viaHop && <b style={{ color: "var(--entity)", marginRight: 6 }}>via {last?.kind === "topic" ? topicLabel(last.via) : last?.via}</b>}{ranked && c.relevance !== null && c.relevance !== undefined ? `relevance ${pct(c.relevance)}` : ""}{!isPicked && <b style={{ marginLeft: 6 }}>{STATUS_META[c.status].label}</b>}</span>,
      } });
      const srcColor = isPicked ? "var(--accent)" : "var(--line-strong)";
      if (ranked && viaHop) {
        edges.push({ id: `w-${c.id}`, source: "walk", target: c.id, type: "default", label: isPicked ? `${c.hops ?? c.path?.length} hop${(c.hops ?? 1) === 1 ? "" : "s"}` : undefined, style: { stroke: isPicked ? "var(--entity)" : "var(--line-strong)", opacity: isPicked ? 1 : 0.5 } });
        // The bridge: the search hit this fact was reached from.
        const from = c.path?.[0]?.from_fact;
        if (from && shown.some((s) => s.id === from)) edges.push({ id: `p-${c.id}`, source: from, target: c.id, type: "default", style: { stroke: "var(--entity)", strokeDasharray: "3 3", opacity: 0.8 } });
      } else if (ranked) {
        if (c.vec_rank) edges.push({ id: `v-${c.id}`, source: "vec", target: c.id, type: "default", label: isPicked ? `#${c.vec_rank}` : undefined, style: { stroke: srcColor, opacity: isPicked ? 1 : 0.5 } });
        if (c.fts_rank) edges.push({ id: `k-${c.id}`, source: "kw", target: c.id, type: "default", label: isPicked ? `#${c.fts_rank}` : undefined, style: { stroke: srcColor, opacity: isPicked ? 1 : 0.5, strokeDasharray: "4 3" } });
      } else {
        edges.push({ id: `a-${c.id}`, source: "all", target: c.id, type: "default", style: { stroke: srcColor, opacity: isPicked ? 1 : 0.5 } });
      }
      if (isPicked) edges.push({ id: `b-${c.id}`, source: c.id, target: "block", type: "default", animated: true, label: num(c.score, 2), style: { stroke: "var(--accent-2)", strokeWidth: 1.6 }, markerEnd: arrow("var(--accent-2)") });
    });
    return { nodes, edges, height: Math.min(860, Math.max(380, h + 60)) };
  }, [ex, markdownTokens]);

  return (
    <div className="flow" style={{ height }}>
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.1, maxZoom: 1 }} minZoom={0.2}
        nodesDraggable={false} proOptions={{ hideAttribution: true }}
        onNodeClick={(_, n) => { if (!["q", "vec", "kw", "walk", "all", "block"].includes(n.id)) openFact(n.id); }}>
        <Background gap={22} size={1} color="var(--line)" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

function RankingTable({ ex }: { ex: Explain }) {
  const { openFact } = useApp();
  const tip = useTip();
  const [show, setShow] = useState<"all" | "picked" | "cut">("all");
  const ranked = ex.mode === "ranked";
  const order = [...ex.candidates].sort((a, b) =>
    (a.status === "picked" ? 0 : 1) - (b.status === "picked" ? 0 : 1) || (b.score ?? -1) - (a.score ?? -1) || (b.relevance ?? -1) - (a.relevance ?? -1));
  let rank = 0;
  const ranks = new Map(order.map((c) => [c.id, c.status === "picked" ? ++rank : null]));
  const rows = order.filter((c) => show === "all" || (show === "picked" ? c.status === "picked" : c.status !== "picked"));
  const maxScore = Math.max(0.0001, ...order.map((c) => c.score ?? 0));
  return (
    <Card title="Every candidate and why" sub={`${ex.candidates.length} considered`} flush
      right={<Seg label="Show" value={show} onChange={setShow} options={[{ value: "all", label: "All" }, { value: "picked", label: "Picked" }, { value: "cut", label: "Left out" }]} />}>
      <div className="table-wrap">
        <table className="t">
          <thead>
            <tr>
              <th>Fact</th>
              {ranked && <th>Found by</th>}
              {ranked && <th className="n">Vector</th>}
              {ranked && <th className="n">Keyword</th>}
              {ranked && <th>Relevance</th>}
              <th className="n">Fresh</th>
              <th className="n">Trust</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => {
              const meta = STATUS_META[c.status];
              const levels = c.grade_levels;
              return (
                <tr key={c.id} className={`click ${c.status === "picked" ? "" : "dim"}`} onClick={() => openFact(c.id)} {...rowKeys(() => openFact(c.id))}>
                  <td style={{ minWidth: 300 }}>
                    <div className="row" style={{ gap: 6, marginBottom: 4 }}>
                      <Pill tone={meta.tone} title={meta.help}>{ranks.get(c.id) ? `#${ranks.get(c.id)} ` : ""}{meta.label}</Pill>
                      <CategoryChip category={c.category} small />
                      {c.scope !== ex.chain.at(-1) && <span className="small faint mono">{c.scope}</span>}
                    </div>
                    <div style={{ color: "var(--ink)" }}>{c.text}</div>
                    <div className="why muted" style={{ marginTop: 4 }}>{why(c, ex, ranks.get(c.id) ?? null)}</div>
                    {c.path && c.path.length > 0 && <div style={{ marginTop: 6 }}><PathLine path={c.path} ex={ex} onOpen={openFact} /></div>}
                  </td>
                  {ranked && (
                    <td style={{ minWidth: 110 }}>
                      <Pill tone={walked(c) ? "warn" : "muted"} title={FOUND_BY_META[c.found_by].help}>{FOUND_BY_META[c.found_by].label}</Pill>
                      {walked(c) && c.walk_score !== undefined && <div className="small faint num" style={{ marginTop: 3 }}>walk {num(c.walk_score, 3)}</div>}
                    </td>
                  )}
                  {ranked && <td className="n">{c.vec_rank ? <>#{c.vec_rank}<div className="small faint">{num(c.similarity)}</div></> : "—"}</td>}
                  {ranked && <td className="n">{c.fts_rank ? `#${c.fts_rank}` : "—"}</td>}
                  {ranked && (
                    <td style={{ minWidth: 120 }}>
                      {c.relevance !== null && c.relevance !== undefined ? (
                        <div className="stack" style={{ gap: 4 }} {...tip(levels ? (
                          <>
                            <b>Jev's grade</b>
                            {Object.entries(levels).map(([k, v]) => <div key={k}>{GRADE_LEVELS[Number(k)] ?? k}: {pct(v)}</div>)}
                          </>
                        ) : "Relevance from search rank (grading off)")}>
                          <Meter value={c.relevance} tone="purple" />
                          <span className="small num">{pct(c.relevance)}</span>
                        </div>
                      ) : <span className="faint">—</span>}
                    </td>
                  )}
                  <td className="n">{num(c.freshness_factor)}</td>
                  <td className="n">{num(c.trust_factor)}</td>
                  <td style={{ minWidth: 110 }}>
                    {c.score !== null ? <div className="stack" style={{ gap: 4 }}><Meter value={c.score} max={maxScore} /><span className="small num">{num(c.score, 3)}</span></div> : <span className="faint">—</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {ranked && (
        <div className="formula" style={{ padding: "10px 14px", borderTop: "1px solid var(--line)" }}>
          score = relevance × (0.6 + 0.4 × freshness) × (0.5 + 0.5 × trust) × recall-history boost{ex.walk_score_factor ? ` × ${ex.walk_score_factor} if reached by a hop` : ""}. Picked in score order until the budget or the {ex.k_max}-fact limit is reached.
        </div>
      )}
    </Card>
  );
}

export function RecallView({ ex, markdown, tokens }: { ex: Explain; markdown: string; tokens: number }) {
  const toast = useToast();
  const used = Math.min(1, tokens / Math.max(1, ex.budget_limit));
  return (
    <>
      <div className="grid cols-3">
        <div className="card kpi">
          <span className="label">Mode</span>
          <span className="v" style={{ fontSize: 22 }}><Pill tone={ex.mode === "ranked" ? "accent" : "purple"}>{ex.mode}</Pill></span>
          <span className="d">{ex.mode_reason}</span>
        </div>
        <div className="card kpi">
          <span className="label">Budget used</span>
          <span className="v">{tokens}<small>/ {ex.budget_limit} tokens</small></span>
          <Meter value={used} tone="purple" />
          <span className="d">The whole chain holds {ex.chain_tokens.toLocaleString()} tokens across {ex.chain_facts.toLocaleString()} facts.</span>
        </div>
        <div className="card kpi">
          <span className="label">Time</span>
          <span className="v">{ms(ex.timings.total_ms)}</span>
          <span className="d">{Object.entries(ex.timings).filter(([k]) => k !== "total_ms").map(([k, v]) => `${k.replace("_ms", "")} ${ms(v)}`).join(" · ")}</span>
        </div>
      </div>
      <Card title="Retrieval path" sub="From the question to the memory block. Faded facts were considered and left out; open the table below for the reason.">
        <RetrievalPath ex={ex} markdownTokens={tokens} />
      </Card>
      <div className="split">
        <Card title="Funnel" sub="How many facts survived each step">
          <Funnel ex={ex} />
          <div className="small muted" style={{ marginTop: 12 }}>
            Scope chain: {ex.chain.map((s, i) => <code key={s}>{i ? " → " : ""}{s}</code>)}
            {ex.shadowed > 0 && <> · {ex.shadowed} fact{ex.shadowed > 1 ? "s" : ""} shadowed in this chain</>}
            · as of {fmtDate(ex.as_of, true)}
          </div>
          {ex.walk && (
            <div className="small muted" style={{ marginTop: 8 }}>
              <b style={{ color: "var(--ink)" }}>Walk:</b> {ex.walk.seeds} seeds, {ex.walk.hops} hop{ex.walk.hops === 1 ? "" : "s"}, {ex.walk.reached} facts reached, {ex.walk.expanded} sent to grading
              {ex.walk.over_cap > 0 && <> ({ex.walk.over_cap} over the cap)</>}.
              {ex.walk.hubs_cut.length > 0
                ? <> Not walked as hubs (over {ex.walk.hub_threshold} facts): {ex.walk.hubs_cut.map((h) => `${h.type === "topic" ? topicLabel(h.name) : h.name} (${h.degree})`).join(", ")}.</>
                : <> No entity was common enough to be cut as a hub.</>}
            </div>
          )}
        </Card>
        <Card title="Memory block the agent receives" right={<button className="btn ghost sm" onClick={() => navigator.clipboard?.writeText(markdown).then(() => toast("Copied"), () => toast("Copy is blocked here; select the text instead.", true))}><Copy /> Copy</button>}>
          <MemoryMarkdown text={markdown} />
        </Card>
      </div>
      <RankingTable ex={ex} />
    </>
  );
}

export function AskPage() {
  const { scope } = useApp();
  const qc = useQueryClient();
  const config = useQuery({ queryKey: ["config"], queryFn: api.config });
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"auto" | "ranked" | "full">("auto");
  const [budget, setBudget] = useState<number | null>(null);
  const [asOf, setAsOf] = useState("");
  const b = budget ?? config.data?.recall.budget_tokens ?? 4000;

  const run = useMutation({
    mutationFn: () => api.recall(scope, {
      query: query || undefined, budget_tokens: b, force_mode: mode === "auto" ? undefined : mode,
      as_of: asOf ? new Date(asOf).toISOString() : undefined,
    }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["recalls"] }),
  });

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Ask</h1>
          <p>See exactly what an agent at <code>{scope}</code> would receive for a question, and why each fact made it in or not.</p>
        </div>
      </div>
      <Card>
        <form className="stack" onSubmit={(e) => { e.preventDefault(); run.mutate(); }}>
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <input id="q" className="input" style={{ height: 42, fontSize: 15 }} value={query} onChange={(e) => setQuery(e.target.value)}
              placeholder="Which theme should the new dashboard use?" autoFocus aria-label="Question" />
            <button className="btn primary" style={{ height: 42 }} type="submit" disabled={run.isPending || (mode === "ranked" && !query)}>
              {run.isPending ? <Sparkles className="spin" /> : <Search />} Recall
            </button>
          </div>
          <div className="row" style={{ gap: 18 }}>
            <Seg label="Mode" value={mode} onChange={setMode} options={[
              { value: "auto", label: "Automatic" }, { value: "ranked", label: "Always rank" }, { value: "full", label: "Whole memory" },
            ]} />
            <label className="row small" style={{ gap: 8 }}>
              Budget
              <input id="budget" type="range" min={200} max={16000} step={100} value={b} onChange={(e) => setBudget(Number(e.target.value))} style={{ width: 160 }} />
              <span className="num" style={{ minWidth: 80 }}>{b.toLocaleString()} tokens</span>
            </label>
            <label className="row small" style={{ gap: 8 }}>
              As of <input id="asof" className="input" type="datetime-local" value={asOf} onChange={(e) => setAsOf(e.target.value)} style={{ width: 210, minHeight: 30 }} />
            </label>
          </div>
        </form>
      </Card>
      {run.error && <ErrorBox error={run.error} />}
      {run.isPending && <Loading label="Searching, grading with Jev and fitting the budget" />}
      {run.data?.explain && <RecallView ex={run.data.explain} markdown={run.data.markdown} tokens={run.data.token_count} />}
      {!run.data && !run.isPending && <Empty title="Ask something" icon={<Search />}>The result shows the retrieval path, the funnel, the memory block and a score breakdown for every candidate.</Empty>}
    </div>
  );
}

export function RecallPage({ id }: { id: string }) {
  const detail = useQuery({ queryKey: ["recall", id], queryFn: () => api.recallDetail(id) });
  const d = detail.data;
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>{d?.query ? `“${truncate(d.query, 90)}”` : "Recall"}</h1>
          <p>{d ? <>At <code>{d.scope}</code> · {fmtDate(d.created_at, true)} · <span className="mono faint">{short(d.id)}</span></> : "Loading…"}</p>
        </div>
      </div>
      {detail.isLoading && <Loading />}
      {detail.error && <ErrorBox error={detail.error} />}
      {d && (d.explain
        ? <RecallView ex={d.explain} markdown={d.markdown} tokens={d.token_count} />
        : <Card title="Memory block"><p className="small muted">This recall was made before explanations were recorded.</p><MemoryMarkdown text={d.markdown} /></Card>)}
    </div>
  );
}
