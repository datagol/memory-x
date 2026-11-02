import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Background, Controls, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";
import { ArrowRight, Boxes, FileText, Layers, MessageSquare, Plus, Sparkles, Trash2, Wand2 } from "lucide-react";
import { api, type Fact, type IngestTrace, type Neighbor, type TraceCandidate } from "../lib/api";
import { useApp } from "../lib/app";
import { BAND_META, CATEGORY_LABEL, OUTCOME_META, VERDICT_LABEL, catVar, fmtDate, ms, num, pct, short, topicLabel, truncate } from "../lib/meta";
import { nodeTypes } from "../components/FlowNodes";
import { validScope } from "../components/ScopePicker";
import { Card, CategoryChip, Empty, ErrorBox, Loading, Pill, ProbBars, Seg, useTip, useToast } from "../components/ui";

// ---------------------------------------------------------------- composer

type Turn = { role: string; content: string };
const EXAMPLE: Turn[] = [
  { role: "user", content: "btw I switched to light mode, the dark one hurts my eyes. Also I'm on PTO next week." },
];

function ScopeField() {
  const { scope, setScope } = useApp();
  const [value, setValue] = useState(scope);
  useEffect(() => setValue(scope), [scope]);
  const ok = validScope(value.trim());
  const labels = (ok ? value.trim() : scope).split(".");
  const commit = () => { if (ok && value.trim() !== scope) setScope(value.trim()); };
  return (
    <label className="field">
      <span>Store into scope</span>
      <input id="ingest-scope" className="input mono" value={value} onChange={(e) => setValue(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } }}
        placeholder="memory.company_2.user_3.agent_1334" aria-invalid={!ok} style={ok ? undefined : { borderColor: "var(--danger)" }} />
      {ok ? (
        <small>
          Dots separate levels, e.g. <code>memory.company_2.user_3.agent_1334</code>. An agent here also reads{" "}
          {labels.length > 1 ? labels.slice(0, -1).map((_, i) => <code key={i}>{i ? ", " : ""}{labels.slice(0, i + 1).join(".")}</code>) : "no parent scopes"}.
        </small>
      ) : (
        <small style={{ color: "var(--danger)" }}>Use letters, digits, _ or - in each part, separated by dots.</small>
      )}
    </label>
  );
}

export function AddPage() {
  const { scope, go } = useApp();
  const qc = useQueryClient();
  const toast = useToast();
  const [kind, setKind] = useState<"messages" | "text">("messages");
  const [turns, setTurns] = useState<Turn[]>([{ role: "user", content: "" }]);
  const [doc, setDoc] = useState({ text: "", source: "notes", trust: 0.8 });
  const [observed, setObserved] = useState("");

  const run = useMutation({
    mutationFn: () => {
      const input = kind === "messages"
        ? { kind, turns: turns.filter((t) => t.content.trim()).map((t) => ({ role: t.role.trim() || "user", content: t.content })) }
        : { kind, text: doc.text, source: doc.source || "document", trust: doc.trust };
      return api.insert(scope, input, observed ? new Date(observed).toISOString() : undefined);
    },
    onSuccess: (r) => {
      qc.invalidateQueries();
      toast(`${r.added} added · ${r.dropped} duplicate · ${r.superseded} replaced · ${r.shadowed} shadowed`);
      go({ page: "ingest", id: r.ingest_id });
    },
    onError: (e) => toast((e as Error).message, true),
  });

  const empty = kind === "messages" ? !turns.some((t) => t.content.trim()) : !doc.text.trim();
  const setTurn = (i: number, patch: Partial<Turn>) => setTurns((ts) => ts.map((t, j) => (j === i ? { ...t, ...patch } : t)));

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Add memory</h1>
          <p>Paste a conversation or a document. memx extracts facts, sorts them, checks them against what <code>{scope}</code> already knows, and shows you every step.</p>
        </div>
      </div>
      <div className="split">
        <Card
          title="Input"
          right={<Seg label="Input type" value={kind} onChange={setKind} options={[
            { value: "messages", label: "Conversation", icon: <MessageSquare /> },
            { value: "text", label: "Document", icon: <FileText /> },
          ]} />}
        >
          <form className="stack" onSubmit={(e) => { e.preventDefault(); if (!empty) run.mutate(); }}>
            <ScopeField />
            {kind === "messages" ? (
              <>
                {turns.map((t, i) => (
                  <div key={i} className="composer-turn">
                    <select id={`role-${i}`} className="select" value={["user", "assistant"].includes(t.role) ? t.role : "other"}
                      onChange={(e) => setTurn(i, { role: e.target.value === "other" ? "alice" : e.target.value })} aria-label="Speaker">
                      <option value="user">User</option>
                      <option value="assistant">Assistant</option>
                      <option value="other">Named person</option>
                    </select>
                    <div className="stack" style={{ gap: 6 }}>
                      {!["user", "assistant"].includes(t.role) && (
                        <input className="input" value={t.role} onChange={(e) => setTurn(i, { role: e.target.value })} placeholder="Name" aria-label="Speaker name" />
                      )}
                      <textarea id={`turn-${i}`} className="textarea" rows={3} value={t.content} placeholder={i === 0 ? "I lead the data platform team. We never deploy on Fridays." : "Next message"}
                        onChange={(e) => setTurn(i, { content: e.target.value })} aria-label={`Message ${i + 1}`} />
                    </div>
                    <button type="button" className="btn ghost icon-btn" aria-label="Remove message" disabled={turns.length === 1}
                      onClick={() => setTurns((ts) => ts.filter((_, j) => j !== i))}><Trash2 /></button>
                  </div>
                ))}
                <div className="row">
                  <button type="button" className="btn sm" onClick={() => setTurns((ts) => [...ts, { role: ts.at(-1)?.role === "user" ? "assistant" : "user", content: "" }])}><Plus /> Add message</button>
                  <button type="button" className="btn ghost sm" onClick={() => setTurns(EXAMPLE)}><Wand2 /> Use an example</button>
                </div>
              </>
            ) : (
              <>
                <label className="field"><span>Document text</span>
                  <textarea id="doc" className="textarea" rows={10} value={doc.text} onChange={(e) => setDoc({ ...doc, text: e.target.value })} placeholder="Paste notes, a wiki page, meeting minutes…" />
                </label>
                <div className="grid cols-2">
                  <label className="field"><span>Source label</span>
                    <input id="source" className="input" value={doc.source} onChange={(e) => setDoc({ ...doc, source: e.target.value })} />
                  </label>
                  <label className="field"><span>Trust · {doc.trust.toFixed(1)}</span>
                    <input id="trust" type="range" min={0} max={1} step={0.1} value={doc.trust} onChange={(e) => setDoc({ ...doc, trust: Number(e.target.value) })} />
                    <small>Lower trust ranks these facts lower at recall time.</small>
                  </label>
                </div>
              </>
            )}
            <label className="field"><span>When it was said (optional)</span>
              <input id="observed" className="input" type="datetime-local" value={observed} onChange={(e) => setObserved(e.target.value)} style={{ maxWidth: 260 }} />
              <small>Defaults to now. Relative dates like “next week” resolve against this.</small>
            </label>
            <div className="row end">
              <span className="small muted">Into <code>{scope}</code></span>
              <button className="btn primary" type="submit" disabled={empty || run.isPending}>
                {run.isPending ? <><Sparkles className="spin" /> Extracting and reconciling…</> : <>Add to memory <ArrowRight /></>}
              </button>
            </div>
            {run.error && <ErrorBox error={run.error} />}
          </form>
        </Card>
        <Card title="What happens next">
          <ol className="timeline">
            {[
              ["Extract", "Haiku reads the input in windows of about 2,500 tokens and writes standalone facts, keeping numbers, names and dates."],
              ["Sort", "Jev picks one of nine categories and checks whether the text tries to instruct the AI."],
              ["Find neighbors", "Vector and keyword search over this scope and its parents. Similarity bands decide which pairs Jev must judge."],
              ["Decide", "Jev says whether the new fact is the same, replaces, is covered by, or relates to each neighbor."],
              ["Commit", "Under a scope lock: insert, replace, hide a parent fact for this scope, link, or reinforce. Every decision is logged."],
            ].map(([t, d]) => (
              <li key={t}><b>{t}</b><div className="small muted" style={{ marginTop: 2 }}>{d}</div></li>
            ))}
          </ol>
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- explanations

function decisionText(c: TraceCandidate, facts: Record<string, Fact>, bandHigh: number): string {
  const v = c.verdicts ?? [];
  const judged = v.filter((x) => x.source === "jev");
  if (c.outcome === "rejected") {
    return c.reason === "injection"
      ? `Jev rated it ${pct(c.sort?.injection)} likely to be an instruction aimed at the AI, above the reject line, so it was not stored.`
      : "Longer than 600 characters, so it was not stored.";
  }
  if (c.outcome === "dropped") {
    const same = v.find((x) => x.neighbor_id === c.same_as);
    const target = c.same_as ? `“${truncate(facts[c.same_as]?.text ?? "an existing fact", 80)}”` : "an existing fact";
    const retired = c.superseded_ids?.length
      ? ` It also retired ${c.superseded_ids.length} outdated fact${c.superseded_ids.length > 1 ? "s" : ""} in narrower scopes that contradicted it.`
      : "";
    if (same?.source === "hash") return `Exactly the same text as ${target}. Nothing new was stored; the existing fact was reinforced.${retired}`;
    if (same?.source === "auto") return `Near-identical to ${target} (similarity ${num(same.sim)}, above ${bandHigh}) with no new words, so no judge was needed. The existing fact was reinforced.${retired}`;
    return `Jev judged it the same as ${target} (${pct(same?.probability)}). The existing fact was reinforced instead of storing a copy.${retired}`;
  }
  if (c.outcome === "superseded_on_arrival") {
    return "Jev found an existing fact that already says this more completely or more recently, so it was stored only as history.";
  }
  const parts: string[] = [];
  if (c.note) parts.push("Flagged as a possible injection: stored, but hidden from agents and not compared with other facts.");
  if (c.superseded_ids?.length) parts.push(`Replaces ${c.superseded_ids.length} older fact${c.superseded_ids.length > 1 ? "s" : ""} here or in narrower scopes.`);
  if (c.shadowed_ids?.length) parts.push(`Hides ${c.shadowed_ids.length} parent-scope fact${c.shadowed_ids.length > 1 ? "s" : ""} for this scope and below; the parent keeps it.`);
  if (c.related_ids?.length) parts.push(`Linked as related to ${c.related_ids.length} fact${c.related_ids.length > 1 ? "s" : ""}, so they render together.`);
  if (!parts.length) {
    if (!(c.neighbors ?? []).length) parts.push("Nothing similar was stored yet, so it was added as a new fact.");
    else if (!judged.length) parts.push("Every neighbor was below the similarity band, so it was added as a new fact without asking the judge.");
    else parts.push("Jev judged every close neighbor unrelated, so it was added as a new fact.");
  }
  return parts.join(" ");
}

// ---------------------------------------------------------------- similarity axis

function SimAxis({ neighbors, low, high }: { neighbors: Neighbor[]; low: number; high: number }) {
  const tip = useTip();
  const color = (b: Neighbor["band"]) =>
    b === "judge" ? "var(--indigo)" : b === "auto" || b === "exact" ? "var(--ok)" : "var(--faint)";
  const top = neighbors.reduce<Neighbor | null>((a, n) => (!a || n.sim > a.sim ? n : a), null);
  return (
    <div>
      <div className="simaxis" aria-label="Neighbor similarity">
        <div className="track" />
        <div className="zone" style={{ left: `${low * 100}%`, width: `${(high - low) * 100}%`, background: "var(--indigo-weak)" }} />
        <div className="zone" style={{ left: `${high * 100}%`, right: 0, background: "var(--ok-bg)" }} />
        {[low, high].map((t) => <div key={t} className="tick" style={{ left: `${t * 100}%` }}><span>{t.toFixed(2)}</span></div>)}
        <span className="end l">0</span><span className="end r">1</span>
        {neighbors.map((n) => (
          <div key={n.id} tabIndex={0} className={`pt ${n === top ? "top" : ""}`} style={{ left: `${Math.max(0, Math.min(1, n.sim)) * 100}%`, background: color(n.band) }}
            {...tip(<><b>{BAND_META[n.band].label}</b> · similarity {n.sim.toFixed(3)}<br />{truncate(n.text, 140)}<br /><span style={{ opacity: 0.7 }}>{n.scope} · found by {n.via === "fts" ? "keyword" : n.via === "vec" ? "vector" : n.via} search</span></>)} />
        ))}
      </div>
      <div className="legend" style={{ marginTop: 18 }}>
        <span className="item"><span className="dot" style={{ background: "var(--faint)" }} />below {low}: unrelated</span>
        <span className="item"><span className="dot" style={{ background: "var(--indigo)" }} />{low}–{high}: Jev decides</span>
        <span className="item"><span className="dot" style={{ background: "var(--ok)" }} />above {high}: duplicate if no new words</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- path diagram

function PathDiagram({ trace }: { trace: IngestTrace }) {
  const { openFact } = useApp();
  const { nodes, edges, height } = useMemo(() => {
    const X = { input: 0, window: 250, cand: 520, decide: 830, effect: 1110 };
    const GAP = 150;
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const cands = trace.candidates;
    const ing = trace.ingest;
    const edge = (id: string, source: string, target: string, opts: Partial<Edge> & { color?: string } = {}) => {
      const color = opts.color ?? "var(--line-strong)";
      edges.push({ id, source, target, type: "smoothstep", style: { stroke: color, strokeWidth: 1.6 }, markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 }, ...opts });
    };
    let effectY = 0;
    const effectCount = (c: TraceCandidate) =>
      (c.fact_id ? 1 : 0) + (c.same_as ? 1 : 0) + (c.superseded_by ? 1 : 0)
      + (c.superseded_ids?.length ?? 0) + (c.shadowed_ids?.length ?? 0) + (c.related_ids?.length ?? 0);
    const candY = new Map<number, number>();
    let cursor = 0;
    cands.forEach((c) => { candY.set(c.index, cursor); cursor += Math.max(GAP, effectCount(c) * 112 + 30); });
    const totalH = Math.max(GAP, cursor);
    const words = ing.payload.kind === "messages" ? `${ing.payload.turns.length} messages` : `${ing.payload.text.length.toLocaleString()} characters`;
    nodes.push({ id: "input", type: "stage", position: { x: X.input, y: totalH / 2 - 60 }, data: {
      kicker: <><Layers /> Input</>, title: ing.kind === "messages" ? "Conversation" : `Document · ${ing.source}`,
      body: <>{words}<br />observed {fmtDate(ing.observed_at)}</>, tone: "query",
    } });
    trace.windows.forEach((w) => {
      const mine = cands.filter((c) => c.window === w.index);
      const ys = mine.map((c) => candY.get(c.index)!);
      const y = ys.length ? (Math.min(...ys) + Math.max(...ys)) / 2 : (w.index * totalH) / Math.max(1, trace.windows.length);
      nodes.push({ id: `w${w.index}`, type: "stage", position: { x: X.window, y }, data: {
        kicker: <><Boxes /> Window {w.index + 1}</>, title: `${w.extracted ?? 0} facts extracted`,
        body: `${w.units} unit${w.units === 1 ? "" : "s"} · ${w.chars.toLocaleString()} chars${w.context_units ? ` · ${w.context_units} context` : ""}`, width: 210,
      } });
      edge(`in-w${w.index}`, "input", `w${w.index}`, { color: "var(--indigo)" });
    });
    cands.forEach((c) => {
      const y = candY.get(c.index)!;
      const cat = c.sort?.category ?? c.category ?? "other";
      nodes.push({ id: `c${c.index}`, type: "fact", position: { x: X.cand, y }, data: {
        text: c.text, category: cat, status: "active",
        badges: c.sort ? <span>{pct(c.sort.confidence)} sure</span> : undefined, width: 260,
      } });
      edge(`w-c${c.index}`, `w${c.window}`, `c${c.index}`);
      const meta = c.outcome ? OUTCOME_META[c.outcome] : { label: "Pending", tone: "muted" };
      const judged = (c.verdicts ?? []).filter((v) => v.source === "jev").length;
      const near = (c.neighbors ?? []).filter((n) => n.band !== "below").length;
      nodes.push({ id: `d${c.index}`, type: "stage", position: { x: X.decide, y }, data: {
        kicker: <>Decision</>, title: <Pill tone={meta.tone}>{meta.label}</Pill>,
        body: `${(c.neighbors ?? []).length} neighbors · ${near} in band · ${judged} judged`, width: 210,
      } });
      edge(`c-d${c.index}`, `c${c.index}`, `d${c.index}`);
      // effects
      const effects: { id: string; label: string; color: string; status?: Fact["status"] }[] = [];
      if (c.fact_id) effects.push({ id: c.fact_id, label: c.outcome === "superseded_on_arrival" ? "stored as history" : "new fact", color: "var(--ok)" });
      if (c.same_as) effects.push({ id: c.same_as, label: "reinforced", color: "var(--muted)" });
      (c.superseded_ids ?? []).forEach((id) => effects.push({ id, label: "replaced", color: "var(--warn)" }));
      (c.shadowed_ids ?? []).forEach((id) => effects.push({ id, label: "shadowed here", color: "var(--purple)" }));
      (c.related_ids ?? []).forEach((id) => effects.push({ id, label: "related", color: "var(--indigo)" }));
      if (c.superseded_by) effects.push({ id: c.superseded_by, label: "already covers it", color: "var(--muted)" });
      effectY = Math.max(effectY, y);
      effects.forEach((e, k) => {
        const f = trace.facts[e.id];
        const nid = `f${c.index}-${e.id}`;
        nodes.push({ id: nid, type: "fact", position: { x: X.effect, y: y + k * 112 }, data: {
          text: f?.text ?? short(e.id), category: f?.category ?? "other", status: f?.status ?? "active", scope: f?.scope, width: 270,
        } });
        edge(`d-${nid}`, `d${c.index}`, nid, { color: e.color, label: e.label, animated: e.label === "new fact" });
        effectY = Math.max(effectY, y + k * 112);
      });
    });
    return { nodes, edges, height: Math.max(360, Math.min(900, Math.max(totalH, effectY + 140) + 80)) };
  }, [trace]);

  return (
    <div className="flow" style={{ height }}>
      <ReactFlow
        nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.12, maxZoom: 1 }}
        minZoom={0.2} nodesDraggable={false} proOptions={{ hideAttribution: true }}
        onNodeClick={(_, n) => { if (n.id.startsWith("f")) openFact(n.id.split("-").slice(1).join("-")); }}
      >
        <Background gap={22} size={1} color="var(--line)" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

// ---------------------------------------------------------------- per-candidate detail

function CandidateCard({ c, trace }: { c: TraceCandidate; trace: IngestTrace }) {
  const { openFact } = useApp();
  const meta = c.outcome ? OUTCOME_META[c.outcome] : null;
  const verdictById = new Map((c.verdicts ?? []).map((v) => [v.neighbor_id, v]));
  const shown = (c.neighbors ?? []).filter((n) => n.band !== "below").slice(0, 6);
  const link = (id: string) => (
    <a href="#" onClick={(e) => { e.preventDefault(); openFact(id); }}>{truncate(trace.facts[id]?.text ?? short(id), 70)}</a>
  );
  return (
    <article className="cand">
      <header className="cand-head">
        <span className="idx">#{c.index + 1}</span>
        <div className="txt">{c.text}<div className="small muted" style={{ fontWeight: 400, marginTop: 3 }}>
          said by {c.speaker} · window {c.window + 1}{c.event_date ? ` · event date ${c.event_date}` : ""}{c.time_text ? ` · “${c.time_text}”` : ""}
        </div></div>
        {meta && <Pill tone={meta.tone} title={meta.help}>{meta.label}</Pill>}
      </header>
      <div className="cand-body">
        <section>
          <div className="label">1 · Sort (Jev)</div>
          {c.sort ? (
            <>
              <div className="row" style={{ gap: 6 }}>
                <CategoryChip category={c.sort.category} />
                {c.category_hint && c.category_hint !== c.sort.category && <span className="small muted">extractor guessed {CATEGORY_LABEL[c.category_hint]}</span>}
              </div>
              <ProbBars probs={c.sort.probabilities} labels={CATEGORY_LABEL} limit={4} />
              <div className="small muted">Injection check {pct(c.sort.injection, 1)} · {ms(c.sort.ms)}</div>
              <div className="row small" style={{ gap: 6, marginTop: 4 }}>
                <span className="muted">Theme</span>
                {c.sort.topic
                  ? <span className="chip topic" title={`Jev picked this theme at ${pct(c.sort.topic_confidence)}`}>{topicLabel(c.sort.topic)} <span className="faint">{pct(c.sort.topic_confidence)}</span></span>
                  : <span className="faint">none clear enough</span>}
              </div>
            </>
          ) : <span className="small muted">Not sorted ({c.reason === "too_long" ? "too long" : "skipped"}).</span>}
          <div className="row small" style={{ gap: 6, marginTop: 8, alignItems: "flex-start" }}>
            <span className="muted" style={{ paddingTop: 2 }}>Entities</span>
            {(c.entities ?? []).length ? (
              <div className="row" style={{ gap: 4 }}>
                {(c.entities ?? []).map((e) => (
                  <span key={e.name} className="chip ent" title={
                    e.action === "alias" ? `“${e.name}” merged into “${e.canonical}” (Jev ${pct(e.merge?.p_same)} same)`
                    : e.action === "link" ? `Already known${e.canonical && e.canonical !== e.name ? ` as “${e.canonical}”` : ""}`
                    : e.action === "create" ? "New entity" : e.merge ? `Kept separate from “${e.merge.with}” (Jev ${pct(e.merge.p_same)} same)` : ""}>
                    {e.canonical && e.canonical !== e.name ? <>{e.name} <span className="faint">→ {e.canonical}</span></> : e.name}
                    <span className="faint">{e.type}</span>
                  </span>
                ))}
              </div>
            ) : <span className="faint">none named</span>}
          </div>
        </section>
        <section>
          <div className="label">2 · Neighbors</div>
          {(c.neighbors ?? []).length ? (
            <>
              <SimAxis neighbors={c.neighbors!} low={trace.config.band_low} high={trace.config.band_high} />
              <div className="stack" style={{ gap: 8 }}>
                {shown.map((n) => {
                  const v = verdictById.get(n.id);
                  return (
                    <div key={n.id} className="small" style={{ borderLeft: `3px solid ${catVar(n.category)}`, paddingLeft: 8 }}>
                      <div>{link(n.id)}</div>
                      <div className="row muted" style={{ gap: 6, marginTop: 2 }}>
                        <span className="num">sim {n.sim.toFixed(3)}</span>
                        <span>· {BAND_META[n.band].label}</span>
                        {n.scope !== trace.ingest.scope && <span className="mono">· {n.scope}</span>}
                        {v && <Pill tone={v.label === "unrelated" ? "muted" : v.label === "same" ? "ok" : "accent"}>{VERDICT_LABEL[v.label]} {v.source === "jev" ? pct(v.probability) : ""}</Pill>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : <span className="small muted">No existing facts in this scope or its parents were close enough to compare.</span>}
        </section>
        <section>
          <div className="label">3 · Decision</div>
          <p className="why" style={{ margin: 0 }}>{decisionText(c, trace.facts, trace.config.band_high)}</p>
          {c.fact_id && <div className="small">Stored as {link(c.fact_id)}</div>}
          {(c.superseded_ids ?? []).map((id) => <div key={id} className="small">Replaced: {link(id)}{trace.facts[id] && trace.facts[id].scope !== trace.ingest.scope && <span className="mono faint"> · {trace.facts[id].scope}</span>}</div>)}
          {(c.shadowed_ids ?? []).map((id) => <div key={id} className="small">Hidden here: {link(id)}</div>)}
          {(c.verdicts ?? []).filter((v) => v.source === "jev").length > 0 && (
            <details>
              <summary className="small" style={{ cursor: "pointer", color: "var(--indigo)" }}>Jev's full verdict probabilities</summary>
              <div className="stack" style={{ marginTop: 8 }}>
                {(c.verdicts ?? []).filter((v) => v.source === "jev").map((v) => (
                  <div key={v.neighbor_id} className="stack" style={{ gap: 4 }}>
                    <div className="small muted">vs “{truncate(trace.facts[v.neighbor_id]?.text ?? "", 60)}”</div>
                    <ProbBars probs={v.probabilities} labels={VERDICT_LABEL} limit={5} />
                  </div>
                ))}
              </div>
            </details>
          )}
          {c.reconcile_ms !== undefined && <div className="small faint">Reconciled in {ms(c.reconcile_ms)}</div>}
        </section>
      </div>
    </article>
  );
}

function Timings({ t }: { t?: Record<string, number> }) {
  const tip = useTip();
  if (!t) return null;
  const parts = [
    ["Extract", t.extract_ms, "var(--indigo)"], ["Embed", t.embed_ms, "var(--cat-identity)"],
    ["Sort", t.sort_ms, "var(--purple)"], ["Reconcile + commit", t.reconcile_commit_ms, "var(--cat-preference)"],
  ] as const;
  const total = Math.max(1, t.total_ms);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div style={{ display: "flex", gap: 2, height: 12 }}>
        {parts.map(([k, v, c]) => <div key={k} style={{ width: `${(v / total) * 100}%`, background: c, borderRadius: 3, minWidth: v ? 3 : 0 }} {...tip(<><b>{k}</b> {ms(v)}</>)} />)}
      </div>
      <div className="legend">
        {parts.map(([k, v, c]) => <span key={k} className="item"><span className="dot" style={{ background: c }} />{k} {ms(v)}</span>)}
        <span className="item" style={{ marginLeft: "auto" }}><b>Total {ms(t.total_ms)}</b></span>
      </div>
    </div>
  );
}

export function IngestTraceView({ trace }: { trace: IngestTrace }) {
  const ing = trace.ingest;
  const r = ing.receipt ?? {};
  return (
    <>
      <div className="grid kpis">
        {([["Added", r.added, "ok"], ["Duplicates", r.dropped, "muted"], ["Replaced", r.superseded, "warn"], ["Shadowed", r.shadowed, "purple"], ["History", r.historical, "muted"], ["Rejected", r.rejected, "danger"]] as const).map(([k, v]) => (
          <div key={k} className="card kpi"><span className="label">{k}</span><span className="v">{v ?? 0}</span></div>
        ))}
      </div>
      <Card title="Path through the pipeline" sub="Input → extraction windows → candidate facts → decisions → facts created or changed. Click a fact to open it.">
        {trace.candidates.length ? <PathDiagram trace={trace} /> : <Empty title="No facts extracted">The extractor found nothing worth remembering in this input.</Empty>}
      </Card>
      <div className="grid cols-2">
        <Card title="Timing">
          <Timings t={ing.receipt?.timings} />
          <dl className="kv" style={{ marginTop: 14 }}>
            <dt>Extractor</dt><dd>{ing.extractor_model ?? "—"}</dd>
            <dt>Tokens</dt><dd className="num">{ing.tokens_in.toLocaleString()} in · {ing.tokens_out.toLocaleString()} out</dd>
            <dt>Mode</dt><dd>{ing.mode}</dd>
          </dl>
        </Card>
        <Card title="Input">
          {ing.payload.kind === "messages" ? (
            <div className="stack" style={{ gap: 8, maxHeight: 260, overflowY: "auto" }}>
              {ing.payload.turns.map((t, i) => (
                <div key={i} className="small"><b>{t.role}</b>: <span style={{ whiteSpace: "pre-wrap" }}>{truncate(t.content, 1200)}</span></div>
              ))}
            </div>
          ) : <div className="small" style={{ whiteSpace: "pre-wrap", maxHeight: 260, overflowY: "auto" }}>{truncate(ing.payload.text, 4000)}</div>}
        </Card>
      </div>
      <div className="stack">
        <h2 style={{ margin: "8px 0 0", fontSize: 16 }}>Each candidate, step by step</h2>
        {trace.candidates.map((c) => <CandidateCard key={c.index} c={c} trace={trace} />)}
      </div>
    </>
  );
}

export function IngestPage({ id }: { id: string }) {
  const { go } = useApp();
  const trace = useQuery({ queryKey: ["trace", id], queryFn: () => api.trace(id), refetchInterval: (q) => (q.state.data && ["committed", "failed"].includes(q.state.data.ingest.status) ? false : 1500) });
  const ing = trace.data?.ingest;
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Insert trace</h1>
          <p>{ing ? <>Into <code>{ing.scope}</code> · {fmtDate(ing.created_at, true)} · <span className="mono faint">{short(ing.id)}</span></> : "Loading…"}</p>
        </div>
        <div className="actions">
          {ing && <Pill tone={ing.status === "committed" ? "ok" : ing.status === "failed" ? "danger" : "warn"}>{ing.status}</Pill>}
          <button className="btn" onClick={() => go({ page: "add" })}><Plus /> Add more</button>
        </div>
      </div>
      {trace.isLoading && <Loading />}
      {trace.error && <ErrorBox error={trace.error} />}
      {ing?.error && <ErrorBox error={ing.error} />}
      {trace.data && <IngestTraceView trace={trace.data} />}
    </div>
  );
}
