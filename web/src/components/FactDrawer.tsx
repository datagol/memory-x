import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, GitBranch, Network, Trash2, Workflow } from "lucide-react";
import { api, type Decision, type Fact } from "../lib/api";
import { useApp } from "../lib/app";
import { CATEGORY_LABEL, ENTITY_TYPE_LABEL, VERDICT_LABEL, fmtDate, num, pct, short, topicLabel } from "../lib/meta";
import { CategoryChip, ConfirmButton, Drawer, ErrorBox, Loading, Pill, ProbBars, useToast } from "./ui";

export function statusTone(status: Fact["status"]) {
  return status === "active" ? "ok" : status === "superseded" ? "muted" : "warn";
}

function DecisionLine({ d, factId }: { d: Decision; factId: string }) {
  if (d.step === "sort") {
    const r = d.result;
    return (
      <div className="stack" style={{ gap: 6 }}>
        <div className="small"><b>Categorized</b> by Jev as {CATEGORY_LABEL[r.category as keyof typeof CATEGORY_LABEL] ?? r.category} ({pct(r.confidence)} confident). Injection check {pct(r.injection)}.</div>
        {r.probabilities && <ProbBars probs={r.probabilities} labels={CATEGORY_LABEL} limit={3} />}
      </div>
    );
  }
  if (d.step === "entity") {
    const r = d.result;
    return (
      <div className="small">
        <b>Entity “{r.name}”</b> {r.action === "alias"
          ? <>merged into “{r.with}” by Jev ({pct(r.p_same)} the same thing).</>
          : <>kept separate from “{r.with}” (Jev {pct(r.p_same)} the same thing).</>}
      </div>
    );
  }
  if (d.step === "verdict" || d.step === "auto") {
    const r = d.result;
    const mine = d.subject_fact_id === factId;
    const other = mine ? r.neighbor : r.candidate;
    const how = d.step === "auto" ? "automatically (near-identical text)" : `by Jev (${pct(r.probability)})`;
    return (
      <div className="small">
        <b>{VERDICT_LABEL[r.label as keyof typeof VERDICT_LABEL] ?? r.label}</b> {how}, similarity {num(d.similarity)}.
        <div className="muted" style={{ marginTop: 2 }}>{mine ? "Compared with" : "Compared by new fact"}: “{other}”</div>
      </div>
    );
  }
  return <div className="small">{d.step}</div>;
}

export function FactDrawer() {
  const { factId, openFact, go, scope } = useApp();
  const qc = useQueryClient();
  const toast = useToast();
  const fact = useQuery({ queryKey: ["fact", factId], queryFn: () => api.fact(factId!), enabled: !!factId });
  const hist = useQuery({ queryKey: ["history", factId], queryFn: () => api.history(factId!), enabled: !!factId });

  const del = useMutation({
    mutationFn: () => api.deleteFact(factId!),
    onSuccess: () => { toast("Fact deleted. The deletion is recorded in the forget log."); qc.invalidateQueries(); openFact(null); },
    onError: (e) => toast(String((e as Error).message), true),
  });
  const promote = useMutation({
    mutationFn: (to: string) => api.promote(factId!, to),
    onSuccess: (f) => { toast(`Promoted to ${f.scope}`); qc.invalidateQueries(); openFact(f.id); },
    onError: (e) => toast(String((e as Error).message), true),
  });

  const f = fact.data;
  const parents = f ? f.scope.split(".").slice(0, -1).map((_, i, a) => a.slice(0, i + 1).join(".")).reverse() : [];

  return (
    <Drawer
      open={!!factId}
      onClose={() => openFact(null)}
      title={f ? f.text : "Fact"}
      head={f && (
        <div className="row">
          <CategoryChip category={f.category} />
          <Pill tone={statusTone(f.status)}>{f.status}</Pill>
          {Boolean(f.flags?.injection_suspect) && <Pill tone="danger">Suspected injection</Pill>}
          <span className="small faint mono">{short(f.id)}</span>
        </div>
      )}
    >
      {fact.isLoading && <Loading />}
      {fact.error && <ErrorBox error={fact.error} />}
      {f && (
        <>
          <div className="row">
            <button className="btn sm" onClick={() => { openFact(null); go({ page: "graph", focus: f.id }); }}><Network /> Show in graph</button>
            {f.ingest_id && <button className="btn sm" onClick={() => { openFact(null); go({ page: "ingest", id: f.ingest_id! }); }}><Workflow /> How it was added</button>}
            {f.status === "active" && parents.length > 0 && (
              <select className="select" style={{ width: "auto", minHeight: 28, height: 28, padding: "0 8px", fontSize: 12.5 }}
                value="" onChange={(e) => e.target.value && promote.mutate(e.target.value)} aria-label="Promote to a broader scope">
                <option value="">Promote to…</option>
                {parents.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            )}
            <span style={{ marginLeft: "auto" }} />
            <ConfirmButton className="btn sm danger" confirmLabel="Delete permanently" onConfirm={() => del.mutate()} disabled={del.isPending}>
              <Trash2 /> Delete
            </ConfirmButton>
          </div>

          <dl className="kv">
            <dt>Scope</dt><dd className="mono">{f.scope}{f.scope !== scope && <span className="faint"> (viewing {scope})</span>}</dd>
            <dt>Said by</dt><dd>{f.speaker ?? f.source ?? "—"} · trust {num(f.trust, 1)}</dd>
            <dt>Observed</dt><dd>{fmtDate(f.observed_at, true)}</dd>
            {f.event_at && <><dt>Event date</dt><dd>{fmtDate(f.event_at)}</dd></>}
            {f.stale_after && <><dt>Expires</dt><dd>{fmtDate(f.stale_after)}</dd></>}
            <dt>Sort confidence</dt><dd>{pct(f.sort_confidence)}</dd>
            <dt>Theme</dt><dd>{f.topic ? <span className="chip topic">{topicLabel(f.topic)} <span className="faint">{pct(f.topic_confidence)}</span></span> : <span className="faint">none clear enough</span>}</dd>
            <dt>Mentions</dt>
            <dd>
              {f.entities.length ? (
                <div className="row" style={{ gap: 4 }}>
                  {f.entities.map((e) => (
                    <button key={e.id} className="chip ent" style={{ cursor: "pointer" }} title={`${ENTITY_TYPE_LABEL[e.type]}. Click to see every fact mentioning it.`}
                      onClick={() => { openFact(null); go({ page: "graph", focus: f.id }); }}>
                      {e.name} <span className="faint">{ENTITY_TYPE_LABEL[e.type].toLowerCase()}</span>
                    </button>
                  ))}
                </div>
              ) : <span className="faint">no named entities</span>}
            </dd>
            <dt>Reinforced</dt><dd>{f.reinforced_count}× {f.last_reinforced_at && <span className="muted">· last {fmtDate(f.last_reinforced_at)}</span>}</dd>
            <dt>Recalled</dt><dd>{f.access_count}× {f.last_accessed_at && <span className="muted">· last {fmtDate(f.last_accessed_at, true)}</span>}</dd>
            <dt>Tokens</dt><dd className="num">{f.token_count}</dd>
            {f.superseded_by && <><dt>Replaced by</dt><dd><a href="#" onClick={(e) => { e.preventDefault(); openFact(f.superseded_by!); }}>{short(f.superseded_by)} <ArrowUpRight size={12} /></a> on {fmtDate(f.superseded_at)}</dd></>}
          </dl>

          <div className="stack">
            <div className="label"><GitBranch size={12} style={{ verticalAlign: -1 }} /> Version history</div>
            {hist.isLoading && <Loading />}
            {hist.data && (
              <ol className="timeline">
                {hist.data.versions.map((v) => (
                  <li key={v.fact.id} className={v.fact.id === f.id ? "current" : ""}>
                    <div className="row" style={{ gap: 6 }}>
                      <Pill tone={statusTone(v.fact.status)}>{v.fact.status}</Pill>
                      <span className="small muted">{fmtDate(v.fact.observed_at)} · <span className="mono">{v.fact.scope}</span></span>
                    </div>
                    <div style={{ margin: "6px 0 8px", fontWeight: v.fact.id === f.id ? 600 : 400 }}>
                      {v.fact.id === f.id ? v.fact.text : <a href="#" onClick={(e) => { e.preventDefault(); openFact(v.fact.id); }}>{v.fact.text}</a>}
                    </div>
                    <div className="stack" style={{ gap: 10 }}>
                      {v.decisions.filter((d) => d.subject_fact_id === v.fact.id).map((d) => <DecisionLine key={d.id} d={d} factId={v.fact.id} />)}
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </div>

          {hist.data && hist.data.links.length > 0 && (
            <div className="stack">
              <div className="label">Links</div>
              {hist.data.links.map((l, i) => {
                const other = l.a === f.id ? l.b : l.a;
                const text = l.kind === "shadows"
                  ? (l.a === f.id ? "Hides an ancestor-scope fact for this scope and below" : "Hidden in a narrower scope by")
                  : "Related to";
                return (
                  <div key={i} className="row small">
                    <Pill tone={l.kind === "shadows" ? "purple" : "accent"}>{l.kind}</Pill>
                    <span>{text}</span>
                    <a href="#" className="mono" onClick={(e) => { e.preventDefault(); openFact(other); }}>{short(other)}</a>
                    {l.similarity !== null && <span className="muted">sim {num(l.similarity)}</span>}
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </Drawer>
  );
}
