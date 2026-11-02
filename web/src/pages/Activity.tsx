import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useApp } from "../lib/app";
import { fmtDate, ms, timeAgo, truncate } from "../lib/meta";
import { Card, Empty, ErrorBox, Loading, Pill } from "../components/ui";

type Tab = "ingests" | "recalls" | "jobs";

export function ActivityPage({ tab: initial }: { tab?: string }) {
  const { scope, go } = useApp();
  const [tab, setTab] = useState<Tab>((["ingests", "recalls", "jobs"].includes(initial ?? "") ? initial : "ingests") as Tab);
  const [all, setAll] = useState(false);
  const s = all ? undefined : scope;
  const ingests = useQuery({ queryKey: ["ingests", s, 200], queryFn: () => api.ingests(s, 200), enabled: tab === "ingests" });
  const recalls = useQuery({ queryKey: ["recalls", s, 200], queryFn: () => api.recalls(s, 200), enabled: tab === "recalls" });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: () => api.jobs(), enabled: tab === "jobs", refetchInterval: tab === "jobs" ? 5000 : false });

  const q = tab === "ingests" ? ingests : tab === "recalls" ? recalls : jobs;
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Activity</h1>
          <p>Every insert, question and background job. Open one to replay exactly what happened.</p>
        </div>
        <div className="actions">
          {tab !== "jobs" && (
            <label className="row small" style={{ gap: 6 }}>
              <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> All scopes
            </label>
          )}
        </div>
      </div>
      <Card flush>
        <div className="tabs" role="tablist">
          {(["ingests", "recalls", "jobs"] as Tab[]).map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
              {t === "ingests" ? "Inserts" : t === "recalls" ? "Questions" : "Jobs"}
            </button>
          ))}
        </div>
        {q.isLoading && <Loading />}
        {q.error && <div style={{ padding: 14 }}><ErrorBox error={q.error} /></div>}
        {tab === "ingests" && ingests.data && (ingests.data.length ? (
          <div className="table-wrap"><table className="t">
            <thead><tr><th>Status</th><th>Input</th><th>Scope</th><th className="n">Added</th><th className="n">Dup</th><th className="n">Replaced</th><th className="n">Time</th><th>When</th></tr></thead>
            <tbody>
              {ingests.data.map((r) => (
                <tr key={r.id} className="click" onClick={() => go({ page: "ingest", id: r.id })}>
                  <td><Pill tone={r.status === "committed" ? "ok" : r.status === "failed" ? "danger" : "warn"}>{r.status}</Pill></td>
                  <td style={{ maxWidth: 440 }}>
                    <div className="ellipsis">{truncate(r.preview, 140)}</div>
                    {r.error && <div className="small" style={{ color: "var(--danger)" }}>{truncate(r.error, 140)}</div>}
                  </td>
                  <td className="mono small">{r.scope}</td>
                  <td className="n">{r.receipt?.added ?? "—"}</td>
                  <td className="n">{r.receipt?.dropped ?? "—"}</td>
                  <td className="n">{r.receipt ? (r.receipt.superseded ?? 0) + (r.receipt.shadowed ?? 0) : "—"}</td>
                  <td className="n small muted">{ms(r.receipt?.timings?.total_ms)}</td>
                  <td className="small muted" title={fmtDate(r.created_at, true)}>{timeAgo(r.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
        ) : <Empty title="No inserts yet" />)}
        {tab === "recalls" && recalls.data && (recalls.data.length ? (
          <div className="table-wrap"><table className="t">
            <thead><tr><th>Mode</th><th>Question</th><th>Scope</th><th className="n">Facts</th><th className="n">Tokens</th><th className="n">Latency</th><th>When</th></tr></thead>
            <tbody>
              {recalls.data.map((r) => (
                <tr key={r.id} className="click" onClick={() => go({ page: "recall", id: r.id })}>
                  <td><Pill tone={r.mode === "ranked" ? "accent" : "purple"}>{r.mode}</Pill></td>
                  <td style={{ maxWidth: 440 }}><div className="ellipsis">{r.query ?? <span className="muted">(no query)</span>}</div></td>
                  <td className="mono small">{r.scope}</td>
                  <td className="n">{r.facts}</td>
                  <td className="n">{r.token_count}</td>
                  <td className="n">{ms(r.latency_ms)}</td>
                  <td className="small muted" title={fmtDate(r.created_at, true)}>{timeAgo(r.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
        ) : <Empty title="No questions yet" />)}
        {tab === "jobs" && jobs.data && (jobs.data.length ? (
          <div className="table-wrap"><table className="t">
            <thead><tr><th>Status</th><th>Kind</th><th className="n">Attempts</th><th>Worker</th><th>Last error</th><th>Created</th></tr></thead>
            <tbody>
              {jobs.data.map((j) => (
                <tr key={j.id} className={j.kind === "ingest" ? "click" : ""} onClick={() => j.kind === "ingest" && go({ page: "ingest", id: String(j.payload.ingest_id) })}>
                  <td><Pill tone={j.status === "done" ? "ok" : j.status === "dead" ? "danger" : j.status === "running" ? "accent" : "warn"}>{j.status}</Pill></td>
                  <td>{j.kind}</td>
                  <td className="n">{j.attempts}/{j.max_attempts}</td>
                  <td className="mono small muted">{j.locked_by ?? "—"}</td>
                  <td className="small" style={{ maxWidth: 380, color: j.last_error ? "var(--danger)" : undefined }}>{j.last_error ? truncate(j.last_error, 160) : "—"}</td>
                  <td className="small muted" title={fmtDate(j.created_at, true)}>{timeAgo(j.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
        ) : <Empty title="No jobs" />)}
      </Card>
    </div>
  );
}
