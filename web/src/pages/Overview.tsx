import { useQuery } from "@tanstack/react-query";
import { ArrowRight, PlusCircle, Search } from "lucide-react";
import { api, type Category } from "../lib/api";
import { useApp } from "../lib/app";
import { CATEGORIES, CATEGORY_LABEL, catVar, ms, timeAgo, truncate } from "../lib/meta";
import { Card, Empty, ErrorBox, Loading, Pill, useTip } from "../components/ui";

function Kpi({ label, value, detail }: { label: string; value: React.ReactNode; detail?: React.ReactNode }) {
  return (
    <div className="card kpi">
      <span className="label">{label}</span>
      <span className="v">{value}</span>
      {detail && <span className="d">{detail}</span>}
    </div>
  );
}

function CategoryBars({ rows }: { rows: { category: Category; status: string; n: number }[] }) {
  const tip = useTip();
  const by = new Map<Category, { active: number; superseded: number; expired: number }>();
  for (const r of rows) {
    const e = by.get(r.category) ?? { active: 0, superseded: 0, expired: 0 };
    e[r.status as "active"] += r.n;
    by.set(r.category, e);
  }
  const max = Math.max(1, ...[...by.values()].map((e) => e.active + e.superseded + e.expired));
  const present = CATEGORIES.filter((c) => by.has(c));
  if (!present.length) return <Empty title="No facts yet">Add memory to see how it is categorized.</Empty>;
  return (
    <div>
      {present.map((c) => {
        const e = by.get(c)!;
        const old = e.superseded + e.expired;
        return (
          <div key={c} className="hbar" {...tip(<><b>{CATEGORY_LABEL[c]}</b><br />{e.active} active · {e.superseded} superseded · {e.expired} expired</>)}>
            <span className="row" style={{ gap: 7, flexWrap: "nowrap" }}>
              <span className="chip" style={{ border: 0, padding: 0, background: "none" }}><span className="sw" style={{ background: catVar(c) }} />{CATEGORY_LABEL[c]}</span>
            </span>
            <div className="track">
              <div className="seg-a" style={{ width: `${(e.active / max) * 100}%`, background: catVar(c) }} />
              {old > 0 && <div className="seg-b" style={{ width: `${(old / max) * 100}%`, background: catVar(c) }} />}
            </div>
            <span className="num small" style={{ textAlign: "right" }}>{e.active.toLocaleString()}{old > 0 && <span className="faint"> +{old}</span>}</span>
          </div>
        );
      })}
      <div className="legend" style={{ marginTop: 10 }}>
        <span className="item"><span className="sw" style={{ width: 14, height: 8, borderRadius: 2, background: "var(--muted)" }} />Active</span>
        <span className="item"><span className="sw" style={{ width: 14, height: 8, borderRadius: 2, background: "var(--muted)", opacity: 0.38 }} />Superseded or expired (history)</span>
      </div>
    </div>
  );
}

function PerDay({ rows }: { rows: { day: string; n: number }[] }) {
  const tip = useTip();
  const days: { day: string; n: number }[] = [];
  const map = new Map(rows.map((r) => [r.day, r.n]));
  for (let i = 29; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86400_000).toISOString().slice(0, 10);
    days.push({ day: d, n: map.get(d) ?? 0 });
  }
  const max = Math.max(1, ...days.map((d) => d.n));
  const total = days.reduce((a, d) => a + d.n, 0);
  return (
    <div>
      <div className="cols" aria-label={`Facts created per day, last 30 days: ${total} total`}>
        {days.map((d) => (
          <div key={d.day} className="c" style={{ height: `${(d.n / max) * 100}%` }} {...tip(<><b>{d.n.toLocaleString()} facts</b><br />{d.day}</>)} />
        ))}
      </div>
      <div className="row small faint" style={{ justifyContent: "space-between", marginTop: 6 }}>
        <span>{days[0].day}</span><span>{total.toLocaleString()} in 30 days · peak {max.toLocaleString()}/day</span><span>today</span>
      </div>
    </div>
  );
}

export function Overview() {
  const { scope, go } = useApp();
  const stats = useQuery({ queryKey: ["stats", scope], queryFn: () => api.stats(scope) });
  const ingests = useQuery({ queryKey: ["ingests", scope, 6], queryFn: () => api.ingests(scope, 6) });
  const recalls = useQuery({ queryKey: ["recalls", scope, 6], queryFn: () => api.recalls(scope, 6) });

  const s = stats.data;
  const count = (status: string) => s?.facts.filter((f) => f.status === status).reduce((a, f) => a + f.n, 0) ?? 0;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Overview</h1>
          <p>Everything stored at <code>{scope}</code> and the scopes below it.</p>
        </div>
        <div className="actions">
          <button className="btn" onClick={() => go({ page: "ask" })}><Search /> Ask a question</button>
          <button className="btn primary" onClick={() => go({ page: "add" })}><PlusCircle /> Add memory</button>
        </div>
      </div>
      {stats.error && <ErrorBox error={stats.error} />}
      {stats.isLoading && <Loading />}
      {s && (
        <>
          <div className="grid kpis">
            <Kpi label="Active facts" value={count("active").toLocaleString()} detail={`${s.scopes} scope${s.scopes === 1 ? "" : "s"}`} />
            <Kpi label="History" value={(count("superseded") + count("expired")).toLocaleString()} detail={`${count("superseded")} replaced · ${count("expired")} expired`} />
            <Kpi label="Entities" value={s.entities.toLocaleString()} detail={`${(s.links.mentions ?? 0).toLocaleString()} mentions · ${s.links.related ?? 0} related · ${s.links.shadows ?? 0} shadows`} />
            <Kpi label="Inserts" value={Object.values(s.ingests).reduce((a, b) => a + b, 0).toLocaleString()} detail={s.ingests.failed ? <span style={{ color: "var(--danger)" }}>{s.ingests.failed} failed</span> : "none failed"} />
            <Kpi label="Recalls" value={s.recalls.n.toLocaleString()} detail={s.recalls.n ? `avg ${ms(s.recalls.avg_ms)} · ${s.recalls.avg_tokens} tokens` : "none yet"} />
            <Kpi label="Jev decisions" value={Object.values(s.decisions).reduce((a, b) => a + b, 0).toLocaleString()} detail={`${s.decisions.sort ?? 0} sort · ${s.decisions.verdict ?? 0} verdict · ${s.decisions.grade ?? 0} grade`} />
          </div>
          <div className="grid cols-2">
            <Card title="Facts by category" sub="Active, with history alongside">
              <CategoryBars rows={s.facts} />
            </Card>
            <Card title="New facts per day" sub="Last 30 days">
              <PerDay rows={s.facts_per_day} />
            </Card>
          </div>
        </>
      )}
      <div className="grid cols-2">
        <Card title="Recent inserts" right={<button className="btn ghost sm" onClick={() => go({ page: "activity", tab: "ingests" })}>All <ArrowRight /></button>} flush>
          {ingests.data?.length ? (
            <table className="t"><tbody>
              {ingests.data.map((r) => (
                <tr key={r.id} className="click" onClick={() => go({ page: "ingest", id: r.id })}>
                  <td style={{ width: 90 }}><Pill tone={r.status === "committed" ? "ok" : r.status === "failed" ? "danger" : "warn"}>{r.status}</Pill></td>
                  <td><div className="ellipsis" style={{ maxWidth: 360 }}>{truncate(r.preview, 120)}</div><div className="small faint mono">{r.scope}</div></td>
                  <td className="n small muted">{r.receipt ? `+${r.receipt.added ?? 0}` : "—"}<div className="faint">{timeAgo(r.created_at)}</div></td>
                </tr>
              ))}
            </tbody></table>
          ) : ingests.isLoading ? <Loading /> : <Empty title="No inserts yet" />}
        </Card>
        <Card title="Recent questions" right={<button className="btn ghost sm" onClick={() => go({ page: "activity", tab: "recalls" })}>All <ArrowRight /></button>} flush>
          {recalls.data?.length ? (
            <table className="t"><tbody>
              {recalls.data.map((r) => (
                <tr key={r.id} className="click" onClick={() => go({ page: "recall", id: r.id })}>
                  <td style={{ width: 90 }}><Pill tone={r.mode === "ranked" ? "accent" : "purple"}>{r.mode}</Pill></td>
                  <td><div className="ellipsis" style={{ maxWidth: 360 }}>{r.query ?? <span className="muted">(no query)</span>}</div><div className="small faint mono">{r.scope}</div></td>
                  <td className="n small muted">{r.facts} facts<div className="faint">{timeAgo(r.created_at)}</div></td>
                </tr>
              ))}
            </tbody></table>
          ) : recalls.isLoading ? <Loading /> : <Empty title="No questions yet">Ask one to see how memory is retrieved.</Empty>}
        </Card>
      </div>
    </div>
  );
}
