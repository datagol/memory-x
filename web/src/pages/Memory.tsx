import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, Trash2 } from "lucide-react";
import { api, type Category } from "../lib/api";
import { useApp } from "../lib/app";
import { CATEGORIES, CATEGORY_LABEL, fmtDate, num } from "../lib/meta";
import { statusTone } from "../components/FactDrawer";
import { ScopeSettingsCard } from "../components/ScopeSettings";
import { Card, CategoryChip, ConfirmButton, Empty, ErrorBox, Loading, MemoryMarkdown, Pill, Seg, useToast, rowKeys } from "../components/ui";

export function MemoryPage() {
  const { scope, openFact, setScope } = useApp();
  const qc = useQueryClient();
  const toast = useToast();
  const [status, setStatus] = useState<"active" | "superseded" | "expired" | "">("active");
  const [category, setCategory] = useState<Category | "">("");
  const [ancestors, setAncestors] = useState(true);
  const [search, setSearch] = useState("");

  const rendered = useQuery({ queryKey: ["render", scope], queryFn: () => api.render(scope) });
  const facts = useQuery({
    queryKey: ["facts", scope, status, category, ancestors],
    queryFn: () => api.facts(scope, { status, category: category || undefined, include_ancestors: ancestors, limit: 1000 }),
  });
  const forget = useMutation({
    mutationFn: () => api.deleteScope(scope),
    onSuccess: (r) => {
      toast(`Forgot ${scope}: ${r.facts_removed} facts and ${r.ingests_removed} inserts removed.`);
      qc.invalidateQueries();
      setScope(scope.split(".").slice(0, -1).join(".") || "memory");
    },
    onError: (e) => toast((e as Error).message, true),
  });

  const rows = useMemo(() => {
    const t = search.trim().toLowerCase();
    return (facts.data ?? []).filter((f) => !t || f.text.toLowerCase().includes(t));
  }, [facts.data, search]);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Memory</h1>
          <p>What an agent at <code>{scope}</code> receives, and every fact behind it. Select a fact to see its history and the decisions that shaped it.</p>
        </div>
        <div className="actions">
          {scope.includes(".") ? (
            <ConfirmButton confirmLabel={`Forget ${scope} and everything below`} onConfirm={() => forget.mutate()} disabled={forget.isPending}>
              <Trash2 /> Forget scope
            </ConfirmButton>
          ) : (
            // The top scope holds every tenant's memory; wiping all of it is not a two-click action.
            <button className="btn danger" disabled title={`${scope} is the top scope. Pick a scope below it to forget.`}>
              <Trash2 /> Forget scope
            </button>
          )}
        </div>
      </div>
      <ScopeSettingsCard scope={scope} />
      <div className="split">
        <Card
          title="Agent view"
          sub="The full memory block for this scope and its parents"
          right={rendered.data ? (
            <button className="btn ghost sm" onClick={() => navigator.clipboard?.writeText(rendered.data!).then(() => toast("Copied"), () => toast("Copy is blocked here; select the text instead.", true))}>
              <Copy /> Copy
            </button>
          ) : undefined}
        >
          {rendered.isLoading ? <Loading /> : rendered.error ? <ErrorBox error={rendered.error} /> : <MemoryMarkdown text={rendered.data ?? ""} />}
        </Card>
        <Card
          title="Facts"
          sub={facts.data ? `${rows.length.toLocaleString()} shown` : undefined}
          flush
        >
          <div className="stack" style={{ padding: 12, borderBottom: "1px solid var(--line)" }}>
            <div className="row">
              <Seg label="Status" value={status} onChange={setStatus} options={[
                { value: "active", label: "Active" }, { value: "superseded", label: "Replaced" },
                { value: "expired", label: "Expired" }, { value: "", label: "All" },
              ]} />
              <select className="select" style={{ width: "auto" }} value={category} onChange={(e) => setCategory(e.target.value as Category | "")} aria-label="Category">
                <option value="">All categories</option>
                {CATEGORIES.map((c) => <option key={c} value={c}>{CATEGORY_LABEL[c]}</option>)}
              </select>
              <label className="row small" style={{ gap: 6 }}>
                <input type="checkbox" checked={ancestors} onChange={(e) => setAncestors(e.target.checked)} /> Include parent scopes
              </label>
            </div>
            <input className="input" placeholder="Filter by text" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Filter facts by text" />
          </div>
          {facts.isLoading ? <Loading /> : facts.error ? <div style={{ padding: 12 }}><ErrorBox error={facts.error} /></div> : rows.length === 0 ? (
            <Empty title="No facts match">Change the filters, or add memory to this scope.</Empty>
          ) : (
            <div className="table-wrap" style={{ maxHeight: "calc(100vh - 300px)", overflowY: "auto" }}>
              <table className="t">
                <thead><tr><th>Fact</th><th>Category</th><th className="n">Trust</th><th>Observed</th></tr></thead>
                <tbody>
                  {rows.map((f) => (
                    <tr key={f.id} className={`click ${f.status !== "active" ? "dim" : ""}`} onClick={() => openFact(f.id)} {...rowKeys(() => openFact(f.id))}>
                      <td>
                        <div style={{ textDecoration: f.status !== "active" ? "line-through" : undefined }}>{f.text}</div>
                        <div className="row small faint" style={{ gap: 6, marginTop: 3 }}>
                          {f.scope !== scope && <span className="mono">{f.scope}</span>}
                          {f.status !== "active" && <Pill tone={statusTone(f.status)}>{f.status}</Pill>}
                          {f.reinforced_count > 0 && <span>reinforced {f.reinforced_count}×</span>}
                          {Boolean(f.flags?.injection_suspect) && <Pill tone="danger">suspect</Pill>}
                        </div>
                      </td>
                      <td><CategoryChip category={f.category} small /></td>
                      <td className="n">{num(f.trust, 1)}</td>
                      <td className="small muted" style={{ whiteSpace: "nowrap" }}>{fmtDate(f.observed_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
