import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RotateCcw, SlidersHorizontal } from "lucide-react";
import { api, type ScopeSettingKey, type ScopeSettingValue, type ScopeSettings } from "../lib/api";
import { ErrorBox, Loading, useToast } from "./ui";

type Draft = Record<ScopeSettingKey, ScopeSettingValue>;

const ROWS: { key: ScopeSettingKey; label: string; help: string; show: (v: ScopeSettingValue) => string }[] = [
  { key: "budget_tokens", label: "Recall budget", help: "Tokens an agent here may receive. Below it the whole memory is sent; above it facts are ranked for the question.", show: (v) => `${Number(v).toLocaleString()} tokens` },
  { key: "event_ttl_days", label: "Upcoming events expire", help: "Days after its date that an upcoming event (“on PTO next week”) stops being recalled. Empty or 0: never; events fade with the event half-life instead. Applies to new inserts.", show: (v) => (v ? `${v} days after the date` : "events never expire") },
  { key: "text_trust", label: "Document trust", help: "Default trust for pasted or uploaded documents that don't set one. Lower trust ranks their facts lower. Applies to new inserts.", show: (v) => `document trust ${Number(v).toFixed(1)}` },
  { key: "walk_enabled", label: "Walk shared entities", help: "In ranked recall, also consider facts linked to the top hits through a shared person, team or system.", show: (v) => (v ? "walk on" : "walk off") },
];

function draftOf(s: ScopeSettings): Draft {
  return Object.fromEntries(ROWS.map((r) => [r.key, s.effective[r.key].value])) as Draft;
}

export function ScopeSettingsCard({ scope }: { scope: string }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const q = useQuery({ queryKey: ["scope-settings", scope], queryFn: () => api.scopeSettings(scope) });
  const [draft, setDraft] = useState<Draft | null>(null);
  useEffect(() => { if (q.data) setDraft(draftOf(q.data)); }, [q.data]);

  const save = useMutation({
    mutationFn: (changes: Partial<Record<ScopeSettingKey, ScopeSettingValue | null>>) => api.setScopeSettings(scope, changes),
    onSuccess: (s) => {
      qc.setQueryData(["scope-settings", scope], s);
      qc.invalidateQueries({ queryKey: ["scope-settings"] });
      qc.invalidateQueries({ queryKey: ["render"] });
      toast(`Settings saved for ${scope} and the scopes below it`);
    },
    onError: (e) => toast((e as Error).message, true),
  });

  const s = q.data;
  const dirty = s && draft ? ROWS.filter((r) => draft[r.key] !== s.effective[r.key].value).map((r) => r.key) : [];
  const overridden = s ? ROWS.filter((r) => s.effective[r.key].from !== "default") : [];

  return (
    <div className="card scope-settings">
      <button type="button" className="scope-settings-bar" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <SlidersHorizontal />
        <b>Settings for this scope</b>
        {s ? (
          <span className="summary">
            {ROWS.map((r, i) => (
              <span key={r.key} className={s.effective[r.key].from !== "default" ? "set" : undefined}>
                {i > 0 && " · "}{r.show(s.effective[r.key].value)}
              </span>
            ))}
          </span>
        ) : <span className="summary muted">loading…</span>}
        <span className="toggle">{open ? "Close" : overridden.length ? `Edit · ${overridden.length} changed` : "Edit"}</span>
      </button>
      {open && (
        <div className="card-body stack">
          <p className="small muted" style={{ margin: 0 }}>
            Overrides apply to <code>{scope}</code> and every scope below it, unless a narrower scope sets its own. Everything else is
            server-wide and set in <code>.env</code>; see Settings.
          </p>
          {q.isLoading || !draft || !s ? (q.error ? <ErrorBox error={q.error} /> : <Loading />) : (
            <form className="stack" onSubmit={(e) => {
              e.preventDefault();
              if (dirty.length) save.mutate(Object.fromEntries(dirty.map((k) => [k, draft[k]])));
            }}>
              {ROWS.map((r) => {
                const eff = s.effective[r.key];
                const here = eff.from === scope;
                return (
                  <div key={r.key} className="setting-row">
                    <div className="setting-label">
                      <label htmlFor={`set-${r.key}`}>{r.label}</label>
                      <small>{r.help}</small>
                    </div>
                    <div className="setting-control">
                      {r.key === "walk_enabled" ? (
                        <select id={`set-${r.key}`} className="select" value={draft[r.key] ? "on" : "off"}
                          onChange={(e) => setDraft({ ...draft, [r.key]: e.target.value === "on" })}>
                          <option value="on">On</option><option value="off">Off</option>
                        </select>
                      ) : r.key === "text_trust" ? (
                        <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
                          <input id={`set-${r.key}`} type="range" min={0} max={1} step={0.1} value={Number(draft[r.key])}
                            onChange={(e) => setDraft({ ...draft, [r.key]: Number(e.target.value) })} style={{ flex: 1 }} />
                          <span className="mono small">{Number(draft[r.key]).toFixed(1)}</span>
                        </div>
                      ) : r.key === "event_ttl_days" ? (
                        <input id={`set-${r.key}`} className="input" type="number" min={0} max={3650} step={1}
                          value={draft[r.key] ? String(draft[r.key]) : ""} placeholder="never (empty or 0)"
                          onChange={(e) => setDraft({ ...draft, [r.key]: e.target.value === "" ? 0 : Number(e.target.value) })} />
                      ) : (
                        <input id={`set-${r.key}`} className="input" type="number" value={Number(draft[r.key])}
                          min={200} max={200000} step={100}
                          onChange={(e) => setDraft({ ...draft, [r.key]: Number(e.target.value) })} />
                      )}
                      <div className="setting-source">
                        {here ? <span className="set">Set here</span>
                          : eff.from === "default" ? <span>Server default · <code>{eff.env}</code></span>
                            : <span>Inherited from <code>{eff.from}</code></span>}
                        {here && (
                          <button type="button" className="btn ghost sm" disabled={save.isPending}
                            onClick={() => save.mutate({ [r.key]: null })} title="Remove this scope's override and inherit again">
                            <RotateCcw /> Reset
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              <div className="row end">
                {dirty.length > 0 && (
                  <button type="button" className="btn ghost" onClick={() => setDraft(draftOf(s))}>Discard</button>
                )}
                <button className="btn primary" type="submit" disabled={!dirty.length || save.isPending}>
                  {dirty.length ? `Save ${dirty.length} change${dirty.length > 1 ? "s" : ""} for this scope` : "No changes"}
                </button>
              </div>
            </form>
          )}
        </div>
      )}
    </div>
  );
}
