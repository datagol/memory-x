import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, CornerDownRight, FolderTree, Plus } from "lucide-react";
import { api } from "../lib/api";
import { useApp } from "../lib/app";

export const SEGMENT = /^[A-Za-z0-9_-]{1,64}$/;
export const validScope = (s: string) => s.length > 0 && s.split(".").every((p) => SEGMENT.test(p));

interface ScopeNode { path: string; depth: number; active: number; total: number; known: boolean }

/** Searchable scope tree. Intermediate scopes that were never written to still appear, so the hierarchy reads. */
export function ScopePicker() {
  const { scope, setScope } = useApp();
  const [open, setOpen] = useState(false);
  const [term, setTerm] = useState("");
  const [cursor, setCursor] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const { data } = useQuery({ queryKey: ["scopes"], queryFn: api.scopes, staleTime: 15_000 });

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as globalThis.Node)) setOpen(false); };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const nodes = useMemo(() => {
    const map = new Map<string, ScopeNode>();
    for (const s of data ?? []) {
      const labels = s.path.split(".");
      labels.forEach((_, i) => {
        const p = labels.slice(0, i + 1).join(".");
        const n = map.get(p) ?? { path: p, depth: i, active: 0, total: 0, known: false };
        n.total += s.active;
        if (p === s.path) { n.active = s.active; n.known = true; }
        map.set(p, n);
      });
    }
    return [...map.values()].sort((a, b) => a.path.localeCompare(b.path));
  }, [data]);

  const filtered = useMemo(() => {
    const t = term.trim().toLowerCase();
    return t ? nodes.filter((n) => n.path.toLowerCase().includes(t)) : nodes;
  }, [nodes, term]);

  const candidate = term.trim();
  const canCreate = validScope(candidate) && !nodes.some((n) => n.path === candidate);
  const options = [...filtered.map((n) => n.path), ...(canCreate ? [candidate] : [])];

  const pick = (p: string) => { setScope(p); setOpen(false); setTerm(""); };

  return (
    <div className="scope-picker" ref={ref}>
      <button className="scope-btn" onClick={() => setOpen((o) => !o)} aria-haspopup="listbox" aria-expanded={open}>
        <FolderTree />
        <span className="scope-lbl">Scope</span>
        <span className="path">{scope}</span>
        <ChevronDown style={{ marginLeft: "auto" }} />
      </button>
      {open && (
        <div className="scope-pop">
          <div className="search">
            <input
              autoFocus
              className="input"
              placeholder="Find a scope, or type a new one (memory.acme.team)"
              value={term}
              onChange={(e) => { setTerm(e.target.value); setCursor(0); }}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") { e.preventDefault(); setCursor((c) => Math.min(c + 1, options.length - 1)); }
                if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
                if (e.key === "Enter" && options[cursor]) pick(options[cursor]);
                if (e.key === "Escape") setOpen(false);
              }}
            />
            <span className="small muted">Recall at a scope also reads every parent scope. Counts are active facts in each scope and below it.</span>
          </div>
          <div className="scope-list" role="listbox">
            {filtered.map((n, i) => (
              <button
                key={n.path}
                role="option"
                aria-selected={n.path === scope}
                className={`scope-item ${n.path === scope ? "selected" : ""} ${i === cursor ? "active" : ""}`}
                style={{ paddingLeft: term ? 8 : 8 + n.depth * 14 }}
                onClick={() => pick(n.path)}
              >
                {n.depth > 0 && !term && <CornerDownRight className="tw" />}
                <span className="ellipsis">{term ? n.path : n.path.split(".").pop()}</span>
                <span className="count">{n.total.toLocaleString()}</span>
              </button>
            ))}
            {canCreate && (
              <button className={`scope-item ${cursor === filtered.length ? "active" : ""}`} onClick={() => pick(candidate)}>
                <Plus className="tw" /> Use new scope <b style={{ marginLeft: 4 }}>{candidate}</b>
              </button>
            )}
            {!filtered.length && !canCreate && (
              <div className="small muted" style={{ padding: 10 }}>
                No scope matches. Scope names use letters, digits, _ and -, separated by dots.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
