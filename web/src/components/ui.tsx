import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { AlertTriangle, Inbox, Loader2, X } from "lucide-react";
import type { Category } from "../lib/api";
import { CATEGORY_LABEL, catVar } from "../lib/meta";

export function CategoryChip({ category, small }: { category: Category; small?: boolean }) {
  return (
    <span className="chip" style={small ? { height: 20, fontSize: 11.5 } : undefined}>
      <span className="sw" style={{ background: catVar(category) }} />
      {CATEGORY_LABEL[category] ?? category}
    </span>
  );
}

export function Pill({ tone, children, title, icon }: { tone: string; children: ReactNode; title?: string; icon?: ReactNode }) {
  return (
    <span className={`pill tone-${tone}`} title={title}>
      {icon}
      {children}
    </span>
  );
}

export function Meter({ value, tone, max = 1, title }: { value: number; tone?: "purple" | "ok"; max?: number; title?: string }) {
  const w = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div className={`meter ${tone ?? ""}`} title={title} role="meter" aria-valuenow={value} aria-valuemin={0} aria-valuemax={max}>
      <i style={{ width: `${w}%` }} />
    </div>
  );
}

export function ProbBars({ probs, labels, limit = 9, tone }: {
  probs: Record<string, number>; labels?: Record<string, string>; limit?: number; tone?: "purple";
}) {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, limit);
  const top = entries[0]?.[0];
  return (
    <div className="stack" style={{ gap: 5 }}>
      {entries.map(([k, v]) => (
        <div key={k} className={`prob ${k === top ? "top" : ""}`}>
          <span className="name" title={labels?.[k] ?? k}>{labels?.[k] ?? k}</span>
          <Meter value={v} tone={tone} />
          <span className="val">{(v * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

export function Empty({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="empty">
      {icon ?? <Inbox />}
      <b>{title}</b>
      {children && <p>{children}</p>}
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="empty" aria-busy="true">
      <Loader2 className="spin" />
      <p>{label}…</p>
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="banner tone-danger">
      <AlertTriangle />
      <div>{message}</div>
    </div>
  );
}

export function Card({ title, sub, right, children, flush, className }: {
  title?: ReactNode; sub?: ReactNode; right?: ReactNode; children: ReactNode; flush?: boolean; className?: string;
}) {
  return (
    <section className={`card ${className ?? ""}`}>
      {(title || right) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {sub && <div className="sub">{sub}</div>}
          </div>
          {right && <div className="right">{right}</div>}
        </div>
      )}
      <div className={`card-body ${flush ? "flush" : ""}`}>{children}</div>
    </section>
  );
}

export function Seg<T extends string>({ value, options, onChange, label }: {
  value: T; options: { value: T; label: ReactNode; icon?: ReactNode }[]; onChange: (v: T) => void; label: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Drawer({ open, onClose, title, children, head }: {
  open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; head?: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return createPortal(
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true">
        <div className="drawer-head">
          <div style={{ flex: 1, minWidth: 0, display: "grid", gap: 8 }}>
            {head}
            <h2>{title}</h2>
          </div>
          <button className="btn ghost icon-btn" onClick={onClose} aria-label="Close">
            <X />
          </button>
        </div>
        <div className="drawer-body">{children}</div>
      </aside>
    </>,
    document.body,
  );
}

/** A button that asks for a second click instead of a browser confirm dialog. */
export function ConfirmButton({ onConfirm, children, confirmLabel, className = "btn danger", disabled }: {
  onConfirm: () => void; children: ReactNode; confirmLabel: string; className?: string; disabled?: boolean;
}) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(t);
  }, [armed]);
  return (
    <button
      type="button"
      className={`${className} ${armed ? "confirm" : ""}`}
      disabled={disabled}
      onClick={() => (armed ? (setArmed(false), onConfirm()) : setArmed(true))}
    >
      {armed ? confirmLabel : children}
    </button>
  );
}

// ---- tooltips ----
type TipState = { x: number; y: number; content: ReactNode } | null;
const TipCtx = createContext<(t: TipState) => void>(() => {});

export function TipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TipState>(null);
  return (
    <TipCtx.Provider value={setTip}>
      {children}
      {tip && createPortal(
        <div className="tip" style={{ left: Math.min(tip.x + 12, window.innerWidth - 330), top: tip.y + 14 }}>{tip.content}</div>,
        document.body,
      )}
    </TipCtx.Provider>
  );
}

export function useTip() {
  const set = useContext(TipCtx);
  return useCallback(
    (content: ReactNode) => ({
      onMouseMove: (e: React.MouseEvent) => set({ x: e.clientX, y: e.clientY, content }),
      onMouseLeave: () => set(null),
      onFocus: (e: React.FocusEvent) => {
        const r = (e.target as HTMLElement).getBoundingClientRect();
        set({ x: r.left, y: r.bottom, content });
      },
      onBlur: () => set(null),
    }),
    [set],
  );
}

// ---- toasts ----
type Toast = { id: number; text: string; err?: boolean };
const ToastCtx = createContext<(text: string, err?: boolean) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const next = useRef(1);
  const push = useCallback((text: string, err?: boolean) => {
    const id = next.current++;
    setItems((xs) => [...xs, { id, text, err }]);
    setTimeout(() => setItems((xs) => xs.filter((t) => t.id !== id)), err ? 7000 : 3500);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {items.map((t) => <div key={t.id} className={`toast ${t.err ? "err" : ""}`}>{t.text}</div>)}
      </div>
    </ToastCtx.Provider>
  );
}

export const useToast = () => useContext(ToastCtx);

// ---- memory markdown (the exact block an agent receives) ----
export function MemoryMarkdown({ text }: { text: string }) {
  if (!text.trim()) return <Empty title="Nothing to show">The memory block is empty for this view.</Empty>;
  const lines = text.split("\n");
  const out: ReactNode[] = [];
  type Item = { text: string; children: Item[] };
  let list: Item[] = [];
  const nested = (items: Item[]) => (
    <ul>{items.map((c, j) => <li key={j}><Line text={c.text} />{c.children.length > 0 && nested(c.children)}</li>)}</ul>
  );
  const flush = (key: string) => {
    if (!list.length) return;
    out.push(
      <ul key={key}>
        {list.map((item, i) => (
          <li key={i}>
            <Line text={item.text} />
            {item.children.length > 0 && nested(item.children)}
          </li>
        ))}
      </ul>,
    );
    list = [];
  };
  lines.forEach((line, i) => {
    if (line.startsWith("# ")) { flush(`l${i}`); out.push(<h1 key={i}>{line.slice(2)}</h1>); }
    else if (line.startsWith("## ")) { flush(`l${i}`); out.push(<h2 key={i}>{line.slice(3)}</h2>); }
    else if (line.startsWith("    - ")) { list[list.length - 1]?.children.at(-1)?.children.push({ text: line.slice(6), children: [] }); }
    else if (line.startsWith("  - ")) { list[list.length - 1]?.children.push({ text: line.slice(4), children: [] }); }
    else if (line.startsWith("- ")) { list.push({ text: line.slice(2), children: [] }); }
  });
  flush("end");
  return <div className="mem">{out}</div>;
}

function Line({ text }: { text: string }) {
  const date = text.match(/^\((\d{4}-\d{2}-\d{2})\)\s+/);
  const scope = text.match(/\s\[([A-Za-z0-9_.-]+)\]$/);
  let body = text;
  if (date) body = body.slice(date[0].length);
  if (scope) body = body.slice(0, body.length - scope[0].length);
  return (
    <>
      {date && <span className="date">{date[1]}</span>}
      {body}
      {scope && <span className="scope-tag">{scope[1]}</span>}
    </>
  );
}
