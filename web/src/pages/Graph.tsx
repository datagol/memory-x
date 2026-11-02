import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Background, Controls, MarkerType, Panel, ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type Node, type NodeChange,
} from "@xyflow/react";
import { ChevronDown, ChevronRight, ExternalLink, Info, RotateCcw, X } from "lucide-react";
import { api, type Category, type GraphData } from "../lib/api";
import { useApp } from "../lib/app";
import { CATEGORIES, CATEGORY_LABEL, ENTITY_TYPE_LABEL, catVar, topicLabel, truncate } from "../lib/meta";
import {
  EntityIcon, facing, nodeTypes, type ClusterKind, type ClusterLabelData, type ClusterNodeData, type EntityNodeData, type FactNodeData,
} from "../components/FlowNodes";
import { statusTone } from "../components/FactDrawer";
import { Card, CategoryChip, Empty, ErrorBox, Loading, Pill, Seg } from "../components/ui";

// Geometry. Fact heights are estimated from text length so boxes can be laid out before React Flow measures.
const NODE_W = 260;
const ENT_W = 214;
const ENT_H = 50;
const PAD = 16;
const GAP = 48;
const COL_GAP = 18;
const FACT_GAP = 12;
const LABEL_H = 38;
const CHIP_MAX = 3;

type XY = { x: number; y: number };
type Size = { w: number; h: number };
type GFact = GraphData["facts"][number];
type Ent = GraphData["entities"][number] & { degree: number; hub: boolean };
type Group = "entity" | "scope";

interface Cluster {
  id: string;
  kind: ClusterKind;
  label: string;
  entity?: string;
  facts: string[];
  here?: boolean;
}

interface Model {
  clusters: Cluster[];
  facts: GFact[];
  factById: Map<string, GFact>;
  entityById: Map<string, Ent>;
  nodeEntities: string[];
  hubs: Ent[];
  mentionsByFact: Map<string, string[]>;
  factsByEntity: Map<string, string[]>;
  clusterOf: Map<string, string>;
  sizes: Map<string, Size>;
  auto: Map<string, XY>;
  chips: Map<string, { name: string; type: string }[]>;
}

function factHeight(text: string, footer: boolean) {
  const lines = Math.min(3, Math.max(1, Math.ceil(text.length / 40)));
  return 18 + 20 + 6 + lines * 17.5 + (footer ? 24 : 0);
}

/** Decide which box every fact goes in, then lay the boxes out.
 *
 *  "entity": an entity mentioned by two or more active facts becomes a box headed by that entity, holding the facts
 *  filed under it (each fact goes to its most-mentioned entity). Entities in most facts (hubs, like the company
 *  name) are not boxes; they are listed above the canvas. Facts with no shared entity are grouped by Jev's theme.
 *  Replaced facts sit right under the fact that replaced them.
 *  "scope": one box per scope, broader scopes on the left. */
function buildModel(data: GraphData, hidden: Set<Category>, group: Group): Model {
  const facts = data.facts.filter((f) => !hidden.has(f.category));
  const factById = new Map(facts.map((f) => [f.id, f]));
  const mentionsByFact = new Map<string, string[]>();
  const factsByEntity = new Map<string, string[]>();
  for (const m of data.mentions) {
    if (!factById.has(m.fact)) continue;
    mentionsByFact.set(m.fact, [...(mentionsByFact.get(m.fact) ?? []), m.entity]);
    factsByEntity.set(m.entity, [...(factsByEntity.get(m.entity) ?? []), m.fact]);
  }
  const activeCount = facts.filter((f) => f.status === "active").length;
  const hubMin = Math.max(4, Math.ceil(0.35 * activeCount));
  const entityById = new Map<string, Ent>();
  for (const e of data.entities) {
    const degree = (factsByEntity.get(e.id) ?? []).filter((id) => factById.get(id)?.status === "active").length;
    entityById.set(e.id, { ...e, degree, hub: degree >= hubMin });
  }
  const linking = (id: string) => { const e = entityById.get(id); return !!e && !e.hub && e.degree >= 2; };
  const hubs = [...entityById.values()].filter((e) => e.hub).sort((a, b) => b.degree - a.degree || a.name.localeCompare(b.name));

  const primary = (fid: string): string | null => {
    const es = (mentionsByFact.get(fid) ?? []).filter(linking);
    es.sort((a, b) => entityById.get(b)!.degree - entityById.get(a)!.degree || entityById.get(a)!.name.localeCompare(entityById.get(b)!.name));
    return es[0] ?? null;
  };
  const homeKey = (f: GFact, depth = 0): string => {
    if (f.status !== "active" && f.superseded_by && factById.has(f.superseded_by) && depth < 20) {
      return homeKey(factById.get(f.superseded_by)!, depth + 1);
    }
    const p = primary(f.id);
    if (p) return `e:${p}`;
    if (f.topic) return `t:${f.topic}`;
    return "loose";
  };
  const cmp = (a: string, b: string) => {
    const fa = factById.get(a)!, fb = factById.get(b)!;
    return CATEGORIES.indexOf(fa.category) - CATEGORIES.indexOf(fb.category) || fb.observed_at.localeCompare(fa.observed_at);
  };
  /** Active facts in category order, each followed by the facts it replaced. */
  const order = (list: string[]) => {
    const act = list.filter((id) => factById.get(id)!.status === "active").sort(cmp);
    const rest = list.filter((id) => factById.get(id)!.status !== "active").sort(cmp);
    const out: string[] = [];
    const used = new Set<string>();
    const add = (id: string) => {
      out.push(id);
      used.add(id);
      for (const r of rest) if (!used.has(r) && factById.get(r)!.superseded_by === id) add(r);
    };
    act.forEach(add);
    rest.forEach((r) => !used.has(r) && add(r));
    return out;
  };

  const clusters: Cluster[] = [];
  let nodeEntities: string[] = [];
  if (group === "entity") {
    const groups = new Map<string, string[]>();
    for (const f of facts) { const k = homeKey(f); groups.set(k, [...(groups.get(k) ?? []), f.id]); }
    nodeEntities = [...entityById.values()].filter((e) => linking(e.id)).map((e) => e.id);
    const entityClusters = nodeEntities.map((eid) => ({
      id: `cluster:e:${eid}`, kind: "entity" as const, label: entityById.get(eid)!.name, entity: eid, facts: order(groups.get(`e:${eid}`) ?? []),
    }));
    // Order entity boxes so boxes that link to each other sit next to each other.
    const clusterKey = new Map<string, string>();
    for (const c of entityClusters) for (const f of c.facts) clusterKey.set(f, c.id);
    const weight = new Map<string, number>();
    const bump = (a: string, b: string) => { if (a === b) return; const k = a < b ? `${a}|${b}` : `${b}|${a}`; weight.set(k, (weight.get(k) ?? 0) + 1); };
    for (const [fid, cid] of clusterKey) for (const e of (mentionsByFact.get(fid) ?? []).filter(linking)) bump(cid, `cluster:e:${e}`);
    const w = (a: string, b: string) => weight.get(a < b ? `${a}|${b}` : `${b}|${a}`) ?? 0;
    const size = (c: Cluster) => c.facts.length + (c.entity ? entityById.get(c.entity)!.degree / 100 : 0);
    const remaining = [...entityClusters].sort((a, b) => size(b) - size(a));
    const ordered: Cluster[] = [];
    while (remaining.length) {
      let best = 0;
      if (ordered.length) {
        let bestScore = -1;
        remaining.forEach((c, i) => {
          const score = w(ordered[ordered.length - 1].id, c.id) * 100 + ordered.reduce((acc, o) => acc + w(o.id, c.id), 0) * 10 + size(c);
          if (score > bestScore) { bestScore = score; best = i; }
        });
      }
      ordered.push(remaining.splice(best, 1)[0]);
    }
    clusters.push(...ordered);
    const themes = [...groups].filter(([k]) => k.startsWith("t:")).sort((a, b) => b[1].length - a[1].length);
    for (const [k, list] of themes) clusters.push({ id: `cluster:${k}`, kind: "theme", label: topicLabel(k.slice(2)), facts: order(list) });
    if (groups.has("loose")) clusters.push({ id: "cluster:loose", kind: "loose", label: "No shared entity or theme", facts: order(groups.get("loose")!) });
  } else {
    for (const scope of data.scopes) {
      const mine = facts.filter((f) => f.scope === scope);
      if (!mine.length) continue;
      const byKey = new Map<string, string[]>();
      for (const f of mine) { const k = homeKey(f); byKey.set(k, [...(byKey.get(k) ?? []), f.id]); }
      const list = [...byKey.values()].sort((a, b) => b.length - a.length).flatMap(order);
      clusters.push({ id: `cluster:s:${scope}`, kind: "scope", label: scope, facts: list, here: scope === data.scope });
    }
  }

  const clusterOf = new Map<string, string>();
  for (const c of clusters) {
    if (c.entity) clusterOf.set(c.entity, c.id);
    for (const f of c.facts) clusterOf.set(f, c.id);
  }
  const own = new Map(clusters.filter((c) => c.entity).flatMap((c) => c.facts.map((f) => [f, c.entity!] as const)));
  const chips = new Map<string, { name: string; type: string }[]>();
  const sizes = new Map<string, Size>();
  for (const f of facts) {
    const list = (mentionsByFact.get(f.id) ?? []).filter((e) => e !== own.get(f.id)).map((e) => entityById.get(e)).filter(Boolean) as Ent[];
    list.sort((a, b) => b.degree - a.degree || a.name.localeCompare(b.name));
    chips.set(f.id, list.map((e) => ({ name: e.name, type: e.type })));
    sizes.set(f.id, { w: NODE_W, h: factHeight(f.text, list.length > 0 || f.scope !== data.scope) });
  }
  for (const eid of nodeEntities) sizes.set(eid, { w: ENT_W, h: ENT_H });

  // Inner layout of each box, then the boxes themselves: a masonry grid for groups, one row for scopes.
  const auto = new Map<string, XY>();
  const inner = new Map<string, { offs: Map<string, XY>; size: Size; span: number }>();
  for (const c of clusters) {
    const n = c.facts.length;
    const k = c.kind === "scope" ? Math.max(1, Math.min(4, Math.ceil(Math.sqrt(n / 3)))) : n > 6 ? 2 : 1;
    const header = c.kind === "entity" ? ENT_H + 20 : LABEL_H;
    const per = Math.max(1, Math.ceil(n / k));
    const colH = Array(k).fill(0) as number[];
    const offs = new Map<string, XY>();
    c.facts.forEach((fid, i) => {
      const col = Math.min(k - 1, Math.floor(i / per));
      offs.set(fid, { x: col * (NODE_W + COL_GAP), y: header + colH[col] });
      colH[col] += sizes.get(fid)!.h + FACT_GAP;
    });
    const innerW = k * NODE_W + (k - 1) * COL_GAP;
    if (c.entity) offs.set(c.entity, { x: (innerW - ENT_W) / 2, y: 0 });
    const innerH = n ? header + Math.max(...colH) - FACT_GAP : c.entity ? ENT_H : LABEL_H;
    inner.set(c.id, { offs, size: { w: innerW + 2 * PAD, h: innerH + 2 * PAD }, span: k });
  }
  const origin = new Map<string, XY>();
  if (group === "scope") {
    let x = 0;
    for (const c of clusters) { origin.set(c.id, { x, y: 0 }); x += inner.get(c.id)!.size.w + GAP * 1.5; }
  } else {
    const U = NODE_W + 2 * PAD + GAP;
    const units = clusters.reduce((a, c) => a + inner.get(c.id)!.span, 0);
    const maxSpan = Math.max(1, ...clusters.map((c) => inner.get(c.id)!.span));
    let K = Math.max(1, Math.min(6, Math.round(Math.sqrt(units * 1.6))));
    K = Math.min(Math.max(K, maxSpan), Math.max(units, 1));
    const heights = Array(K).fill(0) as number[];
    for (const c of clusters) {
      const { span, size } = inner.get(c.id)!;
      let start = 0;
      let top = Infinity;
      for (let s = 0; s + span <= K; s++) {
        const h = Math.max(...heights.slice(s, s + span));
        if (h < top) { top = h; start = s; }
      }
      origin.set(c.id, { x: start * U, y: top });
      for (let s = start; s < start + span; s++) heights[s] = top + size.h + GAP;
    }
  }
  for (const c of clusters) {
    const o = origin.get(c.id)!;
    for (const [id, off] of inner.get(c.id)!.offs) auto.set(id, { x: o.x + PAD + off.x, y: o.y + PAD + off.y });
  }
  return { clusters, facts, factById, entityById, nodeEntities, hubs, mentionsByFact, factsByEntity, clusterOf, sizes, auto, chips };
}

type Box = { x: number; y: number; w: number; h: number; members: string[] };

/** Boxes follow their members, so dragging a fact out of its group stretches the box instead of leaving it behind. */
function boxesFor(model: Model, pos: Map<string, XY>): Map<string, Box> {
  const out = new Map<string, Box>();
  for (const c of model.clusters) {
    const members = [...(c.entity ? [c.entity] : []), ...c.facts].filter((m) => pos.has(m));
    if (!members.length) continue;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const m of members) {
      const p = pos.get(m)!, s = model.sizes.get(m)!;
      minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x + s.w); maxY = Math.max(maxY, p.y + s.h);
    }
    const top = c.kind === "entity" ? 0 : LABEL_H;
    out.set(c.id, { x: minX - PAD, y: minY - PAD - top, w: maxX - minX + 2 * PAD, h: maxY - minY + 2 * PAD + top, members });
  }
  return out;
}

/** Positions the person dragged nodes to, kept per scope, view and grouping in this browser only. */
function loadMoved(key: string): Record<string, XY> {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as Record<string, XY>) : {};
  } catch {
    return {};
  }
}

function saveMoved(key: string, moved: Record<string, XY>) {
  try {
    if (Object.keys(moved).length) localStorage.setItem(key, JSON.stringify(moved));
    else localStorage.removeItem(key);
  } catch {
    /* storage unavailable */
  }
}

function Key({ group }: { group: Group }) {
  const [open, setOpenState] = useState(() => { try { return localStorage.getItem("memx.graph.key") === "open"; } catch { return false; } });
  const setOpen = (fn: (o: boolean) => boolean) => setOpenState((o) => {
    const n = fn(o);
    try { localStorage.setItem("memx.graph.key", n ? "open" : "closed"); } catch { /* storage unavailable */ }
    return n;
  });
  return (
    <div className="gkey" style={open ? undefined : { width: "auto" }}>
      <button className="gkey-h" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? <ChevronDown /> : <ChevronRight />} How to read this
      </button>
      {open && (
        <>
          {group === "entity" ? (
            <>
              <div className="krow"><span className="kent"><EntityIcon type="person" /></span><span><b>Entity</b> named in two or more facts: a person, team, system, place…</span></div>
              <div className="krow"><span className="kbox ent" /><span><b>Its box</b> holds the facts filed under it. Drag the entity to move the whole box.</span></div>
              <div className="krow"><span className="kbox" /><span><b>Theme box</b>: facts that share no entity, grouped by Jev's theme. Drag its tab to move it.</span></div>
            </>
          ) : (
            <div className="krow"><span className="kbox" /><span><b>Scope box</b>: where facts are stored. Broader scopes on the left; a child sees its parents.</span></div>
          )}
          <div className="krow"><span className="kfact" /><span><b>Fact</b>. The colored bar is its category; struck through means replaced.</span></div>
          <div className="krow"><span className="echip">Name</span><span>Other entities the fact names</span></div>
          {group === "entity" && <div className="krow"><span className="ln" style={{ borderColor: "var(--entity)" }} /><span>Fact also names this entity</span></div>}
          <div className="krow"><span className="ln dash" style={{ borderColor: "var(--indigo)" }} /><span>Related: both true, same subject. Shown for the selected fact only</span></div>
          <div className="krow"><span className="ln" style={{ borderColor: "var(--purple)" }} /><span>Hides a broader fact at this scope</span></div>
          <div className="krow"><span className="ln" style={{ borderColor: "var(--muted)" }} /><span>Replaced by a newer fact (arrow points to the old one)</span></div>
        </>
      )}
    </div>
  );
}

function Inspector({ model, data, selected, select, open, close }: {
  model: Model; data: GraphData; selected: string; select: (id: string) => void; open: (id: string) => void; close: () => void;
}) {
  const fact = model.factById.get(selected);
  const ent = model.entityById.get(selected);
  if (fact) {
    const ents = (model.mentionsByFact.get(fact.id) ?? []).map((e) => model.entityById.get(e)).filter(Boolean) as Ent[];
    const links = data.edges.filter((e) => e.source === fact.id || e.target === fact.id).filter((e) => model.factById.has(e.source) && model.factById.has(e.target));
    const linkText = (e: GraphData["edges"][number]) => {
      const mine = e.source === fact.id;
      if (e.kind === "related") return "Related to";
      if (e.kind === "shadows") return mine ? "Hides here" : "Hidden below by";
      return mine ? "Replaced" : "Replaced by";
    };
    return (
      <div className="gpanel">
        <div className="gp-h"><span className="k">Fact</span><button className="gp-close" onClick={close} aria-label="Close"><X /></button></div>
        <div className="row" style={{ gap: 6 }}><CategoryChip category={fact.category} small /><Pill tone={statusTone(fact.status)}>{fact.status}</Pill></div>
        <div className="gp-title">{fact.text}</div>
        <dl className="kv" style={{ gridTemplateColumns: "70px minmax(0,1fr)", fontSize: 12 }}>
          <dt>Scope</dt><dd className="mono">{fact.scope}</dd>
          <dt>Theme</dt><dd>{fact.topic ? topicLabel(fact.topic) : <span className="faint">none</span>}</dd>
        </dl>
        <div className="stack" style={{ gap: 6 }}>
          <div className="label">Names</div>
          {ents.length ? (
            <div className="row" style={{ gap: 4 }}>
              {ents.map((e) => <button key={e.id} className="echip big" onClick={() => select(e.id)}><EntityIcon type={e.type} />{e.name}<span className="faint">{e.degree}</span></button>)}
            </div>
          ) : <span className="small faint">No named entities</span>}
        </div>
        {links.length > 0 && (
          <div className="stack" style={{ gap: 4 }}>
            <div className="label">Links</div>
            <div className="gp-list">
              {links.map((l, i) => {
                const other = model.factById.get(l.source === fact.id ? l.target : l.source)!;
                return (
                  <button key={i} className="gp-item" onClick={() => select(other.id)}>
                    <span className="k">{linkText(l)}</span><span>{truncate(other.text, 90)}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
        <button className="btn sm" onClick={() => open(fact.id)}><ExternalLink /> Open full details</button>
      </div>
    );
  }
  if (ent) {
    const fids = model.factsByEntity.get(ent.id) ?? [];
    const co = new Map<string, number>();
    for (const f of fids) for (const e of model.mentionsByFact.get(f) ?? []) if (e !== ent.id) co.set(e, (co.get(e) ?? 0) + 1);
    const together = [...co].sort((a, b) => b[1] - a[1]).slice(0, 8);
    const history = fids.length - ent.degree;
    const note = ent.hub
      ? "Named in most facts here, so its links are not drawn and it has no box of its own."
      : ent.degree >= 2 ? "Facts that name it are linked, and recall can hop between them."
      : "Only one fact names it so far, so it links nothing yet.";
    return (
      <div className="gpanel">
        <div className="gp-h">
          <span className="k"><EntityIcon type={ent.type} />{ENTITY_TYPE_LABEL[ent.type] ?? "Entity"}</span>
          <button className="gp-close" onClick={close} aria-label="Close"><X /></button>
        </div>
        <div className="gp-title">{ent.name}</div>
        <div className="small muted">
          Named in {ent.degree} active fact{ent.degree === 1 ? "" : "s"}{history > 0 ? ` and ${history} replaced` : ""}. {note}
          {ent.aliases?.length ? <> Also written as {ent.aliases.map((a) => `“${a}”`).join(", ")}.</> : null}
        </div>
        <div className="gp-list">
          {fids.map((id) => {
            const f = model.factById.get(id)!;
            return (
              <button key={id} className="gp-item" onClick={() => select(id)} style={f.status !== "active" ? { textDecoration: "line-through", opacity: 0.7 } : undefined}>
                <span className="sw" style={{ background: catVar(f.category) }} /><span>{truncate(f.text, 110)}</span>
              </button>
            );
          })}
        </div>
        {together.length > 0 && (
          <div className="stack" style={{ gap: 6 }}>
            <div className="label">Named together with</div>
            <div className="row" style={{ gap: 4 }}>
              {together.map(([eid, n]) => {
                const e = model.entityById.get(eid);
                return e ? <button key={eid} className="echip big" onClick={() => select(eid)}><EntityIcon type={e.type} />{e.name}<span className="faint">{n}</span></button> : null;
              })}
            </div>
          </div>
        )}
      </div>
    );
  }
  return null;
}

function GraphCanvas({ data, focus, group }: { data: GraphData; focus?: string; group: Group }) {
  const { openFact } = useApp();
  const [hidden, setHidden] = useState<Set<Category>>(new Set());
  const [selected, setSelected] = useState<string | null>(focus ?? null);
  const flow = useReactFlow();
  const storageKey = `memx.layout2.${group}.${data.view}.${data.scope}`;
  const [moved, setMoved] = useState<Record<string, XY>>(() => loadMoved(storageKey));

  const model = useMemo(() => buildModel(data, hidden, group), [data, hidden, group]);
  const pos = useMemo(() => {
    const m = new Map(model.auto);
    for (const [id, p] of Object.entries(moved)) if (m.has(id)) m.set(id, p);
    return m;
  }, [model.auto, moved]);
  const boxes = useMemo(() => boxesFor(model, pos), [model, pos]);
  const posRef = useRef(pos);
  const boxRef = useRef(boxes);
  posRef.current = pos;
  boxRef.current = boxes;

  const moveGroup = (next: Record<string, XY>, clusterId: string, dx: number, dy: number) => {
    const box = boxRef.current.get(clusterId);
    if (!box || (!dx && !dy)) return next;
    const out = { ...next };
    for (const m of box.members) { const p = posRef.current.get(m)!; out[m] = { x: p.x + dx, y: p.y + dy }; }
    return out;
  };
  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setMoved((prev) => {
      let next = prev;
      for (const c of changes) {
        if (c.type !== "position" || !c.position) continue;
        if (c.id.startsWith("label:")) {
          const cid = c.id.slice(6);
          const box = boxRef.current.get(cid);
          if (box) next = moveGroup(next, cid, c.position.x - (box.x + 10), c.position.y - (box.y + 8));
        } else if (model.nodeEntities.includes(c.id)) {
          const p = posRef.current.get(c.id);
          const cid = model.clusterOf.get(c.id);
          if (p && cid) next = moveGroup(next, cid, c.position.x - p.x, c.position.y - p.y);
        } else {
          next = { ...next, [c.id]: c.position };
        }
      }
      return next;
    });
  }, [model]);
  const onNodeDragStop = useCallback(() => setMoved((m) => { saveMoved(storageKey, m); return m; }), [storageKey]);
  const resetLayout = () => { setMoved({}); saveMoved(storageKey, {}); setTimeout(() => flow.fitView({ padding: 0.12, maxZoom: 1, duration: 400 }), 50); };
  const rearranged = Object.keys(moved).length > 0;

  const neighbors = useMemo(() => {
    if (!selected) return null;
    const s = new Set([selected]);
    for (const e of data.edges) {
      if (e.source === selected) s.add(e.target);
      if (e.target === selected) s.add(e.source);
    }
    for (const f of model.factsByEntity.get(selected) ?? []) s.add(f);
    for (const e of model.mentionsByFact.get(selected) ?? []) s.add(e);
    return s;
  }, [selected, data.edges, model]);

  const select = (id: string | null) => {
    setSelected(id);
    const p = id ? pos.get(id) : undefined;
    const sz = id ? model.sizes.get(id) : undefined;
    if (p && sz) flow.setCenter(p.x + sz.w / 2, p.y + sz.h / 2, { zoom: Math.max(flow.getZoom(), 0.8), duration: 400 });
  };

  const nodes: Node[] = [];
  for (const c of model.clusters) {
    const b = boxes.get(c.id);
    if (!b) continue;
    const dim = !!neighbors && !b.members.some((m) => neighbors.has(m));
    nodes.push({
      id: c.id, type: "cluster", position: { x: b.x, y: b.y }, draggable: false, selectable: false, focusable: false,
      style: { width: b.w, height: b.h, pointerEvents: "none" }, data: { kind: c.kind, here: c.here, dim } satisfies ClusterNodeData,
    });
    if (c.kind !== "entity") {
      nodes.push({
        id: `label:${c.id}`, type: "clusterLabel", position: { x: b.x + 10, y: b.y + 8 }, selectable: false,
        data: { kind: c.kind, label: c.label, count: c.facts.length, here: c.here, dim } satisfies ClusterLabelData,
      });
    }
  }
  for (const eid of model.nodeEntities) {
    const e = model.entityById.get(eid)!;
    nodes.push({
      id: eid, type: "entity", position: pos.get(eid)!,
      data: {
        name: e.name, type: e.type, typeLabel: ENTITY_TYPE_LABEL[e.type] ?? "Entity", count: e.degree,
        selected: eid === selected, dim: !!neighbors && !neighbors.has(eid),
      } satisfies EntityNodeData,
    });
  }
  for (const f of model.facts) {
    const chips = model.chips.get(f.id) ?? [];
    nodes.push({
      id: f.id, type: "fact", position: pos.get(f.id)!,
      data: {
        text: f.text, category: f.category, status: f.status, selected: f.id === selected,
        dim: !!neighbors && !neighbors.has(f.id), width: NODE_W,
        chips: chips.slice(0, CHIP_MAX), more: Math.max(0, chips.length - CHIP_MAX),
        scopeTag: f.scope !== data.scope ? f.scope : undefined,
        badges: f.shadowed_here ? <span title="Hidden at this scope by a narrower correction" style={{ color: "var(--purple)" }}>shadowed here</span> : f.reinforced_count > 0 ? <span title="Said again this many times">×{f.reinforced_count + 1}</span> : undefined,
      } satisfies FactNodeData,
    });
  }

  const boxOf = (id: string) => ({ ...pos.get(id)!, ...model.sizes.get(id)! });
  const edges: Edge[] = [];
  // "Related" links are Jev's "both true, same subject" verdicts. Between company facts nearly everything is
  // related to everything, so they are drawn only for the selected fact. Shadows and replacements stay visible.
  const visibleEdges = data.edges.filter((e) => model.factById.has(e.source) && model.factById.has(e.target))
    .filter((e) => e.kind !== "related" || e.source === selected || e.target === selected);
  visibleEdges.forEach((e, i) => {
    const active = neighbors ? (e.source === selected || e.target === selected) : true;
    const color = e.kind === "shadows" ? "var(--purple)" : e.kind === "supersedes" ? "var(--muted)" : "var(--indigo)";
    edges.push({
      id: `e${i}`, source: e.source, target: e.target, type: "default", ...facing(boxOf(e.source), boxOf(e.target)),
      label: active && selected ? (e.kind === "related" ? "related" : e.kind === "supersedes" ? "replaced" : "hides") : undefined,
      style: { stroke: color, strokeWidth: active ? 1.8 : 1, strokeDasharray: e.kind === "related" ? "5 4" : undefined, opacity: active ? 0.9 : 0.08 },
      markerEnd: e.kind === "related" ? undefined : { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
    });
  });
  if (group === "entity") {
    const nodeSet = new Set(model.nodeEntities);
    for (const f of model.facts) {
      for (const e of model.mentionsByFact.get(f.id) ?? []) {
        if (!nodeSet.has(e) || model.clusterOf.get(e) === model.clusterOf.get(f.id)) continue;
        const active = neighbors ? (f.id === selected || e === selected) : true;
        edges.push({
          id: `m:${e}:${f.id}`, source: e, target: f.id, type: "default", ...facing(boxOf(e), boxOf(f.id)),
          style: { stroke: "var(--entity)", strokeWidth: active && selected ? 1.8 : 1.2, opacity: active ? (selected ? 0.95 : 0.55) : 0.06 },
        });
      }
    }
  }

  useEffect(() => {
    if (focus && pos.has(focus)) {
      const p = pos.get(focus)!;
      setTimeout(() => flow.setCenter(p.x + NODE_W / 2, p.y + 40, { zoom: 1, duration: 500 }), 80);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus]);

  const present = CATEGORIES.filter((c) => data.facts.some((f) => f.category === c));
  const themeCount = model.clusters.filter((c) => c.kind === "theme").length;
  return (
    <>
      <div className="row" style={{ padding: "10px 14px", borderBottom: "1px solid var(--line)" }}>
        <span className="small muted">Show</span>
        {present.map((c) => (
          <button key={c} className="chip" style={{ cursor: "pointer", opacity: hidden.has(c) ? 0.4 : 1 }} aria-pressed={!hidden.has(c)}
            onClick={() => setHidden((h) => { const n = new Set(h); n.has(c) ? n.delete(c) : n.add(c); return n; })}>
            <span className="sw" style={{ background: catVar(c) }} />{CATEGORY_LABEL[c]}
          </button>
        ))}
        <span className="small muted" style={{ marginLeft: "auto" }}>
          {model.facts.length} facts
          {group === "entity" ? ` · ${model.nodeEntities.length} linking entit${model.nodeEntities.length === 1 ? "y" : "ies"} · ${themeCount} theme${themeCount === 1 ? "" : "s"}` : ` · ${model.clusters.length} scope${model.clusters.length === 1 ? "" : "s"}`}
        </span>
        {rearranged && (
          <button className="btn sm ghost" onClick={resetLayout} title="Put every node back where the automatic layout places it"><RotateCcw /> Reset layout</button>
        )}
      </div>
      {model.hubs.length > 0 && (
        <div className="row" style={{ padding: "8px 14px", borderBottom: "1px solid var(--line)", gap: 6 }}>
          <span className="small muted" title="Drawing their links would connect almost everything to everything">Named in most facts, so not drawn as boxes:</span>
          {model.hubs.map((h) => (
            <button key={h.id} className={`echip big ${selected === h.id ? "on" : ""}`} onClick={() => setSelected((s) => (s === h.id ? null : h.id))}>
              <EntityIcon type={h.type} />{h.name}<span className="faint">{h.degree}</span>
            </button>
          ))}
        </div>
      )}
      <div className="flow tall">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onNodeDragStop={onNodeDragStop}
          nodesConnectable={false}
          fitView
          fitViewOptions={{ padding: 0.1, maxZoom: 1, minZoom: 0.35 }}
          minZoom={0.1}
          onNodeClick={(_, n) => { if (!n.id.startsWith("cluster:") && !n.id.startsWith("label:")) setSelected((s) => (s === n.id ? null : n.id)); }}
          onNodeDoubleClick={(_, n) => { if (model.factById.has(n.id)) openFact(n.id); }}
          onPaneClick={() => setSelected(null)}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={22} size={1} color="var(--line)" />
          <Controls showInteractive={false} position="bottom-right" />
          {selected && (model.factById.has(selected) || model.entityById.has(selected)) && (
            <Panel position="top-right">
              <Inspector model={model} data={data} selected={selected} select={select} open={openFact} close={() => setSelected(null)} />
            </Panel>
          )}
          <Panel position="bottom-left"><Key group={group} /></Panel>
        </ReactFlow>
      </div>
    </>
  );
}

function loadGroup(): Group {
  try { return localStorage.getItem("memx.graph.group") === "scope" ? "scope" : "entity"; } catch { return "entity"; }
}

export function GraphPage({ focus }: { focus?: string }) {
  const { scope, setScope } = useApp();
  const [chosen, setChosen] = useState<"chain" | "subtree" | null>(null);
  const [history, setHistory] = useState(true);
  const [group, setGroupState] = useState<Group>(loadGroup);
  const setGroup = (g: Group) => { setGroupState(g); try { localStorage.setItem("memx.graph.group", g); } catch { /* storage unavailable */ } };
  const chain = useQuery({ queryKey: ["graph", scope, "chain", history], queryFn: () => api.graph(scope, "chain", history, 400), enabled: chosen !== "subtree" });
  // An empty chain usually means a parent scope (like `memory`) whose facts all live below it.
  const autoSubtree = chosen === null && chain.data !== undefined && chain.data.total === 0;
  const view = chosen ?? (autoSubtree ? "subtree" : "chain");
  const sub = useQuery({ queryKey: ["graph", scope, "subtree", history], queryFn: () => api.graph(scope, "subtree", history, 400), enabled: view === "subtree" });
  const graph = view === "subtree" ? sub : chain;
  const scopes = useQuery({ queryKey: ["scopes"], queryFn: api.scopes, staleTime: 15_000 });
  const children = (scopes.data ?? [])
    .filter((s) => s.path.startsWith(`${scope}.`) && s.active > 0)
    .sort((a, b) => b.active - a.active)
    .slice(0, 12);
  const wide = view === "subtree" && (graph.data?.scopes.length ?? 0) > 8;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Relationships</h1>
          <p>What an agent at this scope knows, grouped by the people, teams and systems it is about. Click anything to see its connections; drag to rearrange.</p>
        </div>
        <div className="actions">
          <Seg label="Group by" value={group} onChange={setGroup} options={[
            { value: "entity", label: "Entities and themes" }, { value: "scope", label: "Scopes" },
          ]} />
          <Seg label="Which scopes" value={view} onChange={(v) => setChosen(v)} options={[
            { value: "chain", label: "This scope + parents" }, { value: "subtree", label: "This scope + children" },
          ]} />
          <Seg label="History" value={history ? "y" : "n"} onChange={(v) => setHistory(v === "y")} options={[
            { value: "y", label: "With history" }, { value: "n", label: "Active only" },
          ]} />
        </div>
      </div>
      {(autoSubtree || wide) && children.length > 0 && (
        <div className="card card-body stack" style={{ gap: 10 }}>
          <div className="small">
            {autoSubtree
              ? <><b>No facts are stored at <code>{scope}</code> itself</b>, so this shows the scopes below it. For the clearest picture, open one of them:</>
              : <>This view spans many scopes. For the clearest picture, open one of them:</>}
          </div>
          <div className="row">
            {children.map((c) => (
              <button key={c.path} className="chip" style={{ cursor: "pointer" }} onClick={() => { setScope(c.path); setChosen(null); }}>
                <span className="mono">{c.path.slice(scope.length + 1)}</span><span className="faint">{c.active}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      <Card flush>
        {graph.isLoading ? <Loading label="Laying out the graph" /> : graph.error ? <div style={{ padding: 14 }}><ErrorBox error={graph.error} /></div> :
          !graph.data?.facts.length ? <Empty title="No facts here yet">Add memory to this scope, or pick another scope at the top.</Empty> : (
            <>
              {graph.data.truncated && (
                <div className="banner tone-warn" style={{ borderRadius: 0 }}>
                  <Info /> Showing the {graph.data.facts.length} most recent of {graph.data.total.toLocaleString()} facts. Pick a narrower scope to see everything.
                </div>
              )}
              <ReactFlowProvider key={`${scope}:${view}:${group}`}><GraphCanvas data={graph.data} focus={focus} group={group} /></ReactFlowProvider>
            </>
          )}
      </Card>
    </div>
  );
}
