import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Background, Controls, MarkerType, Panel, ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type Node } from "@xyflow/react";
import {
  forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY, type Simulation, type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import { ExternalLink, Info, RotateCcw, X } from "lucide-react";
import { api, type Category, type GraphData } from "../lib/api";
import { useApp } from "../lib/app";
import { CATEGORIES, CATEGORY_LABEL, ENTITY_TYPE_LABEL, catVar, topicLabel, truncate } from "../lib/meta";
import { EntityIcon } from "../components/FlowNodes";
import { graphEdgeTypes, graphNodeTypes, type AnchorData, type AnchorKind, type FactDotData } from "../components/GraphNodes";
import { statusTone } from "../components/FactDrawer";
import { Card, CategoryChip, Empty, ErrorBox, Loading, Pill, Seg } from "../components/ui";

// A live network. Facts are dots; names, themes or scopes are circles ("anchors") the facts are tied to by springs.
// Everything can be dragged: pull a node and what is tied to it follows, then settles.
const FACT_D = 14;
const anchorD = (count: number) => 30 + Math.min(52, Math.round(Math.sqrt(count) * 9));

type XY = { x: number; y: number };
type GFact = GraphData["facts"][number];
type Ent = GraphData["entities"][number] & { degree: number; hub: boolean };
/** What the facts are grouped round. "none" draws only the facts and the links between them. */
type Group = "none" | "entity" | "theme" | "scope";

/** A name, theme or scope drawn as a circle. */
interface Anchor {
  id: string;
  kind: AnchorKind;
  label: string;
  type?: string;
  count: number;
  d: number;
  /** A name most facts mention: its links pull less and are drawn fainter. */
  hub?: boolean;
  here?: boolean;
}

/** The facts filed under one name, theme or scope; selecting a group lights them. */
interface ListGroup { id: string; kind: AnchorKind | "loose" | "category"; label: string; type?: string; category?: Category; facts: string[] }

interface Tie { fact: string; anchor: string; hub: boolean }

interface Model {
  anchors: Anchor[];
  /** Every fact tied to every anchor it belongs to: all the names it mentions, or its theme, or its scope. */
  ties: Tie[];
  /** Two names that some fact mentions together, and how many facts do. */
  pairs: { a: string; b: string; n: number }[];
  groups: ListGroup[];
  facts: GFact[];
  factById: Map<string, GFact>;
  entityById: Map<string, Ent>;
  mentionsByFact: Map<string, string[]>;
  factsByEntity: Map<string, string[]>;
}

/** Work out what is drawn and what is tied to what. One rule per view:
 *
 *  "entity" (by name): every name is a circle, and a fact is tied to each name it mentions. Facts that mention
 *  nothing float free. "theme": a circle per theme Jev gave the facts. "scope": a circle per scope.
 *  A replaced fact is tied to the fact that replaced it instead.
 *  Each fact is also filed once: under the most specific name it shares, its theme or its scope. */
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
  const named = (fid: string) => (mentionsByFact.get(fid) ?? []).filter((e) => entityById.has(e));
  const replacedBy = (f: GFact) => (f.status !== "active" && f.superseded_by && factById.has(f.superseded_by) ? f.superseded_by : null);

  const anchors: Anchor[] = [];
  const ties: Tie[] = [];
  if (group === "entity") {
    for (const e of entityById.values()) {
      const n = (factsByEntity.get(e.id) ?? []).length;
      if (n) anchors.push({ id: e.id, kind: "entity", label: e.name, type: e.type, count: Math.max(e.degree, 1), d: anchorD(Math.max(e.degree, 1)), hub: e.hub });
    }
    for (const f of facts) {
      if (replacedBy(f)) continue;
      for (const e of named(f.id)) ties.push({ fact: f.id, anchor: e, hub: entityById.get(e)!.hub });
    }
  } else if (group !== "none") {
    const key = (f: GFact) => (group === "theme" ? (f.topic ? `hub:t:${f.topic}` : null) : `hub:s:${f.scope}`);
    const counts = new Map<string, number>();
    for (const f of facts) {
      const k = key(f);
      if (!k) continue;
      counts.set(k, (counts.get(k) ?? 0) + 1);
      if (!replacedBy(f)) ties.push({ fact: f.id, anchor: k, hub: false });
    }
    for (const [k, n] of counts) {
      const name = k.slice(6);
      anchors.push(group === "theme"
        ? { id: k, kind: "theme", label: topicLabel(name), count: n, d: anchorD(n) }
        // The last part of the path names the scope; the whole path would not fit under the circle.
        : { id: k, kind: "scope", label: name.split(".").pop()!, count: n, d: anchorD(n), here: name === data.scope });
    }
  }
  anchors.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));

  const pairCount = new Map<string, number>();
  if (group === "entity") {
    for (const f of facts) {
      const es = named(f.id).filter((e) => !entityById.get(e)!.hub).sort();
      for (let i = 0; i < es.length; i++) for (let j = i + 1; j < es.length; j++) {
        pairCount.set(`${es[i]}|${es[j]}`, (pairCount.get(`${es[i]}|${es[j]}`) ?? 0) + 1);
      }
    }
  }
  // One shared fact already joins two names through its dot; a band is for names that keep turning up together.
  const pairs = [...pairCount].filter(([, n]) => n >= 2).map(([k, n]) => ({ a: k.split("|")[0], b: k.split("|")[1], n }));

  // The list: one heading per circle that has facts filed under it, biggest first, then the facts with no circle.
  const byDegree = (a: string, b: string) =>
    entityById.get(b)!.degree - entityById.get(a)!.degree || entityById.get(a)!.name.localeCompare(entityById.get(b)!.name);
  const specific = (id: string) => { const e = entityById.get(id)!; return !e.hub && e.degree >= 2; };
  const filedUnder = (f: GFact, depth = 0): string => {
    const by = replacedBy(f);
    if (by && depth < 20) return filedUnder(factById.get(by)!, depth + 1);
    if (group === "none") return `cat:${f.category}`;
    if (group === "scope") return `hub:s:${f.scope}`;
    if (group === "theme") return f.topic ? `hub:t:${f.topic}` : "loose";
    const es = named(f.id);
    return es.filter(specific).sort(byDegree)[0] ?? es.filter((e) => entityById.get(e)!.hub).sort(byDegree)[0] ?? es.sort(byDegree)[0] ?? "loose";
  };
  const filed = new Map<string, string[]>();
  for (const f of facts) { const k = filedUnder(f); filed.set(k, [...(filed.get(k) ?? []), f.id]); }
  const order = (list: string[]) => {
    const cmp = (a: string, b: string) =>
      CATEGORIES.indexOf(factById.get(a)!.category) - CATEGORIES.indexOf(factById.get(b)!.category)
      || factById.get(b)!.observed_at.localeCompare(factById.get(a)!.observed_at);
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
  const groups: ListGroup[] = anchors.filter((a) => filed.has(a.id))
    .map((a) => ({ id: a.id, kind: a.kind, label: a.label, type: a.type, facts: order(filed.get(a.id)!) }))
    .sort((a, b) => b.facts.length - a.facts.length);
  // With no grouping the list still needs headings to scan by: the categories, in their usual order.
  for (const c of CATEGORIES) {
    if (filed.has(`cat:${c}`)) groups.push({ id: `cat:${c}`, kind: "category", label: CATEGORY_LABEL[c], category: c, facts: order(filed.get(`cat:${c}`)!) });
  }
  if (filed.has("loose")) {
    groups.push({ id: "loose", kind: "loose", label: group === "entity" ? "Mention no name" : "No theme", facts: order(filed.get("loose")!) });
  }
  return { anchors, ties, pairs, groups, facts, factById, entityById, mentionsByFact, factsByEntity };
}

type SimNode = SimulationNodeDatum & { id: string; r: number; anchor: boolean; tied: boolean };
type SimLink = SimulationLinkDatum<SimNode> & { dist: number; str: number };

/** The physics: springs along the ties, everything pushing apart, a gentle pull to the middle. It is settled before
 *  the first paint (from fixed starting positions, so the same memory always opens the same way) and then only runs
 *  while something is being dragged. A node that was dragged stays where it was dropped. */
function buildSim(model: Model, data: GraphData, pinned: Record<string, XY>) {
  // With nothing to group round, the links between facts are all that shapes the picture, so they pull properly.
  const bare = model.anchors.length === 0;
  const tiesOf = new Map<string, number>();
  for (const t of model.ties) tiesOf.set(t.fact, (tiesOf.get(t.fact) ?? 0) + 1);
  const size = new Map(model.anchors.map((a) => [a.id, a.d]));
  const nodes: SimNode[] = [
    ...model.anchors.map((a) => ({ id: a.id, r: a.d / 2, anchor: true, tied: true })),
    ...model.facts.map((f) => ({ id: f.id, r: FACT_D / 2, anchor: false, tied: tiesOf.has(f.id) || f.status !== "active" })),
  ];
  for (const n of nodes) { const p = pinned[n.id]; if (p) { n.fx = p.x; n.fy = p.y; } }
  const links: SimLink[] = [];
  for (const t of model.ties) {
    // A fact that mentions several names sits between them; a name most facts mention pulls only weakly.
    links.push({ source: t.fact, target: t.anchor, dist: size.get(t.anchor)! / 2 + 24, str: (t.hub ? 0.22 : 0.75) / Math.sqrt(tiesOf.get(t.fact)!) });
  }
  for (const f of model.facts) {
    if (f.status !== "active" && f.superseded_by && model.factById.has(f.superseded_by)) {
      links.push({ source: f.id, target: f.superseded_by, dist: 18, str: 1 });
    }
  }
  for (const e of data.edges) {
    if (e.kind === "related" && model.factById.has(e.source) && model.factById.has(e.target)) {
      links.push({ source: e.source, target: e.target, dist: bare ? 78 : 64, str: bare ? 0.12 : 0.012 });
    }
  }
  // Each circle has its name written under it. A fact that drifts onto a name is nudged out sideways or downward,
  // whichever is nearer, so names stay readable.
  const labels = model.anchors.map((a) => ({ node: nodes.find((n) => n.id === a.id)!, half: Math.min(a.label.length, 30) * 3.6 + 10 }));
  const halfLabel = new Map(labels.map((l) => [l.node.id, l.half]));
  const dots = nodes.filter((n) => !n.anchor);
  const keepOffLabels = (alpha: number) => {
    for (const { node: a, half } of labels) {
      const top = a.y! + a.r, bottom = top + 24;
      for (const d of dots) {
        if (d.y! < top - 6 || d.y! > bottom + 6 || Math.abs(d.x! - a.x!) > half + 8) continue;
        const side = half + 8 - Math.abs(d.x! - a.x!), down = bottom + 6 - d.y!;
        if (down < side) d.vy! += down * alpha * 0.9;
        else d.vx! += (d.x! < a.x! ? -1 : 1) * side * alpha * 0.9;
      }
    }
  };
  const sim = forceSimulation(nodes)
    .force("link", forceLink<SimNode, SimLink>(links).id((n) => n.id).distance((l) => l.dist).strength((l) => l.str))
    .force("charge", forceManyBody<SimNode>().strength((n) => (n.anchor ? -260 - n.r * 6 : bare ? -150 : -34)).distanceMax(520))
    // Room for the name written under a circle.
    .force("collide", forceCollide<SimNode>((n) => (n.anchor ? Math.max(n.r + 24, halfLabel.get(n.id)! * 0.82) : n.r + (bare ? 6 : 3))).iterations(3))
    // The canvas is wider than tall, so the pull to the middle is weaker sideways. Facts tied to nothing are pulled
    // in a little harder: the circles push them back out, so they settle just round the edge of the network instead
    // of drifting far off and shrinking the whole picture.
    .force("x", forceX<SimNode>(0).strength((n) => (n.tied ? 0.05 : 0.075)))
    .force("y", forceY<SimNode>(0).strength((n) => (n.tied ? 0.09 : 0.13)))
    .force("labels", keepOffLabels)
    .stop();
  sim.tick(280);
  return { sim, byId: new Map(nodes.map((n) => [n.id, n])) };
}

/** Where the person dropped nodes, kept per scope, view and grouping in this browser only. */
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

const ANCHOR_ROW: Record<Exclude<Group, "none">, string> = {
  entity: "Name. Bigger means more facts mention it",
  theme: "Theme. Bigger means more facts",
  scope: "Scope. Bigger means more facts stored there",
};
const TIE_ROW: Record<Exclude<Group, "none">, string> = { entity: "Fact mentions the name", theme: "Fact has the theme", scope: "Fact is stored at the scope" };

function Key({ group }: { group: Group }) {
  return (
    <div className="gkey-strip">
      <span><i className="kdot" />Fact, coloured by category</span>
      <span><i className="kdot past" />Replaced fact</span>
      {group !== "none" && <span><i className={`kbub ${group === "entity" ? "ent" : ""}`} />{ANCHOR_ROW[group]}</span>}
      {group !== "none" && <span><i className="ln" style={{ borderColor: "var(--cat-preference)", borderTopWidth: 1.5 }} />{TIE_ROW[group]}</span>}
      {group === "entity" && <span><i className="ln" style={{ borderColor: "var(--entity)", borderTopWidth: 4, opacity: 0.35 }} />Names often mentioned together</span>}
      <span><i className="ln dash" style={{ borderColor: "var(--accent)" }} />Related facts</span>
      {group === "none" && <span><i className="ln" style={{ borderColor: "var(--muted)" }} />Replaced by a newer fact</span>}
      {group === "none" && <span><i className="ln" style={{ borderColor: "var(--accent-2)" }} />Hides a broader fact here</span>}
      <span className="faint">Drag anything; it stays where you drop it.</span>
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
      ? "Named in most facts here, so its links are drawn fainter and pull less; otherwise it would drag everything into one clump."
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
  // A theme or a scope: the facts gathered round it.
  const gathered = model.groups.find((g) => g.id === selected && (g.kind === "theme" || g.kind === "scope"));
  if (gathered) {
    const n = gathered.facts.length;
    return (
      <div className="gpanel">
        <div className="gp-h">
          <span className="k">{gathered.kind === "theme" ? "Theme" : "Scope"}</span>
          <button className="gp-close" onClick={close} aria-label="Close"><X /></button>
        </div>
        <div className="gp-title">{gathered.label}</div>
        <div className="small muted">{n} fact{n === 1 ? "" : "s"} {gathered.kind === "theme" ? "on this theme." : "stored here."}</div>
        <div className="gp-list">
          {gathered.facts.map((id) => {
            const f = model.factById.get(id)!;
            return (
              <button key={id} className="gp-item" onClick={() => select(id)} style={f.status !== "active" ? { textDecoration: "line-through", opacity: 0.7 } : undefined}>
                <span className="sw" style={{ background: catVar(f.category) }} /><span>{truncate(f.text, 110)}</span>
              </button>
            );
          })}
        </div>
      </div>
    );
  }
  return null;
}

function GraphCanvas({ data, focus, group }: { data: GraphData; focus?: string; group: Group }) {
  const { openFact } = useApp();
  const [hidden, setHidden] = useState<Set<Category>>(new Set());
  const [selected, setSelected] = useState<string | null>(focus ?? null);
  const [hot, setHot] = useState<string | null>(null);
  const [, setFrame] = useState(0);
  const flow = useReactFlow();
  const storageKey = `memx.graph.net.${group}.${data.view}.${data.scope}`;
  const [relMode, setRelModeState] = useState<RelMode>(loadRelMode);
  const setRelMode = (m: RelMode) => { setRelModeState(m); try { localStorage.setItem("memx.graph.related", m); } catch { /* storage unavailable */ } };
  const [pinned, setPinned] = useState<Record<string, XY>>(() => loadMoved(storageKey));
  const dragging = useRef(false);
  const dragged = useRef(false); // a drag ends with a click on the same node; that click must not select it

  const model = useMemo(() => buildModel(data, hidden, group), [data, hidden, group]);
  // Built once per model; what was dropped where is read at that moment and applied by hand afterwards.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const { sim, byId } = useMemo(() => buildSim(model, data, loadMoved(storageKey)), [model, data, storageKey]);
  const simRef = useRef<Simulation<SimNode, undefined>>(sim);
  simRef.current = sim;
  useEffect(() => {
    let raf = 0;
    sim.on("tick", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => setFrame((f) => f + 1)); });
    return () => { sim.on("tick", null); sim.stop(); cancelAnimationFrame(raf); };
  }, [sim]);

  const at = (id: string): XY | null => { const n = byId.get(id); return n ? { x: n.x ?? 0, y: n.y ?? 0 } : null; };

  // Dragging holds the node under the pointer and wakes the physics, so what is tied to it follows. On release the
  // node stays where it was dropped and the rest settles round it.
  const onNodeDragStart = useCallback((_: unknown, n: Node) => {
    const node = byId.get(n.id);
    if (!node) return;
    dragging.current = true;
    node.fx = node.x; node.fy = node.y;
    sim.alphaTarget(0.25).restart();
  }, [byId, sim]);
  const onNodeDrag = useCallback((_: unknown, n: Node) => {
    const node = byId.get(n.id);
    if (!node) return;
    node.fx = n.position.x + node.r; node.fy = n.position.y + node.r;
    dragged.current = true;
  }, [byId]);
  const onNodeDragStop = useCallback((_: unknown, n: Node) => {
    dragging.current = false;
    setTimeout(() => { dragged.current = false; }, 0);
    sim.alphaTarget(0);
    const node = byId.get(n.id);
    if (!node || node.fx == null || node.fy == null) return;
    setPinned((prev) => { const next = { ...prev, [n.id]: { x: node.fx!, y: node.fy! } }; saveMoved(storageKey, next); return next; });
  }, [byId, sim, storageKey]);
  const resetLayout = () => {
    for (const n of byId.values()) { n.fx = null; n.fy = null; }
    setPinned({});
    saveMoved(storageKey, {});
    sim.alpha(0.9).restart();
    setTimeout(() => flow.fitView({ padding: 0.1, maxZoom: 1.3, duration: 500 }), 900);
  };
  const rearranged = Object.keys(pinned).length > 0;

  // What is lit: the selected node and what it touches, or, with nothing selected, whatever is pointed at.
  const lead = selected ?? (dragging.current ? null : hot);
  const neighbors = useMemo(() => {
    if (!lead) return null;
    const s = new Set([lead]);
    for (const e of data.edges) {
      if (e.source === lead) s.add(e.target);
      if (e.target === lead) s.add(e.source);
    }
    for (const t of model.ties) {
      if (t.fact === lead) s.add(t.anchor);
      if (t.anchor === lead) s.add(t.fact);
    }
    for (const g of model.groups) if (g.id === lead) g.facts.forEach((f) => s.add(f));
    return s;
  }, [lead, data.edges, model]);
  const lit = (id: string) => !neighbors || neighbors.has(id);
  const strong = !!selected;

  const select = (id: string | null) => {
    setSelected(id);
    const p = id ? at(id) : null;
    if (p) flow.setCenter(p.x, p.y, { zoom: Math.max(flow.getZoom(), 1), duration: 400 });
  };

  // Every node carries its known size. Without it React Flow hides and re-measures all of them each time the list
  // is rebuilt, which is on every step of the physics.
  const nodes: Node[] = [];
  for (const a of model.anchors) {
    const p = at(a.id)!;
    nodes.push({
      id: a.id, type: "anchor", position: { x: p.x - a.d / 2, y: p.y - a.d / 2 }, width: a.d, height: a.d, zIndex: 2,
      data: {
        kind: a.kind, label: a.label, type: a.type, count: a.count, size: a.d, here: a.here, selected: a.id === selected, dim: !lit(a.id),
      } satisfies AnchorData,
    });
  }
  for (const f of model.facts) {
    const p = at(f.id)!;
    nodes.push({
      id: f.id, type: "fact", position: { x: p.x - FACT_D / 2, y: p.y - FACT_D / 2 }, width: FACT_D, height: FACT_D, zIndex: 3,
      data: {
        text: f.text, category: f.category, status: f.status, selected: f.id === selected, hot: f.id === hot, dim: !lit(f.id),
        shadowed: f.shadowed_here,
      } satisfies FactDotData,
    });
  }

  const ends = { sourceHandle: "s", targetHandle: "t", type: "straight" };
  const arc = { ...ends, type: "arc" };
  const edges: Edge[] = [];
  // Two names some fact mentions together: a soft band, thicker the more facts do.
  for (const p of model.pairs) {
    const on = !neighbors || p.a === lead || p.b === lead;
    edges.push({
      id: `p:${p.a}:${p.b}`, source: p.a, target: p.b, ...ends, zIndex: 0,
      style: { stroke: "var(--entity)", strokeWidth: Math.min(9, 2 + p.n * 2), strokeLinecap: "round", opacity: on ? 0.16 : 0.03 },
    });
  }
  // A fact to each name it mentions (or its theme, or its scope), in the colour of its category.
  for (const t of model.ties) {
    const on = !neighbors || t.fact === lead || t.anchor === lead;
    const f = model.factById.get(t.fact)!;
    edges.push({
      id: `t:${t.fact}:${t.anchor}`, source: t.anchor, target: t.fact, ...ends, zIndex: 1,
      style: {
        stroke: catVar(f.category), strokeWidth: on && neighbors ? 1.6 : 1.1,
        opacity: !on ? 0.05 : neighbors ? 0.9 : t.hub ? 0.22 : 0.5,
      },
    });
  }
  // "Related" links are Jev's "both true, same subject" verdicts. By default they show for the selected fact only;
  // the toolbar switch can draw them all, or none. Shadows and replacements always stay visible.
  const allEdges = data.edges.filter((e) => model.factById.has(e.source) && model.factById.has(e.target));
  const relatedCount = allEdges.filter((e) => e.kind === "related").length;
  // With no grouping the links are the whole picture, so they are always drawn.
  const bare = group === "none";
  const visibleEdges = allEdges.filter((e) => e.kind !== "related" || bare || relMode === "always"
    || (relMode === "selected" && (e.source === selected || e.target === selected)));
  visibleEdges.forEach((e, i) => {
    const active = neighbors ? (e.source === lead || e.target === lead) : true;
    const color = e.kind === "shadows" ? "var(--accent-2)" : e.kind === "supersedes" ? "var(--muted)" : "var(--accent)";
    const related = e.kind === "related";
    edges.push({
      id: `e${i}`, source: e.source, target: e.target, ...arc, zIndex: 1,
      label: active && strong && !related ? (e.kind === "supersedes" ? "replaced" : "hides") : undefined,
      style: {
        stroke: color, strokeWidth: active && strong ? 1.8 : 1, strokeDasharray: related ? "3 4" : undefined,
        opacity: !active ? 0.04 : related && !neighbors ? (bare ? 0.5 : 0.18) : 0.9,
      },
      markerEnd: related ? undefined : { type: MarkerType.ArrowClosed, color, width: 12, height: 12 },
    });
  });

  useEffect(() => {
    const p = focus ? at(focus) : null;
    if (p) setTimeout(() => flow.setCenter(p.x, p.y, { zoom: 1.3, duration: 500 }), 80);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus]);

  const present = CATEGORIES.filter((c) => data.facts.some((f) => f.category === c));
  const plural = (n: number, one: string) => `${n} ${one}${n === 1 ? "" : "s"}`;
  const counted = bare ? plural(allEdges.length, "link") : plural(model.anchors.length, group === "entity" ? "name" : group);
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
        <span style={{ marginLeft: "auto" }} />
        {relatedCount > 0 && !bare && (
          <Seg label="Related links" value={relMode} onChange={setRelMode} options={[
            { value: "always", label: `Related links (${relatedCount})` }, { value: "selected", label: "On click" }, { value: "off", label: "Off" },
          ]} />
        )}
        <span className="small muted">{plural(model.facts.length, "fact")} · {counted}</span>
        {rearranged && (
          <button className="btn sm ghost" onClick={resetLayout} title="Let everything go back to where the physics puts it"><RotateCcw /> Reset layout</button>
        )}
      </div>
      <div className="flow graph">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={graphNodeTypes}
            edgeTypes={graphEdgeTypes}
            onNodeDragStart={onNodeDragStart}
            onNodeDrag={onNodeDrag}
            onNodeDragStop={onNodeDragStop}
            nodesConnectable={false}
            fitView
            fitViewOptions={{ padding: 0.14, maxZoom: 1.3, minZoom: 0.3 }}
            minZoom={0.1}
            onNodeClick={(_, n) => { if (!dragged.current) setSelected((s) => (s === n.id ? null : n.id)); }}
            onNodeDoubleClick={(_, n) => { if (model.factById.has(n.id)) openFact(n.id); }}
            onNodeMouseEnter={(_, n) => setHot(n.id)}
            onNodeMouseLeave={() => setHot(null)}
            onPaneClick={() => setSelected(null)}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={22} size={1} color="var(--line)" />
            <Controls showInteractive={false} position="bottom-right" />
            {selected && (model.factById.has(selected) || model.entityById.has(selected) || model.anchors.some((a) => a.id === selected)) && (
              <Panel position="top-right">
                <Inspector model={model} data={data} selected={selected} select={select} open={openFact} close={() => setSelected(null)} />
              </Panel>
            )}
          </ReactFlow>
        </div>
      <Key group={group} />
    </>
  );
}

type RelMode = "selected" | "always" | "off";

function loadRelMode(): RelMode {
  try {
    const v = localStorage.getItem("memx.graph.related");
    return v === "always" || v === "off" ? v : "selected";
  } catch {
    return "selected";
  }
}

function loadGroup(): Group {
  try {
    const v = localStorage.getItem("memx.graph.group");
    return v === "scope" || v === "theme" || v === "none" ? v : "entity";
  } catch {
    return "entity";
  }
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
          <p>What an agent at this scope knows. Each dot is a fact. Point at anything to see what it touches, click it to read it, and drag anything to rearrange.</p>
        </div>
        <div className="actions">
          <span className="gseg">
            <span>Group by</span>
            <Seg label="Group by" value={group} onChange={setGroup} options={[
              { value: "none", label: "None" }, { value: "entity", label: "Name" }, { value: "theme", label: "Theme" }, { value: "scope", label: "Scope" },
            ]} />
          </span>
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
