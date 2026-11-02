import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { ReactNode } from "react";
import { Building2, FolderKanban, Hash, Layers, MapPin, Package, Server, Tag, Unlink, User, Users, type LucideIcon } from "lucide-react";
import type { Category, FactStatus } from "../lib/api";
import { CATEGORY_LABEL, catVar } from "../lib/meta";

const ENTITY_ICON: Record<string, LucideIcon> = {
  person: User, organization: Building2, team: Users, system: Server, project: FolderKanban, place: MapPin,
  product: Package, other: Tag,
};

export function EntityIcon({ type, size }: { type: string; size?: number }) {
  const Icon = ENTITY_ICON[type] ?? Tag;
  return <Icon size={size} aria-hidden />;
}

const hidden = { opacity: 0, width: 6, height: 6, border: 0, background: "transparent" } as const;

const SIDES = [["r", Position.Right], ["l", Position.Left], ["b", Position.Bottom], ["t", Position.Top]] as const;

/** Source and target handles on every side, so edges can leave and enter on the sides facing each other. */
function Handles() {
  return (
    <>
      {SIDES.map(([id, pos]) => <Handle key={`s-${id}`} id={`s-${id}`} type="source" position={pos} style={hidden} isConnectable={false} />)}
      {[SIDES[1], SIDES[0], SIDES[3], SIDES[2]].map(([id, pos]) => <Handle key={`t-${id}`} id={`t-${id}`} type="target" position={pos} style={hidden} isConnectable={false} />)}
    </>
  );
}

/** Pick handles facing each other for two node boxes. */
export function facing(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }) {
  const dx = b.x + b.w / 2 - (a.x + a.w / 2);
  const dy = b.y + b.h / 2 - (a.y + a.h / 2);
  if (Math.abs(dx) > (a.w + b.w) / 2.4) return dx > 0 ? { sourceHandle: "s-r", targetHandle: "t-l" } : { sourceHandle: "s-l", targetHandle: "t-r" };
  return dy > 0 ? { sourceHandle: "s-b", targetHandle: "t-t" } : { sourceHandle: "s-t", targetHandle: "t-b" };
}

export type FactNodeData = {
  text: string;
  category: Category;
  status: FactStatus;
  scope?: string;
  badges?: ReactNode;
  selected?: boolean;
  dim?: boolean;
  width?: number;
  /** Other entities the fact names, shown as small chips under the text. */
  chips?: { name: string; type: string }[];
  more?: number;
  /** Set when the fact lives in another scope than the one being viewed. */
  scopeTag?: string;
};

export function FactNode({ data }: NodeProps) {
  const d = data as FactNodeData;
  return (
    <div className={`fnode ${d.status} ${d.selected ? "sel" : ""} ${d.dim ? "dim" : ""}`} style={d.width ? { width: d.width } : undefined}>
      <Handles />
      <span className="bar" style={{ background: catVar(d.category) }} />
      <div className="meta">
        <span className="chip" style={{ height: 18, fontSize: 11, padding: "0 6px" }}>
          <span className="sw" style={{ background: catVar(d.category) }} />{CATEGORY_LABEL[d.category]}
        </span>
        {d.status !== "active" && <span>{d.status}</span>}
        {d.badges}
      </div>
      <div className="txt">{d.text}</div>
      {d.scope && <div className="meta mono" style={{ fontSize: 10.5 }}>{d.scope}</div>}
      {(d.chips?.length || d.scopeTag) ? (
        <div className="chips">
          {d.chips?.map((c) => <span key={c.name} className="echip"><EntityIcon type={c.type} />{c.name}</span>)}
          {d.more ? <span className="echip">+{d.more}</span> : null}
          {d.scopeTag && <span className="stag" title="Stored in this scope">{d.scopeTag}</span>}
        </div>
      ) : null}
    </div>
  );
}

export type StageNodeData = { kicker: ReactNode; title: ReactNode; body?: ReactNode; tone?: "query" | "block"; width?: number; dim?: boolean };

export function StageNode({ data }: NodeProps) {
  const d = data as StageNodeData;
  return (
    <div className={`fnode stage ${d.tone ?? ""} ${d.dim ? "dim" : ""}`} style={d.width ? { width: d.width } : undefined}>
      <Handles />
      <div className="k">{d.kicker}</div>
      <div className="big">{d.title}</div>
      {d.body && <div className="small muted" style={{ lineHeight: 1.45 }}>{d.body}</div>}
    </div>
  );
}

export type ScopeNodeData = { path: string; here?: boolean; count?: number };

export function ScopeNode({ data }: NodeProps) {
  const d = data as ScopeNodeData;
  return (
    <div className={`fnode scope ${d.here ? "here" : ""}`}>
      <Handles />
      {d.path}
      {d.count !== undefined && <span className="faint" style={{ marginLeft: 8 }}>{d.count}</span>}
    </div>
  );
}

export type EntityNodeData = { name: string; type: string; typeLabel: string; count: number; selected?: boolean; dim?: boolean };

export function EntityNode({ data }: NodeProps) {
  const d = data as EntityNodeData;
  return (
    <div className={`fnode entity ${d.selected ? "sel" : ""} ${d.dim ? "dim" : ""}`} title={`${d.name}: mentioned by ${d.count} facts. Drag to move its group.`}>
      <Handles />
      <span className="ic"><EntityIcon type={d.type} /></span>
      <div className="tx">
        <div className="nm">{d.name}</div>
        <div className="ty">{d.typeLabel} · in {d.count} facts</div>
      </div>
    </div>
  );
}

export type ClusterKind = "entity" | "theme" | "loose" | "scope";
export type ClusterNodeData = { kind: ClusterKind; here?: boolean; dim?: boolean };

/** The rounded box behind a group of facts. Purely visual: it never takes the pointer, so the canvas pans through it. */
export function ClusterNode({ data }: NodeProps) {
  const d = data as ClusterNodeData;
  return <div className={`cluster ${d.kind} ${d.here ? "here" : ""} ${d.dim ? "dim" : ""}`} />;
}

export type ClusterLabelData = { kind: ClusterKind; label: string; count: number; here?: boolean; dim?: boolean };

/** The tab on a theme or scope box. Dragging it moves the whole box. */
export function ClusterLabel({ data }: NodeProps) {
  const d = data as ClusterLabelData;
  const Icon = d.kind === "theme" ? Hash : d.kind === "scope" ? Layers : Unlink;
  const kicker = d.kind === "theme" ? "Theme" : d.kind === "scope" ? (d.here ? "This scope" : "Scope") : "";
  return (
    <div className={`clabel ${d.here ? "here" : ""} ${d.dim ? "dim" : ""}`} title="Drag to move this group">
      <Icon />{kicker}<b className={d.kind === "scope" ? "mono" : ""}>{d.label}</b><span className="n">{d.count}</span>
    </div>
  );
}

export const nodeTypes = {
  fact: FactNode, stage: StageNode, scope: ScopeNode, entity: EntityNode, cluster: ClusterNode, clusterLabel: ClusterLabel,
};
