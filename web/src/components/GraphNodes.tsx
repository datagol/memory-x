import { BaseEdge, Handle, Position, type EdgeProps, type NodeProps } from "@xyflow/react";
import { Hash, Layers } from "lucide-react";
import type { Category, FactStatus } from "../lib/api";
import { CATEGORY_LABEL, catVar, truncate } from "../lib/meta";
import { EntityIcon } from "./FlowNodes";
import { useTip } from "./ui";

// Nodes of the Relationships graph: a live network of facts (dots) and the names, themes or scopes they hang off.

export type AnchorKind = "entity" | "theme" | "scope";

/** One invisible handle pair in the middle of every node, so links run from centre to centre. */
function Handles() {
  return (
    <>
      <Handle id="s" type="source" position={Position.Top} className="gcentre-handle" isConnectable={false} />
      <Handle id="t" type="target" position={Position.Top} className="gcentre-handle" isConnectable={false} />
    </>
  );
}

export type AnchorData = {
  kind: AnchorKind;
  label: string;
  /** Entity type, for the icon of a name. */
  type?: string;
  count: number;
  size: number;
  here?: boolean;
  selected?: boolean;
  dim?: boolean;
};

const KIND_ICON = { theme: Hash, scope: Layers };

/** A name, theme or scope. The facts linked to it mention that name, have that theme, or are stored at that scope;
 *  the more there are, the bigger it is. */
export function Anchor({ data }: NodeProps) {
  const d = data as AnchorData;
  const Icon = d.kind === "entity" ? null : KIND_ICON[d.kind];
  const icon = d.size >= 56 ? 20 : d.size >= 40 ? 16 : 13;
  return (
    <div className={`ganchor ${d.dim ? "dim" : ""}`}>
      <div className={`gbub ${d.kind} ${d.here ? "here" : ""} ${d.selected ? "sel" : ""}`}>
        {Icon ? <Icon size={icon} aria-hidden /> : <EntityIcon type={d.type ?? "other"} size={icon} />}
        <Handles />
      </div>
      {d.count > 1 && <span className="gcount">{d.count}</span>}
      <div className={`gcap ${d.size < 40 ? "small" : ""} ${d.kind === "scope" ? "mono" : ""}`}>{truncate(d.label, 30)}</div>
    </div>
  );
}

export type FactDotData = {
  text: string;
  category: Category;
  status: FactStatus;
  selected?: boolean;
  /** Pointed at. */
  hot?: boolean;
  dim?: boolean;
  /** Hidden at this scope by a narrower correction. */
  shadowed?: boolean;
};

/** A fact: a dot in the colour of its category. Hollow means replaced or hidden here. Its text shows on hover. */
export function FactDot({ data }: NodeProps) {
  const d = data as FactDotData;
  const tip = useTip();
  const past = d.status !== "active" || d.shadowed;
  const note = d.status !== "active" ? ` · ${d.status}` : d.shadowed ? " · hidden here" : "";
  return (
    <div
      className={`gdot ${past ? "past" : ""} ${d.selected ? "sel" : ""} ${d.hot ? "hot" : ""} ${d.dim ? "dim" : ""}`}
      style={{ ["--c" as string]: catVar(d.category) }}
      {...tip(<><b>{CATEGORY_LABEL[d.category]}{note}</b><br />{truncate(d.text, 180)}</>)}
    >
      <Handles />
    </div>
  );
}

/** A link between facts, drawn as a gentle arc so it reads apart from the straight links to names. */
export function ArcEdge({ sourceX, sourceY, targetX, targetY, style, markerEnd }: EdgeProps) {
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const cx = (sourceX + targetX) / 2 - dy * 0.14;
  const cy = (sourceY + targetY) / 2 + dx * 0.14;
  return <BaseEdge path={`M ${sourceX} ${sourceY} Q ${cx} ${cy} ${targetX} ${targetY}`} style={style} markerEnd={markerEnd} />;
}

export const graphNodeTypes = { anchor: Anchor, fact: FactDot };
export const graphEdgeTypes = { arc: ArcEdge };
