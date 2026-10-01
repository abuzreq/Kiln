import React, { useMemo } from "react";
import { layoutNodes, spinePath } from "./unetLayout.js";
import { resolveTargets } from "../bendTargets.js";

// Tiny strip showing which layers one bend hits, in network order. It lives in
// the card head so a collapsed stack is still scannable: you can see at a glance
// that one bend sits on the decoder and another on the bottleneck, and where
// they overlap. It keeps the sequential layout rather than the map's U -- at
// eight pixels a column the U folds into a blob and tells you less.
const SMALL = { colW: 8, rowH: 6, marginX: 5, top: 5, bottom: 6, rMin: 2, rMax: 4.5 };

export default function BendFootprint({ nodes, targets }) {
  const layout = useMemo(() => layoutNodes(nodes, SMALL), [nodes]);
  const hit = useMemo(() => resolveTargets(targets, nodes), [targets, nodes]);
  if (!layout) return null;
  const { placed, width, height } = layout;
  const n = hit.size;

  return (
    <div className="bend-footprint" title={`${n} of ${placed.length} layers`}>
      <svg width={width} height={height} aria-hidden="true">
        <path d={spinePath(placed)} fill="none" stroke="var(--line)" strokeWidth="1" />
        {placed.map((p) => (
          <circle
            key={p.id}
            cx={p.x}
            cy={p.y}
            r={hit.has(p.id) ? p.r : Math.max(p.r - 0.8, 1.4)}
            fill={hit.has(p.id) ? "var(--accent)" : "var(--text-dim)"}
            opacity={hit.has(p.id) ? 1 : 0.35}
          />
        ))}
      </svg>
      <span className={`sub bend-footprint-count ${n ? "" : "none"}`}>
        {n === 0 ? "no layers" : n === 1 ? "1 layer" : `${n} layers`}
      </span>
    </div>
  );
}
