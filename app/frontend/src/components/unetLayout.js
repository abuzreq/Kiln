import { scaleSqrt } from "d3";

// Shared placement for the UNet bend points. The big map and the per-bend
// footprint sparkline both draw from this so the two can never disagree about
// which dot is which layer.
const DEFAULTS = { colW: 62, rowH: 66, marginX: 40, top: 40, bottom: 60, rMin: 9, rMax: 26 };

export function layoutNodes(nodes, opts = {}) {
  if (!nodes?.length) return null;
  const o = { ...DEFAULTS, ...opts };
  const depthOf = (n) => Math.log2(n.down_factor || 1) || 0;
  const maxDepth = Math.max(...nodes.map(depthOf));
  const maxCh = Math.max(...nodes.map((n) => n.channels || 1));
  const r = scaleSqrt().domain([1, Math.max(maxCh, 2)]).range([o.rMin, o.rMax]);
  const placed = nodes.map((n, i) => ({
    ...n,
    i,
    x: o.marginX + i * o.colW,
    y: o.top + depthOf(n) * o.rowH,
    r: r(n.channels || 1),
  }));
  return {
    placed,
    width: o.marginX * 2 + (nodes.length - 1) * o.colW,
    height: o.top + maxDepth * o.rowH + o.bottom,
  };
}

/** "M x,y L x,y ..." spine through the placed nodes. */
export function spinePath(placed) {
  return placed.map((n, i) => `${i === 0 ? "M" : "L"}${n.x},${n.y}`).join(" ");
}

/** Compact resolution label from the introspect down_factor, e.g. "1/4 res". */
export function resLabel(node) {
  const d = node?.down_factor;
  if (!d || d <= 1) return "full res";
  return `1/${Math.round(d)} res`;
}
