import { scaleSqrt } from "d3";

// Placement for the UNet bend points.
//
// Two layouts live here. `layoutNodes` is the original strip -- one column per
// layer, in sequence -- still used by the per-bend footprint sparkline, where a
// row of dots in a card head is exactly the right picture.
//
// `layoutStructured` is what the model map draws: the network by shape rather
// than by sequence. One band per resolution level, encoder stepping down the
// left, mid at the bottom, decoder climbing the right. The width then follows
// the widest band instead of the layer count, so a 21-layer xurdif model and a
// 50-layer Diffusers one both fit the column they are drawn in.
//
// `collapseBands` is a coarser reading of that same geometry -- one capsule per
// band side -- so the map can change density without anything moving.
const DEFAULTS = { colW: 62, rowH: 66, marginX: 40, top: 40, bottom: 60, rMin: 9, rMax: 26 };

export const STRUCT = {
  colW: 34, rowH: 46, wrapStep: 34, indent: 18, gutter: 74, marginR: 12,
  top: 26, bottom: 34, minContent: 240, maxContent: 440, rMin: 6, rMax: 15,
  capW: 86, capH: 24,
};

/** Resolution level of a node: 0 at full res, 1 at a half, 2 at a quarter... */
export function depthOf(node) {
  const d = Math.log2(node?.down_factor || 1);
  return Number.isFinite(d) ? Math.max(0, Math.round(d)) : 0;
}

function radiusScale(nodes, o) {
  const maxCh = Math.max(...nodes.map((n) => n.channels || 1));
  return scaleSqrt().domain([1, Math.max(maxCh, 2)]).range([o.rMin, o.rMax]);
}

export function layoutNodes(nodes, opts = {}) {
  if (!nodes?.length) return null;
  const o = { ...DEFAULTS, ...opts };
  const maxDepth = Math.max(...nodes.map(depthOf));
  const r = radiusScale(nodes, o);
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

/**
 * The U: nodes placed by resolution level and side.
 *
 * Returns `{ placed, bands, width, height, maxDepth, o }`. Every node keeps its
 * sequence index `i` -- shift+click ranges on the map are ranges through the
 * network, not through the picture -- and gains `depth`, `side` ("L", "R" or
 * "mid") and `wrapRow`.
 */
export function layoutStructured(nodes, opts = {}) {
  if (!nodes?.length) return null;
  const o = { ...STRUCT, ...opts };
  const list = nodes.map((n, i) => ({ ...n, i, resDepth: depthOf(n) }));
  const maxDepth = Math.max(...list.map((n) => n.resDepth));
  // A stage Kiln does not know still has to land somewhere, so anything that is
  // not tagged falls on the side of the bottleneck its position puts it.
  const midAt = list.findIndex((n) => n.stage === "mid");
  list.forEach((n) => {
    if (n.stage === "mid") { n.side = "mid"; return; }
    if (n.stage === "decoder") { n.side = "R"; return; }
    if (n.stage === "encoder") { n.side = "L"; return; }
    n.side = midAt < 0 || n.i < midAt ? "L" : "R";
  });

  // The bottleneck takes a row of its own whenever the deepest level also holds
  // encoder or decoder layers: it is drawn centred, so sharing that row would
  // put it straight through them.
  const hasMid = list.some((n) => n.side === "mid");
  const crowded = list.some((n) => n.side !== "mid" && n.resDepth === maxDepth);
  const midRow = hasMid && crowded ? maxDepth + 1 : maxDepth;
  list.forEach((n) => { n.depth = n.side === "mid" ? midRow : n.resDepth; });
  const lastRow = Math.max(maxDepth, hasMid ? midRow : 0);

  const rows = [];
  for (let d = 0; d <= lastRow; d += 1) rows[d] = { L: [], R: [], mid: [] };
  list.forEach((n) => rows[n.depth][n.side].push(n));

  // Width is settled before anything is placed, from the widest band rather
  // than from the layer count, and then clamped. That is the whole point: the
  // map can no longer grow sideways without bound.
  const need = rows.map((b, d) => 2 * d * o.indent + (b.L.length + b.R.length) * o.colW + 1.5 * o.colW);
  const midNeed = Math.max(...rows.map((b) => b.mid.length)) * o.colW;
  const contentW = Math.max(o.minContent, Math.min(o.maxContent, Math.max(...need, midNeed)));
  const width = o.gutter + contentW + o.marginR;

  // Anything that does not fit a band wraps inside it instead of widening it.
  const capSide = (d) => Math.max(1, Math.floor((contentW / 2 - d * o.indent - 0.75 * o.colW) / o.colW));
  const capMid = Math.max(1, Math.floor(contentW / o.colW));
  const wrapRows = rows.map((b, d) => Math.max(
    Math.ceil(b.L.length / capSide(d)),
    Math.ceil(b.R.length / capSide(d)),
    Math.ceil(b.mid.length / capMid),
    1,
  ));
  const bandTop = [];
  rows.forEach((_, d) => {
    bandTop[d] = d === 0 ? o.top : bandTop[d - 1] + o.rowH + (wrapRows[d - 1] - 1) * o.wrapStep;
  });
  const height = bandTop[lastRow] + (wrapRows[lastRow] - 1) * o.wrapStep + o.bottom;

  const r = radiusScale(list, o);
  const cx = (o.gutter + width - o.marginR) / 2;
  const placed = [];
  rows.forEach((b, d) => {
    const cap = capSide(d);
    const put = (n, w, x) => placed.push({
      ...n, wrapRow: w, x, y: bandTop[d] + w * o.wrapStep, r: r(n.channels || 1),
    });
    b.L.forEach((n, idx) => put(n, Math.floor(idx / cap), o.gutter + d * o.indent + (idx % cap) * o.colW));
    b.R.forEach((n, idx) => {
      const w = Math.floor(idx / cap);
      const len = Math.min(cap, b.R.length - w * cap);
      put(n, w, (width - o.marginR - d * o.indent) - (len - 1 - (idx % cap)) * o.colW);
    });
    b.mid.forEach((n, idx) => {
      const w = Math.floor(idx / capMid);
      const len = Math.min(capMid, b.mid.length - w * capMid);
      put(n, w, cx + ((idx % capMid) - (len - 1) / 2) * o.colW);
    });
  });
  placed.sort((a, b) => a.i - b.i);

  const bands = rows.map((b, d) => {
    const all = [...b.L, ...b.mid, ...b.R].sort((x, y) => x.i - y.i);
    return {
      depth: d,
      y: bandTop[d],
      label: resLabel({ down_factor: 2 ** (all[0]?.resDepth ?? d) }),
      channels: Math.max(0, ...all.map((n) => n.channels || 0)),
      // What bending here tends to do, in words, on the two bands where the
      // answer is clear enough to be worth saying.
      role: d === 0 ? "texture, detail" : d === lastRow ? "composition" : "",
      ids: all.map((n) => n.id),
      sides: { L: b.L.length, R: b.R.length, mid: b.mid.length },
    };
  }).filter((b) => b.ids.length);

  return { placed, bands, width, height, maxDepth: lastRow, o };
}

/** One capsule per band side: the map at arm's length. */
export function collapseBands(layout) {
  if (!layout) return [];
  const { bands, width, o, maxDepth } = layout;
  const byKey = new Map();
  layout.placed.forEach((n) => {
    const key = `${n.depth}:${n.side}`;
    const hit = byKey.get(key);
    if (hit) { hit.ids.push(n.id); hit.count += 1; hit.hasAttention ||= n.type === "attention"; }
    else byKey.set(key, { key, depth: n.depth, side: n.side, ids: [n.id], count: 1, hasAttention: n.type === "attention" });
  });
  const bandY = new Map(bands.map((b) => [b.depth, b.y]));
  const cx = (o.gutter + width - o.marginR) / 2;
  return [...byKey.values()].map((c) => {
    const x = c.side === "L" ? o.gutter + c.depth * o.indent
      : c.side === "R" ? width - o.marginR - o.capW - c.depth * o.indent
        : cx - o.capW / 2;
    const name = c.side === "mid" ? "bottleneck" : `${c.side === "L" ? "enc" : "dec"} ${c.depth}`;
    return {
      ...c,
      x,
      y: (bandY.get(c.depth) ?? o.top) - o.capH / 2,
      w: o.capW,
      h: o.capH,
      label: `${name} · ${c.count}`,
      deepest: c.depth === maxDepth,
    };
  }).sort((a, b) => a.depth - b.depth);
}

/** "M x,y L x,y ..." spine through the placed nodes. */
export function spinePath(placed) {
  return placed.map((n, i) => `${i === 0 ? "M" : "L"}${n.x},${n.y}`).join(" ");
}

/**
 * Polylines through consecutive nodes that share a band, side and wrap row,
 * plus the jumps between those runs. One line through everything would cut a
 * diagonal across the picture every time the sequence changes level.
 */
export function runPaths(placed) {
  const runs = [];
  placed.forEach((n) => {
    const last = runs[runs.length - 1];
    const tail = last?.[last.length - 1];
    if (tail && tail.depth === n.depth && tail.side === n.side && tail.wrapRow === n.wrapRow
        && tail.i === n.i - 1) last.push(n);
    else runs.push([n]);
  });
  return {
    runs: runs.filter((r) => r.length > 1).map(spinePath),
    jumps: runs.slice(1).map((r, k) => {
      const from = runs[k][runs[k].length - 1];
      const to = r[0];
      return `M${from.x},${from.y} L${to.x},${to.y}`;
    }),
  };
}

/** Faint arcs from each band's encoder side to its decoder side: the skips. */
export function skipPaths(layout) {
  if (!layout) return [];
  const byBand = new Map();
  layout.placed.forEach((n) => {
    if (n.side === "mid") return;
    const b = byBand.get(n.depth) || { L: [], R: [] };
    b[n.side].push(n);
    byBand.set(n.depth, b);
  });
  const out = [];
  byBand.forEach((b) => {
    if (!b.L.length || !b.R.length) return;
    const from = b.L[b.L.length - 1];
    const to = b.R[0];
    if (to.x - from.x < 24) return;
    out.push(`M${from.x},${from.y} Q${(from.x + to.x) / 2},${from.y - 12} ${to.x},${to.y}`);
  });
  return out;
}

/** Compact resolution label from the introspect down_factor, e.g. "1/4 res". */
export function resLabel(node) {
  const d = node?.down_factor;
  if (!d || d <= 1) return "full res";
  return `1/${Math.round(d)} res`;
}
