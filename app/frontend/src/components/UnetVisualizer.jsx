import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { tokens } from "../theme.js";
import { heldByActive, holdersIn } from "../bendTargets.js";
import { badgeRun, holdersText } from "./HolderBadges.jsx";
import {
  layoutStructured, collapseBands, runPaths, skipPaths, resLabel,
} from "./unetLayout.js";

// The UNet as a U -- bands by resolution, encoder down the left, bottleneck at
// the bottom, decoder up the right -- for when the structure itself is the
// point. Two densities of the same geometry: "overview" gives one capsule per
// band side, "layers" shows every bend point. (The plain-language way in is
// BendPipeline, which draws no geometry at all.)
//
// In layers, clicking a node toggles it in the FOCUSED bend's targets,
// shift+click takes a range through the network and dragging the background
// sweeps a rectangle. Layers held by other bends keep a dimmed ring, and each
// level carries the numbers of the bends that reach into it, so the whole stack
// stays legible while you edit one.
const STAGE_VARS = {
  encoder: "--stage-encoder",
  mid: "--stage-mid",
  decoder: "--stage-decoder",
  other: "--stage-other",
};

const SIDE_STAGE = { L: "encoder", R: "decoder", mid: "mid" };
const DRAG_SLOP = 4;
// The layout spreads into the width it is given; past that the whole drawing
// may scale up this far to finish filling the card, and no further -- beyond
// it the labels start to look like a different typeface from the rest of Kiln.
const MAX_SCALE = 1.4;
// The layers inside a block, drawn as small dots fanned out just under it.
const SAT_R = 3.4;
const SAT_STEP = 9;
const SAT_GAP = 11;
// Room the satellites need between bands, so they never touch the next row.
const INNER_ROWS = { rowH: 58, wrapStep: 44 };

/** Greedy word wrap for the hover card: SVG text does not wrap by itself. */
function wrapText(text, max = 36) {
  const lines = [];
  let cur = "";
  (text || "").split(" ").forEach((w) => {
    if (cur && `${cur} ${w}`.length > max) { lines.push(cur); cur = w; } else cur = cur ? `${cur} ${w}` : w;
  });
  if (cur) lines.push(cur);
  return lines;
}

export default function UnetVisualizer({
  graph, focusTargets, holders, hasFocus, density = "overview", showInner = false,
  onToggle, onCreateFromNode,
}) {
  const [hover, setHover] = useState(null);
  const [drag, setDrag] = useState(null);
  const [expanded, setExpanded] = useState(() => new Set());
  const [avail, setAvail] = useState(0);
  const anchor = useRef(null);
  const svgRef = useRef(null);
  const boxRef = useRef(null);
  // Live drag geometry and the current handler live in refs so the window
  // listeners are attached once per drag instead of re-bound on every mousemove.
  const dragRef = useRef(null);
  const toggleRef = useRef(onToggle);
  toggleRef.current = onToggle;
  const c = useMemo(
    () => tokens({ ...STAGE_VARS, line: "--line", panel: "--bg-3", text: "--text", dim: "--text-dim" }),
    [],
  );
  // The map is laid out for the box it is drawn in, so a wider card spreads the
  // bands rather than leaving half of itself empty. Changes under 8px are
  // ignored: re-laying out on every pixel would feed the observer its own
  // output. The box only exists once there is a graph to draw, so the observer
  // is attached then rather than on mount.
  const ready = !!graph?.nodes?.length;
  useEffect(() => {
    const el = boxRef.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const measure = () => {
      const w = el.clientWidth;
      setAvail((prev) => (Math.abs(w - prev) >= 8 ? w : prev));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ready]);

  // The layers inside blocks are an advanced, Layers-only view: they open the
  // rows up a little so their dots fit under each block.
  const innerOn = !!(showInner && density === "layers" && graph?.inner?.length);
  const layout = useMemo(
    () => layoutStructured(graph?.nodes, {
      ...(avail ? { available: avail } : {}),
      ...(innerOn ? INNER_ROWS : {}),
    }),
    [graph, avail, innerOn],
  );
  const capsules = useMemo(() => (density === "overview" ? collapseBands(layout) : []), [layout, density]);
  const spine = useMemo(() => (density === "layers" ? runPaths(layout?.placed || []) : null), [layout, density]);
  // Skips the backend can bend on their own become targets, drawn as arcs from
  // the encoder point that feeds them to the decoder block that takes them. A
  // backend without them (Diffusers, for now) keeps the plain decorative arcs.
  const skipArcs = useMemo(() => {
    if (!layout || !graph?.skips?.length) return [];
    const at = new Map(layout.placed.map((n) => [n.id, n]));
    return graph.skips.map((sk) => {
      const a = at.get(sk.from);
      const b = at.get(sk.to);
      if (!a || !b) return null;
      const cx = (a.x + b.x) / 2;
      const cy = Math.min(a.y, b.y) - 14;
      return {
        ...sk, d: `M${a.x},${a.y} Q${cx},${cy} ${b.x},${b.y}`,
        mx: 0.25 * a.x + 0.5 * cx + 0.25 * b.x, my: 0.25 * a.y + 0.5 * cy + 0.25 * b.y,
      };
    }).filter(Boolean);
  }, [layout, graph]);
  const skips = useMemo(() => (skipArcs.length ? [] : skipPaths(layout)), [layout, skipArcs]);

  const innerPlaced = useMemo(() => {
    if (!innerOn || !layout) return [];
    const byParent = new Map();
    graph.inner.forEach((p) => {
      const list = byParent.get(p.parent) || [];
      list.push(p);
      byParent.set(p.parent, list);
    });
    const out = [];
    layout.placed.forEach((n) => {
      const kids = byParent.get(n.id);
      if (!kids) return;
      kids.forEach((p, k) => out.push({
        ...p, extra: "inner", side: n.side, depth: n.depth,
        x: n.x + (k - (kids.length - 1) / 2) * SAT_STEP, y: n.y + n.r + SAT_GAP, r: SAT_R,
        px: n.x, py: n.y + n.r,
      }));
    });
    return out;
  }, [innerOn, layout, graph]);
  // What a box drag can take: the main points, and the inner ones when shown.
  const selectable = useMemo(
    () => (layout ? [...layout.placed, ...innerPlaced] : []),
    [layout, innerPlaced],
  );
  const selectableRef = useRef(selectable);
  selectableRef.current = selectable;

  // With a viewBox, client pixels and drawing units are not the same thing:
  // everything that compares a mouse position to a node has to come through here.
  const localPt = useCallback((e) => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box || !box.width) return { x: 0, y: 0 };
    const k = (layout?.width || 1) / box.width;
    return { x: (e.clientX - box.left) * k, y: (e.clientY - box.top) * k };
  }, [layout]);

  // The marquee finishes on window mouseup so a release outside the svg still lands.
  const dragging = !!drag;
  useEffect(() => {
    if (!dragging) return undefined;
    const move = (e) => {
      const p = localPt(e);
      if (dragRef.current) { dragRef.current.x1 = p.x; dragRef.current.y1 = p.y; }
      setDrag((d) => (d ? { ...d, x1: p.x, y1: p.y } : d));
    };
    const up = (e) => {
      const d = dragRef.current;
      dragRef.current = null;
      setDrag(null);
      if (!d || !layout) return;
      const p = localPt(e);
      if (Math.max(Math.abs(p.x - d.x0), Math.abs(p.y - d.y0)) < DRAG_SLOP) return;
      const [lo, hi] = p.x < d.x0 ? [p.x, d.x0] : [d.x0, p.x];
      const [top, bot] = p.y < d.y0 ? [p.y, d.y0] : [d.y0, p.y];
      const ids = selectableRef.current
        .filter((n) => n.x >= lo && n.x <= hi && n.y >= top && n.y <= bot)
        .map((n) => n.id);
      if (ids.length) toggleRef.current?.(ids, { force: !d.remove });
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [dragging, layout, localPt]);

  /** Target a set of layers, or start a bend on them when nothing is focused. */
  const pick = useCallback((ids, opts) => {
    if (!hasFocus) { onCreateFromNode?.(ids); return; }
    onToggle(ids, opts);
  }, [hasFocus, onToggle, onCreateFromNode]);

  const clickNode = useCallback((n, e) => {
    if (!hasFocus) { onCreateFromNode?.([n.id]); anchor.current = n.i; return; }
    const force = !focusTargets?.has(n.id);
    if (e.shiftKey && anchor.current != null && layout) {
      const [lo, hi] = anchor.current < n.i ? [anchor.current, n.i] : [n.i, anchor.current];
      onToggle(layout.placed.filter((x) => x.i >= lo && x.i <= hi).map((x) => x.id), { force });
    } else {
      onToggle([n.id], { force });
    }
    anchor.current = n.i;
  }, [hasFocus, focusTargets, layout, onToggle, onCreateFromNode]);

  // Left/right walk the network in order; up/down cross bands, which is only a
  // direction worth having now that depth is an axis.
  const nodeKey = useCallback((n, e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); clickNode(n, e); return; }
    const focusIndex = (i) => svgRef.current?.querySelector(`[data-node-index="${i}"]`)?.focus();
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      focusIndex(n.i + (e.key === "ArrowRight" ? 1 : -1));
      return;
    }
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    e.preventDefault();
    const want = n.depth + (e.key === "ArrowDown" ? 1 : -1);
    const same = layout.placed.filter((x) => x.depth === want && x.side === n.side);
    const pool = same.length ? same : layout.placed.filter((x) => x.depth === want);
    if (!pool.length) return;
    const near = pool.reduce((a, b) => (Math.abs(b.x - n.x) < Math.abs(a.x - n.x) ? b : a));
    focusIndex(near.i);
  }, [clickNode, layout]);

  const keyActivate = (fn) => (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fn(e); }
  };

  if (!layout) return <div className="sub" style={{ padding: 20 }}>Select a model to introspect its layers.</div>;
  const { placed, bands, width, height, o } = layout;
  const marquee = drag && Math.max(Math.abs(drag.x1 - drag.x0), Math.abs(drag.y1 - drag.y0)) >= DRAG_SLOP
    ? {
      x: Math.min(drag.x0, drag.x1), y: Math.min(drag.y0, drag.y1),
      w: Math.abs(drag.x1 - drag.x0), h: Math.abs(drag.y1 - drag.y0),
    }
    : null;

  // What letting go of the box would change: the layers it would add (or, with
  // alt, drop), outlined before the release, with their count beside it.
  const preview = marquee
    ? new Set(selectable
      .filter((n) => n.x >= marquee.x && n.x <= marquee.x + marquee.w
        && n.y >= marquee.y && n.y <= marquee.y + marquee.h)
      .filter((n) => (drag.remove ? focusTargets?.has(n.id) : !focusTargets?.has(n.id)))
      .map((n) => n.id))
    : null;

  /** Numbered badges for the other bends in `ids`, from (x, y) along `dir`. */
  const badges = (ids, x, y, dir, key) => badgeRun(holdersIn(holders, ids)).map((h, k) => (
    <g key={`${key}-b${k}`} className="unet-badge" transform={`translate(${x + dir * k * 16},${y})`}
       opacity={h.active ? 1 : 0.6} pointerEvents="none">
      <circle r="7.5" fill={c.panel} stroke={h.active ? c.dim : c.line}
              strokeDasharray={h.active ? undefined : "2 2"} />
      <text y="3.2" textAnchor="middle" fontSize="9" fontWeight="700" fill={h.active ? c.text : c.dim}>
        {h.text}
      </text>
    </g>
  ));

  const state = (ids) => {
    const hit = ids.filter((id) => focusTargets?.has(id)).length;
    return { hit, all: hit === ids.length && hit > 0, some: hit > 0 && hit < ids.length };
  };

  const node = (n) => {
    const isFocus = focusTargets?.has(n.id);
    const isOther = !isFocus && heldByActive(holders, n.id);
    // held only by bends that are off: still worth a trace, but a faint one
    const isOffOnly = !isFocus && !isOther && holders?.has(n.id);
    const fill = c[n.stage] || c[SIDE_STAGE[n.side]] || c.other;
    const attn = n.type === "attention";
    const shape = (r, props) => (attn
      ? <path d={`M0,${-r * 1.2} L${r * 1.2},0 L0,${r * 1.2} L${-r * 1.2},0 Z`} {...props} />
      : <circle r={r} {...props} />);
    return (
      <g
        key={n.id}
        data-node-index={n.i}
        tabIndex={0}
        role="button"
        aria-pressed={!!isFocus}
        aria-label={`${n.label} — ${isFocus ? "targeted" : "not targeted"}`}
        transform={`translate(${n.x},${n.y})`}
        className="unet-node"
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => clickNode(n, e)}
        onKeyDown={(e) => nodeKey(n, e)}
        onMouseEnter={() => setHover(n)}
        onMouseLeave={() => setHover(null)}
      >
        {isFocus && shape(n.r + 5, { fill: "none", stroke: "var(--accent)", strokeWidth: 2.5 })}
        {isOther && shape(n.r + 5, {
          fill: "none", stroke: "var(--accent)", strokeWidth: 1.5, strokeDasharray: "3 3", opacity: 0.45,
        })}
        {isOffOnly && shape(n.r + 5, {
          fill: "none", stroke: c.dim, strokeWidth: 1, strokeDasharray: "2 3", opacity: 0.4,
        })}
        {preview?.has(n.id) && shape(n.r + 8, {
          fill: "none", stroke: "var(--accent)", strokeWidth: 1.5, strokeDasharray: "2 3",
        })}
        {shape(n.r, {
          fill, opacity: isFocus ? 1 : 0.72,
          stroke: hover?.id === n.id ? "#fff" : "transparent", strokeWidth: 2,
        })}
        {/* The dots are small now, so the thing you actually click is not. */}
        <circle r={Math.max(n.r, 12)} fill="transparent" />
      </g>
    );
  };

  const satsOf = new Map();
  innerPlaced.forEach((p) => {
    const list = satsOf.get(p.parent) || [];
    list.push(p);
    satsOf.set(p.parent, list);
  });

  /** A layer inside a block: clicks and box-selects like any point. Kept
   *  right after its block in the DOM, so Tab steps from a block into its
   *  layers and on to the next block. */
  const satellite = (p) => {
    const isFocus = focusTargets?.has(p.id);
    const isOther = !isFocus && heldByActive(holders, p.id);
    const fill = c[p.stage] || c[SIDE_STAGE[p.side]] || c.other;
    const toggle = () => pick([p.id], { force: !isFocus });
    return (
      <g
        key={p.id}
        tabIndex={0}
        role="button"
        aria-pressed={!!isFocus}
        aria-label={`${p.label} — ${isFocus ? "targeted" : "not targeted"}`}
        transform={`translate(${p.x},${p.y})`}
        className="unet-node unet-sat"
        onMouseDown={(e) => e.stopPropagation()}
        onClick={toggle}
        onKeyDown={keyActivate(toggle)}
        onMouseEnter={() => setHover(p)}
        onMouseLeave={() => setHover(null)}
      >
        {isFocus && <circle r={p.r + 3} fill="none" stroke="var(--accent)" strokeWidth="2" />}
        {isOther && (
          <circle r={p.r + 3} fill="none" stroke="var(--accent)" strokeWidth="1.2" strokeDasharray="2 2" opacity="0.5" />
        )}
        {preview?.has(p.id) && (
          <circle r={p.r + 5} fill="none" stroke="var(--accent)" strokeWidth="1.2" strokeDasharray="2 2" />
        )}
        <circle r={p.r} fill={fill} opacity={isFocus ? 1 : 0.6}
                stroke={hover?.id === p.id ? "#fff" : "transparent"} strokeWidth="1.5" />
        <circle r="4.5" fill="transparent" />
      </g>
    );
  };

  /** A skip connection: click the arc to bend that skip alone. */
  const skipArc = (sk) => {
    const on = focusTargets?.has(sk.id);
    const other = !on && heldByActive(holders, sk.id);
    const offOnly = !on && !other && holders?.has(sk.id);
    const hot = hover?.id === sk.id;
    const toggle = () => pick([sk.id], { force: !on });
    return (
      <g
        key={sk.id}
        className={`unet-skip ${on ? "on" : ""}`.trim()}
        role="button"
        tabIndex={0}
        aria-pressed={!!on}
        aria-label={`${sk.label} — ${on ? "targeted" : "not targeted"}`}
        onMouseDown={(e) => e.stopPropagation()}
        onClick={toggle}
        onKeyDown={keyActivate(toggle)}
        onMouseEnter={() => setHover({ ...sk, extra: "skip", x: sk.mx, y: sk.my, r: 5 })}
        onMouseLeave={() => setHover(null)}
      >
        <path d={sk.d} fill="none" stroke="transparent" strokeWidth="14" />
        <path
          d={sk.d} fill="none"
          stroke={on || other || hot ? "var(--accent)" : c.dim}
          strokeWidth={on ? 2.5 : 1.5}
          strokeDasharray={on ? undefined : "4 4"}
          opacity={on ? 1 : hot ? 0.85 : other ? 0.6 : offOnly ? 0.25 : 0.5}
        />
        <circle cx={sk.mx} cy={sk.my} r={on ? 4 : 2.5} fill={on || hot ? "var(--accent)" : c.dim}
                opacity={on || hot ? 1 : 0.7} />
      </g>
    );
  };

  const capsule = (cap) => {
    const { all, some, hit } = state(cap.ids);
    const held = holdersIn(holders, cap.ids);
    const open = () => setExpanded((s) => {
      const next = new Set(s);
      next.add(cap.key);
      return next;
    });
    return (
      <g key={cap.key} className="unet-capsule">
        <g
          role="button"
          tabIndex={0}
          aria-pressed={all}
          aria-label={`${cap.label} layers — ${all ? "targeted" : some ? "partly targeted" : "not targeted"}${
            held.length ? `; also in ${holdersText(held)}` : ""}`}
          onClick={() => pick(cap.ids, { force: !all })}
          onKeyDown={keyActivate(() => pick(cap.ids, { force: !all }))}
        >
          <rect
            x={cap.x} y={cap.y} width={cap.w} height={cap.h} rx={cap.h / 2}
            fill={c[SIDE_STAGE[cap.side]] || c.other}
            opacity={all ? 0.95 : some ? 0.6 : 0.4}
            stroke={all || some ? "var(--accent)" : "transparent"}
            strokeWidth={all ? 2.5 : 1.5}
            strokeDasharray={some && !all ? "3 3" : undefined}
          />
          <text x={cap.x + 10} y={cap.y + cap.h / 2 + 3.5} fontSize="11" fill={c.text}>
            {cap.label}{cap.hasAttention ? " ◆" : ""}
          </text>
          {some && !all && (
            <text x={cap.x + cap.w - 8} y={cap.y + cap.h / 2 + 3.5} fontSize="10" textAnchor="end" fill={c.text}>
              {hit}/{cap.count}
            </text>
          )}
        </g>
        {/* On the capsule's outer top corner, clear of its label. */}
        {cap.side === "L"
          ? badges(cap.ids, cap.x + 6, cap.y - 3, 1, cap.key)
          : badges(cap.ids, cap.x + cap.w - 6, cap.y - 3, -1, cap.key)}
        {cap.count > 1 && (
          <g
            role="button"
            tabIndex={0}
            className="unet-expand"
            aria-label={`Show the ${cap.count} layers in ${cap.label}`}
            transform={`translate(${cap.x + cap.w + 9},${cap.y + cap.h / 2})`}
            onClick={open}
            onKeyDown={keyActivate(open)}
          >
            <circle r="8" fill="transparent" />
            <text y="4" textAnchor="middle" fontSize="12" fill={c.dim}>＋</text>
          </g>
        )}
      </g>
    );
  };

  const collapser = (key, nodes) => {
    const side = nodes[0].side;
    const last = nodes[nodes.length - 1];
    const x = side === "R" ? nodes[0].x - o.colW * 0.6 : last.x + o.colW * 0.6;
    const close = () => setExpanded((s) => {
      const next = new Set(s);
      next.delete(key);
      return next;
    });
    return (
      <g
        key={`${key}-close`}
        role="button"
        tabIndex={0}
        className="unet-expand"
        aria-label="Collapse these layers"
        transform={`translate(${x},${nodes[0].y})`}
        onClick={close}
        onKeyDown={keyActivate(close)}
      >
        <circle r="8" fill="transparent" />
        <text y="4" textAnchor="middle" fontSize="12" fill={c.dim}>−</text>
      </g>
    );
  };

  // In Layers (and in an opened level) the badges sit above the outer end of
  // each band side's first row, where the run starts reading inwards.
  const sideBadges = (list) => {
    const bySide = new Map();
    list.forEach((n) => {
      const key = `${n.depth}:${n.side}`;
      const g = bySide.get(key) || { ids: [], ends: [] };
      g.ids.push(n.id);
      if (!n.wrapRow) g.ends.push(n);
      bySide.set(key, g);
    });
    return [...bySide.entries()].filter(([, g]) => g.ends.length).flatMap(([key, g]) => {
      const outerLeft = key.endsWith(":L");
      const end = g.ends.reduce((a, b) => ((outerLeft ? b.x < a.x : b.x > a.x) ? b : a));
      return badges(g.ids, end.x, end.y - end.r - 10, outerLeft ? 1 : -1, `side-${key}`);
    });
  };

  const expandedNodes = density === "overview"
    ? placed.filter((n) => expanded.has(`${n.depth}:${n.side}`))
    : [];
  const expandedKeys = [...new Set(expandedNodes.map((n) => `${n.depth}:${n.side}`))];

  return (
    <div className="scroll-x" ref={boxRef}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMin meet"
        className={`unet-map ${drag ? "dragging" : ""}`}
        style={{ maxWidth: Math.round(width * MAX_SCALE) }}
        onMouseDown={(e) => {
          if (density !== "layers" || !hasFocus || e.button !== 0) return;
          const p = localPt(e);
          dragRef.current = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, remove: e.altKey };
          setDrag({ ...dragRef.current });
        }}
      >
        {bands.map((b) => {
          const rowState = state(b.ids);
          return (
            <g key={b.depth} className="unet-band">
              <line x1={o.gutter - 8} x2={width - 4} y1={b.y} y2={b.y} stroke={c.line} strokeWidth="1" opacity="0.35" />
              <g
                className="unet-gutter"
                role="button"
                tabIndex={0}
                aria-pressed={rowState.all}
                aria-label={`${b.label} layers — ${rowState.all ? "targeted" : "not targeted"}`}
                onClick={() => pick(b.ids, { force: !rowState.all })}
                onKeyDown={keyActivate(() => pick(b.ids, { force: !rowState.all }))}
              >
                <rect x="0" y={b.y - 11} width={o.gutter - 8} height="22" fill="transparent" />
                <text x="2" y={b.y - 2} fontSize="10.5" fill={rowState.hit ? "var(--accent)" : c.dim}>
                  {b.label}
                </text>
                <text x="2" y={b.y + 9} fontSize="10" fill={c.dim}>
                  {b.role || `${b.channels}ch`}
                </text>
              </g>
            </g>
          );
        })}

        {skips.map((d, i) => (
          <path key={`skip-${i}`} d={d} fill="none" stroke={c.line} strokeWidth="1.5"
                strokeDasharray="4 4" opacity="0.3" />
        ))}
        {skipArcs.map(skipArc)}
        {skipArcs.flatMap((sk) => badges([sk.id], sk.mx, sk.my - 12, 1, `skipb-${sk.id}`))}
        {innerPlaced.map((p) => (
          <line key={`hair-${p.id}`} x1={p.px} y1={p.py} x2={p.x} y2={p.y - p.r}
                stroke={c.line} strokeWidth="1" opacity="0.8" />
        ))}

        {density === "layers" && (
          <>
            {spine.jumps.map((d, i) => (
              <path key={`jump-${i}`} d={d} fill="none" stroke={c.line} strokeWidth="1.5"
                    strokeDasharray="3 4" opacity="0.35" />
            ))}
            {spine.runs.map((d, i) => (
              <path key={`run-${i}`} d={d} fill="none" stroke={c.line} strokeWidth="2" />
            ))}
          </>
        )}

        {marquee && (
          <rect className="unet-marquee" x={marquee.x} y={marquee.y} width={marquee.w} height={marquee.h} />
        )}

        {density === "overview" && capsules.filter((cap) => !expanded.has(cap.key)).map(capsule)}
        {density === "overview" && expandedKeys.map((key) => collapser(key, expandedNodes.filter((n) => `${n.depth}:${n.side}` === key)))}
        {(density === "layers" ? placed : expandedNodes).flatMap((n) => [node(n), ...(satsOf.get(n.id) || []).map(satellite)])}
        {sideBadges(density === "layers" ? placed : expandedNodes)}

        {marquee && preview.size > 0 && (
          <g transform={`translate(${marquee.x + marquee.w},${marquee.y + marquee.h})`} pointerEvents="none">
            <rect x="4" y="2" width={preview.size > 9 ? 34 : 28} height="17" rx="8.5" fill="var(--accent)" />
            <text x={preview.size > 9 ? 21 : 18} y="14" textAnchor="middle" fontSize="11" fontWeight="700"
                  fill="#1a1206">
              {drag.remove ? "−" : "+"}{preview.size}
            </text>
          </g>
        )}

        {hover && (() => {
          // What this point is, what bending it touches (inside blocks), what
          // a click will do, and who else is here.
          const also = holdersIn(holders, [hover.id]);
          const mine = focusTargets?.has(hover.id);
          const act = !hasFocus ? "click to start a bend here"
            : mine ? "in this bend · click to drop" : "click to add to this bend";
          const lines = [
            { text: hover.extra === "skip" || hover.extra === "inner" ? hover.label : `${hover.label} · ${hover.type}`, fill: c.text },
            { text: `${hover.channels}ch · ${hover.h}×${hover.w} · ${resLabel(hover)}`, fill: c.dim },
            ...(hover.extra === "skip" ? [{ text: "bends the skip alone, not the path down", fill: c.dim }] : []),
            ...wrapText(hover.about).map((t) => ({ text: t, fill: c.dim })),
            { text: act, fill: mine ? "var(--accent)" : c.text },
            ...(also.length ? [{ text: `also in ${holdersText(also)}`, fill: c.dim }] : []),
          ];
          const cardW = 214;
          const h = 10 + lines.length * 13;
          const top = hover.y - h - 12 < 4 ? hover.y + hover.r + 8 : hover.y - h - 12;
          return (
            <g transform={`translate(${Math.max(4, Math.min(hover.x - cardW / 2, width - cardW - 4))},${top})`}
               pointerEvents="none">
              <rect width={cardW} height={h} rx="5" fill={c.panel} stroke={c.line} />
              {lines.map((ln, k) => (
                <text key={k} x="8" y={15 + k * 13} fontSize="11" fill={ln.fill}>{ln.text}</text>
              ))}
            </g>
          );
        })()}
      </svg>
      <div className="row wrap" style={{ gap: 14, marginTop: 6 }}>
        {["encoder", "mid", "decoder"].map((k) => (
          <span key={k} className="row center" style={{ gap: 6, fontSize: 12, color: "var(--text-dim)" }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: c[k] }} /> {k}
          </span>
        ))}
        <span className="row center" style={{ gap: 6, fontSize: 12, color: "var(--text-dim)" }}>◆ attention</span>
        {skipArcs.length > 0 && (
          <span className="row center" style={{ gap: 6, fontSize: 12, color: "var(--text-dim)" }}>
            <svg width="20" height="9" aria-hidden="true">
              <path d="M1 8 Q10 0 19 8" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3 3" />
            </svg>
            skip — click to bend it alone
          </span>
        )}
        {innerOn && (
          <span className="row center" style={{ gap: 6, fontSize: 12, color: "var(--text-dim)" }}>
            <svg width="20" height="9" aria-hidden="true">
              <circle cx="4" cy="5" r="2.5" fill="currentColor" /><circle cx="10" cy="5" r="2.5" fill="currentColor" />
              <circle cx="16" cy="5" r="2.5" fill="currentColor" />
            </svg>
            inside blocks
          </span>
        )}
        <span className="row center unet-legend-this" style={{ gap: 6, fontSize: 12, color: "var(--text-dim)" }}>
          <span className="unet-legend-ring" /> this bend
        </span>
        <span className="row center" style={{ gap: 6, fontSize: 12, color: "var(--text-dim)" }}>
          <span className="bend-badge sm">1</span> other bends, by number
        </span>
      </div>
    </div>
  );
}
