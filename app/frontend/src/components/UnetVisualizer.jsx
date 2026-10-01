import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { tokens } from "../theme.js";
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
// sweeps a rectangle. Layers held by other bends keep a dimmed ring so the whole
// stack stays legible while you edit one.
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

export default function UnetVisualizer({
  graph, focusTargets, otherTargets, hasFocus, density = "overview",
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
  // output.
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
  }, []);

  const layout = useMemo(
    () => layoutStructured(graph?.nodes, avail ? { available: avail } : undefined),
    [graph, avail],
  );
  const capsules = useMemo(() => (density === "overview" ? collapseBands(layout) : []), [layout, density]);
  const spine = useMemo(() => (density === "layers" ? runPaths(layout?.placed || []) : null), [layout, density]);
  const skips = useMemo(() => skipPaths(layout), [layout]);

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
      const ids = layout.placed
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

  const state = (ids) => {
    const hit = ids.filter((id) => focusTargets?.has(id)).length;
    return { hit, all: hit === ids.length && hit > 0, some: hit > 0 && hit < ids.length };
  };

  const node = (n) => {
    const isFocus = focusTargets?.has(n.id);
    const isOther = !isFocus && otherTargets?.has(n.id);
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
        {shape(n.r, {
          fill, opacity: isFocus ? 1 : 0.72,
          stroke: hover?.id === n.id ? "#fff" : "transparent", strokeWidth: 2,
        })}
        {/* The dots are small now, so the thing you actually click is not. */}
        <circle r={Math.max(n.r, 12)} fill="transparent" />
      </g>
    );
  };

  const capsule = (cap) => {
    const { all, some, hit } = state(cap.ids);
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
          aria-label={`${cap.label} layers — ${all ? "targeted" : some ? "partly targeted" : "not targeted"}`}
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
        {(density === "layers" ? placed : expandedNodes).map(node)}

        {hover && (
          <g
            transform={`translate(${Math.max(4, Math.min(hover.x - 85, width - 174))},${hover.y - 48 < 4 ? hover.y + hover.r + 8 : hover.y - 48})`}
            pointerEvents="none"
          >
            <rect width="170" height="36" rx="5" fill={c.panel} stroke={c.line} />
            <text x="8" y="15" fontSize="11" fill={c.text}>{hover.label} · {hover.type}</text>
            <text x="8" y="28" fontSize="11" fill={c.dim}>
              {hover.channels}ch · {hover.h}×{hover.w} · {resLabel(hover)}
            </text>
          </g>
        )}
      </svg>
      <div className="row wrap" style={{ gap: 14, marginTop: 6 }}>
        {["encoder", "mid", "decoder"].map((k) => (
          <span key={k} className="row center" style={{ gap: 6, fontSize: 12, color: "var(--text-dim)" }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: c[k] }} /> {k}
          </span>
        ))}
        <span className="row center" style={{ gap: 6, fontSize: 12, color: "var(--text-dim)" }}>◆ attention</span>
      </div>
    </div>
  );
}
