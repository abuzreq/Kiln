import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { tokens } from "../theme.js";
import { layoutNodes, spinePath, resLabel } from "./unetLayout.js";

// U-shaped UNet layer graph, and the primary way you pick layers to bend.
// Clicking a node toggles it in the FOCUSED bend's targets; shift+click takes a
// range and dragging the background sweeps a span. Layers held by other bends in
// the stack keep a dimmed ring so the whole stack stays legible while you edit one.
const STAGE_VARS = {
  encoder: "--stage-encoder",
  mid: "--stage-mid",
  decoder: "--stage-decoder",
  other: "--stage-other",
};

const DRAG_SLOP = 4;

export default function UnetVisualizer({
  graph, focusTargets, otherTargets, hasFocus, onToggle, onCreateFromNode,
}) {
  const [hover, setHover] = useState(null);
  const [drag, setDrag] = useState(null);
  const anchor = useRef(null);
  const svgRef = useRef(null);
  // Live drag geometry and the current handler live in refs so the window
  // listeners are attached once per drag instead of re-bound on every mousemove.
  const dragRef = useRef(null);
  const toggleRef = useRef(onToggle);
  toggleRef.current = onToggle;
  const c = useMemo(
    () => tokens({ ...STAGE_VARS, line: "--line", panel: "--bg-3", text: "--text", dim: "--text-dim" }),
    [],
  );
  const layout = useMemo(() => layoutNodes(graph?.nodes), [graph]);

  const localX = useCallback((clientX) => {
    const box = svgRef.current?.getBoundingClientRect();
    return box ? clientX - box.left : 0;
  }, []);

  // The marquee finishes on window mouseup so a release outside the svg still lands.
  const dragging = !!drag;
  useEffect(() => {
    if (!dragging) return undefined;
    const move = (e) => {
      const x1 = localX(e.clientX);
      if (dragRef.current) dragRef.current.x1 = x1;
      setDrag((d) => (d ? { ...d, x1 } : d));
    };
    const up = (e) => {
      const d = dragRef.current;
      dragRef.current = null;
      setDrag(null);
      if (!d || !layout) return;
      const x1 = localX(e.clientX);
      if (Math.abs(x1 - d.x0) < DRAG_SLOP) return;
      const [lo, hi] = x1 < d.x0 ? [x1, d.x0] : [d.x0, x1];
      const ids = layout.placed.filter((n) => n.x >= lo && n.x <= hi).map((n) => n.id);
      if (ids.length) toggleRef.current?.(ids, { force: !d.remove });
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [dragging, layout, localX]);

  const clickNode = useCallback((n, e) => {
    if (!hasFocus) { onCreateFromNode?.(n.id); anchor.current = n.i; return; }
    const force = !focusTargets?.has(n.id);
    if (e.shiftKey && anchor.current != null && layout) {
      const [lo, hi] = anchor.current < n.i ? [anchor.current, n.i] : [n.i, anchor.current];
      onToggle(layout.placed.slice(lo, hi + 1).map((x) => x.id), { force });
    } else {
      onToggle([n.id], { force });
    }
    anchor.current = n.i;
  }, [hasFocus, focusTargets, layout, onToggle, onCreateFromNode]);

  const nodeKey = useCallback((n, e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); clickNode(n, e); return; }
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    svgRef.current?.querySelector(`[data-node-index="${n.i + step}"]`)?.focus();
  }, [clickNode]);

  if (!layout) return <div className="sub" style={{ padding: 20 }}>Select a model to introspect its layers.</div>;
  const { placed, width, height } = layout;
  const band = drag && Math.abs(drag.x1 - drag.x0) >= DRAG_SLOP
    ? { x: Math.min(drag.x0, drag.x1), w: Math.abs(drag.x1 - drag.x0) }
    : null;

  return (
    <div className="scroll-x">
      <svg
        ref={svgRef}
        width={Math.max(width, 300)}
        height={height}
        className={`unet-map ${drag ? "dragging" : ""}`}
        style={{ display: "block" }}
        onMouseDown={(e) => {
          if (!hasFocus || e.button !== 0) return;
          const x = localX(e.clientX);
          dragRef.current = { x0: x, x1: x, remove: e.altKey };
          setDrag({ ...dragRef.current });
        }}
      >
        <path d={spinePath(placed)} fill="none" stroke={c.line} strokeWidth="2" />
        {band && <rect className="unet-marquee" x={band.x} y="0" width={band.w} height={height} />}
        {placed.map((n) => {
          const isFocus = focusTargets?.has(n.id);
          const isOther = !isFocus && otherTargets?.has(n.id);
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
              {isFocus && <circle r={n.r + 5} fill="none" stroke="var(--accent)" strokeWidth="2.5" />}
              {isOther && (
                <circle r={n.r + 5} fill="none" stroke="var(--accent)" strokeWidth="1.5"
                        strokeDasharray="3 3" opacity="0.45" />
              )}
              <circle r={n.r} fill={c[n.stage] || c.other} opacity={isFocus ? 1 : 0.72}
                      stroke={hover?.id === n.id ? "#fff" : "transparent"} strokeWidth="2" />
              <text y={n.r + 13} textAnchor="middle" fontSize="10" fill={c.dim}>{n.channels}</text>
            </g>
          );
        })}
        {hover && (
          <g transform={`translate(${Math.min(hover.x, width - 170)},${hover.y - 48})`} pointerEvents="none">
            <rect width="170" height="36" rx="5" fill={c.panel} stroke={c.line} />
            <text x="8" y="15" fontSize="11" fill={c.text}>{hover.label} · {hover.type}</text>
            <text x="8" y="28" fontSize="11" fill={c.dim}>
              {hover.channels}ch · {hover.h}×{hover.w} · {resLabel(hover)}
            </text>
          </g>
        )}
      </svg>
      <div className="row" style={{ gap: 14, marginTop: 6 }}>
        {["encoder", "mid", "decoder"].map((k) => (
          <span key={k} className="row center" style={{ gap: 6, fontSize: 12, color: "var(--text-dim)" }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: c[k] }} /> {k}
          </span>
        ))}
      </div>
    </div>
  );
}
