import React, { useRef } from "react";

// When during sampling a bend acts, as one span of the run instead of two
// sliders that clamp each other. Noise is on the left, the finished image on
// the right, so the bar reads the way the image forms. The op catalog gives
// each category its own default window (ops.py SCHEDULE_DEFAULTS), which is
// the case for showing it as a window at all.
const STEP = 0.05;
const BIG = 0.25;
const SHORTCUTS = [
  { id: "whole", label: "Whole run", start: 0, end: 1 },
  { id: "early", label: "Early", start: 0, end: 0.5 },
  { id: "late", label: "Late", start: 0.5, end: 1 },
];

const clamp01 = (v) => Math.min(1, Math.max(0, v));
// toFixed: 3 * 0.05 is 0.15000000000000002, and that ends up in saved stacks.
const snap = (v) => Number((Math.round(clamp01(v) / STEP) * STEP).toFixed(2));
const pct = (v) => `${Math.round(v * 100)}%`;

export default function RunWindow({ start, end, onChange }) {
  const track = useRef(null);
  // Which handle a pointer is moving. "either" while two handles sit on one
  // spot and the pointer has not yet said which way it is going.
  const drag = useRef(null);

  const at = (clientX) => {
    const r = track.current.getBoundingClientRect();
    return snap((clientX - r.left) / (r.width || 1));
  };
  const set = (which, v) => {
    if (which === "start") onChange({ step_start: Math.min(v, end), step_end: end });
    else onChange({ step_start: start, step_end: Math.max(v, start) });
  };

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    const v = at(e.clientX);
    let which = e.target.dataset?.which;
    if (which && start === end) {
      // Stacked handles: only one of them can move at either end of the bar.
      which = end >= 1 ? "start" : start <= 0 ? "end" : "either";
    } else if (!which) {
      // A press on the bar itself moves the nearer handle there.
      which = v <= start ? "start" : v >= end ? "end"
        : (v - start <= end - v ? "start" : "end");
      set(which, v);
    }
    drag.current = which;
    track.current.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    const v = at(e.clientX);
    if (drag.current === "either") {
      if (v === start) return;
      drag.current = v < start ? "start" : "end";
    }
    set(drag.current, v);
  };
  const release = () => { drag.current = null; };

  const onKey = (which) => (e) => {
    const cur = which === "start" ? start : end;
    const lo = which === "start" ? 0 : start;
    const hi = which === "start" ? end : 1;
    const next = {
      ArrowLeft: cur - STEP, ArrowDown: cur - STEP,
      ArrowRight: cur + STEP, ArrowUp: cur + STEP,
      PageDown: cur - BIG, PageUp: cur + BIG,
      Home: lo, End: hi,
    }[e.key];
    if (next == null) return;
    e.preventDefault();
    set(which, snap(Math.min(hi, Math.max(lo, next))));
  };

  const handle = (which, value, label) => (
    <span
      role="slider"
      tabIndex={0}
      className={`run-window-handle ${which}`}
      data-which={which}
      aria-label={label}
      aria-valuemin={which === "start" ? 0 : Math.round(start * 100)}
      aria-valuemax={which === "start" ? Math.round(end * 100) : 100}
      aria-valuenow={Math.round(value * 100)}
      aria-valuetext={`${pct(value)} of the run`}
      style={{ left: `${value * 100}%` }}
      onKeyDown={onKey(which)}
    />
  );

  return (
    <div className="run-window">
      <div className="row between center">
        <div className="section-title mb-0">When in the run</div>
        <span className="run-window-val tnum">{pct(start)} → {pct(end)}</span>
      </div>
      <div
        className="run-window-track"
        ref={track}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={release}
        onPointerCancel={release}
      >
        <span className="run-window-fill" style={{ left: `${start * 100}%`, width: `${(end - start) * 100}%` }} />
        {handle("start", start, "Turns on at")}
        {handle("end", end, "Turns off at")}
      </div>
      <div className="run-window-ends sub">
        <span>noise · first step</span>
        <span>image · last step</span>
      </div>
      <div className="run-window-shortcuts" role="group" aria-label="Common windows">
        {SHORTCUTS.map((s) => (
          <button
            key={s.id}
            type="button"
            className={`btn xs ${s.start === start && s.end === end ? "on" : ""}`.trim()}
            aria-pressed={s.start === start && s.end === end}
            onClick={() => onChange({ step_start: s.start, step_end: s.end })}
          >
            {s.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** The same window at card size: a thin bar with the active span lit. */
export function RunWindowMini({ start, end, hot }) {
  return (
    <span
      className={`run-mini ${hot ? "hot" : ""}`.trim()}
      title={`Acts from ${pct(start)} to ${pct(end)} of the run`}
    >
      <span style={{ left: `${start * 100}%`, width: `${Math.max(0, end - start) * 100}%` }} />
    </span>
  );
}
