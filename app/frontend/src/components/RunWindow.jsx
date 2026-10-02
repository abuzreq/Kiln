import React from "react";
import DualRange from "./DualRange.jsx";

// When during sampling a bend acts, as one span of the run instead of two
// sliders that clamp each other. Noise is on the left, the finished image on
// the right, so the bar reads the way the image forms. The op catalog gives
// each category its own default window (ops.py SCHEDULE_DEFAULTS), which is
// the case for showing it as a window at all.
const SHORTCUTS = [
  { id: "whole", label: "Whole run", start: 0, end: 1 },
  { id: "early", label: "Early", start: 0, end: 0.5 },
  { id: "late", label: "Late", start: 0.5, end: 1 },
];

const pct = (v) => `${Math.round(v * 100)}%`;

export default function RunWindow({ start, end, onChange }) {
  return (
    <div className="run-window">
      <div className="row between center">
        <div className="section-title mb-0">When in the run</div>
        <span className="run-window-val tnum">{pct(start)} → {pct(end)}</span>
      </div>
      <DualRange
        className="run-window-track"
        lo={start}
        hi={end}
        min={0}
        max={1}
        step={0.05}
        bigStep={0.25}
        loLabel="Turns on at"
        hiLabel="Turns off at"
        format={(v) => `${pct(v)} of the run`}
        onChange={(a, b) => onChange({ step_start: a, step_end: b })}
      />
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
