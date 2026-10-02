import React, { useRef } from "react";

// A bar with two handles for a span, lo to hi, on [min, max]. Shared by the run
// window (when a bend acts) and by op settings that are a span of values
// (clamp's min and max): two sliders that clamp each other hide the very thing
// the pair means, which is the space between them.
//
// Drag either handle; press the bar to bring the nearer handle there; arrows,
// page keys and home/end on a focused handle. Values snap to `step`.

function decimals(step) {
  const s = String(step);
  return s.includes(".") ? s.split(".")[1].length : 0;
}

export default function DualRange({
  lo, hi, min = 0, max = 1, step = 0.05, bigStep, onChange,
  loLabel = "From", hiLabel = "To", format = (v) => String(v), className = "",
}) {
  const track = useRef(null);
  // Which handle a pointer is moving. "either" while the two sit on one spot
  // and the pointer has not yet said which way it is going.
  const drag = useRef(null);
  const span = max - min || 1;
  const big = bigStep ?? span / 4;
  const frac = (v) => (v - min) / span;
  // toFixed: 3 * 0.05 is 0.15000000000000002, and that ends up in saved stacks.
  const snap = (v) => {
    const c = Math.min(max, Math.max(min, v));
    return Number((min + Math.round((c - min) / step) * step).toFixed(decimals(step)));
  };

  const at = (clientX) => {
    const r = track.current.getBoundingClientRect();
    return snap(min + ((clientX - r.left) / (r.width || 1)) * span);
  };
  const set = (which, v) => {
    if (which === "lo") onChange(Math.min(v, hi), hi);
    else onChange(lo, Math.max(v, lo));
  };

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    const v = at(e.clientX);
    let which = e.target.dataset?.which;
    if (which && lo === hi) {
      // Stacked handles: only one of them can move at either end of the bar.
      which = hi >= max ? "lo" : lo <= min ? "hi" : "either";
    } else if (!which) {
      which = v <= lo ? "lo" : v >= hi ? "hi" : (v - lo <= hi - v ? "lo" : "hi");
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
      if (v === lo) return;
      drag.current = v < lo ? "lo" : "hi";
    }
    set(drag.current, v);
  };
  const release = () => { drag.current = null; };

  const onKey = (which) => (e) => {
    const cur = which === "lo" ? lo : hi;
    const floor = which === "lo" ? min : lo;
    const ceil = which === "lo" ? hi : max;
    const next = {
      ArrowLeft: cur - step, ArrowDown: cur - step,
      ArrowRight: cur + step, ArrowUp: cur + step,
      PageDown: cur - big, PageUp: cur + big,
      Home: floor, End: ceil,
    }[e.key];
    if (next == null) return;
    e.preventDefault();
    set(which, snap(Math.min(ceil, Math.max(floor, next))));
  };

  const handle = (which, value, label) => (
    <span
      role="slider"
      tabIndex={0}
      className={`dual-range-handle ${which}`}
      data-which={which}
      aria-label={label}
      aria-valuemin={which === "lo" ? min : lo}
      aria-valuemax={which === "lo" ? hi : max}
      aria-valuenow={value}
      aria-valuetext={format(value)}
      style={{ left: `${frac(value) * 100}%` }}
      onKeyDown={onKey(which)}
    />
  );

  return (
    <div
      className={`dual-range ${className}`.trim()}
      ref={track}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={release}
      onPointerCancel={release}
    >
      <span className="dual-range-fill" style={{ left: `${frac(lo) * 100}%`, width: `${(frac(hi) - frac(lo)) * 100}%` }} />
      {handle("lo", lo, loLabel)}
      {handle("hi", hi, hiLabel)}
    </div>
  );
}
