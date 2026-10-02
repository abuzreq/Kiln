import React, { useEffect, useId, useMemo, useRef, useState } from "react";

// Choosing what a bend does. A flat <select> of sixteen names said nothing
// about any of them; this groups them by what they do (the catalog's
// `category`: value, stochastic, spatial, morph, edges) and shows the op's own
// help for whichever one you are on, before you switch to it. Switching resets
// the bend's params and window, so reading first is worth the space.

const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;

export default function OpPicker({ ops, value, onChange }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(value);
  const wrap = useRef(null);
  const btn = useRef(null);
  const list = useRef(null);
  const uid = useId();
  const optId = (name) => `${uid}-op-${name}`;

  const groups = useMemo(() => {
    const order = [];
    const by = {};
    (ops || []).forEach((o) => {
      const c = o.category || "other";
      if (!by[c]) { by[c] = []; order.push(c); }
      by[c].push(o);
    });
    return order.map((cat) => ({ cat, ops: by[cat] }));
  }, [ops]);
  const flat = groups.flatMap((g) => g.ops);
  const current = flat.find((o) => o.name === value);
  const shown = flat.find((o) => o.name === active) || current;

  useEffect(() => {
    if (!open) return undefined;
    setActive(value);
    list.current?.focus();
    const onDown = (e) => { if (!wrap.current?.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, value]);

  // Keep the active option in view by scrolling the list alone. scrollIntoView
  // would scroll the page too, sliding another option under a resting pointer,
  // whose hover then steals the keyboard's place.
  useEffect(() => {
    const box = list.current;
    const el = open && box?.querySelector(`[data-op="${active}"]`);
    if (!el) return;
    const top = el.offsetTop - box.offsetTop;
    if (top < box.scrollTop) box.scrollTop = top;
    else if (top + el.offsetHeight > box.scrollTop + box.clientHeight) {
      box.scrollTop = top + el.offsetHeight - box.clientHeight;
    }
  }, [open, active]);

  const close = () => { setOpen(false); btn.current?.focus(); };
  const choose = (name) => {
    close();
    if (name !== value) onChange(name);
  };

  const onListKey = (e) => {
    const i = flat.findIndex((o) => o.name === active);
    const go = (j) => { e.preventDefault(); setActive(flat[(j + flat.length) % flat.length].name); };
    if (e.key === "ArrowDown") go(i + 1);
    else if (e.key === "ArrowUp") go(i - 1);
    else if (e.key === "Home") go(0);
    else if (e.key === "End") go(flat.length - 1);
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(active); }
    else if (e.key === "Escape") { e.preventDefault(); close(); }
    else if (e.key === "Tab") setOpen(false);
  };

  return (
    <div className="op-picker" ref={wrap}>
      <button
        ref={btn}
        type="button"
        className="op-picker-btn"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`Operation: ${current?.label || value}`}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); setOpen(true); }
        }}
      >
        <span className="op-picker-name">{current?.label || value}</span>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"
             strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div className="op-picker-pop">
          <div
            ref={list}
            role="listbox"
            tabIndex={-1}
            aria-label="Operation"
            aria-activedescendant={optId(active)}
            className="op-picker-list"
            onKeyDown={onListKey}
          >
            {groups.map((g) => (
              <div key={g.cat} role="group" aria-labelledby={`${uid}-cat-${g.cat}`} className="op-picker-group">
                <span id={`${uid}-cat-${g.cat}`} className="op-picker-cat">{g.cat}</span>
                <span className="op-picker-opts">
                  {g.ops.map((o) => (
                    <span
                      key={o.name}
                      id={optId(o.name)}
                      role="option"
                      data-op={o.name}
                      aria-selected={o.name === value}
                      className={`op-picker-opt ${o.name === active ? "active" : ""} ${o.name === value ? "sel" : ""}`.trim()}
                      onMouseMove={() => { if (active !== o.name) setActive(o.name); }}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => choose(o.name)}
                    >
                      {o.label}
                    </span>
                  ))}
                </span>
              </div>
            ))}
          </div>
          {shown && (
            <p className="op-picker-help" aria-live="polite">
              <b>{shown.label}</b> — {shown.help}
              {shown.schedule && (
                <span className="sub"> Acts {pct(shown.schedule.start)} → {pct(shown.schedule.end)} of the run by default.</span>
              )}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
