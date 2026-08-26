import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

/** Hover/focus tooltip rendered into a portal.
 *
 *  It used to be an absolutely-positioned child, which meant any ancestor with
 *  `overflow: auto` clipped it — so the help on bend parameters and model rows,
 *  the controls that need it most, was the most likely to be cut off. Fixed
 *  positioning in a portal escapes every scroll container.
 */
export function Tooltip({ text, children }) {
  const anchor = useRef(null);
  const bubble = useRef(null);
  const [rect, setRect] = useState(null);
  const [pos, setPos] = useState(null);

  const show = useCallback(() => {
    if (anchor.current) setRect(anchor.current.getBoundingClientRect());
  }, []);
  const hide = useCallback(() => { setRect(null); setPos(null); }, []);

  useLayoutEffect(() => {
    if (!rect || !bubble.current) return;
    const b = bubble.current.getBoundingClientRect();
    const m = 8;
    const left = Math.max(m, Math.min(
      rect.left + rect.width / 2 - b.width / 2,
      window.innerWidth - b.width - m,
    ));
    const above = rect.top - b.height - 6;
    setPos({ left, top: above < m ? rect.bottom + 6 : above });
  }, [rect]);

  useEffect(() => {
    if (!rect) return undefined;
    // any scroll slides the anchor out from under the bubble
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    return () => {
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
    };
  }, [rect, hide]);

  if (!text) return children;

  return (
    <>
      <span
        ref={anchor}
        className="tip-wrap"
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
      >
        {children}
      </span>
      {rect && createPortal(
        <span
          ref={bubble}
          className={`tooltip${text.includes("\n") ? " pre-line" : ""}`}
          role="tooltip"
          style={pos ? { left: pos.left, top: pos.top } : { left: 0, top: 0, opacity: 0 }}
        >
          {text}
        </span>,
        document.body,
      )}
    </>
  );
}

export function TipLabel({ children, tip }) {
  if (!tip) return children;
  return (
    <Tooltip text={tip}>
      <span className="has-tip">
        {children}
        <span className="tip-mark" aria-hidden="true">?</span>
      </span>
    </Tooltip>
  );
}

export function Slider({ label, value, min, max, step = 1, onChange, fmt, tip, disabled }) {
  const display = fmt ? fmt(value) : value;
  return (
    <div className={`slider-row ${disabled ? "disabled" : ""}`}>
      <div className="lab">
        <TipLabel tip={tip}><span>{label}</span></TipLabel>
        <span className="val">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-label={label}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

export function Field({ label, children, tip }) {
  return (
    <label className="field">
      <TipLabel tip={tip}><span>{label}</span></TipLabel>
      {children}
    </label>
  );
}

export function Text({ label, value, onChange, placeholder, tip, disabled }) {
  return (
    <Field label={label} tip={tip}>
      <input type="text" value={value} placeholder={placeholder} disabled={disabled} onChange={(e) => onChange(e.target.value)} />
    </Field>
  );
}

export function Num({ label, value, onChange, min, max, step, tip, disabled }) {
  return (
    <Field label={label} tip={tip}>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value === "" ? "" : parseFloat(e.target.value))}
      />
    </Field>
  );
}

// `ariaLabel` is for the selects that sit under a heading instead of a label of
// their own: with label="" there is no text to name the control and no `?` to
// hang a tip on, so screen readers would otherwise announce it as unlabelled.
export function Select({ label, value, onChange, options, tip, disabled, ariaLabel }) {
  const opts = options || [];
  return (
    <Field label={label} tip={tip}>
      <select
        value={value}
        disabled={disabled}
        aria-label={ariaLabel}
        onChange={(e) => onChange(e.target.value)}
      >
        {opts.map((o) => {
          const val = typeof o === "string" ? o : o.value;
          const lab = typeof o === "string" ? o : o.label;
          // `title` gives each option its own hover tooltip in the native list
          const hint = typeof o === "string" ? undefined : o.title;
          return (
            <option key={val} value={val} title={hint}>
              {lab}
            </option>
          );
        })}
      </select>
    </Field>
  );
}

export function Progress({ value }) {
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round((value || 0) * 100)}>
      <div style={{ width: `${Math.round((value || 0) * 100)}%` }} />
    </div>
  );
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>;
}

export function Loading({ children = "Loading…" }) {
  return (
    <div className="empty loading-row">
      <span className="spinner" aria-hidden />
      <span>{children}</span>
    </div>
  );
}

export function Disclose({ title, children, defaultOpen = false, extra, tip, className = "" }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`card ${className}`.trim()}>
      <div
        className="disclose-head"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen((o) => !o); } }}
      >
        <span className="chev" aria-hidden="true">{open ? "▾" : "▸"}</span>
        <h3><TipLabel tip={tip}>{title}</TipLabel></h3>
        {extra}
      </div>
      {open && <div className="disclose-body">{children}</div>}
    </div>
  );
}

function focusables(root) {
  if (!root) return [];
  return [...root.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])")]
    .filter((el) => !el.disabled && el.offsetParent !== null);
}

export function Modal({ title, children, onClose, footer, wide = false }) {
  const box = useRef(null);
  const prev = useRef(typeof document !== "undefined" ? document.activeElement : null);

  useEffect(() => {
    const first = focusables(box.current)[0];
    first?.focus();
    const onKey = (e) => {
      if (e.key === "Escape") { e.preventDefault(); onClose?.(); return; }
      if (e.key !== "Tab") return;
      const list = focusables(box.current);
      if (!list.length) return;
      const firstEl = list[0];
      const lastEl = list[list.length - 1];
      if (e.shiftKey && document.activeElement === firstEl) {
        e.preventDefault();
        lastEl.focus();
      } else if (!e.shiftKey && document.activeElement === lastEl) {
        e.preventDefault();
        firstEl.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      if (prev.current && typeof prev.current.focus === "function") prev.current.focus();
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className={`modal ${wide ? "modal-wide" : ""}`}
        ref={box}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="row between center modal-head">
          <h3 id="modal-title">{title}</h3>
          <button type="button" className="btn ghost sm" onClick={onClose} aria-label="Close">✕</button>
        </div>
        {children}
        {footer && <div className="row modal-foot">{footer}</div>}
      </div>
    </div>
  );
}

export function ConfirmModal({
  title, body, confirmLabel = "Confirm", cancelLabel = "Cancel",
  danger, extra, onCancel, onConfirm,
}) {
  return (
    <Modal
      title={title}
      onClose={onCancel}
      footer={<>
        <button type="button" className="btn ghost" onClick={onCancel}>{cancelLabel}</button>
        <button type="button" className={`btn ${danger ? "danger" : "primary"}`} onClick={onConfirm}>{confirmLabel}</button>
      </>}
    >
      {typeof body === "string" ? <p className="hint mb-0">{body}</p> : body}
      {extra}
    </Modal>
  );
}

export function Chips({ options, selected, onToggle }) {
  return (
    <div className="row wrap gap-2">
      {options.map((o) => (
        <button
          key={o}
          type="button"
          className={`pill chip ${selected.includes(o) ? "on" : ""}`}
          onClick={() => onToggle(o)}
          aria-pressed={selected.includes(o)}
        >
          {o}
        </button>
      ))}
    </div>
  );
}

export function SubNav({ tabs, value, onChange, ariaLabel, className, busy }) {
  const onKeyDown = (e) => {
    const i = tabs.findIndex((t) => t.id === value);
    if (i < 0) return;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const dir = e.key === "ArrowRight" ? 1 : -1;
      onChange(tabs[(i + dir + tabs.length) % tabs.length].id);
    }
  };
  return (
    <div className={`subnav ${className || ""}`} role="tablist" aria-label={ariaLabel} onKeyDown={onKeyDown}>
      {tabs.map((t) => {
        const running = !!busy?.[t.id];
        return (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={value === t.id}
            aria-busy={running || undefined}
            className={`${value === t.id ? "on" : ""}${running ? " busy" : ""}`.trim()}
            onClick={() => onChange(t.id)}
            title={running ? `${t.tip || t.label} — running` : t.tip}
          >
            {t.label}
            {running && <span className="tab-runbar" aria-hidden="true" />}
          </button>
        );
      })}
    </div>
  );
}

export function Seg({ tabs, value, onChange, ariaLabel, className, size, busy }) {
  const onKeyDown = (e) => {
    const i = tabs.findIndex((t) => t.id === value);
    if (i < 0) return;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const dir = e.key === "ArrowRight" ? 1 : -1;
      onChange(tabs[(i + dir + tabs.length) % tabs.length].id);
    }
  };
  return (
    <div
      className={`seg ${size === "sm" ? "seg-sm" : ""} ${className || ""}`.trim()}
      role="tablist"
      aria-label={ariaLabel}
      onKeyDown={onKeyDown}
    >
      {tabs.map((t) => {
        const running = !!busy?.[t.id];
        return (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={value === t.id}
            aria-busy={running || undefined}
            className={`${value === t.id ? "on" : ""}${running ? " busy" : ""}`.trim()}
            onClick={() => onChange(t.id)}
          >
            <Tooltip text={running ? `${t.tip} — something is running here` : t.tip}>
              <span>{t.label}</span>
            </Tooltip>
            {running && <span className="tab-runbar" aria-hidden="true" />}
          </button>
        );
      })}
    </div>
  );
}

export function DeleteBtn({ onClick, label = "Delete" }) {
  return (
    <button type="button" className="btn ghost sm danger" aria-label={label} onClick={onClick}>
      <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
        <path fill="currentColor" d="M5.5 1h5l.7 1H14v1.5H2V2h2.8l.7-1zM3.5 5h9l-.7 9.2A1.5 1.5 0 0 1 10.3 15.5H5.7a1.5 1.5 0 0 1-1.5-1.3L3.5 5z" />
      </svg>
    </button>
  );
}
