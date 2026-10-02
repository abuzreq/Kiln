import React, { useEffect, useRef, useState } from "react";
import { Slider, Select } from "./ui.jsx";
import BendFootprint from "./BendFootprint.jsx";
import RunWindow, { RunWindowMini } from "./RunWindow.jsx";
import { explicitTargets, isGroup, resolveTargets } from "../bendTargets.js";
import { bendAmount } from "../bendSynopsis.js";

function defaultsFor(opDef) {
  const p = {};
  (opDef?.params || []).forEach((param) => { p[param.name] = param.default; });
  return p;
}

function scheduleFor(opDef) {
  const s = opDef?.schedule || { start: 0, end: 1 };
  return {
    step_start: s.start ?? 0,
    step_end: s.end ?? 1,
  };
}

// Help text comes from the op catalog (app/core/craft/ops.py) so the two can
// never drift apart; these are only last-resort fallbacks.
function amountTip(opDef, amountDef) {
  return amountDef?.help
    || opDef?.help
    || `Primary setting for the ${opDef?.label || "bend"} operation.`;
}

function paramTip(param, opDef) {
  return param.help || `Parameter “${param.label}” for the ${opDef?.label || "bend"} operation.`;
}

function ParamControl({ param, opDef, value, onChange }) {
  const tip = paramTip(param, opDef);
  if (param.kind === "select") {
    return <Select label={param.label} value={value} onChange={onChange} options={param.options} tip={tip} />;
  }
  return (
    <Slider
      label={param.label}
      value={value ?? param.default}
      min={param.min ?? 0}
      max={param.max ?? 1}
      step={param.step ?? (param.kind === "int" ? 1 : 0.01)}
      onChange={onChange}
      tip={tip}
    />
  );
}

/** One bend in the stack. It only says what the bend is -- op, amount, where
 *  it hooks in, when it acts -- and picks it; editing happens in the panel
 *  beside the map, which has room the card never did. Its states are the
 *  ones that change what a run does: off, and on but hooked into nothing. */
function BendCard({ b, i, opDef, nodes, focused, dragging, onFocus, onKeyMove, dragProps }) {
  const amount = bendAmount(b, opDef);
  const targets = b.targets || [];
  const hits = resolveTargets(targets, nodes).size;
  // Before the graph loads every bend resolves to nothing; only warn once the
  // layers are known, or when there is genuinely nothing chosen.
  const idle = targets.length === 0 || (nodes?.length > 0 && hits === 0);
  const where = targets.length === 1 && isGroup(targets[0]) ? targets[0] : null;
  const name = opDef?.label || b.op;
  const cls = [
    "bend-card",
    focused && "focused",
    !b.active && "off",
    b.active && idle && "idle",
    dragging && "dragging",
  ].filter(Boolean).join(" ");

  return (
    <div className={cls} draggable {...dragProps}>
      <div
        className="bend-card-head"
        role="button"
        tabIndex={0}
        data-bend-card={b.id}
        aria-pressed={focused}
        aria-label={`Bend ${i + 1}: ${name}${b.active ? "" : ", off"}${idle ? ", no layers" : ""}`}
        aria-keyshortcuts="Alt+ArrowLeft Alt+ArrowRight"
        title="Drag to reorder, or Alt+← / Alt+→"
        onClick={() => onFocus(b.id)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onFocus(b.id); return; }
          if (e.altKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
            e.preventDefault();
            onKeyMove(e.key === "ArrowLeft" ? -1 : 1);
          }
        }}
      >
        <div className="bend-card-head-main">
          <svg className="bend-grip" width="8" height="14" viewBox="0 0 8 14" aria-hidden="true">
            <circle cx="2" cy="2" r="1.2" /><circle cx="6" cy="2" r="1.2" />
            <circle cx="2" cy="7" r="1.2" /><circle cx="6" cy="7" r="1.2" />
            <circle cx="2" cy="12" r="1.2" /><circle cx="6" cy="12" r="1.2" />
          </svg>
          <span className={`bend-badge ${focused ? "on" : ""}`.trim()} aria-hidden="true">{i + 1}</span>
          <span className="bend-card-op grow">
            <b>{name}</b>
            {amount != null && <span className="sub tnum"> {amount}</span>}
          </span>
          {!b.active && <span className="bend-off-tag">off</span>}
        </div>
        {idle && b.active ? (
          <span className="bend-card-warn">No layers yet — no effect</span>
        ) : (
          <BendFootprint nodes={nodes} targets={targets} label={where} />
        )}
        <RunWindowMini start={b.step_start ?? 0} end={b.step_end ?? 1} hot={focused} />
      </div>
    </div>
  );
}

/** The focused bend's settings, beside the map that sets its layers.
 *
 *  `groups` are the named target groups from /craft/ops; `toggleGroup` is the
 *  workspace's, so taking a group here behaves exactly as it did on the chips
 *  that used to sit above the map. */
export function BendInspector({
  b, index, opDef, ops, nodes, groups, note, update, remove, toggleGroup,
}) {
  if (!b) {
    return (
      <div className="bend-inspector empty">
        <p className="sub mb-0">
          Pick a bend in the stack to edit it here, or click the map to start one where you click.
        </p>
      </div>
    );
  }

  const amountKey = opDef?.amount_param;
  const amountDef = amountKey ? (opDef?.params || []).find((p) => p.name === amountKey) : null;
  const extraParams = (opDef?.params || []).filter((p) => p.name !== amountKey);
  const labels = Object.fromEntries((nodes || []).map((n) => [n.id, n.label || n.id]));
  const explicit = explicitTargets(b, nodes);
  const targets = b.targets || [];
  const hit = resolveTargets(targets, nodes).size;
  const setParam = (name, v) => update(b.id, { params: { ...b.params, [name]: v } });

  return (
    <div className="bend-inspector" aria-label={`Bend ${index + 1} settings`}>
      <div className="bend-inspector-head">
        <span className="bend-badge on" aria-hidden="true">{index + 1}</span>
        <select
          className="grow"
          value={b.op}
          aria-label="Operation"
          onChange={(e) => {
            const nd = ops.find((o) => o.name === e.target.value);
            update(b.id, { op: e.target.value, params: defaultsFor(nd), ...scheduleFor(nd) });
          }}
        >
          {ops.map((o) => <option key={o.name} value={o.name}>{o.label}</option>)}
        </select>
        <button
          type="button"
          role="switch"
          aria-checked={b.active}
          aria-label={`Bend ${index + 1} on`}
          title={b.active ? "On — click to switch this bend off" : "Off — click to switch it on"}
          className={`switch ${b.active ? "on" : ""}`.trim()}
          onClick={() => update(b.id, { active: !b.active })}
        >
          <span />
        </button>
      </div>
      {opDef?.help && (
        <p className="sub bend-inspector-help">
          {opDef.category && <span className="bend-cat">{opDef.category}</span>}
          {opDef.help}
        </p>
      )}

      {amountDef && (
        <Slider
          label="Amount"
          value={b.params[amountKey] ?? amountDef.default}
          min={amountDef.min ?? 0}
          max={amountDef.max ?? 1}
          step={amountDef.step ?? (amountDef.kind === "int" ? 1 : 0.01)}
          onChange={(v) => setParam(amountKey, v)}
          tip={amountTip(opDef, amountDef)}
        />
      )}
      {extraParams.map((param) => (
        <ParamControl key={param.name} param={param} opDef={opDef} value={b.params[param.name]}
          onChange={(v) => setParam(param.name, v)} />
      ))}

      <div className="bend-inspector-section">
        <div className="row between center">
          <div className="section-title mb-0">Where</div>
          <div className="row center gap-2">
            <span className="sub tnum">{hit} / {nodes?.length || 0} layers</span>
            {targets.length > 0 && (
              <button type="button" className="btn ghost xs" onClick={() => update(b.id, { targets: [] })}>
                Clear
              </button>
            )}
          </div>
        </div>
        {targets.length === 0 && (
          <p className="callout mb-0">
            No layers chosen yet — this bend has no effect until you pick at least one. Click the
            map, or take a group below.
          </p>
        )}
        {explicit.length > 0 && (
          <div className="row wrap gap-1">
            {explicit.map((t) => (
              <button type="button" key={t} className="pill chip on" title={t}
                onClick={() => update(b.id, { targets: targets.filter((x) => x !== t) })}>
                {labels[t] || t} ✕
              </button>
            ))}
          </div>
        )}
        {note?.length > 0 && (
          <p className="sub mb-0">
            Expanded <b>{note.join(", ")}</b> into single layers so you could switch one off.
            Take the group again to fold it back.
          </p>
        )}
        {groups?.length > 0 && (
          <div className="row wrap center gap-1">
            <span className="sub">Take</span>
            {groups.map((g) => (
              <button type="button" key={g}
                className={`pill chip ${targets.includes(g) ? "on" : ""}`.trim()}
                aria-pressed={targets.includes(g)}
                onClick={() => toggleGroup(g)}>{g}</button>
            ))}
          </div>
        )}
      </div>

      <RunWindow
        start={b.step_start ?? 0}
        end={b.step_end ?? 1}
        onChange={(patch) => update(b.id, patch)}
      />

      <div className="bend-inspector-foot">
        <button type="button" className="btn sm danger ghost" onClick={() => remove(b.id)}>
          Remove bend
        </button>
      </div>
    </div>
  );
}

export default function BendEditor({
  ops, nodes, stack, setStack, focusedId, setFocusedId, addBend,
  headExtra, beforeStack,
}) {
  const opMap = Object.fromEntries(ops.map((o) => [o.name, o]));
  // { from, at }: the card being dragged and the gap it would land in, as an
  // insertion index (0 = before the first card, stack.length = after the last).
  const [drag, setDrag] = useState(null);
  const refocus = useRef(null);

  // Reordering moves the card's DOM node, which drops keyboard focus; put it
  // back so Alt+arrow can be pressed again and again.
  useEffect(() => {
    if (!refocus.current) return;
    document.querySelector(`[data-bend-card="${refocus.current}"]`)?.focus();
    refocus.current = null;
  });

  const moveTo = (from, at) => {
    const to = at > from ? at - 1 : at;
    if (to === from || to < 0 || to >= stack.length) return;
    const s = [...stack];
    const [b] = s.splice(from, 1);
    s.splice(to, 0, b);
    setStack(s);
  };
  // A drop that would put the card back where it is shows no line.
  const lands = (k) => drag && drag.at === k && k !== drag.from && k !== drag.from + 1;

  const dragProps = (b, i) => ({
    onDragStart: (e) => {
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", b.id);
      setDrag({ from: i, at: null });
    },
    onDragOver: (e) => {
      if (!drag) return;
      e.preventDefault();
      const r = e.currentTarget.getBoundingClientRect();
      const at = i + (e.clientX > r.left + r.width / 2 ? 1 : 0);
      if (at !== drag.at) setDrag({ ...drag, at });
    },
    onDrop: (e) => {
      if (!drag) return;
      e.preventDefault();
      if (drag.at != null) moveTo(drag.from, drag.at);
      setDrag(null);
    },
    onDragEnd: () => setDrag(null),
  });

  const slot = (k) => {
    const edge = k === 0 || k === stack.length;
    return (
      <span key={`slot-${k}`} className={`bend-join ${edge ? "edge" : ""} ${lands(k) ? "drop" : ""}`.trim()}
        aria-hidden="true">
        {!edge && "›"}
      </span>
    );
  };

  return (
    <div className="col bend-editor">
      <div className="row between center">
        <div className="section-title mb-0">
          Bend stack ({stack.length})
        </div>
        <div className="row center gap-2">
          {headExtra}
        </div>
      </div>
      <p className="hint mb-0">
        Applied left to right — drag a card to reorder. Pick one to edit it beside the map.
      </p>

      {beforeStack || null}

      <div className="bend-stack">
        {stack.length === 0 && (
          <div className="empty bend-stack-empty">No bends yet. Click the map to start one, or add one here.</div>
        )}
        {stack.map((b, i) => (
          <React.Fragment key={b.id}>
            {slot(i)}
            <BendCard
              b={b}
              i={i}
              opDef={opMap[b.op]}
              nodes={nodes}
              focused={focusedId === b.id}
              dragging={drag?.from === i}
              onFocus={setFocusedId}
              onKeyMove={(dir) => { refocus.current = b.id; moveTo(i, dir < 0 ? i - 1 : i + 2); }}
              dragProps={dragProps(b, i)}
            />
          </React.Fragment>
        ))}
        {stack.length > 0 && slot(stack.length)}
        <button type="button" className="bend-add-tile" onClick={addBend} disabled={!ops.length}>
          <span aria-hidden="true">+</span>
          Add bend
        </button>
      </div>
    </div>
  );
}
