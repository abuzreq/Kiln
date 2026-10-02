import React from "react";
import { Slider, Select, DeleteBtn } from "./ui.jsx";
import BendFootprint from "./BendFootprint.jsx";
import RunWindow, { RunWindowMini } from "./RunWindow.jsx";
import { explicitTargets, resolveTargets } from "../bendTargets.js";
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
 *  beside the map, which has room the card never did. */
function BendCard({ b, i, opDef, nodes, focused, onFocus, update, remove, move }) {
  const amount = bendAmount(b, opDef);
  return (
    <div className={`bend-card ${b.active ? "" : "dim"} ${focused ? "focused" : ""}`.trim()}>
      <div
        className="bend-card-head"
        role="button"
        tabIndex={0}
        aria-pressed={focused}
        aria-label={`Bend ${i + 1}: ${opDef?.label || b.op}`}
        onClick={() => onFocus(b.id)}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onFocus(b.id); } }}
      >
        <div className="bend-card-head-main">
          <label className="row center" onClick={(e) => e.stopPropagation()}>
            <input type="checkbox" checked={b.active} onChange={(e) => update(b.id, { active: e.target.checked })}
              aria-label={`Bend ${i + 1} on`} />
          </label>
          <span className={`bend-badge ${focused ? "on" : ""}`.trim()} aria-hidden="true">{i + 1}</span>
          <span className="bend-card-op grow">
            <b>{opDef?.label || b.op}</b>
            {amount != null && <span className="sub tnum"> {amount}</span>}
          </span>
          <div className="row gap-1" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="btn ghost sm" onClick={() => move(i, -1)} aria-label="Move up">↑</button>
            <button type="button" className="btn ghost sm" onClick={() => move(i, 1)} aria-label="Move down">↓</button>
            <DeleteBtn onClick={() => remove(b.id)} label="Remove bend" />
          </div>
        </div>
        <BendFootprint nodes={nodes} targets={b.targets} />
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
  ops, nodes, stack, setStack, focusedId, setFocusedId, addBend, updateBend, removeBend,
  headExtra, beforeStack,
}) {
  const opMap = Object.fromEntries(ops.map((o) => [o.name, o]));
  const move = (i, dir) => {
    const j = i + dir;
    if (j < 0 || j >= stack.length) return;
    const s = [...stack];
    [s[i], s[j]] = [s[j], s[i]];
    setStack(s);
  };

  return (
    <div className="col bend-editor">
      <div className="row between center">
        <div className="section-title mb-0">
          Bend stack ({stack.length})
        </div>
        <div className="row center gap-2">
          {headExtra}
          <button type="button" className="btn sm primary" onClick={addBend} disabled={!ops.length}>+ Add bend</button>
        </div>
      </div>
      <p className="hint mb-0">
        Applied left to right. Pick a bend to edit it beside the map.
      </p>

      {beforeStack || null}

      {stack.length === 0 && <div className="empty">No bends yet. Click a layer on the map, or add one here.</div>}

      <div className="bend-stack">
        {stack.map((b, i) => (
          <BendCard
            key={b.id}
            b={b}
            i={i}
            opDef={opMap[b.op]}
            nodes={nodes}
            focused={focusedId === b.id}
            onFocus={setFocusedId}
            update={updateBend}
            remove={removeBend}
            move={move}
          />
        ))}
      </div>
    </div>
  );
}
