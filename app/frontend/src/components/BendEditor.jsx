import React from "react";
import { Slider, Select, DeleteBtn, Tooltip } from "./ui.jsx";
import BendFootprint from "./BendFootprint.jsx";
import { explicitTargets } from "../bendTargets.js";

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
  return param.help || `Parameter ‘${param.label}’ for the ${opDef?.label || "bend"} operation.`;
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

function BendCard({
  b, i, opDef, ops, nodes, labels, focused, note,
  onFocus, update, remove, move, clearTargets, dropTarget,
}) {
  const amountKey = opDef?.amount_param;
  const amountDef = amountKey ? (opDef?.params || []).find((p) => p.name === amountKey) : null;
  const extraParams = (opDef?.params || []).filter((p) => p.name !== amountKey);
  const explicit = explicitTargets(b, nodes);

  return (
    <div className={`bend-card ${b.active ? "" : "dim"} ${focused ? "focused" : "collapsed"}`}>
      <div
        className="bend-card-head"
        role="button"
        tabIndex={0}
        aria-expanded={focused}
        onClick={() => onFocus(b.id)}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onFocus(b.id); } }}
      >
        <div className="bend-card-head-main">
          <label className="row center gap-2" onClick={(e) => e.stopPropagation()}>
            <input type="checkbox" checked={b.active} onChange={(e) => update(b.id, { active: e.target.checked })}
              aria-label="Enable this bend" />
          </label>
          <span className="bend-card-chev">{focused ? "▾" : "▸"}</span>
          <select
            value={b.op}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => {
              const nd = ops.find((o) => o.name === e.target.value);
              update(b.id, { op: e.target.value, params: defaultsFor(nd), ...scheduleFor(nd) });
            }}
            className="grow"
          >
            {ops.map((o) => <option key={o.name} value={o.name}>{o.label}</option>)}
          </select>
          <Tooltip text={opDef?.help || opDef?.label || ""}>
            <span className="pill">{opDef?.category}</span>
          </Tooltip>
          <div className="row gap-1" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="btn ghost sm" onClick={() => move(i, -1)} aria-label="Move up">↑</button>
            <button type="button" className="btn ghost sm" onClick={() => move(i, 1)} aria-label="Move down">↓</button>
            <DeleteBtn onClick={() => remove(b.id)} label="Remove bend" />
          </div>
        </div>
        <BendFootprint nodes={nodes} targets={b.targets} />
      </div>

      {focused && (
        <div className="bend-card-body">
          {opDef?.help && <p className="hint mb-2">{opDef.help}</p>}
          {amountDef && (
            <Slider
              label="Amount"
              value={b.params[amountKey] ?? amountDef.default}
              min={amountDef.min ?? 0}
              max={amountDef.max ?? 1}
              step={amountDef.step ?? (amountDef.kind === "int" ? 1 : 0.01)}
              onChange={(v) => update(b.id, { params: { ...b.params, [amountKey]: v } })}
              tip={amountTip(opDef, amountDef)}
            />
          )}

          <div className="row between center mt-2">
            <div className="section-title mb-0">Targets</div>
            {(b.targets || []).length > 0 && (
              <button type="button" className="btn ghost sm" onClick={() => clearTargets(b.id)}>Clear</button>
            )}
          </div>
          <p className="hint mb-2">
            Set on the map: click a layer, shift+click for a range, drag for a span, or use the group
            buttons above it.
          </p>
          {explicit.length > 0 && (
            <div className="row wrap gap-2 mb-2">
              {explicit.map((t) => (
                <button type="button" key={t} className="pill chip on" onClick={() => dropTarget(b, t)}
                  title={t}>{labels[t] || t} ✕</button>
              ))}
            </div>
          )}
          {note?.length > 0 && (
            <p className="hint mb-2">
              Expanded <b>{note.join(", ")}</b> into individual layers so you could switch one off.
              Click the group again to collapse it back.
            </p>
          )}

          <Slider
            label="Start (run fraction)"
            value={b.step_start}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => update(b.id, { step_start: Math.min(v, b.step_end) })}
            tip="When during sampling this bend turns on (0 = from the start)."
          />
          <Slider
            label="End (run fraction)"
            value={b.step_end}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => update(b.id, { step_end: Math.max(v, b.step_start) })}
            tip="When during sampling this bend turns off (1 = until the last step)."
          />

          {extraParams.length > 0 && (
            <div className="section-title mt-2">More settings</div>
          )}
          {extraParams.map((param) => (
            <ParamControl key={param.name} param={param} opDef={opDef} value={b.params[param.name]}
              onChange={(v) => update(b.id, { params: { ...b.params, [param.name]: v } })} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function BendEditor({
  ops, nodes, stack, setStack, focusedId, setFocusedId, addBend, updateBend, note,
}) {
  const opMap = Object.fromEntries(ops.map((o) => [o.name, o]));
  const labels = Object.fromEntries((nodes || []).map((n) => [n.id, n.label || n.id]));

  const update = updateBend;
  const remove = (id) => {
    const next = stack.filter((b) => b.id !== id);
    setStack(next);
    if (focusedId === id) setFocusedId(next[0]?.id || null);
  };
  const move = (i, dir) => {
    const j = i + dir;
    if (j < 0 || j >= stack.length) return;
    const s = [...stack];
    [s[i], s[j]] = [s[j], s[i]];
    setStack(s);
  };

  const dropTarget = (b, t) => update(b.id, { targets: b.targets.filter((x) => x !== t) });
  const clearTargets = (id) => update(id, { targets: [] });

  return (
    <div className="col bend-editor">
      <div className="row between center">
        <div className="section-title mb-0">
          Bend stack ({stack.length})
        </div>
        <button type="button" className="btn sm primary" onClick={addBend} disabled={!ops.length}>+ Add bend</button>
      </div>
      <p className="hint mb-0">
        Each bend hooks targeted layers and rewrites activations as they pass through. Pick a bend to
        edit it — the map then shows and sets that bend&apos;s layers.
      </p>

      {stack.length === 0 && <div className="empty">No bends yet. Click a layer on the map, or add one here.</div>}

      <div className="bend-stack">
        {stack.map((b, i) => (
          <BendCard
            key={b.id}
            b={b}
            i={i}
            opDef={opMap[b.op]}
            ops={ops}
            nodes={nodes}
            labels={labels}
            focused={focusedId === b.id}
            note={note?.bendId === b.id ? note.groups : null}
            onFocus={setFocusedId}
            update={update}
            remove={remove}
            move={move}
            clearTargets={clearTargets}
            dropTarget={dropTarget}
          />
        ))}
      </div>
    </div>
  );
}
