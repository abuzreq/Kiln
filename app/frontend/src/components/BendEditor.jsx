import React, { useEffect, useRef, useState } from "react";
import { Slider, Select, TipLabel } from "./ui.jsx";
import BendFootprint from "./BendFootprint.jsx";
import RunWindow, { RunWindowMini } from "./RunWindow.jsx";
import DualRange from "./DualRange.jsx";
import OpPicker from "./OpPicker.jsx";
import { explicitTargets, isGroup, resolveTargets } from "../bendTargets.js";
import { depthOf } from "./unetLayout.js";
import { bendAmount } from "../bendSynopsis.js";
import { DiceIcon, UndoIcon } from "./icons.jsx";

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

// Params that mean one thing between them, shown as one control: an offset is
// an X and a Y, and clamp's bounds are a span whose width is the whole point.
const PAIRS = [
  { keys: ["dx", "dy"], kind: "xy" },
  { keys: ["min", "max"], kind: "span" },
];

const decimalsOf = (step) => (String(step).split(".")[1] || "").length;

/** A number box that only commits whole, in-range values: half-typed text
 *  ("-", "1.") stays in the box instead of landing in the bend's params. */
function NumField({ label, value, def, onChange }) {
  const [draft, setDraft] = useState(null);
  const step = def.step ?? (def.kind === "int" ? 1 : 0.01);
  const commit = (text) => {
    const v = parseFloat(text);
    if (!Number.isFinite(v)) return;
    const c = Math.min(def.max ?? Infinity, Math.max(def.min ?? -Infinity, v));
    onChange(def.kind === "int" ? Math.round(c) : Number(c.toFixed(decimalsOf(step))));
  };
  return (
    <label className="num-field">
      <span>{label}</span>
      <input
        type="number"
        inputMode="decimal"
        value={draft ?? value ?? def.default}
        min={def.min}
        max={def.max}
        step={step}
        onChange={(e) => { setDraft(e.target.value); commit(e.target.value); }}
        onBlur={() => setDraft(null)}
      />
    </label>
  );
}

/** Two params as one control (see PAIRS). */
function PairControl({ kind, a, b, opDef, params, setParam, setParams }) {
  const va = params[a.name] ?? a.default;
  const vb = params[b.name] ?? b.default;
  if (kind === "xy") {
    // "Shift X" and "Shift Y" are one "Shift"
    const label = a.label.replace(/\s*X$/i, "") || "Offset";
    return (
      <div className="param-pair">
        <div className="row between center">
          <TipLabel tip={opDef?.help}><span className="param-pair-label">{label}</span></TipLabel>
          <span className="sub tnum">{a.min} to {a.max} feature px</span>
        </div>
        <div className="param-xy">
          <NumField label="X" value={va} def={a} onChange={(v) => setParam(a.name, v)} />
          <NumField label="Y" value={vb} def={b} onChange={(v) => setParam(b.name, v)} />
        </div>
      </div>
    );
  }
  const step = a.step ?? 0.1;
  const fmt = (v) => v.toFixed(decimalsOf(step));
  return (
    <div className="param-pair">
      <div className="row between center">
        <TipLabel tip={`${a.help || ""} ${b.help || ""}`.trim() || opDef?.help}>
          <span className="param-pair-label">Keep between</span>
        </TipLabel>
        <span className="sub tnum">{fmt(va)} … {fmt(vb)}</span>
      </div>
      <DualRange
        lo={va}
        hi={vb}
        min={a.min ?? -5}
        max={b.max ?? 5}
        step={step}
        loLabel={a.label}
        hiLabel={b.label}
        format={fmt}
        onChange={(lo, hi) => setParams({ [a.name]: lo, [b.name]: hi })}
      />
      <div className="row between sub param-pair-ends">
        <span>{a.min ?? -5}</span>
        <span>narrower: flatter, harder bands</span>
        <span>{b.max ?? 5}</span>
      </div>
    </div>
  );
}

// Past this many single layers, Where sums them up by stage instead: eighteen
// chips ("the whole network minus one layer per stage") buried the panel.
const LIST_EACH_MAX = 6;
const STAGE_NAME = { encoder: "encoder", mid: "bottleneck", decoder: "decoder" };

/** The level a layer sits on, named as the Levels map names its capsules. */
function levelOf(n) {
  if (n.extra === "skip") return { key: "skip", label: "skips" };
  if (n.stage === "mid") return { key: "mid", label: "bottleneck" };
  const side = n.stage === "encoder" ? "enc" : n.stage === "decoder" ? "dec" : n.stage;
  return { key: `${n.stage}:${depthOf(n)}`, label: `${side} ${depthOf(n)}` };
}

/** The targets beyond the main layers, said briefly: "+1 skip +2 inner". */
export function extrasText(targets, nodes, points) {
  if (!points?.length) return "";
  const main = new Set((nodes || []).map((n) => n.id));
  const kind = Object.fromEntries(points.filter((p) => p.extra).map((p) => [p.id, p.extra]));
  let skips = 0;
  let inner = 0;
  resolveTargets(targets, points).forEach((id) => {
    if (main.has(id)) return;
    if (kind[id] === "skip") skips += 1;
    else if (kind[id] === "inner") inner += 1;
  });
  return [skips && `+${skips} skip${skips === 1 ? "" : "s"}`, inner && `+${inner} inner`]
    .filter(Boolean).join(" ");
}

/** Buckets of `nodes` by `keyOf`, in network order, with the chosen ids in each. */
function bucket(nodes, chosen, keyOf) {
  const out = [];
  const byKey = new Map();
  (nodes || []).forEach((n) => {
    const k = keyOf(n);
    let entry = byKey.get(k.key);
    if (!entry) { entry = { ...k, ids: [], total: 0 }; byKey.set(k.key, entry); out.push(entry); }
    entry.total += 1;
    if (chosen.has(n.id)) entry.ids.push(n.id);
  });
  return out.filter((e) => e.ids.length);
}

/** Where a bend acts, as chips that each take their layers away on click:
 *  whole groups first, then single layers -- one by one while there are few,
 *  as one chip per stage when there are many, with every layer (by level)
 *  one click away in a box of its own. */
function WhereChips({ targets, nodes, onChange }) {
  const [listEach, setListEach] = useState(false);
  const labels = Object.fromEntries((nodes || []).map((n) => [n.id, n.label || n.id]));
  const groups = targets.filter(isGroup);
  const explicit = explicitTargets({ targets }, nodes);
  const drop = (ids) => onChange(targets.filter((t) => !ids.includes(t)));
  const many = explicit.length > LIST_EACH_MAX;
  const chosen = new Set(explicit);
  const stages = many
    ? bucket(nodes, chosen, (n) => (n.extra === "skip" ? { key: "skip", label: "skips" }
      : n.extra === "inner" ? { key: "inner", label: "inside blocks" }
        : { key: n.stage, label: STAGE_NAME[n.stage] || n.stage }))
    : [];
  const levels = many && listEach ? bucket(nodes, chosen, levelOf) : [];
  const names = (ids) => ids.map((id) => labels[id] || id).join(", ");

  // Under a level heading the chip only needs the layer's role in it:
  // "enc 1 · down" under "enc 2" reads as "down" (its output is at that level).
  const single = (t, short = false) => {
    const full = labels[t] || t;
    const text = short && full.includes(" · ") ? full.split(" · ").slice(1).join(" · ") : full;
    return (
      <button type="button" key={t} className="pill chip on where-chip" title={`${full} (${t}) — click to drop`}
        onClick={() => drop([t])}>
        <span>{text}</span> ✕
      </button>
    );
  };

  if (!groups.length && !explicit.length) return null;
  return (
    <div className="where">
      <div className="row wrap gap-1">
        {groups.map((g) => (
          <button type="button" key={g} className="pill chip on where-chip"
            title={`Every ${g === "all" ? "" : `${g} `}layer, as a group — click to drop`}
            onClick={() => drop([g])}>
            <span>{g}</span> ✕
          </button>
        ))}
        {!many && explicit.map((t) => single(t))}
        {stages.map((st) => (
          <button type="button" key={st.key} className="pill chip on where-chip"
            title={`${names(st.ids)} — click to drop all ${st.ids.length}`}
            onClick={() => drop(st.ids)}>
            <span>{st.label}</span>
            <span className="tnum where-count">
              {st.ids.length === st.total ? `all ${st.total}` : `${st.ids.length}/${st.total}`}
            </span>
            ✕
          </button>
        ))}
      </div>
      {many && (
        <>
          <button type="button" className="btn ghost xs where-toggle" aria-expanded={listEach}
            onClick={() => setListEach((v) => !v)}>
            {listEach ? "Hide single layers ▾" : `Show all ${explicit.length} layers ▸`}
          </button>
          {listEach && (
            <div className="where-list">
              {levels.map((lv) => (
                <div key={lv.key} className="where-level">
                  <span className="where-level-name tnum">{lv.label}</span>
                  <span className="where-level-chips">{lv.ids.map((t) => single(t, true))}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** One bend in the stack. It only says what the bend is -- op, amount, where
 *  it hooks in, when it acts -- and picks it; editing happens in the panel
 *  beside the stack, which has room the card never did. Its states are the
 *  ones that change what a run does: off, and on but hooked into nothing. */
function BendCard({ b, i, opDef, nodes, points, focused, dragging, onFocus, onKeyMove, dragProps }) {
  const amount = bendAmount(b, opDef);
  const targets = b.targets || [];
  const hits = resolveTargets(targets, points || nodes).size;
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
          <BendFootprint nodes={nodes} targets={targets} label={where}
            extra={extrasText(targets, nodes, points)} />
        )}
        <RunWindowMini start={b.step_start ?? 0} end={b.step_end ?? 1} hot={focused} />
      </div>
    </div>
  );
}

/** The focused bend's settings, beside the stack, under the map that sets its layers.
 *
 *  `groups` are the named target groups from /craft/ops; `toggleGroup` is the
 *  workspace's, so taking a group here behaves exactly as it did on the chips
 *  that used to sit above the map. */
export function BendInspector({
  b, index, opDef, ops, nodes, points, note, update, remove, duplicate, onRoll, onUndoRoll,
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
  const targets = b.targets || [];
  const hit = resolveTargets(targets, nodes).size;
  const setParam = (name, v) => update(b.id, { params: { ...b.params, [name]: v } });
  const setParams = (patch) => update(b.id, { params: { ...b.params, ...patch } });

  return (
    <div className="bend-inspector" aria-label={`Bend ${index + 1} settings`}>
      <div className="bend-inspector-head">
        <span className="bend-badge on" aria-hidden="true">{index + 1}</span>
        <OpPicker
          ops={ops}
          value={b.op}
          onChange={(name) => {
            const nd = ops.find((o) => o.name === name);
            update(b.id, { op: name, params: defaultsFor(nd), ...scheduleFor(nd) });
          }}
        />
        {onRoll && (
          <button type="button" className="btn icon sm" onClick={onRoll}
            aria-label="Roll a random bend"
            title={(b.targets || []).length
              ? "Roll a random op, amount and timing. Where it acts stays as you set it."
              : "Roll a random op, amount, timing and layers"}>
            <DiceIcon size={15} />
          </button>
        )}
        {onUndoRoll && (
          <button type="button" className="btn icon sm" onClick={onUndoRoll}
            aria-label="Back to the bend before the roll" title="Back to the bend before the roll">
            <UndoIcon size={14} />
          </button>
        )}
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
      {extraParams.map((param) => {
        const pair = PAIRS.find((pr) => pr.keys.includes(param.name)
          && pr.keys.every((k) => extraParams.some((x) => x.name === k)));
        if (pair) {
          if (param.name !== pair.keys[0]) return null;
          const second = extraParams.find((x) => x.name === pair.keys[1]);
          return (
            <PairControl key={pair.keys.join("-")} kind={pair.kind} a={param} b={second}
              opDef={opDef} params={b.params} setParam={setParam} setParams={setParams} />
          );
        }
        return (
          <ParamControl key={param.name} param={param} opDef={opDef} value={b.params[param.name]}
            onChange={(v) => setParam(param.name, v)} />
        );
      })}

      <div className="bend-inspector-section">
        <div className="row between center">
          <div className="section-title mb-0">Where</div>
          <div className="row center gap-2">
            <span className="sub tnum">
              {hit} / {nodes?.length || 0} layers {extrasText(targets, nodes, points)}
            </span>
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
            map to choose where it acts.
          </p>
        )}
        <WhereChips key={b.id} targets={targets} nodes={points || nodes}
          onChange={(next) => update(b.id, { targets: next })} />
        {note?.length > 0 && (
          <p className="sub mb-0">
            Expanded <b>{note.join(", ")}</b> into single layers so you could switch one off.
          </p>
        )}
      </div>

      <RunWindow
        start={b.step_start ?? 0}
        end={b.step_end ?? 1}
        onChange={(patch) => update(b.id, patch)}
      />

      <div className="bend-inspector-foot">
        <button type="button" className="btn sm ghost" onClick={() => duplicate(b.id)}
          title="A copy right after this one, to try a variation without losing the original">
          Duplicate
        </button>
        <button type="button" className="btn sm danger ghost" onClick={() => remove(b.id)}>
          Remove bend
        </button>
      </div>
    </div>
  );
}

export default function BendEditor({
  ops, nodes, points, stack, setStack, focusedId, setFocusedId, addBend, addRandomBend,
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
        Applied left to right — drag a card to reorder. Pick one to edit its settings.
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
              points={points}
              focused={focusedId === b.id}
              dragging={drag?.from === i}
              onFocus={setFocusedId}
              onKeyMove={(dir) => { refocus.current = b.id; moveTo(i, dir < 0 ? i - 1 : i + 2); }}
              dragProps={dragProps(b, i)}
            />
          </React.Fragment>
        ))}
        {stack.length > 0 && slot(stack.length)}
        <div className="bend-add-col">
          <button type="button" className="bend-add-tile" onClick={addBend} disabled={!ops.length}>
            <span aria-hidden="true">+</span>
            Add bend
          </button>
          {addRandomBend && (
            <button type="button" className="bend-add-tile dice" onClick={addRandomBend}
              disabled={!ops.length} title="Add a bend with a random op, amount, timing and layers">
              <DiceIcon size={14} />
              Random
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
