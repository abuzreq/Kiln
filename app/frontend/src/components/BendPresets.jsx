import React, { useMemo } from "react";
import { RunWindowMini } from "./RunWindow.jsx";
import { resolveTargets } from "../bendTargets.js";
import {
  bendHeadline, bendPresetSummary, bendPresetSynopsis, formatTargets, opIndex,
} from "../bendSynopsis.js";

// The presets popover: save the current stack, start from one of Kiln's
// starters, or load one of your own. A starter is chosen by what it does, so
// each card leads with its own sentence (the `notes` that used to be a
// tooltip), then the bends it is made of, each with where and when it acts.

const STAGES = [
  { group: "encoder", label: "encoder" },
  { group: "mid", label: "bottleneck" },
  { group: "decoder", label: "decoder" },
];

/** The first sentence of a starter's notes: the card's one line. */
function lead(notes) {
  const t = (notes || "").trim();
  const m = t.match(/^.*?[.!?](\s|$)/);
  return (m ? m[0] : t).trim();
}

/** Which stages a preset reaches on this model, sized by their layer counts. */
function StageStrip({ bends, nodes }) {
  const hit = useMemo(() => {
    const ids = new Set();
    (bends || []).forEach((b) => resolveTargets(b.targets, nodes).forEach((id) => ids.add(id)));
    return ids;
  }, [bends, nodes]);
  const parts = STAGES.map((s) => {
    const ids = (nodes || []).filter((n) => n.stage === s.group).map((n) => n.id);
    return { ...s, size: ids.length, on: ids.some((id) => hit.has(id)) };
  }).filter((s) => s.size);
  if (!parts.length) return null;
  const reach = parts.filter((s) => s.on).map((s) => s.label);
  return (
    <span className="preset-stages" title={reach.length ? `Reaches the ${reach.join(", ")}` : "Reaches no layers of this model"}>
      {parts.map((s) => (
        <span key={s.group} className={s.on ? "on" : ""}
          style={{ flexGrow: s.size, "--stage": `var(--stage-${s.group})` }} />
      ))}
    </span>
  );
}

function StarterCard({ p, ops, nodes, onLoad, onAdd }) {
  const opMap = opIndex(ops);
  const name = p.name.replace(/^starter-/, "");
  return (
    <li className="preset-card" title={bendPresetSynopsis(p, ops)}>
      <div className="row between center gap-2">
        <span className="preset-name">{name}</span>
        <StageStrip bends={p.bends} nodes={nodes} />
      </div>
      <p className="preset-lead">{lead(p.notes)}</p>
      <ul className="preset-bends">
        {(p.bends || []).map((b, i) => (
          <li key={b.id || i}>
            <span className="tnum">{bendHeadline(b, opMap[b.op])} → {formatTargets(b.targets)}</span>
            <RunWindowMini start={b.step_start ?? 0} end={b.step_end ?? 1} />
          </li>
        ))}
      </ul>
      <div className="row gap-1">
        <button type="button" className="btn xs primary" onClick={() => onLoad(p.name)}>Load</button>
        <button type="button" className="btn xs ghost" onClick={() => onAdd(p.name)}
          title="Keep your bends and add this preset's after them">
          + Add to stack
        </button>
      </div>
    </li>
  );
}

export default function BendPresets({
  starters, saved, ops, nodes, saveName, setSaveName, canSave, onSave, onLoad, onAdd,
}) {
  return (
    <div className="bend-presets">
      <form
        className="preset-save"
        onSubmit={(e) => { e.preventDefault(); onSave(); }}
      >
        <label className="preset-save-field">
          <span>Save current stack as</span>
          <input
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="melt-decoder"
            spellCheck={false}
          />
        </label>
        <button type="submit" className="btn sm primary" disabled={!canSave}>Save</button>
      </form>
      <p className="sub preset-note">Saved stacks show up in Create too.</p>

      {starters.length > 0 && (
        <section>
          <div className="row between center preset-head">
            <div className="section-title mb-0">Starters</div>
            <span className="sub">each changes one thing · load it, compare, then edit</span>
          </div>
          <ul className="preset-grid">
            {starters.map((p) => (
              <StarterCard key={p.name} p={p} ops={ops} nodes={nodes} onLoad={onLoad} onAdd={onAdd} />
            ))}
          </ul>
        </section>
      )}

      <section>
        <div className="section-title preset-head">Saved</div>
        {saved.length === 0 ? (
          <p className="sub mb-0">Nothing saved yet. Name the stack above to keep it.</p>
        ) : (
          <ul className="preset-rows">
            {saved.map((p) => (
              <li key={p.name} title={bendPresetSynopsis(p, ops)}>
                <span className="preset-row-name">{p.name}</span>
                <span className="sub preset-row-sum">{bendPresetSummary(p, ops)}</span>
                <button type="button" className="btn xs ghost" onClick={() => onAdd(p.name)}
                  aria-label={`Add ${p.name} to the stack`} title="Add to stack">+</button>
                <button type="button" className="btn xs" onClick={() => onLoad(p.name)}>Load</button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
