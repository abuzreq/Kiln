import React, { useMemo, useState } from "react";
import { Seg } from "./ui.jsx";
import { RunWindowMini } from "./RunWindow.jsx";
import { resolveTargets } from "../bendTargets.js";
import {
  bendHeadline, bendPresetSummary, bendPresetSynopsis, formatTargets, opIndex,
} from "../bendSynopsis.js";
// Pictures of the starters, rendered by scripts/make_example_media.py.
import MEDIA from "../exampleMedia.json";

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

/** What the preset does, seen: a render with it, the same seed without it inset. */
function PresetShot({ src, plain, name }) {
  if (!src) return null;
  return (
    <div className="preset-shot">
      <img src={src} alt={`${name}, on a sample model`} loading="lazy" />
      {plain && (
        <span className="preset-shot-plain" title="The same seed without the bend">
          <img src={plain} alt="" loading="lazy" />
          <span>without</span>
        </span>
      )}
    </div>
  );
}

function StarterCard({ p, ops, nodes, onLoad, onAdd }) {
  const opMap = opIndex(ops);
  const name = p.name.replace(/^starter-/, "");
  return (
    <li className="preset-card" title={bendPresetSynopsis(p, ops)}>
      <PresetShot src={MEDIA.bends?.[p.name]} plain={MEDIA.bends?.plain} name={name} />
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

// Saved and Starters are tabs. Shown together, six starter cards filled the
// first screen and a long saved list became a scroll box inside a scrolling
// popover. Someone with saved stacks of their own is mostly after those.
const TAB_KEY = "kiln.bendPresetsTab";
// A filter box appears once the saved list is longer than a glance.
const FILTER_FROM = 7;

function loadTab(hasSaved) {
  try {
    const v = localStorage.getItem(TAB_KEY);
    if (v === "saved" || v === "starters") return v;
  } catch { /* private window */ }
  return hasSaved ? "saved" : "starters";
}

export default function BendPresets({
  starters, saved, ops, nodes, saveName, setSaveName, canSave, onSave, onLoad, onAdd,
}) {
  const [tab, setTabState] = useState(() => loadTab(saved.length > 0));
  const [filter, setFilter] = useState("");
  const setTab = (t) => {
    setTabState(t);
    try { localStorage.setItem(TAB_KEY, t); } catch { /* the session keeps it */ }
  };
  const q = filter.trim().toLowerCase();
  const shown = q
    ? saved.filter((p) => `${p.name} ${bendPresetSummary(p, ops)}`.toLowerCase().includes(q))
    : saved;

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

      <div className="row between center preset-tabs">
        <Seg
          ariaLabel="Which presets"
          size="sm"
          value={tab}
          onChange={setTab}
          tabs={[
            { id: "saved", label: `Saved (${saved.length})`, tip: "Stacks you saved" },
            { id: "starters", label: `Starters (${starters.length})`, tip: "Kiln's recipes: each changes one thing" },
          ]}
        />
        {tab === "starters" && (
          <span className="sub">shown on the sample model {MEDIA.model} · load one, generate, then edit</span>
        )}
        {tab === "saved" && saved.length >= FILTER_FROM && (
          <input
            className="preset-filter"
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder={`Filter ${saved.length} saved…`}
            aria-label="Filter saved presets by name or what they do"
          />
        )}
      </div>

      {tab === "starters" && (
        <ul className="preset-grid">
          {starters.map((p) => (
            <StarterCard key={p.name} p={p} ops={ops} nodes={nodes} onLoad={onLoad} onAdd={onAdd} />
          ))}
        </ul>
      )}

      {tab === "saved" && (saved.length === 0 ? (
        <p className="sub mb-0">
          Nothing saved yet. Name the stack above to keep it, or start from a starter.
        </p>
      ) : shown.length === 0 ? (
        <p className="sub mb-0">No saved preset matches “{filter.trim()}”.</p>
      ) : (
        <ul className="preset-rows">
          {shown.map((p) => (
            <li key={p.name} title={`${p.name}\n\n${bendPresetSynopsis(p, ops)}`}>
              {p.thumbnail
                ? <img className="preset-row-thumb" src={p.thumbnail} alt="" loading="lazy" />
                : <span className="preset-row-thumb" aria-hidden="true" />}
              <span className="preset-row-name">{p.name}</span>
              <span className="sub preset-row-sum">{bendPresetSummary(p, ops)}</span>
              <button type="button" className="btn xs ghost" onClick={() => onAdd(p.name)}
                aria-label={`Add ${p.name} to the stack`} title="Add to stack">+</button>
              <button type="button" className="btn xs" onClick={() => onLoad(p.name)}>Load</button>
            </li>
          ))}
        </ul>
      ))}
    </div>
  );
}
