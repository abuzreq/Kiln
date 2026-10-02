import React, { useMemo, useState } from "react";
import { heldByActive, holdersIn } from "../bendTargets.js";
import HolderBadges, { holdersText } from "./HolderBadges.jsx";

// The model as a signal path: what the image passes through, in order, with
// where the current bend sits marked on it. No geometry and no layer names --
// three stages you can bend whole, each with a strip of ticks (one per layer,
// in order) so "how much of the decoder" shows without opening anything, and a
// numbered row inside any stage you want to be specific about. This is the plain-language way into bending; the
// map (UnetVisualizer) is there when the structure itself is the point.
const STAGES = [
  { group: "encoder", label: "Encoder", sub: "reads the image, detail first" },
  { group: "mid", label: "Bottleneck", sub: "the whole image at its coarsest" },
  { group: "decoder", label: "Decoder", sub: "paints it back, detail last" },
];

export default function BendPipeline({
  graph, focusTargets, holders, hasFocus, activeGroups,
  onToggle, onToggleGroup, onCreateFromNode,
}) {
  const [open, setOpen] = useState(null);
  const nodes = graph?.nodes;

  const stages = useMemo(() => STAGES.map((s) => {
    const layers = (nodes || []).filter((n) => n.stage === s.group);
    return { ...s, layers, ids: layers.map((n) => n.id) };
  }).filter((s) => s.layers.length), [nodes]);

  if (!stages.length) {
    return <div className="sub" style={{ padding: 20 }}>Select a model to introspect its layers.</div>;
  }

  /** Target these layers, or start a bend on them when nothing is focused. */
  const pick = (ids, opts) => {
    if (!hasFocus) { onCreateFromNode?.(ids); return; }
    onToggle(ids, opts);
  };

  const row = (stage) => {
    const { layers } = stage;
    const half = Math.ceil(layers.length / 2);
    const attn = layers.filter((n) => n.type === "attention");
    const quick = [
      { id: "all", label: "all", ids: stage.ids },
      { id: "early", label: "early", ids: stage.ids.slice(0, half) },
      { id: "late", label: "late", ids: stage.ids.slice(half) },
      ...(attn.length ? [{ id: "attn", label: "attention", ids: attn.map((n) => n.id) }] : []),
    ];
    // The pick that says exactly what this bend holds in this stage, if any.
    const held = stage.ids.filter((id) => focusTargets?.has(id));
    const matches = (q) => q.ids.length === held.length && q.ids.every((id) => focusTargets?.has(id));
    const current = held.length ? quick.find(matches)?.id : null;
    return (
      <div className="bend-pipe-layers" key={`${stage.group}-layers`}>
        <div className="row wrap gap-2">
          {layers.map((n, i) => {
            const on = focusTargets?.has(n.id);
            const other = !on && heldByActive(holders, n.id);
            const also = holdersIn(holders, [n.id]);
            return (
              <button
                type="button"
                key={n.id}
                className={`pill chip num ${on ? "on" : ""} ${other ? "other" : ""}`.trim()}
                aria-pressed={on}
                title={`${n.label} · ${n.type} · ${n.channels}ch${also.length ? ` · also in ${holdersText(also)}` : ""}`}
                onClick={() => pick([n.id], { force: !on })}
              >
                {i}{n.type === "attention" ? " ◆" : ""}
              </button>
            );
          })}
        </div>
        <div className="row wrap gap-2 mt-1">
          <span className="sub">take</span>
          {quick.map((q) => (
            <button type="button" key={q.id}
              className={`btn ghost sm ${current === q.id ? "on" : ""}`.trim()}
              aria-pressed={current === q.id}
              onClick={() => pick(q.ids, { force: true })}>{q.label}</button>
          ))}
          <span className="sub">
            {layers.length === 1 ? "1 layer" : `${layers.length} layers, numbered in order`}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="bend-pipe-wrap">
      <div className="bend-pipe">
        <span className="bend-pipe-end">
          <span className="bend-pipe-swatch noise" aria-hidden="true" />
          noise
        </span>
        {stages.map((s) => {
          const hit = s.ids.filter((id) => focusTargets?.has(id)).length;
          const others = holdersIn(holders, s.ids);
          const held = activeGroups?.includes(s.group);
          const all = held || (hit > 0 && hit === s.ids.length);
          const isOpen = open === s.group;
          return (
            <React.Fragment key={s.group}>
              <span className="bend-pipe-arrow" aria-hidden="true">→</span>
              <div
                className={`bend-pipe-stage ${all ? "on" : hit ? "some" : ""} ${isOpen ? "open" : ""}`.trim()}
                style={{ "--stage": `var(--stage-${s.group})` }}
              >
                <button
                  type="button"
                  className="bend-pipe-pill"
                  aria-pressed={all ? true : hit ? "mixed" : false}
                  title={`Bend the whole ${s.label.toLowerCase()}`}
                  onClick={() => onToggleGroup?.(s.group)}
                >
                  <span className="bend-pipe-title">
                    <b>{s.label}</b>
                    <span className="bend-pipe-count tnum">
                      {all ? `all ${s.ids.length}` : hit ? `${hit} of ${s.ids.length}` : `${s.ids.length} layers`}
                      <HolderBadges list={others} />
                    </span>
                  </span>
                  <span className="sub">{s.sub}</span>
                  <span className="bend-pipe-ticks" aria-hidden="true">
                    {s.layers.map((n) => {
                      const on = focusTargets?.has(n.id);
                      const other = !on && heldByActive(holders, n.id);
                      return <i key={n.id} className={on ? "on" : other ? "other" : ""} />;
                    })}
                  </span>
                </button>
                {s.layers.length > 1 && (
                  <button
                    type="button"
                    className="bend-pipe-open"
                    aria-expanded={isOpen}
                    aria-label={isOpen ? `Hide the layers in ${s.label}` : `Pick single layers in ${s.label}`}
                    onClick={() => setOpen(isOpen ? null : s.group)}
                  >
                    {isOpen ? "Hide layers ▾" : "Pick single layers ▸"}
                  </button>
                )}
              </div>
            </React.Fragment>
          );
        })}
        <span className="bend-pipe-arrow" aria-hidden="true">→</span>
        <span className="bend-pipe-end">
          <span className="bend-pipe-swatch image" aria-hidden="true" />
          image
        </span>
      </div>
      {stages.filter((s) => s.group === open).map(row)}
    </div>
  );
}
