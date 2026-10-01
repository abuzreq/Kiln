import React, { useMemo, useState } from "react";
import { resolveTargets } from "../bendTargets.js";

// The model as a signal path: what the image passes through, in order, with
// where the current bend sits marked on it. No geometry and no layer names --
// three stages you can bend whole, and a numbered strip inside any stage you
// want to be specific about. This is the plain-language way into bending; the
// map (UnetVisualizer) is there when the structure itself is the point.
const STAGES = [
  { group: "encoder", label: "Encoder", sub: "reads the image, detail first" },
  { group: "mid", label: "Bottleneck", sub: "the whole image at its coarsest" },
  { group: "decoder", label: "Decoder", sub: "paints it back, detail last" },
];

export default function BendPipeline({
  graph, focusTargets, otherTargets, hasFocus, activeGroups,
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
    return (
      <div className="bend-pipe-layers" key={`${stage.group}-layers`}>
        <div className="row wrap gap-2">
          {layers.map((n, i) => {
            const on = focusTargets?.has(n.id);
            const other = !on && otherTargets?.has(n.id);
            return (
              <button
                type="button"
                key={n.id}
                className={`pill chip num ${on ? "on" : ""} ${other ? "other" : ""}`.trim()}
                aria-pressed={on}
                title={`${n.label} · ${n.type} · ${n.channels}ch`}
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
            <button type="button" key={q.id} className="btn ghost sm"
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
        <span className="bend-pipe-end">Noise</span>
        {stages.map((s) => {
          const hit = s.ids.filter((id) => focusTargets?.has(id)).length;
          const others = s.ids.filter((id) => !focusTargets?.has(id) && otherTargets?.has(id)).length;
          const held = activeGroups?.includes(s.group);
          const all = held || (hit > 0 && hit === s.ids.length);
          const isOpen = open === s.group;
          return (
            <React.Fragment key={s.group}>
              <span className="bend-pipe-arrow" aria-hidden="true">→</span>
              <div className={`bend-pipe-stage ${all ? "on" : hit ? "some" : ""}`.trim()}>
                <button
                  type="button"
                  className="bend-pipe-pill"
                  aria-pressed={all}
                  title={`Bend the whole ${s.label.toLowerCase()}`}
                  onClick={() => onToggleGroup?.(s.group)}
                >
                  <b>{s.label}</b>
                  <span className="sub">{s.sub}</span>
                  <span className="bend-pipe-count">
                    {all ? `all ${s.ids.length}` : hit ? `${hit} of ${s.ids.length}` : `${s.ids.length} layers`}
                    {others > 0 && <i title={`${others} held by other bends`}> +{others}</i>}
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
                    {isOpen ? "−" : "+"}
                  </button>
                )}
              </div>
            </React.Fragment>
          );
        })}
        <span className="bend-pipe-arrow" aria-hidden="true">→</span>
        <span className="bend-pipe-end">Image</span>
      </div>
      {stages.filter((s) => s.group === open).map(row)}
    </div>
  );
}
