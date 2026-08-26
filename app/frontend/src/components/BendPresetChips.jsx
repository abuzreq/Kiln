import React from "react";
import { Tooltip } from "./ui.jsx";
import { bendPresetSynopsis } from "../bendSynopsis.js";

export default function BendPresetChips({ presets, value, onChange, allowNone = true, label = "Load:", ops }) {
  if (!presets?.length) return null;
  return (
    <div className="bend-preset-list mt-2">
      {label && <span className="sub">{label}</span>}
      <div className="row wrap gap-2 mt-1">
        {allowNone && (
          <button type="button" className={`btn sm ${!value ? "on" : ""}`} onClick={() => onChange("")}>
            — none —
          </button>
        )}
        {presets.map((p) => (
          <span key={p.name} className="row center gap-1 bend-preset-chip">
            <button
              type="button"
              className={`btn sm ${value === p.name ? "on" : ""}`}
              onClick={() => onChange(p.name)}
            >
              {p.name}
            </button>
            <Tooltip text={bendPresetSynopsis(p, ops)}>
              <span className="tip-mark" tabIndex={0} role="img" aria-label={`Bend stack for ${p.name}`}>?</span>
            </Tooltip>
          </span>
        ))}
      </div>
    </div>
  );
}
