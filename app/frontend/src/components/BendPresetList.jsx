import React from "react";
import { Tooltip } from "./ui.jsx";
import { bendPresetSynopsis, bendPresetSummary } from "../bendSynopsis.js";

/** Bend presets as a compact list, grouped.
 *
 *  Was a wall of wrapping chips, which cost a lot of vertical space for names
 *  alone. A row per preset fits more in less, and has room for the one-line
 *  summary of what it does — which is what you actually pick by. The full
 *  synopsis (a starter's description, then its ops) stays on hover.
 *
 *  `groups` is [{ label, presets }]; empty groups render nothing.
 */
export default function BendPresetList({ groups, onLoad, ops }) {
  const shown = (groups || []).filter((g) => g.presets?.length);
  if (!shown.length) return null;
  return (
    <div className="bend-preset-list">
      {shown.map((g) => (
        <div key={g.label}>
          <div className="section-title">{g.label}</div>
          <ul className="bend-preset-rows">
            {g.presets.map((p) => (
              <li key={p.name}>
                <Tooltip text={bendPresetSynopsis(p, ops)}>
                  <button
                    type="button"
                    className="bend-preset-row"
                    onClick={() => onLoad(p.name)}
                  >
                    {/* Starters are already under a "Starters" heading, so the
                        prefix in their stored name is noise in the list. */}
                    <span className="name">
                      {p.builtin ? p.name.replace(/^starter-/, "") : p.name}
                    </span>
                    <span className="sub summary">{bendPresetSummary(p, ops)}</span>
                  </button>
                </Tooltip>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
