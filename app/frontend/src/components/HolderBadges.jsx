import React from "react";

// The numbered badges that say which other bends hold part of the map. The
// number is the bend's place in the stack -- the same number its card shows --
// so "1" on the encoder means the first card, without a colour per bend to
// fight the stage colours for. Off bends stay listed, faded, because they come
// back the moment they are switched on.

const MAX_SHOWN = 2;

/** Badges to draw for `list` ([{ n, active }]): the first few, then "+k". */
export function badgeRun(list) {
  if (!list?.length) return [];
  if (list.length <= MAX_SHOWN) return list.map((h) => ({ ...h, text: String(h.n) }));
  const shown = list.slice(0, MAX_SHOWN - 1).map((h) => ({ ...h, text: String(h.n) }));
  const rest = list.slice(MAX_SHOWN - 1);
  return [...shown, { n: rest[0].n, active: rest.some((h) => h.active), text: `+${rest.length}`, more: true }];
}

/** Words for a tooltip or a screen reader: "bends 1 and 3 (off)". */
export function holdersText(list) {
  if (!list?.length) return "";
  const parts = list.map((h) => `${h.n}${h.active ? "" : " (off)"}`);
  const joined = parts.length === 1 ? parts[0]
    : `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
  return `${list.length === 1 ? "bend" : "bends"} ${joined}`;
}

/** HTML badges, for the Simple view and anywhere outside the SVG map. */
export default function HolderBadges({ list }) {
  const run = badgeRun(list);
  if (!run.length) return null;
  return (
    <span className="holder-badges" title={`Also held by ${holdersText(list)}`}>
      {run.map((h) => (
        <span key={`${h.n}-${h.text}`} className={`bend-badge sm ${h.active ? "" : "off"}`.trim()}>
          {h.text}
        </span>
      ))}
    </span>
  );
}
