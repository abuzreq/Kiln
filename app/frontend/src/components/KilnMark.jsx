import React from "react";

// Kiln's mark: a stepped brick arch, which is a kiln door and also a U-Net
// upside down, with dashed skip connections between matching blocks. It is
// the app's status light too: "cold" before anything has been made, "warm"
// when idle, "firing" while any work runs.

// [x, y, w, h] on a 32-unit grid, left leg bottom-up, the top, right leg top-down.
const BLOCKS = [
  [4, 22, 5, 7], [5.5, 15.6, 5, 5.4], [8, 10.4, 4.6, 4.2], [12, 6.4, 8, 3.2],
  [19.4, 10.4, 4.6, 4.2], [21.5, 15.6, 5, 5.4], [23, 22, 5, 7],
];
// One skip beside the top block and two at each wider level, hottest at the
// top, where heat rises to.
const SKIPS = [
  ["M12.6 12.5h6.8", "#ffe0a8"],
  ["M10.5 17.4h11", "#ffc06a"],
  ["M10.5 19.4h11", "#ffb347"],
  ["M9 24h14", "#ff8a52"],
  ["M9 26.6h14", "#ff6a2b"],
];

// Below 28 px the fine dashes turn to noise: chunkier blocks, one coarser
// skip per level.
const BLOCKS_SMALL = [
  [3.5, 21.5, 6, 7.5], [5.5, 15, 5.5, 5.6], [8, 9.6, 5, 4.6], [12, 5.6, 8, 3.6],
  [19, 9.6, 5, 4.6], [21, 15, 5.5, 5.6], [22.5, 21.5, 6, 7.5],
];
const SKIPS_SMALL = [
  ["M13 12h6", "#ffd08a"],
  ["M11 17.8h10", "#ffb347"],
  ["M9.5 25.2h13", "#ff6a2b"],
];

export default function KilnMark({ state = "warm", size = 24, title }) {
  const small = size < 28;
  const blocks = small ? BLOCKS_SMALL : BLOCKS;
  const skips = small ? SKIPS_SMALL : SKIPS;
  return (
    <svg
      className={`kiln-mark ${state}${small ? " small" : ""}`}
      width={size}
      height={size}
      viewBox="0 0 32 32"
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
    >
      {title && <title>{title}</title>}
      <g className="kiln-mark-blocks">
        {blocks.map(([x, y, w, h]) => (
          <rect key={`${x},${y}`} x={x} y={y} width={w} height={h} rx="0.8" />
        ))}
      </g>
      <g
        fill="none"
        strokeLinecap="round"
        strokeWidth={small ? 1.7 : 1.1}
        strokeDasharray={small ? "2.6 1.6" : "1.2 1"}
      >
        {skips.map(([d, color], i) => (
          <path
            key={d}
            className="kiln-mark-skip"
            d={d}
            stroke={color}
            style={{ animationDelay: `${-i * 0.45}s` }}
          />
        ))}
      </g>
    </svg>
  );
}
