import React, { useMemo } from "react";
import { parseSpec } from "../attnLayout.js";

// A block sketch of the UNet the Train form describes, redrawn as the channel
// multipliers are typed. Encoder levels left to right, the bottleneck, then the
// decoder mirrored; block height follows resolution (halved per level), block
// width follows channel count (base width x multiplier). Attention is tagged on
// the levels the layout names, so the per-level pickers and the picture agree.
//
// Stage colours are the same tokens the Craft map uses; SVG attributes take
// var() directly, as BendFootprint already does.

/** Multipliers string or array -> [1, 2, 2, 2], skipping anything non-numeric. */
export function parseMults(mults) {
  const raw = Array.isArray(mults) ? mults : String(mults || "").split(",");
  return raw.map((x) => parseInt(String(x).trim(), 10)).filter((n) => Number.isFinite(n) && n >= 1);
}

/**
 * The levels a network of these multipliers has, shallowest first, then the
 * bottleneck: { loc, label, ch, res, mult }. `loc` is the attention location the
 * spec uses ("-1" is the last encoder level, "mid" the bottleneck), so a picker
 * row and a block can be looked up by the same key.
 */
export function levelInfo(mults, imageSize, baseDim = 64) {
  const m = parseMults(mults);
  const size = Number(imageSize) || 0;
  const depth = m.length;
  const levels = m.map((mult, i) => ({
    loc: String(i - depth),
    label: `Level ${i - depth}`,
    mult,
    ch: baseDim * mult,
    res: size ? size / 2 ** i : null,
  }));
  if (depth) {
    levels.push({
      loc: "mid", label: "Bottleneck", mult: m[depth - 1],
      ch: baseDim * m[depth - 1], res: size ? size / 2 ** depth : null,
    });
  }
  return levels;
}

const fmtRes = (r) => (r == null ? "" : Number.isInteger(r) ? `${r}` : r.toFixed(1));

const H_MAX = 72;   // tallest block, level 0
const H_MIN = 9;
const W_MIN = 14;
const W_SPAN = 22;  // extra width at the widest level
const GAP = 8;
const TOP = 16;     // room for the channel label above
const BOTTOM = 30;  // room for the resolution label and the attention tag

export default function ArchSketch({ mults, imageSize, attn, baseDim = 64 }) {
  const levels = useMemo(() => levelInfo(mults, imageSize, baseDim), [mults, imageSize, baseDim]);
  const attnAt = useMemo(() => parseSpec(attn) || {}, [attn]);
  const depth = levels.length ? levels.length - 1 : 0;

  if (!depth) {
    return <p className="sub arch-sketch-empty">Type the multipliers as comma-separated numbers, e.g. 1,2,2,2.</p>;
  }

  const enc = levels.slice(0, depth);
  const mid = levels[depth];
  const maxMult = Math.max(...levels.map((l) => l.mult));
  const widthOf = (l) => W_MIN + W_SPAN * (l.mult / maxMult);
  const heightOf = (i) => Math.max(H_MAX / 2 ** i, H_MIN);

  // encoder, bottleneck, decoder (mirror), laid out on one baseline
  const blocks = [];
  let x = 4;
  const push = (l, i, stage) => {
    const w = widthOf(l);
    const h = heightOf(i);
    blocks.push({ ...l, stage, x, w, h, y: TOP + H_MAX - h });
    x += w + GAP;
  };
  enc.forEach((l, i) => push(l, i, "encoder"));
  push(mid, depth, "mid");
  enc.slice().reverse().forEach((l, k) => push({ ...l, loc: null }, depth - 1 - k, "decoder"));
  const width = x;
  const height = TOP + H_MAX + BOTTOM;

  const step = 2 ** depth;
  const size = Number(imageSize) || 0;
  const misfit = size && size % step;
  const below = size - (size % step);
  const above = below + step;

  return (
    <div className="arch-sketch">
      <div className="scroll-x">
        <svg width={Math.max(width, 120)} height={height} role="img"
          aria-label={`${depth} encoder levels, bottleneck at ${fmtRes(mid.res)}px`}>
          <line x1="0" x2={width} y1={TOP + H_MAX + 0.5} y2={TOP + H_MAX + 0.5} stroke="var(--line)" />
          {blocks.map((b, k) => {
            const kind = b.loc ? attnAt[b.loc] : null;
            const cx = b.x + b.w / 2;
            const showRes = b.stage !== "decoder";
            return (
              <g key={k}>
                <rect x={b.x} y={b.y} width={b.w} height={b.h} rx="2"
                  fill={`var(--stage-${b.stage})`} opacity={b.stage === "decoder" ? 0.35 : 0.85} />
                {kind && (
                  <rect x={b.x - 2.5} y={b.y - 2.5} width={b.w + 5} height={b.h + 5} rx="3.5"
                    fill="none" stroke="var(--accent)" strokeWidth="1.5" />
                )}
                {b.stage !== "decoder" && (
                  <text x={cx} y={b.y - 4} textAnchor="middle" fill="var(--text-dim)">{b.ch}</text>
                )}
                {showRes && b.res != null && (
                  <text x={cx} y={TOP + H_MAX + 12} textAnchor="middle" fill="var(--text-dim)">{fmtRes(b.res)}</text>
                )}
                {kind && (
                  <text x={cx} y={TOP + H_MAX + 24} textAnchor="middle" fill="var(--accent)">{kind}</text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <p className="sub mt-0 mb-0">
        {depth} level{depth === 1 ? "" : "s"}: the image is halved {depth} time{depth === 1 ? "" : "s"} down to{" "}
        {fmtRes(mid.res)}px at the bottleneck. Numbers above are channel widths, below are pixel sizes;
        an orange frame marks attention.
      </p>
      {misfit ? (
        <p className="sub mt-0 mb-0" style={{ color: "var(--warn)" }}>
          {size}px does not divide by {step}, so this network cannot train at that size. Use {below || step} or {above}.
        </p>
      ) : null}
    </div>
  );
}
