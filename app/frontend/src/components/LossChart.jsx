import React, { useMemo, useState } from "react";
import { Modal } from "./ui.jsx";
import { tokens } from "../theme.js";

function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0];
  if (min === max) return [min];
  const span = max - min;
  const raw = span / Math.max(count - 1, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step * 0.001; v += step) ticks.push(v);
  return ticks.length ? ticks : [min, max];
}

function formatLoss(v) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(1);
  if (a >= 10) return v.toFixed(2);
  if (a >= 1) return v.toFixed(3);
  if (a >= 0.01) return v.toFixed(4);
  return v.toExponential(2);
}

function ChartSvg({ points, height, large = false }) {
  const W = large ? 880 : 520;
  const H = height;
  const padL = large ? 72 : 48;
  const padR = large ? 24 : 16;
  const padT = large ? 28 : 20;
  const padB = large ? 56 : 36;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xs = points.map((p) => p.step);
  const ys = points.map((p) => p.loss);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const yPad = Math.max((maxY - minY) * 0.08, 1e-6);
  const yLo = minY - yPad;
  const yHi = maxY + yPad;

  const sx = (x) => padL + ((x - minX) / Math.max(maxX - minX, 1)) * plotW;
  const sy = (y) => padT + plotH - ((y - yLo) / Math.max(yHi - yLo, 1e-6)) * plotH;

  const xTicks = niceTicks(minX, maxX, large ? 6 : 4);
  const yTicks = niceTicks(yLo, yHi, large ? 6 : 4);

  const d = points.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.step).toFixed(1)},${sy(p.loss).toFixed(1)}`).join(" ");
  const fs = large ? 13 : 10;
  const fsTitle = large ? 14 : 11;
  const {
    grid: gridColor, axis: axisColor, text: textColor, accent, accent2,
  } = tokens({
    grid: "--grid", axis: "--axis", text: "--text-dim",
    accent: "--accent", accent2: "--accent-2",
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="loss-chart-svg" style={{ display: "block" }}>
      <defs>
        <linearGradient id="lossGrad" x1="0" x2="1">
          <stop offset="0" stopColor={accent} />
          <stop offset="1" stopColor={accent2} />
        </linearGradient>
      </defs>
      {yTicks.map((t) => (
        <g key={`yg-${t}`}>
          <line x1={padL} y1={sy(t)} x2={W - padR} y2={sy(t)} stroke={gridColor} strokeWidth="1" />
          <text x={padL - 10} y={sy(t) + 4} fill={textColor} fontSize={fs} textAnchor="end">{formatLoss(t)}</text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g key={`xg-${t}`}>
          <line x1={sx(t)} y1={padT} x2={sx(t)} y2={padT + plotH} stroke={gridColor} strokeWidth="1" strokeDasharray="3 4" />
          <text x={sx(t)} y={H - padB + 22} fill={textColor} fontSize={fs} textAnchor="middle">{Math.round(t)}</text>
        </g>
      ))}
      <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH} stroke={axisColor} strokeWidth="1.5" />
      <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke={axisColor} strokeWidth="1.5" />
      <path d={d} fill="none" stroke="url(#lossGrad)" strokeWidth={large ? 2.5 : 2} strokeLinejoin="round" strokeLinecap="round" />
      <text x={W / 2} y={H - 10} fill={textColor} fontSize={fsTitle} textAnchor="middle">Training step</text>
      <text
        x={16}
        y={padT + plotH / 2}
        fill={textColor}
        fontSize={fsTitle}
        textAnchor="middle"
        transform={`rotate(-90 16 ${padT + plotH / 2})`}
      >
        Loss
      </text>
    </svg>
  );
}

export default function LossChart({ points, height = 140 }) {
  const [open, setOpen] = useState(false);
  const valid = points && points.length >= 2;

  const stats = useMemo(() => {
    if (!valid) return null;
    const ys = points.map((p) => p.loss);
    const lastPt = points[points.length - 1];
    return {
      min: Math.min(...ys),
      max: Math.max(...ys),
      last: ys[ys.length - 1],
      step: lastPt.step,
      start: points[0].step,
    };
  }, [points, valid]);

  if (!valid) {
    return <div className="sub empty">Loss curve appears once training reports steps.</div>;
  }

  return (
    <>
      <button
        type="button"
        className="loss-chart-btn"
        onClick={() => setOpen(true)}
        aria-label="Enlarge loss chart"
        title="Click to enlarge"
      >
        <ChartSvg points={points} height={height} />
        <span className="sub loss-chart-hint">Click to enlarge</span>
      </button>
      {open && (
        <Modal
          title="Training loss"
          wide
          onClose={() => setOpen(false)}
          footer={<button type="button" className="btn ghost" onClick={() => setOpen(false)}>Close</button>}
        >
          {stats && (
            <div className="loss-stats">
              <div><span className="loss-stat-label">Latest</span><b>{formatLoss(stats.last)}</b><span className="sub"> @ step {stats.step}</span></div>
              <div><span className="loss-stat-label">Min</span><b>{formatLoss(stats.min)}</b></div>
              <div><span className="loss-stat-label">Max</span><b>{formatLoss(stats.max)}</b></div>
              <div><span className="loss-stat-label">Steps</span><b>{stats.start}–{stats.step}</b></div>
            </div>
          )}
          <ChartSvg points={points} height={420} large />
        </Modal>
      )}
    </>
  );
}
