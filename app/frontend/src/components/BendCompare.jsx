import React, { useEffect, useRef, useState } from "react";
import { bendHeadline, bendPresetSynopsis, formatTargets, opIndex } from "../bendSynopsis.js";

// Looking at a bend's effect: the same seed sampled without and with the
// stack, shown four ways. Side by side is honest about both pictures; a wipe
// and a flip put them in the same place, which is where a small change shows;
// the difference image (enlarged view only) shows where they differ at all.

export const MODES = [
  { id: "split", label: "Split", tip: "The two side by side" },
  { id: "wipe", label: "Wipe", tip: "One picture: drag the line between without and with" },
  { id: "flip", label: "Flip", tip: "One picture: click it to swap between without and with" },
];
export const MODES_LARGE = [
  ...MODES,
  { id: "diff", label: "Difference", tip: "Where the two differ: the brighter, the more" },
];

// The difference image is stretched so this share of pixels stays below full
// brightness. A fixed gain could not serve both ends: a subtle bend stayed
// black, and a strong one turned nearly every pixel orange.
const DIFF_PERCENTILE = 0.98;
const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;

function Tag({ side, children }) {
  return <span className={`cmp-tag ${side}`}>{children}</span>;
}

function Missing({ what }) {
  return <div className="cmp-missing sub">No {what} image</div>;
}

function Split({ plain, bent }) {
  return (
    <div className="cmp-split">
      <figure className="cmp-frame">
        {plain ? <img src={plain} alt="Without bends" draggable={false} /> : <Missing what="unbent" />}
        <Tag side="left">Without</Tag>
      </figure>
      <figure className="cmp-frame">
        {bent ? <img src={bent} alt="With bends" draggable={false} /> : <Missing what="bent" />}
        <Tag side="left hot">With bends</Tag>
      </figure>
    </div>
  );
}

/** One picture, without on the left of the line and with on the right. */
function Wipe({ plain, bent }) {
  const [at, setAt] = useState(50);
  const box = useRef(null);
  const drag = useRef(false);
  const pos = (x) => {
    const r = box.current.getBoundingClientRect();
    return Math.max(0, Math.min(100, ((x - r.left) / (r.width || 1)) * 100));
  };
  const key = (e) => {
    const next = {
      ArrowLeft: at - 5, ArrowDown: at - 5, ArrowRight: at + 5, ArrowUp: at + 5,
      PageDown: at - 25, PageUp: at + 25, Home: 0, End: 100,
    }[e.key];
    if (next == null) return;
    e.preventDefault();
    setAt(Math.max(0, Math.min(100, next)));
  };
  return (
    <div
      className="cmp-frame cmp-wipe"
      ref={box}
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        drag.current = true;
        box.current.setPointerCapture?.(e.pointerId);
        setAt(pos(e.clientX));
      }}
      onPointerMove={(e) => { if (drag.current) setAt(pos(e.clientX)); }}
      onPointerUp={() => { drag.current = false; }}
      onPointerCancel={() => { drag.current = false; }}
    >
      <img src={bent} alt="With bends" draggable={false} />
      <img src={plain} alt="Without bends" draggable={false} className="cmp-wipe-top"
           style={{ clipPath: `inset(0 ${100 - at}% 0 0)` }} />
      <span className="cmp-wipe-line" style={{ left: `${at}%` }} aria-hidden="true" />
      <span
        role="slider"
        tabIndex={0}
        className="cmp-wipe-handle"
        style={{ left: `${at}%` }}
        aria-label="Wipe between without and with bends"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(at)}
        aria-valuetext={`${Math.round(at)}% shows the image without bends`}
        onKeyDown={key}
      />
      <Tag side="left">Without</Tag>
      <Tag side="right hot">With bends</Tag>
    </div>
  );
}

/** One picture; a click (or Space) swaps it. */
function Flip({ plain, bent }) {
  const [showPlain, setShowPlain] = useState(false);
  return (
    <button
      type="button"
      className="cmp-frame cmp-flip"
      aria-pressed={showPlain}
      aria-label={showPlain ? "Showing without bends. Click to show with bends" : "Showing with bends. Click to show without"}
      onClick={() => setShowPlain((v) => !v)}
    >
      <img src={showPlain ? plain : bent} alt="" draggable={false} />
      <Tag side={showPlain ? "left" : "left hot"}>{showPlain ? "Without" : "With bends"}</Tag>
      <span className="cmp-flip-hint">click to flip</span>
    </button>
  );
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("image did not load"));
    im.src = src;
  });
}

/** Per-pixel difference, stretched to the pair: black where nothing changed,
 *  full orange from the 98th percentile up. */
function Diff({ plain, bent }) {
  const canvas = useRef(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let gone = false;
    setFailed(false);
    Promise.all([loadImage(plain), loadImage(bent)]).then(([a, b]) => {
      if (gone || !canvas.current) return;
      const w = b.naturalWidth;
      const h = b.naturalHeight;
      const cv = canvas.current;
      cv.width = w;
      cv.height = h;
      const ctx = cv.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(a, 0, 0, w, h);
      const pa = ctx.getImageData(0, 0, w, h).data;
      ctx.drawImage(b, 0, 0, w, h);
      const pb = ctx.getImageData(0, 0, w, h).data;
      const diff = new Uint8Array(w * h);
      const hist = new Uint32Array(256);
      for (let i = 0, k = 0; i < pa.length; i += 4, k += 1) {
        const d = Math.round((Math.abs(pa[i] - pb[i]) + Math.abs(pa[i + 1] - pb[i + 1])
          + Math.abs(pa[i + 2] - pb[i + 2])) / 3);
        diff[k] = d;
        hist[d] += 1;
      }
      let ceil = 255;
      for (let v = 0, seen = 0; v < 256; v += 1) {
        seen += hist[v];
        if (seen >= diff.length * DIFF_PERCENTILE) { ceil = Math.max(v, 1); break; }
      }
      const out = ctx.createImageData(w, h);
      for (let k = 0; k < diff.length; k += 1) {
        const d = Math.min(255, (diff[k] / ceil) * 255);
        // warm ramp, so "a lot" reads as the accent rather than as white
        out.data[k * 4] = d;
        out.data[k * 4 + 1] = d * 0.6;
        out.data[k * 4 + 2] = d * 0.35;
        out.data[k * 4 + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
    }).catch(() => { if (!gone) setFailed(true); });
    return () => { gone = true; };
  }, [plain, bent]);
  return (
    <figure className="cmp-frame cmp-diff">
      {failed
        ? <Missing what="difference" />
        : <canvas ref={canvas} role="img" aria-label="Where the two images differ; brighter is more different" />}
      <Tag side="left">Difference · scaled to this pair</Tag>
    </figure>
  );
}

export function CompareViewer({ plain, bent, mode, large = false }) {
  const both = plain && bent;
  let body;
  if (!both || mode === "split") body = <Split plain={plain} bent={bent} />;
  else if (mode === "wipe") body = <Wipe plain={plain} bent={bent} />;
  else if (mode === "flip") body = <Flip plain={plain} bent={bent} />;
  else body = <Diff plain={plain} bent={bent} />;
  return <div className={`cmp-viewer ${large ? "large" : ""}`.trim()}>{body}</div>;
}

/** "seed 42 · 50 steps · 64px", plus whether the unbent side was reused. */
export function CompareMeta({ meta }) {
  if (!meta) return null;
  const size = Array.isArray(meta.size) ? meta.size.join("×") : meta.size ? `${meta.size}px` : null;
  return (
    <div className="cmp-meta">
      <span className="sub tnum">
        seed {meta.seed ?? "random"}{meta.steps ? ` · ${meta.steps} steps` : ""}{size ? ` · ${size}` : ""}
      </span>
      {meta.reused && <span className="pill good" title="Same model, settings and seed: only the bent side was sampled">plain reused</span>}
    </div>
  );
}

/** The bends behind a result, one numbered chip each. */
export function StackChips({ stack, ops }) {
  const opMap = opIndex(ops);
  const active = (stack || []).map((b, i) => ({ b, n: i + 1 })).filter(({ b }) => b.active !== false);
  if (!active.length) return null;
  return (
    <ul className="cmp-stack">
      {active.map(({ b, n }) => (
        <li key={b.id || n}>
          <span className="bend-badge sm">{n}</span>
          <span className="tnum">
            {bendHeadline(b, opMap[b.op])} → {formatTargets(b.targets)}
            {((b.step_start ?? 0) > 0 || (b.step_end ?? 1) < 1) && ` · ${pct(b.step_start)}–${pct(b.step_end)}`}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** Earlier results this session. Clicking one only shows it; bringing its
 *  bends back is a separate, explicit act, because it replaces the stack. */
export function CompareHistory({ history, shownId, ops, onShow, onRestore, restorable }) {
  if (!history?.length) return null;
  const shown = history.find((h) => h.id === shownId);
  return (
    <div className="cmp-history">
      <div className="row between center">
        <div className="section-title mb-0">Earlier tries</div>
        <span className="sub">this session</span>
      </div>
      <div className="cmp-history-row">
        {history.map((h, i) => {
          const n = history.length - i;
          const on = h.id === shownId;
          return (
            <button
              key={h.id}
              type="button"
              className={`cmp-try ${on ? "on" : ""}`.trim()}
              aria-pressed={on}
              title={bendPresetSynopsis({ bends: h.stack }, ops)}
              onClick={() => onShow(h.id)}
            >
              <img src={h.bent} alt={`Try ${n}`} draggable={false} />
              <span className="tnum">{i === 0 ? "latest" : `try ${n}`}</span>
            </button>
          );
        })}
      </div>
      {shown && restorable && (
        <div className="row between center cmp-restore">
          <span className="sub">These bends differ from your stack.</span>
          <button type="button" className="btn xs" onClick={() => onRestore(shown)}>Restore these bends</button>
        </div>
      )}
    </div>
  );
}
