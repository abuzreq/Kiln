import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay } from "../screens/playContext.jsx";
import { Slider, Select, Num, Tooltip, Popover } from "./ui.jsx";
import {
  BrushIcon, WandIcon, ShapeIcon, ContrastIcon, MoveIcon,
  RectIcon, EllipseIcon, PolygonIcon, PatternIcon, AddIcon, NewMaskIcon, EyeIcon,
  ChevronDownIcon,
} from "./icons.jsx";
import { contrastPreview } from "../contrastMask.js";
import { cachedStroke } from "../selection.js";
import { maskUrlToCache } from "../layers.js";

// The mask tools, in rail order. Move comes first and is the default, so a
// stray click on the picture never paints. Each has a bare-letter key (no
// modifier, never while typing), which leaves Ctrl+Z/Y, Ctrl+D, Ctrl+Shift+I
// and Space-to-pan as they were. The Contrast tool is shown as Split: it
// divides the picture in two, and "Contrast" is also a post-process slider.
export const MASK_TOOLS = [
  { id: "move", key: "V", label: "Move", Icon: MoveIcon, tip: "Move: drag the mask around, or nudge it with the arrow keys" },
  { id: "brush", key: "B", label: "Brush", Icon: BrushIcon, tip: "Brush: paint the mask freehand" },
  { id: "wand", key: "W", label: "Wand", Icon: WandIcon, tip: "Wand: click a colour on the picture to take everything like it nearby" },
  { id: "shape", key: "U", label: "Shape", Icon: ShapeIcon, tip: "Shape: rectangles, ellipses, polygons, or a pattern" },
  { id: "contrast", key: "C", label: "Split", Icon: ContrastIcon, tip: "Split: divide the picture in two by brightness or by local contrast, and take one side" },
];
const SHAPE_KINDS = [
  { id: "rect", label: "Rectangle", Icon: RectIcon, tip: "Rectangle: drag on the canvas; Shift for a square" },
  { id: "ellipse", label: "Ellipse", Icon: EllipseIcon, tip: "Ellipse: drag on the canvas; Shift for a circle" },
  { id: "polygon", label: "Polygon", Icon: PolygonIcon, tip: "Polygon: click corners on the canvas; Enter or double-click closes" },
  { id: "pattern", label: "Pattern", Icon: PatternIcon, tip: "Pattern: blobs, cells, stripes, a split, or scattered shapes, from a seed" },
];
const SPLIT_METHODS = [
  { value: "luminance", label: "Brightness" },
  { value: "contrast", label: "Local contrast" },
];
// The two sides of a split. "Foreground/Background" claimed more than the maths
// delivers -- a luminance split separates dark from light, nothing more.
const SPLIT_SIDES = [
  { id: "foreground", label: "Side A", luminance: "Darker", contrast: "Detailed" },
  { id: "background", label: "Side B", luminance: "Lighter", contrast: "Flat" },
];

const isTyping = (e) => {
  const t = e.target;
  const tag = t?.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t?.isContentEditable;
};

/** The pixel size of the picture on the canvas, once it has loaded. */
export function useFrameSize(frame) {
  const [size, setSize] = useState(null);
  useEffect(() => {
    if (!frame) { setSize(null); return undefined; }
    let live = true;
    const im = new Image();
    im.onload = () => { if (live) setSize({ w: im.naturalWidth, h: im.naturalHeight }); };
    im.src = frame;
    return () => { live = false; };
  }, [frame]);
  return size;
}

/** The tools, in a rail against the picture's left edge. Below them, M: show
 *  or hide the hatching of the mask being edited. */
export function ToolRail() {
  const { maskTool, setMaskTool, activeMask, setMaskVisible, tab } = usePlay();
  const maskShown = !!activeMask && activeMask.visible !== false;

  useEffect(() => {
    if (tab !== "create") return undefined;
    const onKey = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat || isTyping(e)) return;
      const k = e.key.toUpperCase();
      const tool = MASK_TOOLS.find((t) => t.key === k);
      if (tool) { e.preventDefault(); setMaskTool(tool.id); return; }
      if (k === "M" && activeMask) { e.preventDefault(); setMaskVisible(activeMask.id, !maskShown); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tab, setMaskTool, activeMask, maskShown, setMaskVisible]);

  return (
    <nav className="tool-rail" aria-label="Mask tools">
      {MASK_TOOLS.map((t) => (
        <Tooltip key={t.id} text={`${t.tip} (${t.key})`}>
          <button
            type="button"
            className={`tool-rail-btn ${maskTool === t.id ? "on" : ""}`}
            onClick={() => setMaskTool(t.id)}
            aria-label={`${t.label} (${t.key})`}
            aria-pressed={maskTool === t.id}
          >
            <t.Icon size={18} />
            <kbd aria-hidden="true">{t.key}</kbd>
          </button>
        </Tooltip>
      ))}
      <span className="tool-rail-sep" aria-hidden="true" />
      <Tooltip text={activeMask
        ? `${maskShown ? "Hide" : "Show"} the hatching of ${activeMask.name}. It still counts when filling. (M)`
        : "No mask yet: paint one to show it"}
      >
        <button
          type="button"
          className="tool-rail-btn"
          onClick={() => activeMask && setMaskVisible(activeMask.id, !maskShown)}
          disabled={!activeMask}
          aria-label="Show the mask (M)"
          aria-pressed={maskShown}
        >
          <EyeIcon off={!maskShown} size={18} />
          <kbd aria-hidden="true">M</kbd>
        </button>
      </Tooltip>
    </nav>
  );
}

/** Which mask strokes go into, and a way to start another. */
function IntoMask() {
  const { inpaintMasks, activeMask, selectEntity, addMask } = usePlay();
  return (
    <Popover
      label="Mask to edit"
      triggerClass="bar-into"
      align="start"
      trigger={<>Into <strong>{activeMask?.name || "a new mask"}</strong><ChevronDownIcon /></>}
    >
      {(close) => (
        <div className="bar-menu">
          <div className="bar-menu-title">Strokes go into</div>
          {inpaintMasks.map((m) => (
            <button
              key={m.id}
              type="button"
              className={`bar-menu-item ${m.id === activeMask?.id ? "on" : ""}`}
              onClick={() => { selectEntity(m.id); close(); }}
            >
              <span className="grow">{m.name}</span>
              <span className="sub">{m.enabled ? "used" : "off"}</span>
            </button>
          ))}
          <button type="button" className="bar-menu-item" onClick={() => { addMask(); close(); }}>
            <AddIcon /> <span className="grow">New mask</span>
          </button>
        </div>
      )}
    </Popover>
  );
}

/** A one-line hint that shortens with an ellipsis; hover shows all of it. */
function Hint({ text }) {
  return <span className="bar-hint" title={text}>{text}</span>;
}

/** A compact range for the options bar: label, track, value. */
function BarRange({ label, value, min, max, step = 1, onChange, fmt = (v) => v, tip, width = 96 }) {
  return (
    <Tooltip text={tip}>
      <label className="bar-range">
        <span>{label}</span>
        <input
          type="range" min={min} max={max} step={step} value={value} style={{ width }}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <b className="tnum">{fmt(value)}</b>
      </label>
    </Tooltip>
  );
}

function BarSeg({ ariaLabel, items, value, onChange }) {
  return (
    <div className="seg seg-sm" role="group" aria-label={ariaLabel}>
      {items.map((it) => (
        <button key={it.id} type="button" className={value === it.id ? "on" : ""} aria-pressed={value === it.id}
          onClick={() => onChange(it.id)} title={it.tip}>
          {it.Icon && <it.Icon />}{it.label}
        </button>
      ))}
    </div>
  );
}

/** Everything the Split tool knows, kept here so it survives switching tools. */
function useSplit() {
  const { toast } = useApp();
  const { frame, applyContrastMask, maskTool } = usePlay();
  const [method, setMethod] = useState("luminance");
  const [brightness, setBrightness] = useState(128);
  const [contrast, setContrast] = useState(0);
  const [soften, setSoften] = useState(2);
  const [smoothOn, setSmoothOn] = useState(false);
  const [smoothRadius, setSmoothRadius] = useState(2);
  const [side, setSide] = useState("foreground");
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);

  const opts = useMemo(() => {
    const base = method === "luminance"
      ? { method: "luminance", autoThreshold: false, threshold: brightness, soften }
      : { method: "contrast", autoThreshold: true, thresholdBias: contrast, soften };
    return { ...base, presmooth: smoothOn ? smoothRadius : 0 };
  }, [method, brightness, contrast, soften, smoothOn, smoothRadius]);

  // The previews follow the picture and the settings by themselves while the
  // tool is up: there is no Calculate step to forget after a run or a nudge.
  useEffect(() => {
    setPreview(null);
    if (maskTool !== "contrast" || !frame) return undefined;
    let live = true;
    const t = setTimeout(async () => {
      setBusy(true);
      try {
        const p = await contrastPreview(frame, opts);
        if (live) setPreview(p);
      } catch (e) {
        if (live) toast(e.message || "Could not split the picture", "error");
      } finally {
        if (live) setBusy(false);
      }
    }, 250);
    return () => { live = false; clearTimeout(t); };
  }, [frame, opts, maskTool, toast]);

  const sideLabel = (id) => {
    const def = SPLIT_SIDES.find((x) => x.id === id);
    return method === "luminance" ? def?.luminance : def?.contrast;
  };
  const region = (id) => (method === "contrast"
    ? (id === "foreground" ? "high" : "low")
    : (id === "foreground" ? "dark" : "light"));

  // The split reads the stack, not the screen: with post-process on the two
  // differ, and the fill this selects for reads the stack.
  const apply = async (target = "active") => {
    if (!frame) { toast("Put an image on the canvas first", "error"); return; }
    setBusy(true);
    try {
      const label = sideLabel(side);
      const ok = await applyContrastMask(frame, { ...opts, region: region(side) }, { target, name: label || "Split" });
      if (ok) {
        toast(target === "new"
          ? `Made a mask of the ${(label || side).toLowerCase()} side`
          : `Selected the ${(label || side).toLowerCase()} side`, "success");
      } else {
        toast("Could not build the selection", "error");
      }
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setBusy(false);
    }
  };

  return {
    method, setMethod, brightness, setBrightness, contrast, setContrast, soften, setSoften,
    smoothOn, setSmoothOn, smoothRadius, setSmoothRadius, side, setSide,
    preview, busy, apply, sideLabel,
  };
}

function SplitOptions({ split }) {
  const { frame } = usePlay();
  const s = split;
  const sides = SPLIT_SIDES.map((x) => ({ id: x.id, label: s.sideLabel(x.id) }));
  if (!frame) return <Hint text="Put an image on the canvas to split it" />;
  return (
    <>
      <BarSeg ariaLabel="Split by" value={s.method} onChange={s.setMethod}
        items={SPLIT_METHODS.map((m) => ({ id: m.value, label: m.label,
          tip: m.value === "luminance" ? "Light from dark" : "Busy areas from flat ones" }))} />
      {s.method === "luminance" ? (
        <BarRange label="Threshold" value={s.brightness} min={1} max={255} onChange={s.setBrightness}
          tip="Brightness level (1–255) that separates the two sides" width={90} />
      ) : (
        <BarRange label="Bias" value={s.contrast} min={-40} max={40} onChange={s.setContrast}
          tip="Nudge the auto-detected local-contrast split" width={90} />
      )}
      <BarSeg ariaLabel="Side to take" value={s.side} onChange={s.setSide} items={sides} />
      <Tooltip text={`Add the ${(s.sideLabel(s.side) || "").toLowerCase()} side to the mask. Replaces any earlier split in it; brush and shape strokes stay.`}>
        <button type="button" className="btn sm primary" onClick={() => s.apply("active")} disabled={s.busy}>
          <AddIcon /> {s.busy ? "Working…" : "Add to mask"}
        </button>
      </Tooltip>
      <Popover label="More split settings" triggerClass="btn ghost sm" trigger={<>More <ChevronDownIcon /></>}
        panelClass="bar-pop">
        <div className="bar-pop-body">
          <div className="contrast-split-previews">
            {SPLIT_SIDES.map((x) => (
              <button
                key={x.id}
                type="button"
                className={`contrast-split-tile ${s.side === x.id ? "on" : ""}`}
                onClick={() => s.setSide(x.id)}
              >
                {s.preview?.[x.id]
                  ? <img src={s.preview[x.id]} alt={`${s.sideLabel(x.id)} side`} />
                  : <span className="sub">{s.busy ? "…" : x.label}</span>}
                <span className="contrast-split-label">{s.sideLabel(x.id)}</span>
              </button>
            ))}
          </div>
          {s.preview && (
            <p className="sub mb-0">Split at {s.preview.cut}{s.method === "luminance" ? " brightness" : " contrast"}</p>
          )}
          <Slider label="Edge soften" value={s.soften} min={0} max={8} step={1} onChange={s.setSoften}
            tip="Softens the edge of the mask the split makes." />
          <label className="row center gap-2">
            <input type="checkbox" checked={s.smoothOn} onChange={(e) => s.setSmoothOn(e.target.checked)} />
            <span className="sub">Blur first (fewer speckles)</span>
          </label>
          {s.smoothOn && (
            <Slider label="Blur radius" value={s.smoothRadius} min={1} max={6} step={1} onChange={s.setSmoothRadius}
              tip="Blur the picture before splitting it, to reduce speckles." />
          )}
          <button type="button" className="btn sm" onClick={() => s.apply("new")} disabled={s.busy}>
            <NewMaskIcon /> Put this side in a new mask
          </button>
        </div>
      </Popover>
    </>
  );
}

/** The active tool's settings, in one line above the picture. */
export function ToolOptions() {
  const {
    maskTool, brushSize, setBrushSize, brushHard, setBrushHard, eraser, setEraser,
    wandTolerance, setWandTolerance, shapeKind, setShapeKind, genShape, setGenShape,
    polyCount, addStroke, addMaskWithStroke, activeMask, frame,
  } = usePlay();
  const split = useSplit();
  const frameSize = useFrameSize(frame);
  const maskName = activeMask?.name || "the mask";
  const tool = MASK_TOOLS.find((t) => t.id === maskTool) || MASK_TOOLS[0];

  const paintErase = (
    <BarSeg ariaLabel="Paint or erase" value={eraser ? "erase" : "paint"} onChange={(v) => setEraser(v === "erase")}
      items={[
        { id: "paint", label: "Paint", tip: "Add to the mask" },
        { id: "erase", label: "Erase", tip: "Cut out of the mask" },
      ]} />
  );

  let body = null;
  if (maskTool === "brush") {
    body = (
      <>
        {paintErase}
        <BarSeg ariaLabel="Brush edge" value={brushHard ? "hard" : "soft"} onChange={(v) => setBrushHard(v === "hard")}
          items={[{ id: "soft", label: "Soft" }, { id: "hard", label: "Hard" }]} />
        <BarRange label="Size" value={brushSize} min={8} max={160} step={2} onChange={setBrushSize}
          tip="Brush diameter in canvas pixels" />
      </>
    );
  } else if (maskTool === "wand") {
    body = (
      <>
        {paintErase}
        <BarRange label="Tolerance" value={wandTolerance} min={0} max={100} onChange={setWandTolerance}
          tip="How close a neighbouring pixel's colour must be to join the mask" />
        <Hint text="Click a colour on the picture" />
      </>
    );
  } else if (maskTool === "shape") {
    body = (
      <>
        <div className="seg seg-sm seg-icons" role="group" aria-label="Shape kind">
          {SHAPE_KINDS.map((k) => (
            <Tooltip key={k.id} text={k.tip}>
              <button type="button" className={shapeKind === k.id ? "on" : ""} onClick={() => setShapeKind(k.id)}
                aria-label={k.label} aria-pressed={shapeKind === k.id}>
                <k.Icon />
              </button>
            </Tooltip>
          ))}
        </div>
        {paintErase}
        {shapeKind === "pattern" ? (
          <Popover
            label="Pattern"
            triggerClass="btn sm"
            panelClass="bar-pop"
            trigger={<>{(GEN_KINDS.find((k) => k.id === genShape.kind) || GEN_KINDS[0]).label} · {Math.round(genShape.coverage * 100)}%<ChevronDownIcon /></>}
          >
            <div className="bar-pop-body">
              <PatternPanel
                genShape={genShape}
                setGenShape={setGenShape}
                canvasSize={frameSize}
                eraser={eraser}
                addStroke={addStroke}
                addMaskWithStroke={addMaskWithStroke}
                maskName={maskName}
              />
            </div>
          </Popover>
        ) : (
          <Hint text={shapeKind === "polygon"
            ? (polyCount ? `${polyCount} corner${polyCount === 1 ? "" : "s"} — Enter closes, Esc cancels` : "Click to place corners")
            : `Drag to draw ${shapeKind === "ellipse" ? "an ellipse" : "a rectangle"}; Shift keeps it ${shapeKind === "ellipse" ? "round" : "square"}`} />
        )}
      </>
    );
  } else if (maskTool === "contrast") {
    body = <SplitOptions split={split} />;
  } else {
    body = <Hint text="Drag the mask to move it · arrow keys nudge 1 px, Shift+arrow 10" />;
  }

  return (
    <div className="tool-options" role="group" aria-label={`${tool.label} options`}>
      <IntoMask />
      {body}
    </div>
  );
}

// The shape generator's kinds and the one or two settings each exposes. The
// backend does the work (app/core/tools/masks.py); this only names the knobs.
const GEN_KINDS = [
  { id: "blobs", label: "Blobs", tip: "Organic islands from layered noise",
    params: [{ key: "scale", label: "Scale", min: 2, max: 8, step: 1, def: 4, tip: "Bigger numbers, smaller blobs" }] },
  { id: "cells", label: "Cells", tip: "A patchwork of cells, some of them selected",
    params: [
      { key: "cells", label: "Cells", min: 4, max: 32, step: 1, def: 14, tip: "How many cells the canvas is cut into" },
      { key: "jitter", label: "Wobble", min: 0, max: 1, step: 0.05, def: 0.35, tip: "How much the cell borders wander" },
    ] },
  { id: "stripes", label: "Stripes", tip: "Parallel bands at an angle",
    params: [
      { key: "angle", label: "Angle", min: 0, max: 179, step: 1, def: 45 },
      { key: "count", label: "Bands", min: 1, max: 12, step: 1, def: 5 },
      { key: "wobble", label: "Wobble", min: 0, max: 1, step: 0.05, def: 0.5, tip: "How much the band edges wander" },
    ] },
  { id: "split", label: "Split", tip: "The canvas cut into two or three pieces along a wavy line",
    params: [
      { key: "pieces", label: "Pieces", min: 2, max: 3, step: 1, def: 2 },
      { key: "wave", label: "Wave", min: 0, max: 1, step: 0.05, def: 0.6, tip: "How much the cut wanders" },
      { key: "orientation", label: "Cut", options: [{ value: "v", label: "Left to right" }, { value: "h", label: "Top to bottom" }], def: "v" },
    ] },
  { id: "shapes", label: "Scatter", tip: "Scattered circles and stars, some with holes",
    params: [
      { key: "count", label: "Shapes", min: 1, max: 12, step: 1, def: 6 },
      { key: "holes", label: "Holes", options: [{ value: "yes", label: "Some" }, { value: "no", label: "None" }], def: "yes" },
    ] },
];

const randomSeed31 = () => Math.floor(Math.random() * 2 ** 31);

/** The Pattern arm of the Shape tool: a kind, its knobs, a seed, a live
 *  preview, and Add to mask. Every Add is one stroke; Shuffle rerolls. */
function PatternPanel({ genShape, setGenShape, canvasSize, eraser, addStroke, addMaskWithStroke, maskName }) {
  const { toast } = useApp();
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const kind = GEN_KINDS.find((k) => k.id === genShape.kind) || GEN_KINDS[0];
  const w = canvasSize?.w || 512;
  const h = canvasSize?.h || 512;
  const params = { ...Object.fromEntries(kind.params.map((p) => [p.key, p.def])), ...(genShape.params || {}) };
  // What the backend takes: the same keys, with the select-style knobs decoded.
  const backendParams = () => {
    const out = { ...params };
    if ("holes" in out) out.holes = out.holes !== "no";
    return out;
  };
  const body = (pw, ph) => ({
    kind: genShape.kind, width: pw, height: ph, seed: genShape.seed,
    coverage: genShape.coverage, soften: genShape.soften, invert: false, params: backendParams(),
  });
  const set = (patch) => setGenShape({ ...genShape, ...patch });
  const setParam = (key, v) => set({ params: { ...params, [key]: v } });

  useEffect(() => {
    let live = true;
    const scale = 256 / Math.max(w, h);
    const t = setTimeout(() => {
      api.post("/tools/mask", body(Math.max(16, Math.round(w * scale)), Math.max(16, Math.round(h * scale))))
        .then((d) => { if (live) setPreview(d.mask); })
        .catch(() => { if (live) setPreview(null); });
    }, 200);
    return () => { live = false; clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [genShape.kind, genShape.seed, genShape.coverage, genShape.soften, JSON.stringify(params), w, h]);

  const add = async (target = "active") => {
    setBusy(true);
    try {
      const d = await api.post("/tools/mask", body(w, h));
      const { cache } = await maskUrlToCache(d.mask);
      const stroke = {
        ...cachedStroke("generated", cache, eraser && target === "active" ? "subtract" : "add"),
        gen: { kind: genShape.kind, seed: genShape.seed, coverage: genShape.coverage, soften: genShape.soften, params: backendParams() },
      };
      if (target === "new") addMaskWithStroke(stroke, kind.label);
      else addStroke(stroke);
    } catch (e) { toast(e.message, "error"); }
    setBusy(false);
  };

  return (
    <div className="gen-panel">
      <div className="seg seg-sm mb-2" role="group" aria-label="Pattern">
        {GEN_KINDS.map((k) => (
          <button key={k.id} type="button" className={genShape.kind === k.id ? "on" : ""} title={k.tip}
            onClick={() => set({ kind: k.id, params: {} })}>{k.label}</button>
        ))}
      </div>
      <div className="gen-body">
        <div className="gen-knobs">
          <Slider label="Coverage" value={Math.round(genShape.coverage * 100)} min={5} max={80} step={1}
            onChange={(v) => set({ coverage: v / 100 })} fmt={(v) => `${v} %`}
            tip="Roughly how much of the canvas the shape covers." />
          <Slider label="Soften" value={genShape.soften} min={0} max={24} step={1}
            onChange={(v) => set({ soften: v })}
            tip="Blur the shape's edge, so it selects at partial strength there, like a soft brush." />
          {kind.params.map((p) => (p.options ? (
            <Select key={p.key} label={p.label} value={String(params[p.key])} options={p.options}
              onChange={(v) => setParam(p.key, v)} tip={p.tip} />
          ) : (
            <Slider key={p.key} label={p.label} value={Number(params[p.key])} min={p.min} max={p.max} step={p.step}
              onChange={(v) => setParam(p.key, v)} tip={p.tip} />
          )))}
          <div className="row center gap-2">
            <Num label="Seed" value={genShape.seed} onChange={(v) => set({ seed: Math.max(0, Math.round(Number(v) || 0)) })}
              tip="The same seed and settings give the same shape." />
            <button type="button" className="btn sm" onClick={() => set({ seed: randomSeed31() })} title="A different shape with the same settings">Shuffle</button>
          </div>
        </div>
        <div className="gen-preview" title="What Add to mask will add, white where the mask goes">
          {preview ? <img src={preview} alt="" /> : <span className="sub">…</span>}
        </div>
      </div>
      <AddToMaskButtons
        thing="shape"
        maskName={maskName}
        eraser={eraser}
        busy={busy}
        onAdd={() => add("active")}
        onNew={() => add("new")}
      />
    </div>
  );
}

/** The pair every way of producing an area ends in: into the mask being
 *  painted, or into a mask of its own. Other masks are left as they are --
 *  a composition is several masks on at once. */
function AddToMaskButtons({ onAdd, onNew, eraser, busy, disabled, maskName, thing }) {
  const addText = eraser ? `Cut this ${thing} out of ${maskName}` : `Add this ${thing} to ${maskName}`;
  return (
    <div className="add-pair mt-2">
      <Tooltip text={addText}>
        <button type="button" className="btn sm primary" onClick={onAdd} disabled={busy || disabled}>
          <AddIcon /> {busy ? "Working…" : eraser ? "Cut from mask" : "Add to mask"}
        </button>
      </Tooltip>
      <Tooltip text={eraser
        ? "Erase cuts out of a mask; there is nothing to cut from a new one"
        : `Put this ${thing} in a mask of its own, on and selected. Other masks stay as they are.`}
      >
        <button type="button" className="btn sm" onClick={onNew} disabled={busy || disabled || eraser}>
          <NewMaskIcon /> New mask
        </button>
      </Tooltip>
    </div>
  );
}
