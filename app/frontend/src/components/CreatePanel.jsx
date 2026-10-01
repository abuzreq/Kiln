import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay } from "../screens/playContext.jsx";
import { Slider, Select, Num, Disclose, TipLabel, Tooltip } from "./ui.jsx";
import {
  BrushIcon, WandIcon, ShapeIcon, ContrastIcon, MoveIcon, PaintIcon, EraseIcon,
  RectIcon, EllipseIcon, PolygonIcon, PatternIcon, UndoIcon, InvertIcon, ClearIcon,
  NewMaskIcon, AddIcon,
} from "./icons.jsx";
import {
  buildSamplePayload, buildInpaintPayload, changeToParams, effectiveSteps, skippedSteps,
  fillSizeFor, solverCanResample, LIVE_PARAM_KEYS,
  liveEditLabels, joinLabels } from "../sampleSettings.jsx";
import { contrastPreview, compositePostprocWithMask } from "../contrastMask.js";
import { cachedStroke } from "../selection.js";
import { maskUrlToCache } from "../layers.js";
import { bendPresetSynopsis, bendPresetSummary } from "../bendSynopsis.js";

// The mask tools and the Shape tool's kinds. Icons only in the rows (five
// tools would not fit a sidebar with their words); the label is the tooltip,
// the aria-label, and the name shown beside the row for the one that is on.
const MASK_TOOLS = [
  { id: "brush", label: "Brush", Icon: BrushIcon, tip: "Brush: paint freehand" },
  { id: "wand", label: "Wand", Icon: WandIcon, tip: "Wand: click a colour on the canvas to select everything like it nearby" },
  { id: "shape", label: "Shape", Icon: ShapeIcon, tip: "Shape: rectangles, ellipses, polygons, or a pattern" },
  { id: "contrast", label: "Contrast", Icon: ContrastIcon, tip: "Contrast: split the canvas in two by brightness or by local contrast, and take one side" },
  { id: "move", label: "Move", Icon: MoveIcon, tip: "Move: drag the mask around, or nudge it with the arrow keys" },
];
const SHAPE_KINDS = [
  { id: "rect", label: "Rectangle", Icon: RectIcon, tip: "Rectangle: drag on the canvas; Shift for a square" },
  { id: "ellipse", label: "Ellipse", Icon: EllipseIcon, tip: "Ellipse: drag on the canvas; Shift for a circle" },
  { id: "polygon", label: "Polygon", Icon: PolygonIcon, tip: "Polygon: click corners on the canvas; Enter or double-click closes" },
  { id: "pattern", label: "Pattern", Icon: PatternIcon, tip: "Pattern: blobs, cells, stripes, a split, or scattered shapes, from a seed" },
];

const DEFAULT_PP = { contrast: 1, gamma: 1, saturation: 1, eqhist: 0, unsharp: 0, noise: 0 };

const SPLIT_METHODS = [
  { value: "luminance", label: "Brightness" },
  { value: "contrast", label: "Local contrast" },
];

// The two sides of a split. "Foreground/Background" claimed more than the maths
// delivers — a luminance split separates dark from light, nothing more.
const SPLIT_SIDES = [
  { id: "foreground", label: "Side A", luminance: "Darker", contrast: "Detailed" },
  { id: "background", label: "Side B", luminance: "Lighter", contrast: "Flat" },
];

function isIdentityPostproc(pp) {
  if (!pp) return true;
  return (
    (pp.contrast ?? 1) === 1
    && (pp.gamma ?? 1) === 1
    && (pp.saturation ?? 1) === 1
    && (pp.eqhist ?? 0) === 0
    && (pp.unsharp ?? 0) === 0
    && (pp.noise ?? 0) === 0
  );
}

function loadStored(key) {
  try { return localStorage.getItem(key) || ""; } catch { return ""; }
}

function resolveBends(presets, name) {
  if (!name) return null;
  return presets.find((b) => b.name === name)?.bends || null;
}

export default function CreatePanel() {
  const { toast, modelPath, models, ops } = useApp();
  const {
    generateIntoLayer, fillIntoLayer, pushHistory, frame, setPostFrame,
    sampleParams, openDocument,
    brushSize, setBrushSize, brushHard, setBrushHard, eraser, setEraser,
    maskTool, setMaskTool, wandTolerance, setWandTolerance,
    shapeKind, setShapeKind, genShape, setGenShape, polygonRef, polyCount, addStroke, addMaskWithStroke,
    getMaskDataUrl, applyContrastMask, frameCard, activeLayer,
    activeMask, maskPixels, hasMask, liveMasks, invertMask, clearMask, setMaskParam, nudgeMask,
    undo, canUndo, redo, canRedo, tab, setLivePreview, livePreviewOn,
    job, genRunning, genPaused, canvasIsBlank, runs, trackRuns,
  } = usePlay();
  const layerName = activeLayer?.name || "the active layer";

  // The same model object Play hands the settings panel — needed to tell which
  // sampling controls are live (see the pause toast below).
  const model = useMemo(
    () => (models || []).find((m) => m.path === modelPath) || null,
    [models, modelPath],
  );

  const [ppOn, setPpOn] = useState(false);
  const [pp, setPp] = useState(DEFAULT_PP);
  const [bendPresets, setBendPresets] = useState([]);
  const [genBendPreset, setGenBendPreset] = useState(() => loadStored("kiln.genBendPreset"));
  const [srFactor, setSrFactor] = useState(2);
  const [srSharpen, setSrSharpen] = useState(0.5);
  const [srBusy, setSrBusy] = useState(false);
  const [genChange, setGenChange] = useState(0.7);

  // Everything that scopes to a masked area lives on the mask itself, so there
  // is one Change slider rather than two that looked alike and took turns being
  // the one that mattered. Defaults here cover "no mask yet", where the
  // controls are shown but inert.
  const sel = activeMask?.params || {};
  const regionChange = sel.change ?? 0.65;
  const feather = sel.feather ?? 8;
  // RePaint resampling, on at 2 passes by default: the fill agrees with what
  // surrounds it far better than in a single pass, and most of that gain is
  // already there at two. 1 is off, which is how fills used to work.
  const resample = sel.harmonize ?? 2;
  const regionBendPreset = sel.bendPreset ?? "";
  const maskName = activeMask?.name || "the mask";

  // A fill on a blank canvas has to run the whole schedule. Change works by
  // skipping the early, high-noise steps to preserve what is already there --
  // and on empty pixels there is nothing worth preserving, so a partial run
  // just hands back the blank it started from. So the run uses full Change on
  // a blank canvas, and the slider says so. This is an override at run time,
  // not a write to the mask: the first cut wrote 1 into the mask's own
  // setting, and nothing ever wrote it back, so the first fill after the
  // first generation quietly ran at full strength.
  const fillLocked = hasMask && canvasIsBlank;
  const effectiveRegionChange = fillLocked ? 1 : regionChange;

  const canResample = solverCanResample(sampleParams.sampler);
  // One control, two homes: a mask carries its own Change and bend preset, and
  // the whole-canvas equivalents stay on the panel. Which one the control is
  // editing is spelled out in its label rather than left to be inferred.
  const changeValue = hasMask ? effectiveRegionChange : genChange;
  const setChangeValue = (v) => (
    hasMask ? setMaskParam(activeMask?.id, "change", v) : setGenChange(v)
  );
  // Without a mask, the canvas is what the run works from -- there is no
  // separate init image any more. Below full Change it is img2img over the
  // flattened stack; at full Change, or on a blank canvas, it is a new image.
  // One rule for both paths: the canvas is the source and the mask only says
  // where. The init slot was the last place the mode was invisible.
  const reworkCanvas = !hasMask && !canvasIsBlank && genChange < 1;
  const bendValue = hasMask ? (sel.bendPreset ?? "") : genBendPreset;
  const setBendValue = (v) => (
    hasMask ? setMaskParam(activeMask?.id, "bendPreset", v) : setGenBendPreset(v)
  );
  const [variations, setVariations] = useState(1);
  const [splitMethod, setSplitMethod] = useState("luminance");
  const [brightnessThreshold, setBrightnessThreshold] = useState(128);
  const [contrastThreshold, setContrastThreshold] = useState(0);
  const [splitSoften, setSplitSoften] = useState(2);
  const [smoothOn, setSmoothOn] = useState(false);
  const [smoothRadius, setSmoothRadius] = useState(2);
  const [maskSide, setMaskSide] = useState("foreground");
  const [splitPreview, setSplitPreview] = useState(null);
  const [splitBusy, setSplitBusy] = useState(false);
  // Recompute the split whenever the picture being worked on changes: after a
  // generation, and when an init image is set or swapped. A flag rather than a
  // direct call, because both of those commit new state during the same render
  // and calculating inline would still see the previous image.
  const [splitPending, setSplitPending] = useState(false);
  // Job id -> the live settings it was started (or last resumed) with. Resume
  // sends only what differs, and with a queue each run has its own baseline.
  const snapshots = useRef({});
  const ppSource = frame;
  const ppGen = useRef(0);
  const [canvasSize, setCanvasSize] = useState(null);

  // Region fill runs at the canvas's own resolution, so the user needs to see it.
  useEffect(() => {
    if (!frame) { setCanvasSize(null); return undefined; }
    let live = true;
    const im = new Image();
    im.onload = () => { if (live) setCanvasSize({ w: im.naturalWidth, h: im.naturalHeight }); };
    im.src = frame;
    return () => { live = false; };
  }, [frame]);

  useEffect(() => {
    api.get("/library/bends").then(setBendPresets).catch(() => {});
  }, []);

  useEffect(() => {
    if (tab !== "create") return undefined;
    const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const onKey = (e) => {
      const t = e.target;
      const tag = t?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t?.isContentEditable) return;
      // Arrow keys nudge the active mask while the Move tool is up: one canvas
      // pixel, ten with Shift.
      // A polygon in progress: Enter closes, Esc cancels, Backspace drops a corner.
      if (maskTool === "shape" && shapeKind === "polygon" && polyCount > 0 && !e.ctrlKey && !e.metaKey) {
        if (e.key === "Enter") { e.preventDefault(); polygonRef.current.close(); return; }
        if (e.key === "Escape") { e.preventDefault(); polygonRef.current.cancel(); return; }
        if (e.key === "Backspace") { e.preventDefault(); polygonRef.current.pop(); return; }
      }
      if (ARROWS[e.key] && maskTool === "move" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (!activeMask) return;
        e.preventDefault();
        const step = e.shiftKey ? 10 : 1;
        nudgeMask(activeMask.id, ARROWS[e.key][0] * step, ARROWS[e.key][1] * step);
        return;
      }
      if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "z" && !e.shiftKey) {
        if (!canUndo) return;
        e.preventDefault();
        undo();
      } else if ((k === "z" && e.shiftKey) || k === "y") {
        if (!canRedo) return;
        e.preventDefault();
        redo();
      } else if (k === "d" && !e.shiftKey) {
        e.preventDefault();
        if (activeMask) clearMask(activeMask.id);
      } else if (k === "i" && e.shiftKey) {
        e.preventDefault();
        invertMask(activeMask?.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tab, canUndo, undo, canRedo, redo, clearMask, invertMask, activeMask, maskTool, nudgeMask,
    shapeKind, polyCount, polygonRef]);

  useEffect(() => {
    localStorage.setItem("kiln.genBendPreset", genBendPreset);
  }, [genBendPreset]);

  const splitOpts = useMemo(() => {
    const base = splitMethod === "luminance"
      ? { method: "luminance", autoThreshold: false, threshold: brightnessThreshold, soften: splitSoften }
      : { method: "contrast", autoThreshold: true, thresholdBias: contrastThreshold, soften: splitSoften };
    return { ...base, presmooth: smoothOn ? smoothRadius : 0 };
  }, [splitMethod, brightnessThreshold, contrastThreshold, splitSoften, smoothOn, smoothRadius]);

  // Invalidate calculated preview when inputs change so Apply requires Calculate again.
  useEffect(() => {
    setSplitPreview(null);
  }, [frame, splitOpts]);

  // Post-process reads the flattened stack and writes to a display slot beside
  // it, rather than editing the stack. So it can never feed on its own output,
  // Unprocessed is just the stack itself, and turning it off is free.
  useEffect(() => {
    if (!ppSource) { setPostFrame(null); return undefined; }
    if (!ppOn || isIdentityPostproc(pp)) { setPostFrame(null); return undefined; }
    const gen = ++ppGen.current;
    const t = setTimeout(async () => {
      try {
        const mask = getMaskDataUrl();
        const r = await api.post("/perform/postproc", { image: ppSource, postproc: pp });
        let image = r.image;
        if (mask) {
          image = await compositePostprocWithMask(ppSource, r.image, mask);
        }
        if (gen === ppGen.current) setPostFrame(image);
      } catch { /* ignore transient errors while dragging sliders */ }
    }, 220);
    return () => clearTimeout(t);
  }, [pp, ppOn, ppSource, setPostFrame, getMaskDataUrl, maskPixels]);

  const regionForSide = (side) => {
    if (splitMethod === "contrast") {
      return side === "foreground" ? "high" : "low";
    }
    return side === "foreground" ? "dark" : "light";
  };

  // The split reads the stack, not the screen: with post-process on the two
  // differ, and the fill this selects for reads the stack.
  const applySplit = async (side = maskSide, target = "active") => {
    if (!frame) { toast("Put an image on the canvas first", "error"); return; }
    if (!splitPreview) { toast("Calculate contrast regions first", "error"); return; }
    setSplitBusy(true);
    try {
      const def = SPLIT_SIDES.find((x) => x.id === side);
      const label = splitMethod === "luminance" ? def?.luminance : def?.contrast;
      const ok = await applyContrastMask(
        frame, { ...splitOpts, region: regionForSide(side) }, { target, name: label || "Contrast" },
      );
      if (ok) {
        setMaskSide(side);
        toast(target === "new"
          ? `Made a mask of the ${(label || side).toLowerCase()} side`
          : `Selected the ${(label || side).toLowerCase()} side`, "success");
      } else {
        toast("Could not build the selection", "error");
      }
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setSplitBusy(false);
    }
  };

  // A generation replaces the canvas, which invalidates any split that was on
  // screen. Recomputing it here means the contrast tiles are ready to use
  // straight away instead of needing a manual Calculate after every run.
  useEffect(() => {
    if (!splitPending || !frame || splitBusy) return;
    setSplitPending(false);
    calculateSplit();
    // calculateSplit is recreated every render; the flag is cleared above, so
    // this cannot re-enter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splitPending, frame]);

  const calculateSplit = async () => {
    if (!frame) { toast("Put an image on the canvas first", "error"); return; }
    setSplitBusy(true);
    try {
      const p = await contrastPreview(frame, splitOpts);
      setSplitPreview(p);
    } catch (e) {
      setSplitPreview(null);
      toast(e.message || "Contrast calculate failed", "error");
    } finally {
      setSplitBusy(false);
    }
  };

  /** Every frame a finished job produced, oldest first. */
  const framesOf = (done) => {
    const frames = done.detail?.frames;
    const framesRaw = done.detail?.frames_raw;
    const cards = done.detail?.cards;
    const cardFor = (i) => cards?.[i] ?? done.detail?.card ?? null;
    if (frames?.length) {
      return frames.map((img, i) => ({ img, raw: framesRaw?.[i] ?? null, card: cardFor(i) }));
    }
    if (done.detail?.frame) {
      return [{ img: done.detail.frame, raw: done.detail.frame_raw ?? null, card: cardFor(0) }];
    }
    return [];
  };

  /** A whole-canvas generation lands in the active layer, replacing what it
   *  had. The layer is the user's; Kiln never adds one. Extra variations go to
   *  Results, from where any of them can be put into a layer.
   *
   *  Unless it was part of a queue (`run.toResults`, decided by Play's landing
   *  rule): then every image goes to Results and the layer is left alone. A
   *  run cancelled from the queue strip (`run.discard`) lands nowhere. */
  const finishGeneration = (done, run = {}) => {
    delete snapshots.current[done.id];
    if (run.discard) return;
    if (done.status === "error") toast(done.message, "error");
    if (run.toResults) {
      if (done.status === "done" || done.status === "cancelled") {
        const out = framesOf(done);
        out.forEach((f) => pushHistory(f.img, f.raw, f.card));
        saveQueued(done, out, run.folder);
      }
      return;
    }
    setLivePreview(null);
    if (done.status === "done") setSplitPending(true);
    if (done.status === "done" || done.status === "cancelled") {
      const out = framesOf(done);
      out.forEach((f, i) => {
        if (i === out.length - 1) generateIntoLayer(f.img, f.raw, f.card);
        else pushHistory(f.img, f.raw, f.card);
      });
      if (out.length > 1) {
        toast(`${out.length} variations — the last is in ${layerName}, the rest in Results`, "success");
      } else if (out.length && done.status === "cancelled") {
        toast(`Stopped — kept in ${layerName}`, "success");
      }
    }
  };

  /** A queue's results are also written to disk, recipe in each PNG.
   *
   *  Results keeps only the last HISTORY_MAX images, and a long queue would
   *  push its first ones out. Saved here, when the run lands, because it is
   *  the landing rule -- decided in the browser -- that sent them to Results,
   *  and it can do that after the server has already finished the run. */
  const saveQueued = async (done, out, folder) => {
    if (!folder) return;
    try {
      for (const f of out) {
        const seed = f.card?.params?.seed;
        await api.post("/perform/capture", {
          image: f.img, card: f.card, folder,
          name: `${done.id}-seed${seed ?? "x"}`,
        });
      }
    } catch (e) { toast(`Could not save to captures/${folder}: ${e.message}`, "error"); }
  };

  /** A fill lands in the active layer, over what the layer had in that area.
   *
   *  Undo takes it back. To keep two attempts side by side, duplicate the
   *  layer first, or make a new one: toggling between them is the comparison.
   */
  const finishFill = async (done, maskSrc, maskIds = []) => {
    delete snapshots.current[done.id];
    if (done.status === "error") toast(done.message, "error");
    if (done.status === "done") setSplitPending(true);
    if (done.status === "done" || done.status === "cancelled") {
      const out = framesOf(done);
      for (let i = 0; i < out.length - 1; i += 1) pushHistory(out[i].img, out[i].raw, out[i].card);
      const pick = out[out.length - 1];
      if (pick) {
        await fillIntoLayer(pick.img, pick.card, maskSrc, { maskIds });
        pushHistory(pick.img, pick.raw, pick.card);
      }
      if (out.length > 1) {
        toast(`${out.length} variations — the last is in ${layerName}, the rest in Results`, "success");
      }
    }
    setLivePreview(null);
  };

  /** One masked run. Returns when the job settles.
   *
   *  `init` is the base the fill works from and `mask` the exact mask sent,
   *  both passed in rather than read here: the mask is what the result is
   *  punched to afterwards, and it has to be the one the backend blended with,
   *  not whatever the overlay holds by the time the job finishes.
   */
  const runFill = async ({ init, mask, batchSize = 1 }) => {
    // The masks that are on now are the ones this fill uses; they turn off
    // when it lands, so remember them here rather than reading the list later.
    const usedMasks = liveMasks.map((m) => m.id);
    if (!modelPath) { toast("Pick a model first", "error"); return null; }
    const bendName = regionBendPreset || genBendPreset || "";
    const genBends = resolveBends(bendPresets, regionBendPreset)
      || resolveBends(bendPresets, genBendPreset);
    // Change and Feather belong to the mask. Brush hardness does not: it is a
    // tool setting, and the first cut let a hard brush force full Change and
    // zero feather on a mask that may have been painted soft.
    const mapped = changeToParams(effectiveRegionChange, sampleParams.steps, true);
    const body = buildInpaintPayload(sampleParams, {
      model_path: modelPath,
      init_image: init,
      mask,
      bends: genBends,
      bend_preset: bendName,
      feather,
      overrides: { ...mapped, resample: canResample ? resample : 1 },
      batch_size: batchSize,
    });
    body.live_preview = livePreviewOn;
    try {
      const { job: j } = await api.post("/perform/inpaint", body);
      snapshots.current[j.id] = snapshotLive();
      const [landed] = trackRuns([j], {
        fill: true, settle: (done) => finishFill(done, mask, usedMasks),
      });
      return await landed;
    } catch (e) {
      toast(e.message, "error"); setLivePreview(null);
      return null;
    }
  };

  const run = async () => {
    if (!modelPath) { toast("Pick a model first", "error"); return; }
    const mask = getMaskDataUrl();

    if (mask && !frame) {
      toast("Put an image on the canvas to fill a mask", "error");
      return;
    }

    const batchSize = Math.max(1, Math.min(4, Math.round(variations) || 1));

    if (mask) {
      // A fill is punched to the mask it was sent with and lands over the layer
      // as it is when it finishes; behind a queue, that canvas is anyone's guess.
      if (runs.length) {
        toast("A fill starts when nothing else is running — wait for the queue, or cancel it", "error");
        return;
      }
      await runFill({ init: frame, mask, batchSize });
      return;
    }
    try {
      const genBends = resolveBends(bendPresets, genBendPreset);
      const mapped = changeToParams(genChange, sampleParams.steps, reworkCanvas);
      const body = buildSamplePayload(sampleParams, {
        model_path: modelPath,
        bends: genBends,
        bend_preset: genBendPreset || "",
        init_image: reworkCanvas ? frame : null,
        postproc: ppOn ? pp : {},
        overrides: mapped,
        batch_size: batchSize,
      });
      body.live_preview = livePreviewOn;
      const { job: j } = await api.post("/perform/sample", body);
      snapshots.current[j.id] = snapshotLive();
      trackRuns([j], { settle: finishGeneration });
    } catch (e) { toast(e.message, "error"); }
  };

  // What the live settings were when the run was paused. Resume sends only what
  // actually changed, so this has to cover every key the panel lets you edit --
  // including the guidance ones, which is how a prompt swap reaches the sampler.
  const snapshotLive = () => Object.fromEntries(
    LIVE_PARAM_KEYS.map((k) => [k, sampleParams[k]]),
  );

  const pause = async () => {
    if (!job) return;
    if (!snapshots.current[job.id]) snapshots.current[job.id] = snapshotLive();
    try {
      await api.post(`/jobs/${job.id}/pause`);
      // Name only the controls actually on screen: Create maps its Change slider
      // onto noise_level, which hides it, and Eta is hidden for the fast solvers.
      const live = joinLabels(
        liveEditLabels(sampleParams, model, { noise_level: "The Change slider in Create" }),
        "or",
      );
      toast(`Paused — edit ${live}, then Resume`, "success");
    } catch (e) { toast(e.message, "error"); }
  };

  const resume = async () => {
    if (!job) return;
    const snap = snapshots.current[job.id] || {};
    const updates = {};
    for (const k of LIVE_PARAM_KEYS) {
      if (sampleParams[k] !== snap[k]) updates[k] = sampleParams[k];
    }
    if (updates.seed === "") updates.seed = null;
    try {
      await api.post(`/jobs/${job.id}/resume`, updates);
      snapshots.current[job.id] = snapshotLive();
    } catch (e) { toast(e.message, "error"); }
  };

  const stopAndSave = async () => {
    if (job) await api.post(`/jobs/${job.id}/cancel`);
  };

  /** Upscaling flattens. Every layer would have to be resampled to stay
   *  aligned, and resampling each one separately is worse than resampling the
   *  composite, so the stack collapses to a single layer at the new size. The
   *  masks stay: their strokes are normalised and replay at the new size. */
  const upscale = async () => {
    const src = frame;
    if (!src) return;
    setSrBusy(true);
    try {
      const r = await api.post("/tools/superres", { image: src, factor: srFactor, sharpen: srSharpen });
      const card = frameCard ? { ...frameCard, upscaled: srFactor } : null;
      await openDocument(r.image, card, `Upscaled ${srFactor}x`, { keepMasks: true });
      pushHistory(r.image, null, card);
      toast(`Upscaled to ${r.size[0]}×${r.size[1]} — layers flattened`, "success");
    } catch (e) { toast(e.message, "error"); }
    setSrBusy(false);
  };

  const setPpField = (k, v) => setPp((s) => ({ ...s, [k]: v }));

  const inFlight = runs.length > 0;
  // Where this click's result goes: Results whenever it joins a queue. More
  // runs means clicking Queue again -- there is no count to confuse with
  // Variations.
  const queueing = !hasMask && inFlight;
  const generateLabel = hasMask ? "Fill mask" : inFlight ? "Queue" : "Generate";

  const runSteps = hasMask
    ? effectiveSteps(effectiveRegionChange, sampleParams.steps, true)
    : effectiveSteps(genChange, sampleParams.steps, reworkCanvas);
  const stepsTrimmed = runSteps < sampleParams.steps;

  const activeMults = (models || []).find((m) => m.path === modelPath)?.mults;
  const fill = hasMask && canvasSize
    ? fillSizeFor(canvasSize.w, canvasSize.h, activeMults)
    : null;
  const scaledFill = fill && canvasSize && (fill.w !== canvasSize.w || fill.h !== canvasSize.h);

  const stepText = stepsTrimmed
    ? `${runSteps} of ${sampleParams.steps} steps`
    : `${runSteps} steps`;
  const generateHint = hasMask
    ? `${maskName} · ${stepText}`
    : reworkCanvas
      ? `Whole canvas · reworked · ${sampleParams.image_size}px · ${stepText}`
      : `Whole canvas · new image · ${sampleParams.image_size}px · ${stepText}`;

  // Each option carries its own stack summary, so hovering one in the dropdown
  // says what it will actually do rather than just naming it.
  const bendOptions = [
    { value: "", label: "— none —", title: "No layer bending." },
    ...bendPresets.map((b) => ({
      value: b.name,
      label: `${b.name} — ${bendPresetSummary(b, ops)}`,
      title: bendPresetSynopsis(b, ops),
    })),
  ];
  const presetByName = (n) => bendPresets.find((b) => b.name === n);
  const bendTip = (selected, base) => {
    const p = presetByName(selected);
    return p ? `${base}\n\n${bendPresetSynopsis(p, ops)}` : base;
  };

  return (
    <div className="col create-panel">
      {/* ——— 1. Generate ——————————————————————————————— */}
      <div className="card">
        <h3>
          <TipLabel tip={hasMask
            ? `${maskName} is on, so this reworks only that area and leaves the rest alone. `
              + "When the fill lands, the mask turns off, so the next run is the whole canvas again; turn it back on to try another model or setting on the same area."
            : canvasIsBlank
              ? "Makes a new image. Mask an area below to rework only that part of the canvas instead."
              : "Reworks the whole canvas by the amount of Change; at full Change it makes a new image. "
                + "Mask an area below to rework only that part instead."}
          >
            Generate
          </TipLabel>
        </h3>

        {/* One Change slider, scoped by whatever is selected. There used to be
            two of these with the same name in different sections, only one of
            which did anything at any moment. */}
        {(hasMask || !canvasIsBlank) && (
          <Slider
            label={hasMask ? `Change · ${maskName}` : "Change · whole canvas"}
            value={changeValue}
            min={0}
            max={1}
            step={0.05}
            disabled={fillLocked}
            onChange={setChangeValue}
            fmt={(v) => (fillLocked
              ? `full · blank canvas · ${sampleParams.steps} steps`
              : !hasMask && v >= 1
                ? `new image · ${sampleParams.steps} steps`
                : `${v < 0.3 ? "subtle" : v < 0.7 ? "medium" : "strong"} · ${effectiveSteps(v, sampleParams.steps, true)} steps`)}
            tip={(hasMask
              ? `How strongly the model restyles ${maskName}.`
              : "How far to move from what is on the canvas. Subtle keeps more of it; strong "
                + "invents more; at full the canvas is ignored and a new image is made.")
              + "\n\nWhat it does is skip steps. The first steps of the schedule are the high-noise "
              + "ones that would wipe out what is already there, so the run starts partway down "
              + "instead: the lower the Change, the more of those steps are skipped and the less is "
              + "altered. Right now it skips "
              + `${skippedSteps(changeValue, sampleParams.steps, true)} of ${sampleParams.steps} steps`
              + ", which is why the run is shorter than the Steps box says. It also sets how much "
              + "extra noise is mixed in at each remaining step."
              + (hasMask
                ? "\n\nThis one belongs to the mask, and is remembered with it."
                : "")
              + (fillLocked
                ? "\n\nThe canvas is blank, so there is nothing to keep and the fill runs the whole "
                  + "schedule. The slider comes back once something is underneath."
                : "")}
          />
        )}

        <div className="row gap-2 wrap">
          <div className="w-100">
            <Num
              label="Variations"
              value={variations}
              onChange={(v) => setVariations(Math.max(1, Math.min(4, Math.round(v) || 1)))}
              min={1}
              max={4}
              step={1}
              tip="How many seeds to sample in one GPU run (1–4). With a fixed Seed, uses seed, seed+1, …. All land together for comparison."
            />
          </div>

          {bendPresets.length > 0 && (
            <div className="grow">
              <Select
                label={hasMask ? `Bend preset · ${maskName}` : "Bend preset"}
                value={bendValue}
                onChange={setBendValue}
                options={bendOptions}
                tip={bendTip(
                  bendValue,
                  hasMask
                    ? `Optional UNet tweaks saved in Create ▸ Bend, applied when filling ${maskName}. Remembered with it.`
                    : "Optional layer tweaks saved in Create ▸ Bend, applied to the whole canvas.",
                )}
              />
            </div>
          )}
        </div>
        {bendValue && presetByName(bendValue) && (
          <p className="hint mb-2 mono-hint">
            {bendPresetSynopsis(presetByName(bendValue), ops)}
          </p>
        )}

        <div className="create-generate-box">
          <span className="sub">
            {genPaused
              ? `Paused · step ${job?.detail?.step || "?"} / ${job?.detail?.total || "?"}`
              : variations > 1
                ? `${generateHint} · ${variations} variations`
                : generateHint}
          </span>
          {/* Where the result lands, and at what size. The layer is chosen in
              the Layers panel; this is only the consequence, said next to the
              button, with the reasons a hover away. */}
          <span className="sub block mt-1">
            {queueing ? (
              <TipLabel tip={
                "While a queue exists, every run lands in Results and the layer is left alone: "
                + "otherwise each run would replace the layer in turn, and which one stayed would "
                + "depend on which finished last. Put any of them into a layer from Results."
                + "\n\nEach image is also saved, with its recipe, to one folder per queue under "
                + "captures, so a long queue cannot push its first images out of Results."
              }
              >
                Into <strong>Results</strong> · saved to captures
              </TipLabel>
            ) : (
            <TipLabel tip={[
              hasMask
                ? `The result lands in ${layerName}, over what that layer has in the masked area. `
                  + "It starts from the canvas as shown: hide layers to start from what is under them."
                : `The result lands in ${layerName}, replacing what it has. `
                  + "Add or duplicate a layer first to keep this attempt apart.",
              fill && canvasSize
                ? (scaledFill
                  ? `Fills at ${fill.w}×${fill.h} and is scaled back to the ${canvasSize.w}×${canvasSize.h} canvas on return.`
                  : `Fills at ${fill.w}×${fill.h}, the canvas's own size.`)
                : null,
              stepsTrimmed
                ? `Keeping part of the image means starting partway down the schedule, so ${sampleParams.steps - runSteps} early steps are skipped. Raise Change to use more.`
                : null,
            ].filter(Boolean).join("\n\n")}
            >
              Into <strong>{layerName}</strong>
              {fill && canvasSize ? ` · ${fill.w}×${fill.h}` : ""}
              {stepsTrimmed ? ` · ${sampleParams.steps - runSteps} steps skipped` : ""}
            </TipLabel>
            )}
          </span>
          {/* Generate never locks: while anything is in flight it queues the
              settings as they are at the click. Fills are the exception, see run(). */}
          <button
            type="button"
            className="btn primary w-full mt-2"
            onClick={run}
            disabled={!modelPath || (hasMask && inFlight)}
            title={hasMask && inFlight ? "A fill starts when nothing else is running" : undefined}
          >
            {generateLabel}
          </button>
          {genRunning && (
            <div className="row gap-2 mt-2">
              {genPaused ? (
                <button type="button" className="btn primary grow" onClick={resume}>Resume</button>
              ) : (
                <button type="button" className="btn grow" onClick={pause}>Pause</button>
              )}
              <button type="button" className="btn danger grow" onClick={stopAndSave}>Stop &amp; save</button>
            </div>
          )}
          {/* The fill's edge belongs to the mask, but it is decided at the moment
              of filling, so it sits with the button that fills. */}
          {hasMask && !genRunning && (
            <div className="fill-edge mt-2">
          <div className="section-title">
            <TipLabel tip="How the fill meets what is around it. Both settings belong to the mask and are remembered with it.">
              Fill edge
            </TipLabel>
          </div>
          <Slider
            label="Feather"
            value={feather}
            min={0}
            max={32}
            step={1}
            disabled={!activeMask}
            onChange={(v) => setMaskParam(activeMask?.id, "feather", Math.round(v))}
            tip={"Soft edge blend where the fill meets the rest of the canvas."
              + (activeMask ? `\n\nRemembered with ${maskName}.` : "\n\nMask an area first.")}
          />

          <Slider
            label={canResample ? "Harmonize" : "Harmonize — needs DDIM or DPM-Solver++"}
            value={canResample ? resample : 1}
            min={1}
            max={8}
            step={1}
            onChange={(v) => setMaskParam(activeMask?.id, "harmonize", Math.round(v))}
            disabled={!canResample || !activeMask}
            fmt={(v) => (v <= 1 ? "off" : `${v} passes · about ${v}x slower`)}
            tip={"Lets the fill settle into what is around it, instead of only matching at the "
              + "edge. On at 2 passes by default; turn it off for the old single-pass behaviour, "
              + "or when you need the speed."
              + "\n\nEach pass is another trip over the same ground, so higher is slower."
              + "\n\nNeeds the DDIM or DPM-Solver++ sampler."}
          />
          {!canResample && (
            <p className="hint mb-0">
              Switch the sampler to DDIM or DPM-Solver++ in Sample settings to enable this.
            </p>
          )}
            </div>
          )}
        </div>
      </div>

      {/* ——— 2. Mask —————————————————————————————————— */}
      <Disclose
        title="Mask"
        defaultOpen
        className="create-section"
        tip="Paint where a run may change things. Masks are rows in the Layers panel and stay until you hide or delete them."
      >
        <div className="row between center wrap gap-2 mb-2">
          <span className="sub grow">
            <TipLabel tip="Strokes go into this mask. Pick another mask row in the Layers panel to paint into that one instead; with no row picked, the topmost mask that is on takes them. Painting into a mask turns it on and shows it.">
              Into <strong>{maskName}</strong>
            </TipLabel>
          </span>
          <div className="row center gap-2">
            <Tooltip text="Step back one edit (Ctrl+Z)">
              <button type="button" className="btn ghost sm icon" onClick={undo} disabled={!canUndo} aria-label="Undo">
                <UndoIcon />
              </button>
            </Tooltip>
            <Tooltip text="Swap masked for unmasked (Ctrl+Shift+I)">
              <button type="button" className="btn ghost sm icon" onClick={() => invertMask(activeMask?.id)} aria-label="Invert">
                <InvertIcon />
              </button>
            </Tooltip>
            <Tooltip text="Empty the mask, keeping its row and settings (Ctrl+D)">
              <button type="button" className="btn ghost sm icon" onClick={() => activeMask && clearMask(activeMask.id)} disabled={!activeMask?.strokes.length} aria-label="Clear">
                <ClearIcon />
              </button>
            </Tooltip>
          </div>
        </div>

        {/* How the mask gets made: one row of tools, one at a time, icons only
            so five fit a sidebar; the name of the one that is on sits beside
            the row, and every button says what it is on hover. What follows
            the row is that tool's own controls, and nothing else. */}
        <div className="row center gap-2 mb-2 tool-row">
          <div className="seg seg-icons" role="group" aria-label="Mask tool">
            {MASK_TOOLS.map((t) => (
              <Tooltip key={t.id} text={t.tip}>
                <button type="button" className={maskTool === t.id ? "on" : ""} onClick={() => setMaskTool(t.id)} aria-label={t.label} aria-pressed={maskTool === t.id}>
                  <t.Icon />
                </button>
              </Tooltip>
            ))}
          </div>
          <span className="tool-name">
            {MASK_TOOLS.find((t) => t.id === maskTool)?.label}
            {maskTool === "shape" && <span className="sub"> · {SHAPE_KINDS.find((k) => k.id === shapeKind)?.label}</span>}
          </span>
        </div>

        {(maskTool === "brush" || maskTool === "wand" || maskTool === "shape") && (
          <div className="row gap-2 mb-2 wrap center">
            {maskTool === "shape" && (
              <div className="seg seg-icons" role="group" aria-label="Shape kind">
                {SHAPE_KINDS.map((k) => (
                  <Tooltip key={k.id} text={k.tip}>
                    <button type="button" className={shapeKind === k.id ? "on" : ""} onClick={() => setShapeKind(k.id)} aria-label={k.label} aria-pressed={shapeKind === k.id}>
                      <k.Icon />
                    </button>
                  </Tooltip>
                ))}
              </div>
            )}
            <div className="seg seg-sm" role="group" aria-label="Paint or erase">
              <button type="button" className={!eraser ? "on" : ""} onClick={() => setEraser(false)} title="Add to the mask"><PaintIcon /> Paint</button>
              <button type="button" className={eraser ? "on" : ""} onClick={() => setEraser(true)} title="Cut out of the mask"><EraseIcon /> Erase</button>
            </div>
            {maskTool === "brush" && (
              <div className="seg seg-sm" role="group" aria-label="Brush edge">
                <button type="button" className={!brushHard ? "on" : ""} onClick={() => setBrushHard(false)}>Soft</button>
                <button type="button" className={brushHard ? "on" : ""} onClick={() => setBrushHard(true)}>Hard</button>
              </div>
            )}
          </div>
        )}

        {maskTool === "brush" && (
          <Slider label="Size" value={brushSize} min={8} max={160} step={2} onChange={setBrushSize}
            tip="Brush diameter in canvas pixels." />
        )}

        {maskTool === "wand" && (
          <Slider
            label="Tolerance"
            value={wandTolerance}
            min={0}
            max={100}
            step={1}
            onChange={setWandTolerance}
            tip="How similar a neighboring pixel's color must be to join the mask. Click the canvas to add it."
          />
        )}

        {maskTool === "move" && (
          <p className="hint mb-2">
            <TipLabel tip={`Drag anywhere on the canvas to move ${maskName}, or press the arrow keys: 1 px, or 10 with Shift. With any tool, the label on the mask's box drags it too.`}>
              Drag the mask, or use the arrow keys
            </TipLabel>
          </p>
        )}

        {maskTool === "shape" && (
          shapeKind === "pattern" ? (
            <PatternPanel
              genShape={genShape}
              setGenShape={setGenShape}
              canvasSize={canvasSize}
              eraser={eraser}
              addStroke={addStroke}
              addMaskWithStroke={addMaskWithStroke}
              maskName={maskName}
            />
          ) : (
            <p className="hint mb-2">
              <TipLabel tip={shapeKind === "polygon"
                ? "Each click places a corner. Enter, a double-click, or a click on the first corner closes it; Backspace removes the last corner; Esc abandons it. Erase makes a polygon that cuts out of the mask instead."
                : "Press and drag on the canvas. Shift keeps it square or round. Erase makes a shape that cuts out of the mask instead. Shapes are hard-edged: Feather softens the fill at its edge."}
              >
                {shapeKind === "polygon"
                  ? (polyCount ? `${polyCount} corner${polyCount === 1 ? "" : "s"} placed — Enter closes, Esc cancels` : "Click on the canvas to place corners")
                  : `Drag on the canvas to draw ${shapeKind === "ellipse" ? "an ellipse" : "a rectangle"}`}
              </TipLabel>
            </p>
          )
        )}

        {maskTool === "contrast" && (
          <div className="contrast-arm">
            <div className="row gap-2">
              <div className="grow">
                <Select label="Split by" value={splitMethod} onChange={setSplitMethod} options={SPLIT_METHODS}
                  tip="Brightness splits light from dark; local contrast splits busy areas from flat ones." />
              </div>
              <div className="grow">
                {splitMethod === "luminance" ? (
                  <Slider label="Threshold" value={brightnessThreshold} min={1} max={255} step={1}
                    onChange={setBrightnessThreshold}
                    tip="Brightness level (1–255) that separates the two sides." />
                ) : (
                  <Slider label="Threshold" value={contrastThreshold} min={-40} max={40} step={1}
                    onChange={setContrastThreshold}
                    tip="Nudge the auto-detected local-contrast split." />
                )}
              </div>
            </div>
            <div className="row gap-2">
              <div className="grow">
                <Slider label="Edge soften" value={splitSoften} min={0} max={8} step={1} onChange={setSplitSoften}
                  tip="Softens the edge of the generated region mask." />
              </div>
              <div className="grow">
                <label className="row center gap-2 mb-2 mt-2">
                  <input type="checkbox" checked={smoothOn} onChange={(e) => setSmoothOn(e.target.checked)} />
                  <span className="sub">Blur first (fewer speckles)</span>
                </label>
                {smoothOn && (
                  <Slider label="Blur radius" value={smoothRadius} min={1} max={6} step={1} onChange={setSmoothRadius}
                    tip="Blur the canvas before calculating the split, to reduce speckles." />
                )}
              </div>
            </div>
            <div className="row gap-2 mb-2 center">
              <button
                type="button"
                className="btn sm"
                onClick={calculateSplit}
                // splitPreview is cleared by the effect above whenever the canvas or
                // any split setting changes, so "we have a preview" is exactly "and
                // nothing has moved since". No second piece of state to keep in sync.
                disabled={!frame || splitBusy || !!splitPreview}
              >
                {splitBusy ? "Calculating…" : splitPreview ? "Up to date" : "Calculate"}
              </button>
              {splitPreview && (
                <span className="sub">Split at {splitPreview.cut}{splitMethod === "luminance" ? " brightness" : " contrast"}</span>
              )}
              {!frame && <span className="sub">Put an image on the canvas first</span>}
            </div>
            <div className="contrast-split-previews">
              {SPLIT_SIDES.map((side) => (
                <button
                  key={side.id}
                  type="button"
                  className={`contrast-split-tile ${maskSide === side.id ? "on" : ""}`}
                  onClick={() => setMaskSide(side.id)}
                  disabled={!splitPreview}
                >
                  {splitPreview?.[side.id]
                    ? <img src={splitPreview[side.id]} alt={`${side.label} preview`} />
                    : <span className="sub">{side.label}</span>}
                  <span className="contrast-split-label">
                    {splitMethod === "luminance" ? side.luminance : side.contrast}
                  </span>
                </button>
              ))}
            </div>
            <AddToMaskButtons
              thing="side"
              maskName={maskName}
              eraser={false}
              busy={splitBusy && !!splitPreview}
              disabled={!splitPreview || splitBusy}
              onAdd={() => applySplit(maskSide, "active")}
              onNew={() => applySplit(maskSide, "new")}
              addTip="Replaces any earlier contrast side in this mask; brush and shape strokes stay."
            />
          </div>
        )}

        {!hasMask && (
          <p className="hint mb-0 mt-2">Paint a mask to fill an area.</p>
        )}
      </Disclose>

      {/* ——— 3. Finish ————————————————————————————————— */}
      <Disclose
        title="Finish"
        defaultOpen
        className="create-section"
        extra={ppOn ? <span className="pill on">post-process on</span> : null}
        tip={"Adjustments applied to the finished image, not to sampling. "
          + "Post-process follows the mask when one is on; upscale always uses the whole canvas."}
      >
        <div className="row between center wrap gap-2 mb-2">
          <span className="sub grow">
            <TipLabel tip={hasMask
              ? "With a mask on, the adjustments apply inside it only. They are a display stage over the layers, not an edit to them, so Download and Capture bake them in but the layers stay as they are."
              : "Applied to the whole canvas as a display stage over the layers, not an edit to them: Download and Capture bake it in, the layers stay as they are."}
            >
              {hasMask ? `Inside ${maskName}` : "Whole canvas"}
            </TipLabel>
          </span>
          <label className="row center gap-2">
            <input
              type="checkbox"
              checked={ppOn}
              onChange={(e) => setPpOn(e.target.checked)}
            />
            <span className="sub">Post-process</span>
          </label>
        </div>

        <Slider label="Contrast" value={pp.contrast} min={0.5} max={2} step={0.05}
          onChange={(v) => setPpField("contrast", v)} disabled={!ppOn}
          tip="Boost or flatten contrast after sampling." />
        <Slider label="Gamma" value={pp.gamma} min={0.5} max={2} step={0.05}
          onChange={(v) => setPpField("gamma", v)} disabled={!ppOn}
          tip="Brighten (lower) or darken (higher) midtones." />
        <Slider label="Sharpen" value={pp.unsharp} min={0} max={4} step={0.1}
          onChange={(v) => setPpField("unsharp", v)} disabled={!ppOn}
          tip="Unsharp-mask strength on the finished image." />

        <div className="section-title mt-2">
          <TipLabel tip="Enlarges the finished image with Lanczos resampling. Ignores masks, and flattens the layer stack into one layer at the new size; the masks are kept.">
            Upscale
          </TipLabel>
        </div>
        <div className="row gap-2 center">
          <div className="w-100">
            <Num label="Scale" value={srFactor} onChange={setSrFactor} min={2} max={4} step={1}
              tip="Upscale factor for the canvas image." />
          </div>
          <div className="grow">
            <Slider label="Sharpen" value={srSharpen} min={0} max={3} step={0.1} onChange={setSrSharpen}
              tip="Unsharp-mask strength applied after enlarging." />
          </div>
          <button
            type="button"
            className="btn sm self-end mb-2"
            onClick={upscale}
            disabled={srBusy || !frame}
          >
            {srBusy ? "…" : "Upscale"}
          </button>
        </div>
      </Disclose>
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
 *  a composition is several masks on at once, and the bar already says so. */
function AddToMaskButtons({ onAdd, onNew, eraser, busy, disabled, maskName, thing, addTip }) {
  const addText = eraser ? `Cut this ${thing} out of ${maskName}` : `Add this ${thing} to ${maskName}`;
  return (
    <div className="add-pair mt-2">
      <Tooltip text={addTip ? `${addText}. ${addTip}` : addText}>
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
