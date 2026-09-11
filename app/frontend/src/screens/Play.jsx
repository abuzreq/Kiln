import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CreatePanel from "../components/CreatePanel.jsx";
import { BendWorkspace } from "./Craft.jsx";
import SweepPanel from "./Sweep.jsx";
import Merge from "./Merge.jsx";
import { useApp } from "../state.jsx";
import { api, downloadPost } from "../api.js";
import { Progress, Tooltip } from "../components/ui.jsx";
import { PlayCtx, usePlay, fileToDataUrl } from "./playContext.jsx";
import {
  loadSampleParams, saveSampleParams, SampleSettingsPanel, paramsFromCard, cardLabel,
  LIVE_PARAM_KEYS,
} from "../sampleSettings.jsx";
import ModelPicker from "../components/ModelPicker.jsx";
import { buildContrastMask, floodFillMask, countMaskPixels } from "../contrastMask.js";
import {
  newSelection, brushStroke, cachedStroke, invertStroke, alphaToCache, overlayToCache,
  rasterize, paintBrushPoint,
} from "../selection.js";

const HISTORY_MAX = 24;
// Snapshots used to be PNG data URLs, one per mask edit. Undo is now "drop the
// last stroke and replay", so an entry is a stroke list -- cheap enough to keep
// more of, and it covers deselecting as well as painting.
const SELECTION_UNDO_MAX = 40;
const CANVAS_TABS = new Set(["create"]);

// Canvas bounds. The floor is a size a brush can still be aimed inside; the
// ceiling is well past what these models sample at, and a fill is scaled down
// to MAX_FILL_SIDE anyway, so nothing is gained by going bigger.
const CANVAS_MIN = 64;
const CANVAS_MAX = 2048;

function clampCanvasSide(v) {
  const n = Math.round(Number(v) || 0);
  if (!n) return CANVAS_MIN;
  return Math.max(CANVAS_MIN, Math.min(CANVAS_MAX, n));
}

/** A transparent-black PNG — the ground a composition is painted on. */
function blankImage(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  // New canvases are already (0,0,0,0); clear so that stays true if a browser
  // ever changes the default.
  c.getContext("2d").clearRect(0, 0, w, h);
  return c.toDataURL("image/png");
}

export default function Play() {
  const {
    toast, playTab, bendStack, setBendStack, modelPath, setModelPath, setTabBusy, models,
  } = useApp();
  const tab = playTab;
  // The settings panel needs the selected model, not just its path: whether the
  // EMA toggle does anything, and what image sizes this model can actually
  // produce, are properties of the model rather than of the sampler.
  const activeModel = useMemo(
    () => (models || []).find((m) => m.path === modelPath) || null,
    [models, modelPath],
  );
  const [frame, setFrameState] = useState(null);
  const [frameRaw, setFrameRaw] = useState(null);
  const [frameCard, setFrameCard] = useState(null);
  // Per-tab state that has to outlive the tab's own component (see usePlayState).
  const [tabState, setTabState] = useState({});
  const [pendingCard, setPendingCard] = useState(null);
  const [showRaw, setShowRaw] = useState(false);
  const [initImage, setInitImage] = useState(null);
  const [history, setHistory] = useState([]);
  const [progress, setProgress] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [brushSize, setBrushSize] = useState(48);
  const [brushHard, setBrushHard] = useState(false);
  const [eraser, setEraser] = useState(false);
  const [maskTool, setMaskTool] = useState("brush");
  const [wandTolerance, setWandTolerance] = useState(30);
  const [maskVersion, setMaskVersion] = useState(0);
  // The selection the user owns: null when nothing is selected. Its strokes are
  // the source of truth; the mask canvas is a cache of replaying them.
  const [selection, setSelection] = useState(null);
  const [undoLen, setUndoLen] = useState(0);
  // Candidates from a fill, held until accepted or discarded, so trying a
  // second model does not destroy the result of the first.
  const [staging, setStaging] = useState(null);
  // In-progress frames during a staged run, kept off `frame` so a live preview
  // cannot overwrite the image the fill is running against.
  const [livePreview, setLivePreview] = useState(null);
  const [job, setJob] = useState(null);
  // The blank canvas people paint regions onto. Its size is remembered here
  // rather than read from the sampler's image size: a composition can be any
  // shape, and region fill works at the canvas's own aspect anyway.
  // The exact data URL of the blank ground newCanvas last laid down. "Is the
  // canvas still blank" is then a question about the image itself, not about who
  // called setFrame -- the postprocessing effect re-sets the current frame
  // unchanged on mount, and a flag flipped inside setFrame was cleared by that
  // before anything had actually been drawn.
  const blankFrameRef = useRef(null);
  const [canvasSize, setCanvasSizeState] = useState(() => {
    const n = clampCanvasSide(loadSampleParams().image_size || 512);
    return { w: n, h: n };
  });
  const maskRef = useRef(null);
  const heroRef = useRef(null);
  const syncMaskOverlayRef = useRef(() => {});
  const undoRef = useRef([]);
  // A mirror of `selection`, so the pointer handlers that add strokes can read
  // the current value without going stale between renders.
  const selectionRef = useRef(null);
  const [sampleParams, setSampleParamsState] = useState(loadSampleParams);

  const setTabStateKey = useCallback((key, valueOrFn, initial) => {
    setTabState((s) => {
      const prev = key in s ? s[key] : initial;
      const next = typeof valueOrFn === "function" ? valueOrFn(prev) : valueOrFn;
      return Object.is(prev, next) ? s : { ...s, [key]: next };
    });
  }, []);

  const bumpMask = useCallback(() => setMaskVersion((v) => v + 1), []);
  useEffect(() => { selectionRef.current = selection; }, [selection]);

  /** Replay a selection onto the mask canvas at its current size. */
  const rasterizeNow = useCallback((sel, size) => {
    const c = maskRef.current;
    if (!c) return;
    const w = size?.w || c.width;
    const h = size?.h || c.height;
    if (!w || !h) return;
    rasterize(sel, c, w, h);
    syncMaskOverlayRef.current();
    bumpMask();
  }, [bumpMask]);

  const pushUndo = useCallback((snap) => {
    undoRef.current = [snap, ...undoRef.current].slice(0, SELECTION_UNDO_MAX);
    setUndoLen(undoRef.current.length);
  }, []);

  const clearUndo = useCallback(() => {
    undoRef.current = [];
    setUndoLen(0);
  }, []);

  /** Append a stroke. The caller has usually already drawn it live, so this
   *  records it for undo and replay rather than repainting. */
  const addStroke = useCallback((stroke, { redraw = false } = {}) => {
    const prev = selectionRef.current;
    pushUndo(prev);
    const base = prev || newSelection();
    const next = { ...base, enabled: true, strokes: [...base.strokes, stroke] };
    selectionRef.current = next;
    setSelection(next);
    if (redraw) rasterizeNow(next);
    else bumpMask();
  }, [pushUndo, rasterizeNow, bumpMask]);

  const undoSelection = useCallback(() => {
    const stack = undoRef.current;
    if (!stack.length) return;
    const [prev, ...rest] = stack;
    undoRef.current = rest;
    setUndoLen(rest.length);
    selectionRef.current = prev;
    setSelection(prev);
    rasterizeNow(prev);
  }, [rasterizeNow]);

  /** Drop the selection. Undoable, so a mis-click is not destructive. */
  const deselect = useCallback(() => {
    const prev = selectionRef.current;
    if (!prev) return;
    pushUndo(prev);
    selectionRef.current = null;
    setSelection(null);
    rasterizeNow(null);
  }, [pushUndo, rasterizeNow]);

  /** Invert on an empty selection means "select everything", which is useful. */
  const invertSelection = useCallback(() => {
    addStroke(invertStroke(), { redraw: true });
  }, [addStroke]);

  const setSelectionEnabled = useCallback((on) => {
    const prev = selectionRef.current;
    if (!prev) return;
    const next = { ...prev, enabled: !!on };
    selectionRef.current = next;
    setSelection(next);
  }, []);

  const renameSelection = useCallback((name) => {
    const prev = selectionRef.current;
    if (!prev) return;
    const next = { ...prev, name: name || prev.name };
    selectionRef.current = next;
    setSelection(next);
  }, []);

  const setSelectionParam = useCallback((k, v) => {
    const prev = selectionRef.current;
    if (!prev) return;
    const next = { ...prev, params: { ...prev.params, [k]: v } };
    selectionRef.current = next;
    setSelection(next);
  }, []);

  const setSampleParam = useCallback((k, v) => {
    setSampleParamsState((s) => {
      const next = { ...s, [k]: v };
      saveSampleParams(next);
      return next;
    });
  }, []);

  const mergeSampleParams = useCallback((patch) => {
    setSampleParamsState((s) => {
      const next = { ...s, ...patch };
      saveSampleParams(next);
      return next;
    });
  }, []);

  const pushHistory = useCallback((img, raw, card) => {
    if (!img) return;
    setHistory((h) => [
      {
        id: Math.random().toString(36).slice(2),
        img,
        raw: raw || null,
        card: card || null,
        at: Date.now(),
      },
      ...h.filter((x) => x.img !== img),
    ].slice(0, HISTORY_MAX));
  }, []);

  const setFrame = useCallback((img, raw, card) => {
    setFrameState(img || null);
    if (raw !== undefined) setFrameRaw(raw || null);
    if (card !== undefined) setFrameCard(card || null);
  }, []);

  const commitFrame = useCallback((img, raw, card) => {
    setFrame(img, raw, card);
    pushHistory(img, raw, card);
  }, [setFrame, pushHistory]);

  /** Put an earlier result back on the canvas, keeping the selection so you can
   *  branch from it. Any candidates still being compared are abandoned — the
   *  image they were filled from is the thing being replaced. */
  const restoreHistory = useCallback((entry) => {
    if (!entry) return;
    setStaging(null);
    setLivePreview(null);
    setFrame(entry.img, entry.raw ?? null, entry.card ?? null);
  }, [setFrame]);

  const removeHistory = useCallback((id) => {
    setHistory((h) => h.filter((x) => x.id !== id));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  /** Set (or drop) the init image.
   *
   *  This used to clear the selection, on the grounds that a mask belongs to
   *  the image it was drawn on. But filling a selection reads the *canvas*, not
   *  the init — an init only affects a full-canvas generation, so it cannot
   *  invalidate a selection drawn over the canvas. The selection stays.
   */
  const applyInit = useCallback((src) => {
    setInitImage(src || null);
  }, []);

  const useAsInit = useCallback(() => {
    const src = showRaw && frameRaw ? frameRaw : frame;
    if (!src) return;
    applyInit(src);
    toast("Canvas set as init", "success");
  }, [frame, frameRaw, showRaw, applyInit, toast]);

  const useHistoryAsInit = useCallback((entry) => {
    if (!entry?.img) return;
    applyInit(entry.raw || entry.img);
    toast("Set as init", "success");
  }, [applyInit, toast]);

  /** Drop the init image, keeping whatever is on the canvas. */
  const clearInit = useCallback(() => {
    applyInit(null);
    toast("Init cleared", "success");
  }, [applyInit, toast]);

  /** A blank canvas to paint on: image, init, selection and recipe all reset.
   *
   *  Deliberately *not* set as the init image. An untouched blank canvas is a
   *  surface to select on, not a picture to work from — with nothing selected,
   *  Generate should make a new image rather than img2img from empty pixels.
   *
   *  This is the one place a selection is dropped without the user saying so,
   *  and it is the right one: the canvas it belonged to no longer exists.
   */
  const newCanvas = useCallback((w, h, { quiet = false } = {}) => {
    const width = clampCanvasSide(w);
    const height = clampCanvasSide(h);
    const ground = blankImage(width, height);
    blankFrameRef.current = ground;
    setCanvasSizeState({ w: width, h: height });
    setFrameState(ground);
    setFrameRaw(null);
    setFrameCard(null);
    setPendingCard(null);
    setStaging(null);
    setLivePreview(null);
    applyInit(null);
    selectionRef.current = null;
    setSelection(null);
    rasterizeNow(null);
    clearUndo();
    if (!quiet) toast(`Blank canvas — ${width}x${height}`, "success");
  }, [applyInit, toast, rasterizeNow, clearUndo]);

  /** Resizing is the same act as starting over: a new blank ground at that size. */
  const setCanvasSize = useCallback((patch) => {
    setCanvasSizeState((prev) => ({
      w: clampCanvasSide(patch.w ?? prev.w),
      h: clampCanvasSide(patch.h ?? prev.h),
    }));
  }, []);

  /** Wipe back to a blank canvas at the size already set. */
  const clearCanvas = useCallback(() => {
    newCanvas(canvasSize.w, canvasSize.h, { quiet: true });
    toast("Canvas cleared", "success");
  }, [newCanvas, canvasSize.w, canvasSize.h, toast]);

  /** Put an image's recorded recipe back into the sampler (and pick its model). */
  const applyCard = useCallback((card) => {
    const params = paramsFromCard(card);
    if (!params) { toast("This image has no Kiln settings", "warn"); return false; }
    mergeSampleParams(params);
    if (card.model_path && card.model_path !== modelPath) setModelPath(card.model_path);
    setPendingCard(null);
    toast(`Settings restored — ${cardLabel(card) || "sampler updated"}`, "success");
    return true;
  }, [mergeSampleParams, modelPath, setModelPath, toast]);

  const lockSeed = useCallback((seed) => {
    if (seed == null) return;
    setSampleParam("seed", seed);
    toast(`Seed locked to ${seed}`, "success");
  }, [setSampleParam, toast]);

  const loadFile = useCallback(async (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const url = await fileToDataUrl(file);
    // FileReader gives us the file's raw bytes, so any recipe Kiln embedded on
    // export survives the trip and can be offered back to the user.
    let card = null;
    try {
      const r = await api.post("/perform/read-params", { image: url });
      card = r.card || null;
    } catch { /* not fatal — the image still lands on the canvas */ }
    setStaging(null);
    setLivePreview(null);
    setFrame(url, null, card);
    applyInit(url);
    pushHistory(url, null, card);
    setPendingCard(card && card.params ? card : null);
    toast(card?.params ? "On canvas — Kiln settings found" : "On canvas", "success");
  }, [setFrame, pushHistory, toast]);

  /** The mask as the backend wants it: white where the fill should happen.
   *
   *  Returns null for a disabled selection, which is how the enable toggle
   *  works without every caller having to check it.
   */
  const getMaskDataUrl = useCallback(() => {
    const c = maskRef.current;
    if (!c || !selection?.enabled) return null;
    const ctx = c.getContext("2d");
    const img = ctx.getImageData(0, 0, c.width, c.height);
    const { data } = img;
    let painted = false;
    for (let i = 0; i < data.length; i += 4) {
      const a = data[i + 3];
      if (a > 8) painted = true;
      data[i] = data[i + 1] = data[i + 2] = a;
      data[i + 3] = 255;
    }
    if (!painted) return null;
    const out = document.createElement("canvas");
    out.width = c.width;
    out.height = c.height;
    out.getContext("2d").putImageData(img, 0, 0);
    return out.toDataURL("image/png");
  }, [selection]);

  /** A contrast split becomes one stroke, replacing whatever came before it.
   *
   *  Replacing rather than adding is what the two preview tiles imply: picking
   *  the other side should give you the other side, not both sides at once.
   */
  const applyContrastMask = useCallback(async (imageSrc, options) => {
    const maskCanvas = maskRef.current;
    const hero = heroRef.current;
    const img = hero?.querySelector("img");
    if (!maskCanvas || !img || !imageSrc) return false;

    await new Promise((resolve) => {
      if (img.complete && img.naturalWidth > 0) resolve();
      else img.onload = () => resolve();
    });

    const { overlay, w, h } = await buildContrastMask(imageSrc, options);
    if (!overlay || !w || !h) return false;

    const stroke = cachedStroke("split", overlayToCache(overlay, w, h), "add");
    const prev = selectionRef.current;
    pushUndo(prev);
    const base = prev || newSelection();
    const kept = base.strokes.filter((s) => s.type !== "split");
    const next = { ...base, enabled: true, strokes: [...kept, stroke] };
    selectionRef.current = next;
    setSelection(next);
    rasterizeNow(next, { w, h });
    return true;
  }, [pushUndo, rasterizeNow]);

  const applyFloodFillMask = useCallback(async (imageSrc, x, y, options = {}) => {
    const maskCanvas = maskRef.current;
    if (!maskCanvas || !imageSrc) return false;

    const erase = !!options.eraser;
    const { alpha, w, h } = await floodFillMask(imageSrc, {
      x, y,
      tolerance: options.tolerance ?? wandTolerance,
      soften: options.soften ?? 0,
    });
    if (!alpha || !w || !h) return false;

    // The flood fill is baked in rather than re-evaluated on replay: it is a
    // function of the pixels underneath, and a fill changes those, so a live
    // recomputation would move the selection under the user.
    const stroke = cachedStroke("wand", alphaToCache(alpha, w, h), erase ? "subtract" : "add");
    if (maskCanvas.width !== w || maskCanvas.height !== h) {
      addStroke(stroke, { redraw: true });
      return true;
    }
    // Same size, so the new stroke can be composited straight on rather than
    // replaying the whole list.
    const ctx = maskCanvas.getContext("2d");
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    ctx.drawImage(stroke.cache, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    syncMaskOverlayRef.current();
    addStroke(stroke);
    return true;
  }, [wandTolerance, addStroke]);

  // Busy state is derived here rather than inside each tab: Play stays mounted,
  // so the indicator keeps updating after you navigate away from a running job.
  const busyByTab = {
    create: !!(job && job.status === "running"),
    // the compare and the parameter sweep are both long Bend runs
    bend: !!tabState["bend.busy"] || !!tabState["bend.sweep"]?.busy,
    merge: !!tabState["merge.busy"],
    sweep: tabState["sweep.job"]?.status === "running",
  };
  useEffect(() => {
    Object.entries(busyByTab).forEach(([t, b]) => setTabBusy(t, b));
  }, [busyByTab.create, busyByTab.bend, busyByTab.merge, busyByTab.sweep, setTabBusy]);

  // On the Sweep tab these parameters are decided per cell by the grid, so the
  // shared panel should stop pretending its values apply.
  const sweptParams = [
    tabState["sweep.param"] ?? "seed",
    tabState["sweep.twoD"] ? (tabState["sweep.param2"] ?? "eta") : null,
  ].filter(Boolean);

  /** Stage fill results instead of committing them.
   *
   *  `frame` stays the image the fill ran against, so a second fill with a
   *  different model starts from the same place rather than stacking on top of
   *  the first one's output. That is what makes comparing models possible.
   */
  const stageResults = useCallback((items, base) => {
    if (!items?.length) return;
    setLivePreview(null);
    setStaging((prev) => {
      if (prev) {
        return { ...prev, items: [...prev.items, ...items], index: prev.items.length };
      }
      return { base, items, index: 0 };
    });
  }, []);

  const setStageIndex = useCallback((i) => {
    setStaging((prev) => {
      if (!prev) return prev;
      const n = prev.items.length;
      return { ...prev, index: ((i % n) + n) % n };
    });
  }, []);

  const acceptStaging = useCallback(() => {
    setStaging((prev) => {
      const pick = prev?.items?.[prev.index];
      if (pick) commitFrame(pick.img, pick.raw, pick.card);
      return null;
    });
    setLivePreview(null);
  }, [commitFrame]);

  const discardStaging = useCallback(() => {
    setStaging((prev) => {
      if (prev) setFrame(prev.base.img, prev.base.raw, prev.base.card);
      return null;
    });
    setLivePreview(null);
  }, [setFrame]);

  const staged = staging ? staging.items[staging.index] : null;
  // What the fill runs against: the pre-fill image while staging, so trying a
  // second model compares rather than compounds.
  const fillBase = staging ? staging.base : { img: frame, raw: frameRaw, card: frameCard };
  const displayFrame = livePreview || (staged ? staged.img : frame);
  const displayRaw = livePreview ? null : (staged ? staged.raw : frameRaw);
  const canvasImage = showRaw && displayRaw ? displayRaw : displayFrame;
  // Still the untouched blank ground: nothing has been generated, dropped or
  // restored over it. A fill has to know, because a partial run started
  // from empty pixels gives back the empty pixels it started from.
  const canvasIsBlank = !!frame && frame === blankFrameRef.current && !staging;
  const activeSeed = (staged ? staged.card : frameCard)?.params?.seed ?? null;
  const genRunning = !!(job && job.status === "running");
  const genPaused = genRunning && !!job?.detail?.paused;
  const canUndoSelection = undoLen > 0;
  const selectionPixels = useMemo(
    () => countMaskPixels(maskRef.current),
    // maskVersion is the signal; the pixels live on a ref.
    [maskVersion],
  );
  const hasSelection = !!selection?.enabled && selectionPixels > 0;

  // Play opens on a blank canvas rather than an empty box, so the brush and
  // region tools have something to work on from the first moment. Once only:
  // Clear canvas makes its own blank one, and re-running this on every empty
  // frame would make clearing look like it had failed.
  const startedBlank = useRef(false);
  useEffect(() => {
    if (startedBlank.current) return;
    startedBlank.current = true;
    if (!frame) newCanvas(canvasSize.w, canvasSize.h, { quiet: true });
  }, [frame, newCanvas, canvasSize.w, canvasSize.h]);

  useEffect(() => {
    const onPaste = (e) => {
      const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
      if (f) { e.preventDefault(); loadFile(f); }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [loadFile]);

  const value = {
    tab, frame, frameRaw, showRaw, setShowRaw, canvasImage, initImage, setInitImage,
    applyInit, clearInit, clearCanvas,
    frameCard, setFrameCard, pendingCard, setPendingCard, applyCard, lockSeed, activeSeed,
    history, setFrame, commitFrame, pushHistory, restoreHistory, removeHistory, clearHistory,
    useAsInit, useHistoryAsInit, loadFile,
    progress, setProgress,
    brushSize, setBrushSize, brushHard, setBrushHard, eraser, setEraser,
    maskTool, setMaskTool, wandTolerance, setWandTolerance,
    maskRef, heroRef,
    syncMaskOverlayRef, getMaskDataUrl, applyContrastMask, applyFloodFillMask,
    selection, selectionPixels, hasSelection, addStroke, rasterizeNow,
    deselect, invertSelection, setSelectionEnabled, renameSelection, setSelectionParam,
    undoSelection, canUndoSelection,
    staging, staged, fillBase, stageResults, setStageIndex, acceptStaging, discardStaging,
    setLivePreview,
    sampleParams, setSampleParam, mergeSampleParams, maskVersion, bumpMask,
    canvasSize, setCanvasSize, newCanvas, canvasIsBlank,
    tabState, setTabStateKey,
    job, setJob, genRunning, genPaused,
  };

  return (
    <PlayCtx.Provider value={value}>
      <div className="play">
        <div className="play-top">
          <div className="card play-model-picker">
            <ModelPicker />
          </div>
          <div className="play-sample-settings">
            <SampleSettingsPanel
              params={sampleParams}
              setParam={setSampleParam}
              model={activeModel}
              // Create maps its Change slider onto skip + noise_level and passes
              // them as overrides, so whatever the panel shows for noise level
              // is discarded on that tab. Saying so beats a slider that quietly
              // does nothing.
              overriddenBy={tab === "create" ? { noise_level: "The Change slider in Create" } : null}
              sweptBy={tab === "sweep" ? sweptParams : null}
              disabled={genRunning && !genPaused}
              editable={genPaused ? LIVE_PARAM_KEYS : null}
              stepsMin={genPaused ? (job?.detail?.step || 1) : undefined}
            />
          </div>
        </div>
        <div className={`play-body ${CANVAS_TABS.has(tab) ? "" : "solo"}`}>
          {tab === "create" && <div className="play-tools"><CreatePanel /></div>}
          {CANVAS_TABS.has(tab) && (
            <div
              className={`play-stage ${dragOver ? "drop-on" : ""} ${history.length ? "with-results" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; if (f) loadFile(f); }}
            >
              <PlayCanvas brushable={!!canvasImage} />
              <ResultsRail />
            </div>
          )}
          {tab === "bend" && <div className="play-solo"><BendWorkspace stack={bendStack} setStack={setBendStack} /></div>}
          {tab === "merge" && <div className="play-solo"><Merge /></div>}
          {tab === "sweep" && <div className="play-solo"><SweepPanel /></div>}
        </div>
      </div>
    </PlayCtx.Provider>
  );
}

function PlayCanvas({ brushable }) {
  const { toast } = useApp();
  const {
    frameRaw, showRaw, setShowRaw, initImage, progress, heroRef, canvasImage,
    frameCard, pendingCard, setPendingCard, applyCard, activeSeed, useAsInit,
    clearInit, clearCanvas, canvasSize, setCanvasSize, newCanvas, loadFile,
    selection, selectionPixels, deselect, invertSelection, setSelectionEnabled,
    renameSelection, staging, staged,
  } = usePlay();
  const shown = canvasImage;
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const openRef = useRef(null);
  // While staging, everything in this header describes the candidate on screen
  // rather than the image it was filled from.
  const shownCard = staged ? staged.card : frameCard;
  const shownRaw = staged ? staged.raw : frameRaw;

  const download = async () => {
    if (!shown) return;
    setBusy(true);
    try {
      // Routed through the API so the PNG keeps its embedded recipe — a plain
      // <a download> on the canvas data URL would hand over a stripped file.
      const seed = shownCard?.params?.seed;
      await downloadPost(
        "/perform/export",
        { image: shown, card: shownCard, filename: seed != null ? `kiln-${seed}` : "kiln" },
        seed != null ? `kiln-${seed}.png` : "kiln.png",
      );
    } catch (e) { toast(e.message, "error"); }
    setBusy(false);
  };

  return (
    <div className="col">
      <div className="card canvas-card">
        <div className="row between center mb-2">
          <h3 className="mb-0">Canvas</h3>
          <div className="row center gap-2">
            {progress && <span className="sub">{progress.message}</span>}
            {activeSeed != null && !progress && (
              <span className="pill mono" title="Seed that produced this image">seed {activeSeed}</span>
            )}
            {shownRaw && (
              <Tooltip text="Show the image straight from the sampler, before post-processing and upscaling. Useful for judging what the model actually produced.">
                <label className="row center gap-1 has-tip">
                  <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} />
                  <span className="sub">Unprocessed</span>
                </label>
              </Tooltip>
            )}
          </div>
        </div>
        {progress && <Progress value={progress.value} />}
        {pendingCard && (
          <div className="callout row between center wrap gap-2">
            <span>
              This image carries Kiln settings{cardLabel(pendingCard) ? ` — ${cardLabel(pendingCard)}` : ""}.
            </span>
            <div className="row gap-2">
              <button type="button" className="btn sm primary" onClick={() => applyCard(pendingCard)}>
                Restore settings
              </button>
              <button type="button" className="btn ghost sm" onClick={() => setPendingCard(null)}>
                Dismiss
              </button>
            </div>
          </div>
        )}
        {selection && (
          <div className={`selection-bar ${selection.enabled ? "" : "off"}`} role="status">
            <Tooltip text={selection.enabled
              ? "Turn the selection off without losing it — the next run treats the whole canvas."
              : "Turn the selection back on."}>
              <button
                type="button"
                className={`sel-eye ${selection.enabled ? "on" : ""}`}
                aria-pressed={selection.enabled}
                aria-label={selection.enabled ? "Disable selection" : "Enable selection"}
                onClick={() => setSelectionEnabled(!selection.enabled)}
              >
                {selection.enabled ? "◉" : "○"}
              </button>
            </Tooltip>
            {renaming ? (
              <input
                type="text"
                className="sel-name-input"
                defaultValue={selection.name}
                autoFocus
                aria-label="Selection name"
                onBlur={(e) => { renameSelection(e.target.value.trim()); setRenaming(false); }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") { renameSelection(e.target.value.trim()); setRenaming(false); }
                  if (e.key === "Escape") setRenaming(false);
                }}
              />
            ) : (
              <button type="button" className="sel-name" onClick={() => setRenaming(true)} title="Rename">
                {selection.name}
              </button>
            )}
            <span className="sub sel-note">
              {selectionPixels.toLocaleString()} px —{" "}
              {selection.enabled
                ? "the next run changes only this area"
                : "off, so the next run treats the whole canvas"}
            </span>
            <div className="spacer" />
            <Tooltip text="Swap what is selected for what is not. With nothing selected, this selects the whole canvas.">
              <button type="button" className="btn ghost sm" onClick={invertSelection}>Invert</button>
            </Tooltip>
            <Tooltip text="Drop the selection. Ctrl+Z brings it back.">
              <button type="button" className="btn ghost sm" onClick={() => deselect()}>Deselect</button>
            </Tooltip>
          </div>
        )}
        {staging && <StagingBar />}
        <div className={`hero ${brushable ? "brushable" : ""}`} ref={heroRef}>
          {shown ? <img src={shown} alt="canvas" /> : (
            <span className="sub">Generate, drop, or paste an image.</span>
          )}
          {shown && <MaskOverlay active={brushable} />}
        </div>
        <div className="row wrap center mt-2 gap-2 canvas-size-row">
          <span className="sub">Canvas</span>
          <input
            type="number" className="canvas-size-input" aria-label="Canvas width"
            value={canvasSize.w} min={CANVAS_MIN} max={CANVAS_MAX} step={64}
            onChange={(e) => setCanvasSize({ w: e.target.value })}
          />
          <span className="sub">x</span>
          <input
            type="number" className="canvas-size-input" aria-label="Canvas height"
            value={canvasSize.h} min={CANVAS_MIN} max={CANVAS_MAX} step={64}
            onChange={(e) => setCanvasSize({ h: e.target.value })}
          />
          <Tooltip text="Start again on a blank canvas at this size. Whatever is on the canvas now stays in Results, so this does not lose it.">
            <button type="button" className="btn sm" onClick={() => newCanvas(canvasSize.w, canvasSize.h)}>
              New canvas
            </button>
          </Tooltip>
          <Tooltip text="Open an image from disk. Dropping one onto the canvas, or pasting it, does the same thing.">
            <button type="button" className="btn sm ghost" onClick={() => openRef.current?.click()}>
              Open image…
            </button>
          </Tooltip>
          <input
            ref={openRef}
            type="file"
            accept="image/*"
            className="hidden-file"
            onChange={(e) => { loadFile(e.target.files?.[0]); e.target.value = ""; }}
          />
        </div>
        <div className="row wrap mt-2 gap-2">
          {shown && (
            <Tooltip text="Download the canvas as a PNG. The sampling settings that made it are written into the file, so dropping it back into Kiln restores them.">
              <button type="button" className="btn sm" onClick={download} disabled={busy}>
                {busy ? "Preparing…" : "Download"}
              </button>
            </Tooltip>
          )}
          {shown && <CaptureButton image={shown} card={frameCard} label="Capture" />}
          {shown && (
            <Tooltip text="Use the current canvas as the starting image for the next full generation (img2img). This clears the mask — a mask belongs to the image it was painted on.">
              <button type="button" className="btn sm" onClick={useAsInit}>Use as init</button>
            </Tooltip>
          )}
          {initImage && (
            <Tooltip text="An init image is set: the next full generation starts from it. Remove it to generate from scratch again.">
              <span className="pill on init-pill">
                init set
                <button
                  type="button"
                  className="pill-x"
                  aria-label="Clear init image"
                  onClick={clearInit}
                >
                  ✕
                </button>
              </span>
            </Tooltip>
          )}
          <div className="spacer" />
          {shown && (
            <Tooltip text="Back to a blank canvas at the current size — drops the init and any painted mask with it. The image itself stays in Results.">
              <button type="button" className="btn ghost sm" onClick={clearCanvas}>Clear canvas</button>
            </Tooltip>
          )}
        </div>
      </div>
    </div>
  );
}

/** Save an image into the workspace captures folder, recipe and all. */
function CaptureButton({ image, card, label = "Save", className = "btn sm" }) {
  const { toast } = useApp();
  const capture = async () => {
    if (!image) return;
    try {
      await api.post("/perform/capture", { image, card: card || null });
      toast(card?.params?.seed != null ? `Captured · seed ${card.params.seed}` : "Captured", "success");
    } catch (e) { toast(e.message, "error"); }
  };
  return (
    <Tooltip text="Copy into Kiln's captures folder and Library, with its settings embedded in the PNG.">
      <button type="button" className={className} onClick={capture}>{label}</button>
    </Tooltip>
  );
}

/** Create's output column: every image this session produced, newest first.
 *
 *  Sits beside the canvas rather than under it — a vertical rail keeps the
 *  full history in view while you work instead of scrolling sideways through it.
 */
function ResultsRail() {
  const {
    history, frame, restoreHistory, removeHistory, clearHistory,
    useHistoryAsInit, applyCard,
  } = usePlay();
  if (!history.length) return null;

  return (
    <aside className="card results-rail" aria-label="Results">
      <div className="row between center mb-2">
        <h3 className="mb-0">Results <span className="sub">· {history.length}</span></h3>
        <button type="button" className="btn ghost sm" onClick={clearHistory}>Clear</button>
      </div>
      <div className="results-rail-list">
        {history.map((h, i) => {
          const selected = h.img === frame;
          const seed = h.card?.params?.seed;
          return (
            <div key={h.id} className={`play-result ${selected ? "on" : ""}`}>
              <button
                type="button"
                className="play-result-thumb"
                onClick={() => restoreHistory(h)}
                aria-label={`Result ${i + 1}${seed != null ? `, seed ${seed}` : ""}`}
                title={cardLabel(h.card) || `Result ${i + 1}`}
              >
                <img src={h.img} alt="" />
                <span className="play-result-idx">#{i + 1}</span>
              </button>
              <span className="play-result-seed mono">{seed != null ? `seed ${seed}` : "—"}</span>
              <div className="play-result-actions">
                <Tooltip text="Use as init image">
                  <button type="button" className="btn xs" onClick={() => useHistoryAsInit(h)}>Init</button>
                </Tooltip>
                <CaptureButton image={h.img} card={h.card} label="Save" className="btn xs" />
                <Tooltip text={h.card?.params ? "Load this image's settings into the sampler" : "No settings recorded for this image"}>
                  <button
                    type="button"
                    className="btn xs"
                    onClick={() => applyCard(h.card)}
                    disabled={!h.card?.params}
                  >
                    Set
                  </button>
                </Tooltip>
                <Tooltip text="Remove from results">
                  <button type="button" className="btn xs ghost" onClick={() => removeHistory(h.id)} aria-label="Remove">×</button>
                </Tooltip>
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}

function MaskOverlay({ active }) {
  const {
    maskRef, heroRef, brushSize, brushHard, eraser, canvasImage,
    syncMaskOverlayRef, maskTool, applyFloodFillMask, wandTolerance,
    addStroke, rasterizeNow, selection,
  } = usePlay();
  const drawing = useRef(false);
  const last = useRef(null);
  const wandBusy = useRef(false);
  // Points of the stroke in progress, normalised, recorded as they are painted.
  const points = useRef([]);
  const radiusRef = useRef(0);
  const selRef = useRef(selection);
  useEffect(() => { selRef.current = selection; }, [selection]);

  const syncSize = () => {
    const c = maskRef.current;
    const hero = heroRef.current;
    const img = hero?.querySelector("img");
    if (!c || !img) return;
    const r = img.getBoundingClientRect();
    const hr = hero.getBoundingClientRect();
    c.style.left = `${r.left - hr.left}px`;
    c.style.top = `${r.top - hr.top}px`;
    c.style.width = `${r.width}px`;
    c.style.height = `${r.height}px`;
    // Resize only once the new frame has actually decoded. A freshly generated
    // image reports naturalWidth 0 for the first moments after its src changes,
    // and this used to fall back to 512 and resize the canvas to it -- which
    // clears the canvas. Every click during that window painted a stroke and
    // then wiped it on the next one, so the brush looked dead for a second or
    // two after each generation. The load listener below re-runs this once the
    // real dimensions exist.
    if (!img.complete || !img.naturalWidth || !img.naturalHeight) return;
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (c.width !== w || c.height !== h) {
      // Setting width/height clears the canvas, so the selection has to be put
      // back. It used to be rescaled from the old raster; now the strokes are
      // replayed at the new size, which is both sharper and the reason stroke
      // coordinates are stored normalised. rasterize() sets the dimensions
      // itself, so the next syncSize sees a match and this cannot recurse.
      rasterizeNow(selRef.current, { w, h });
    }
  };

  useEffect(() => {
    syncMaskOverlayRef.current = syncSize;
    syncSize();
    const img = heroRef.current?.querySelector("img");
    // Two things move the overlay out from under the pointer: the image
    // finishing its decode (new size), and the layout reflowing around it (new
    // position). Watch for both -- before this, nothing re-synced after a
    // generation until the next click, and that click was spent re-syncing
    // instead of painting.
    img?.addEventListener("load", syncSize);
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(syncSize) : null;
    if (img && ro) ro.observe(img);
    window.addEventListener("resize", syncSize);
    return () => {
      syncMaskOverlayRef.current = () => {};
      img?.removeEventListener("load", syncSize);
      ro?.disconnect();
      window.removeEventListener("resize", syncSize);
    };
  }, [canvasImage, syncMaskOverlayRef]);

  const pointAt = (e) => {
    const c = maskRef.current;
    const r = c.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * c.width,
      y: ((e.clientY - r.top) / r.height) * c.height,
      scale: c.width / Math.max(r.width, 1),
    };
  };

  /** Paint live, and record the point for the stroke being built.
   *
   *  Painting incrementally rather than replaying the stroke list on every
   *  pointermove keeps the hot path exactly as cheap as it was; the list is
   *  only replayed on undo, invert and resize.
   */
  const paint = (e) => {
    if (!active) return;
    const c = maskRef.current;
    if (!c) return;
    const pt = pointAt(e);
    const ctx = c.getContext("2d");
    const radius = (brushSize / 2) * pt.scale;
    radiusRef.current = radius;
    paintBrushPoint(ctx, last.current, pt, radius, { hard: brushHard, erase: eraser });
    ctx.globalCompositeOperation = "source-over";
    last.current = { x: pt.x, y: pt.y };
    points.current.push({ x: pt.x / c.width, y: pt.y / c.height });
  };

  const endStroke = () => {
    const c = maskRef.current;
    if (drawing.current && points.current.length && c?.width) {
      addStroke(brushStroke({
        points: points.current,
        size: (radiusRef.current * 2) / c.width,
        hard: brushHard,
        mode: eraser ? "subtract" : "add",
      }));
    }
    drawing.current = false;
    last.current = null;
    points.current = [];
  };

  const wandClick = async (e) => {
    if (!active || !canvasImage || wandBusy.current) return;
    const c = maskRef.current;
    if (!c) return;
    const pt = pointAt(e);
    wandBusy.current = true;
    try {
      await applyFloodFillMask(canvasImage, pt.x, pt.y, { tolerance: wandTolerance, eraser });
    } finally {
      wandBusy.current = false;
    }
  };

  // A disabled selection still shows, faintly: otherwise "off" and "deleted"
  // look identical and the eye toggle appears to have thrown the work away.
  const dim = selection && !selection.enabled;

  return (
    <canvas
      className={`mask-overlay ${active ? "on" : ""} ${maskTool === "wand" ? "wand" : ""} ${dim ? "dim" : ""}`}
      ref={maskRef}
      role="img"
      aria-label={maskTool === "wand"
        ? "Selection overlay — click to select a matching area"
        : "Selection overlay — drag to paint a selection"}
      onPointerDown={(e) => {
        if (maskTool === "wand") {
          wandClick(e);
          return;
        }
        // Throws if the pointer id is not an active pointer, which is the case
        // for synthetic events. Capture is a nicety -- losing it should not
        // cost the stroke.
        try { e.target.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
        syncSize();
        drawing.current = true;
        last.current = null;
        points.current = [];
        paint(e);
      }}
      onPointerMove={(e) => { if (maskTool === "brush" && drawing.current) paint(e); }}
      onPointerUp={endStroke}
      onPointerCancel={endStroke}
    />
  );
}

/** Which fields actually differ across a set of candidates.
 *
 *  Labelling a comparison by seed is useless when the seed is the one thing
 *  held fixed — the useful label is whatever was varied to produce the set.
 */
const META_KEYS = ["model", "bend", "change", "seed"];

function metaText(key, v) {
  if (v == null || v === "") return key === "bend" ? "no bend" : null;
  if (key === "change") return `change ${Math.round(v * 100)}%`;
  if (key === "seed") return `seed ${v}`;
  return String(v);
}

function stageLabels(items) {
  const varying = META_KEYS.filter(
    (k) => new Set(items.map((it) => it.meta?.[k] ?? "")).size > 1,
  );
  const use = varying.length ? varying : ["seed"];
  return items.map((it, i) => {
    const bits = use.map((k) => metaText(k, it.meta?.[k])).filter(Boolean);
    return bits.length ? bits.join(" · ") : `Try ${i + 1}`;
  });
}

/** Candidates from one or more fills, held until accepted or discarded. */
function StagingBar() {
  const { staging, setStageIndex, acceptStaging, discardStaging } = usePlay();
  const labels = useMemo(() => stageLabels(staging.items), [staging.items]);

  useEffect(() => {
    const onKey = (e) => {
      const t = e.target;
      const tag = t?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t?.isContentEditable) return;
      if (e.key === "ArrowLeft") { e.preventDefault(); setStageIndex(staging.index - 1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); setStageIndex(staging.index + 1); }
      else if (e.key === "Enter") { e.preventDefault(); acceptStaging(); }
      else if (e.key === "Escape") { e.preventDefault(); discardStaging(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [staging.index, setStageIndex, acceptStaging, discardStaging]);

  return (
    <div className="staging-bar">
      <div className="row between center wrap gap-2 mb-2">
        <span className="sub">
          <strong className="staging-title">Trying {staging.items.length}</strong>
          {" "}— the canvas still holds the image these ran against. Nothing is changed until you accept.
        </span>
        <div className="row gap-2">
          <Tooltip text="Put the highlighted result on the canvas and into Results. (Enter)">
            <button type="button" className="btn sm primary" onClick={acceptStaging}>Accept</button>
          </Tooltip>
          <Tooltip text="Throw all of these away and go back to the image you started from. (Esc)">
            <button type="button" className="btn ghost sm" onClick={discardStaging}>Discard</button>
          </Tooltip>
        </div>
      </div>
      <div className="staging-list">
        {staging.items.map((it, i) => (
          <button
            key={it.id}
            type="button"
            className={`staging-item ${i === staging.index ? "on" : ""}`}
            onClick={() => setStageIndex(i)}
            title={labels[i]}
          >
            <img src={it.img} alt="" />
            <span className="staging-label">{labels[i]}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
