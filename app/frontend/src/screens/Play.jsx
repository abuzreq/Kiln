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

const HISTORY_MAX = 24;
const MASK_UNDO_MAX = 12;
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
  const [maskUndoLen, setMaskUndoLen] = useState(0);
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
  const maskUndoRef = useRef([]);
  const [sampleParams, setSampleParamsState] = useState(loadSampleParams);

  const setTabStateKey = useCallback((key, valueOrFn, initial) => {
    setTabState((s) => {
      const prev = key in s ? s[key] : initial;
      const next = typeof valueOrFn === "function" ? valueOrFn(prev) : valueOrFn;
      return Object.is(prev, next) ? s : { ...s, [key]: next };
    });
  }, []);

  const bumpMask = useCallback(() => setMaskVersion((v) => v + 1), []);
  // clearMask/clearMaskUndo are defined further down; applyInit needs them.
  const clearMaskRef = useRef(null);
  const clearMaskUndoRef = useRef(null);

  const pushMaskUndo = useCallback(() => {
    const c = maskRef.current;
    if (!c || !c.width || !c.height) return;
    // Keep the stack on a ref so mid-stroke pushes do not re-render (re-render
    // was wiping in-progress brush paint from the overlay canvas).
    const url = c.toDataURL("image/png");
    maskUndoRef.current = [url, ...maskUndoRef.current].slice(0, MASK_UNDO_MAX);
    setMaskUndoLen(maskUndoRef.current.length);
  }, []);

  const clearMaskUndo = useCallback(() => {
    maskUndoRef.current = [];
    setMaskUndoLen(0);
  }, []);

  const undoMask = useCallback(() => {
    const stack = maskUndoRef.current;
    if (!stack.length) return;
    const [prev, ...rest] = stack;
    maskUndoRef.current = rest;
    setMaskUndoLen(rest.length);
    const c = maskRef.current;
    if (!c) return;
    const im = new Image();
    im.onload = () => {
      const w = im.naturalWidth || im.width;
      const h = im.naturalHeight || im.height;
      if (c.width !== w || c.height !== h) {
        c.width = w;
        c.height = h;
      } else {
        c.getContext("2d").clearRect(0, 0, c.width, c.height);
      }
      const ctx = c.getContext("2d");
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.drawImage(im, 0, 0);
      syncMaskOverlayRef.current();
      bumpMask();
    };
    im.src = prev;
  }, [bumpMask]);

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

  const restoreHistory = useCallback((entry) => {
    if (!entry) return;
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
   *  A mask is painted against one specific picture, so swapping the init out
   *  from under it would apply an old selection to a new image. Changing the
   *  init clears the mask unless a caller explicitly opts out.
   */
  const applyInit = useCallback((src, { keepMask = false } = {}) => {
    setInitImage((prev) => {
      if (prev !== src && !keepMask) {
        clearMaskRef.current?.({ skipUndo: true });
        clearMaskUndoRef.current?.();
      }
      return src || null;
    });
  }, []);

  const useAsInit = useCallback(() => {
    const src = showRaw && frameRaw ? frameRaw : frame;
    if (!src) return;
    applyInit(src);
    toast("Canvas set as init — mask cleared", "success");
  }, [frame, frameRaw, showRaw, applyInit, toast]);

  const useHistoryAsInit = useCallback((entry) => {
    if (!entry?.img) return;
    applyInit(entry.raw || entry.img);
    toast("Set as init — mask cleared", "success");
  }, [applyInit, toast]);

  /** Drop the init image, keeping whatever is on the canvas. */
  const clearInit = useCallback(() => {
    applyInit(null);
    toast("Init cleared", "success");
  }, [applyInit, toast]);

  /** A blank canvas to paint on: image, init, mask and recipe all reset.
   *
   *  Deliberately *not* set as the init image. An untouched blank canvas is a
   *  surface to select on, not a picture to work from — with no region painted,
   *  Generate should make a new image rather than img2img from empty pixels.
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
    applyInit(null);
    clearMaskRef.current?.({ skipUndo: true });
    clearMaskUndoRef.current?.();
    if (!quiet) toast(`Blank canvas — ${width}x${height}`, "success");
  }, [applyInit, toast]);

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
    setFrame(url, null, card);
    applyInit(url);
    pushHistory(url, null, card);
    setPendingCard(card && card.params ? card : null);
    toast(card?.params ? "On canvas — Kiln settings found" : "On canvas", "success");
  }, [setFrame, pushHistory, toast]);

  useEffect(() => { clearMaskRef.current = clearMask; });
  useEffect(() => { clearMaskUndoRef.current = clearMaskUndo; });

  const clearMask = useCallback((opts = {}) => {
    const c = maskRef.current;
    if (!c) return;
    if (!opts.skipUndo) pushMaskUndo();
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    bumpMask();
  }, [bumpMask, pushMaskUndo]);

  const getMaskDataUrl = useCallback(() => {
    const c = maskRef.current;
    if (!c) return null;
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
  }, []);

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

    pushMaskUndo();
    maskCanvas.width = w;
    maskCanvas.height = h;
    const ctx = maskCanvas.getContext("2d");
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.putImageData(overlay, 0, 0);
    syncMaskOverlayRef.current();
    bumpMask();
    return true;
  }, [bumpMask, pushMaskUndo]);

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

    pushMaskUndo();
    if (maskCanvas.width !== w || maskCanvas.height !== h) {
      maskCanvas.width = w;
      maskCanvas.height = h;
    }
    const ctx = maskCanvas.getContext("2d");
    const existing = ctx.getImageData(0, 0, w, h);
    const overlay = new ImageData(w, h);
    for (let i = 0; i < w * h; i += 1) {
      const o = i * 4;
      let a;
      if (erase) {
        a = alpha[i] > 8 ? 0 : existing.data[o + 3];
      } else {
        a = Math.max(existing.data[o + 3], alpha[i]);
      }
      overlay.data[o] = 255;
      overlay.data[o + 1] = 122;
      overlay.data[o + 2] = 69;
      overlay.data[o + 3] = a > 8 ? a : 0;
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.putImageData(overlay, 0, 0);
    syncMaskOverlayRef.current();
    bumpMask();
    return true;
  }, [bumpMask, wandTolerance, pushMaskUndo]);

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

  const canvasImage = showRaw && frameRaw ? frameRaw : frame;
  // Still the untouched blank ground: nothing has been generated, dropped or
  // restored over it. Region fill has to know, because a partial run started
  // from empty pixels gives back the empty pixels it started from.
  const canvasIsBlank = !!frame && frame === blankFrameRef.current;
  const activeSeed = frameCard?.params?.seed ?? null;
  const genRunning = !!(job && job.status === "running");
  const genPaused = genRunning && !!job?.detail?.paused;
  const canUndoMask = maskUndoLen > 0;

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
    syncMaskOverlayRef, clearMask, getMaskDataUrl, applyContrastMask, applyFloodFillMask,
    pushMaskUndo, undoMask, clearMaskUndo, canUndoMask,
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
    frame, frameRaw, showRaw, setShowRaw, initImage, progress, heroRef,
    frameCard, pendingCard, setPendingCard, applyCard, activeSeed, useAsInit,
    clearInit, clearCanvas, canvasSize, setCanvasSize, newCanvas, loadFile,
    maskRef, maskVersion, clearMask,
  } = usePlay();
  const shown = showRaw && frameRaw ? frameRaw : frame;
  const [busy, setBusy] = useState(false);
  const openRef = useRef(null);
  // Whether a selection is live decides what the Generate button does, so it
  // belongs next to the picture rather than folded into a sidebar section --
  // the orange overlay says an area is marked, not that the next run will be
  // confined to it.
  const maskPixels = useMemo(() => countMaskPixels(maskRef.current), [maskVersion, maskRef]);

  const download = async () => {
    if (!shown) return;
    setBusy(true);
    try {
      // Routed through the API so the PNG keeps its embedded recipe — a plain
      // <a download> on the canvas data URL would hand over a stripped file.
      const seed = frameCard?.params?.seed;
      await downloadPost(
        "/perform/export",
        { image: shown, card: frameCard, filename: seed != null ? `kiln-${seed}` : "kiln" },
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
            {frameRaw && (
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
        {maskPixels > 0 && (
          <div className="callout row between center wrap gap-2" role="status">
            <span>
              <strong>Selection active</strong> — {maskPixels.toLocaleString()} px.
              {" "}The next run changes only this area; the rest of the canvas is kept.
            </span>
            <button type="button" className="btn ghost sm" onClick={() => clearMask()}>
              Deselect
            </button>
          </div>
        )}
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
    syncMaskOverlayRef, bumpMask, maskTool, applyFloodFillMask, wandTolerance,
    pushMaskUndo,
  } = usePlay();
  const drawing = useRef(false);
  const last = useRef(null);
  const wandBusy = useRef(false);
  const strokePushed = useRef(false);

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
      // Setting width/height clears the canvas, so the mask is carried over on
      // another canvas. This used to go through toDataURL + Image.onload, which
      // was wrong twice over: the encode is a synchronous PNG on the main
      // thread (and this runs on *every* live preview frame, each a different
      // size, so a 50-step run paid for 50 of them -- that is the stall people
      // hit when brushing right after generating), and the restore landed a
      // frame later, wiping anything painted in between. drawImage is immediate
      // and scales in one step.
      const snap = document.createElement("canvas");
      snap.width = c.width;
      snap.height = c.height;
      const had = c.width > 0 && c.height > 0;
      if (had) snap.getContext("2d").drawImage(c, 0, 0);
      c.width = w;
      c.height = h;
      if (had) c.getContext("2d").drawImage(snap, 0, 0, w, h);
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

  const stamp = (ctx, x, y, radius) => {
    ctx.globalCompositeOperation = eraser ? "destination-out" : "source-over";
    if (brushHard) {
      ctx.fillStyle = eraser ? "rgba(0,0,0,1)" : "rgba(255,122,69,0.7)";
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    const g = ctx.createRadialGradient(x, y, 0, x, y, radius);
    if (eraser) {
      g.addColorStop(0, "rgba(0,0,0,0.85)");
      g.addColorStop(0.55, "rgba(0,0,0,0.35)");
      g.addColorStop(1, "rgba(0,0,0,0)");
    } else {
      g.addColorStop(0, "rgba(255,122,69,0.55)");
      g.addColorStop(0.55, "rgba(255,122,69,0.22)");
      g.addColorStop(1, "rgba(255,122,69,0)");
    }
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  };

  const paint = (e) => {
    if (!active) return;
    const c = maskRef.current;
    if (!c) return;
    const r = c.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * c.width;
    const y = ((e.clientY - r.top) / r.height) * c.height;
    const ctx = c.getContext("2d");
    const radius = (brushSize / 2) * (c.width / Math.max(r.width, 1));
    const prev = last.current;
    if (prev && brushHard) {
      ctx.globalCompositeOperation = eraser ? "destination-out" : "source-over";
      ctx.strokeStyle = eraser ? "rgba(0,0,0,1)" : "rgba(255,122,69,0.7)";
      ctx.lineWidth = radius * 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(prev.x, prev.y);
      ctx.lineTo(x, y);
      ctx.stroke();
    } else if (prev && !brushHard) {
      const dx = x - prev.x;
      const dy = y - prev.y;
      const dist = Math.hypot(dx, dy);
      const step = Math.max(radius * 0.35, 1);
      const n = Math.max(1, Math.ceil(dist / step));
      for (let i = 1; i <= n; i++) {
        stamp(ctx, prev.x + (dx * i) / n, prev.y + (dy * i) / n, radius);
      }
    } else {
      stamp(ctx, x, y, radius);
    }
    last.current = { x, y };
  };

  const endStroke = () => {
    if (drawing.current) bumpMask();
    drawing.current = false;
    last.current = null;
    strokePushed.current = false;
  };

  const wandClick = async (e) => {
    if (!active || !canvasImage || wandBusy.current) return;
    const c = maskRef.current;
    if (!c) return;
    const r = c.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * c.width;
    const y = ((e.clientY - r.top) / r.height) * c.height;
    wandBusy.current = true;
    try {
      await applyFloodFillMask(canvasImage, x, y, { tolerance: wandTolerance, eraser });
    } finally {
      wandBusy.current = false;
    }
  };

  return (
    <canvas
      className={`mask-overlay ${active ? "on" : ""} ${maskTool === "wand" ? "wand" : ""}`}
      ref={maskRef}
      onPointerDown={(e) => {
        if (maskTool === "wand") {
          wandClick(e);
          return;
        }
        e.target.setPointerCapture(e.pointerId);
        syncSize();
        if (!strokePushed.current) {
          pushMaskUndo();
          strokePushed.current = true;
        }
        drawing.current = true;
        last.current = null;
        paint(e);
      }}
      onPointerMove={(e) => { if (maskTool === "brush" && drawing.current) paint(e); }}
      onPointerUp={endStroke}
      onPointerCancel={endStroke}
    />
  );
}
