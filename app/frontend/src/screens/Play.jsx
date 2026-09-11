import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CreatePanel from "../components/CreatePanel.jsx";
import { BendWorkspace } from "./Craft.jsx";
import SweepPanel from "./Sweep.jsx";
import Merge from "./Merge.jsx";
import { useApp } from "../state.jsx";
import { api, downloadPost } from "../api.js";
import { Progress, Slider, Tooltip } from "../components/ui.jsx";
import { PlayCtx, usePlay, fileToDataUrl } from "./playContext.jsx";
import {
  loadSampleParams, saveSampleParams, SampleSettingsPanel, paramsFromCard, cardLabel,
  LIVE_PARAM_KEYS,
} from "../sampleSettings.jsx";
import ModelPicker from "../components/ModelPicker.jsx";
import { buildContrastMask, floodFillMask, countMaskPixels } from "../contrastMask.js";
import {
  brushStroke, cachedStroke, invertStroke, alphaToCache, overlayToCache,
  paintBrushPoint,
} from "../selection.js";
import {
  newRasterLayer, newInpaintMask, newDocument, uniqueName, uniqueFrom, activeMasks,
  flattenLayers, compositeMasks, punchToMask, layerAlphaToCache, reorder,
} from "../layers.js";

const HISTORY_MAX = 24;
// One entry is a whole document -- both entity arrays and the selected row.
// That is only stroke lists and image references, so it is cheap enough to keep
// plenty of, and it makes Undo mean the same thing everywhere.
const UNDO_MAX = 40;
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
  // `frame` is the flattened stack — a cache, never written to directly.
  const [frame, setFrameState] = useState(null);
  // Post-process is a display stage over the flattened stack rather than an
  // edit to it, so there is no longer a separate "raw" frame to keep: the
  // flattened stack *is* the unprocessed image, and Unprocessed just shows it.
  const [postFrame, setPostFrame] = useState(null);
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
  // Canvas entities, in two groups, after InvokeAI's controlLayers store.
  // Raster layers composite bottom-to-top to make the picture; inpaint masks
  // say where a fill may change it. Both are ordered arrays, both addressed by
  // id, and neither is ever destroyed as a side effect of generating.
  const [rasterLayers, setRasterLayers] = useState([]);
  const [inpaintMasks, setInpaintMasks] = useState([]);
  // The row highlighted in the Layers panel. When it is a mask, it is also what
  // the brush paints into.
  const [selectedId, setSelectedId] = useState(null);
  const [undoLen, setUndoLen] = useState(0);
  // In-progress frames from a running job, kept off the layer stack so a live
  // preview never becomes a layer of its own.
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
  // A mirror of the entity state, so the pointer handlers that add strokes can
  // read the current value without going stale between renders.
  const docRef = useRef({ rasterLayers: [], inpaintMasks: [], selectedId: null });
  const [sampleParams, setSampleParamsState] = useState(loadSampleParams);

  const setTabStateKey = useCallback((key, valueOrFn, initial) => {
    setTabState((s) => {
      const prev = key in s ? s[key] : initial;
      const next = typeof valueOrFn === "function" ? valueOrFn(prev) : valueOrFn;
      return Object.is(prev, next) ? s : { ...s, [key]: next };
    });
  }, []);

  const bumpMask = useCallback(() => setMaskVersion((v) => v + 1), []);
  useEffect(() => {
    docRef.current = { rasterLayers, inpaintMasks, selectedId };
  }, [rasterLayers, inpaintMasks, selectedId]);

  /** Redraw the mask overlay: the union of every enabled mask. */
  const rasterizeMasks = useCallback((masks, size) => {
    const c = maskRef.current;
    if (!c) return;
    const w = size?.w || c.width;
    const h = size?.h || c.height;
    if (!w || !h) return;
    compositeMasks(masks || [], c, w, h);
    syncMaskOverlayRef.current();
    bumpMask();
  }, [bumpMask]);

  const pushUndo = useCallback((snap) => {
    undoRef.current = [snap, ...undoRef.current].slice(0, UNDO_MAX);
    setUndoLen(undoRef.current.length);
  }, []);

  const clearUndo = useCallback(() => {
    undoRef.current = [];
    setUndoLen(0);
  }, []);

  /** The one mutator every entity edit goes through.
   *
   *  `fn` gets the current document and returns the parts of it that change.
   *  Snapshotting the whole document here is what makes one Undo cover a
   *  painted stroke, a deleted layer, a reorder and an accepted fill alike --
   *  which is the behaviour that replaced the staging area's Accept/Discard.
   */
  const edit = useCallback((fn, { redraw = true, undo = true } = {}) => {
    const prev = docRef.current;
    const patch = typeof fn === "function" ? fn(prev) : fn;
    if (!patch) return;
    const next = { ...prev, ...patch };
    if (undo) pushUndo(prev);
    docRef.current = next;
    if (patch.rasterLayers) setRasterLayers(patch.rasterLayers);
    if (patch.inpaintMasks) setInpaintMasks(patch.inpaintMasks);
    if ("selectedId" in patch) setSelectedId(patch.selectedId);
    if (redraw && patch.inpaintMasks) rasterizeMasks(patch.inpaintMasks);
  }, [pushUndo, rasterizeMasks]);

  const undo = useCallback(() => {
    const stack = undoRef.current;
    if (!stack.length) return;
    const [prev, ...rest] = stack;
    undoRef.current = rest;
    setUndoLen(rest.length);
    docRef.current = prev;
    setRasterLayers(prev.rasterLayers);
    setInpaintMasks(prev.inpaintMasks);
    setSelectedId(prev.selectedId);
    rasterizeMasks(prev.inpaintMasks);
  }, [rasterizeMasks]);

  /** Which mask the brush paints into.
   *
   *  The selected row if it is a mask, else the topmost enabled one. Falling
   *  back means the tools always have a target, so painting on a fresh canvas
   *  works without first understanding the panel.
   */
  const paintTargetId = useCallback((doc) => {
    const sel = doc.inpaintMasks.find((m) => m.id === doc.selectedId && !m.locked);
    if (sel) return sel.id;
    const open = [...doc.inpaintMasks].reverse().find((m) => m.enabled && !m.locked);
    return open?.id || null;
  }, []);

  const patchEntity = (list, id, patch) => list.map(
    (e) => (e.id === id ? { ...e, ...(typeof patch === "function" ? patch(e) : patch) } : e),
  );

  /** Append a stroke to the mask being painted, creating one if there is none. */
  const addStroke = useCallback((stroke, { redraw = true } = {}) => {
    edit((doc) => {
      let masks = doc.inpaintMasks;
      let id = paintTargetId(doc);
      if (!id) {
        const m = newInpaintMask({ name: uniqueName(masks, "Inpaint Mask") });
        masks = [...masks, m];
        id = m.id;
      }
      return {
        inpaintMasks: patchEntity(masks, id, (m) => ({
          enabled: true,
          strokes: [...m.strokes, stroke],
        })),
        selectedId: id,
      };
    }, { redraw });
  }, [edit, paintTargetId]);

  const setEntityEnabled = useCallback((id, on) => {
    edit((doc) => ({
      rasterLayers: patchEntity(doc.rasterLayers, id, { enabled: !!on }),
      inpaintMasks: patchEntity(doc.inpaintMasks, id, { enabled: !!on }),
    }));
  }, [edit]);

  const setEntityLocked = useCallback((id, on) => {
    edit((doc) => ({
      rasterLayers: patchEntity(doc.rasterLayers, id, { locked: !!on }),
      inpaintMasks: patchEntity(doc.inpaintMasks, id, { locked: !!on }),
    }), { redraw: false });
  }, [edit]);

  const renameEntity = useCallback((id, name) => {
    if (!name) return;
    edit((doc) => ({
      rasterLayers: patchEntity(doc.rasterLayers, id, { name }),
      inpaintMasks: patchEntity(doc.inpaintMasks, id, { name }),
    }), { redraw: false });
  }, [edit]);

  const setLayerOpacity = useCallback((id, v) => {
    edit((doc) => ({
      rasterLayers: patchEntity(doc.rasterLayers, id, { opacity: Math.max(0, Math.min(1, v)) }),
    }), { redraw: false });
  }, [edit]);

  const deleteEntity = useCallback((id) => {
    edit((doc) => ({
      rasterLayers: doc.rasterLayers.filter((e) => e.id !== id),
      inpaintMasks: doc.inpaintMasks.filter((e) => e.id !== id),
      selectedId: doc.selectedId === id ? null : doc.selectedId,
    }));
  }, [edit]);

  const moveEntity = useCallback((id, delta) => {
    edit((doc) => ({
      rasterLayers: reorder(doc.rasterLayers, id, delta),
      inpaintMasks: reorder(doc.inpaintMasks, id, delta),
    }));
  }, [edit]);

  const addMask = useCallback(() => {
    edit((doc) => {
      const m = newInpaintMask({ name: uniqueName(doc.inpaintMasks, "Inpaint Mask") });
      return { inpaintMasks: [...doc.inpaintMasks, m], selectedId: m.id };
    });
  }, [edit]);

  /** Empty a mask without deleting it — the row, its name and its settings stay. */
  const clearMask = useCallback((id) => {
    edit((doc) => ({ inpaintMasks: patchEntity(doc.inpaintMasks, id, { strokes: [] }) }));
  }, [edit]);

  /** Invert on an empty mask means "select everything", which is useful. */
  const invertMask = useCallback((id) => {
    edit((doc) => {
      const target = id || paintTargetId(doc);
      if (!target) return null;
      return {
        inpaintMasks: patchEntity(doc.inpaintMasks, target, (m) => ({
          enabled: true,
          strokes: [...m.strokes, invertStroke()],
        })),
      };
    });
  }, [edit, paintTargetId]);

  const setMaskParam = useCallback((id, k, v) => {
    edit((doc) => {
      const target = id || paintTargetId(doc);
      if (!target) return null;
      return {
        inpaintMasks: patchEntity(doc.inpaintMasks, target, (m) => ({
          params: { ...m.params, [k]: v },
        })),
      };
    }, { redraw: false, undo: false });
  }, [edit, paintTargetId]);

  const addRasterLayer = useCallback(({ image, card, name }) => {
    edit((doc) => {
      const l = newRasterLayer({
        name: name
          ? uniqueFrom(doc.rasterLayers, name)
          : uniqueName(doc.rasterLayers, "Layer"),
        image,
        card,
      });
      return { rasterLayers: [...doc.rasterLayers, l] };
    }, { redraw: false });
  }, [edit]);

  /** Promote a raster layer's own alpha to a mask.
   *
   *  A fill layer's alpha is exactly the region that fill covered, so this is
   *  how you rework the same area again with a different model or setting --
   *  the workflow the whole redesign is for.
   */
  const useLayerAsMask = useCallback(async (id) => {
    const layer = docRef.current.rasterLayers.find((l) => l.id === id);
    if (!layer?.image) return;
    const { cache } = await layerAlphaToCache(layer.image);
    edit((doc) => {
      const m = {
        ...newInpaintMask({ name: uniqueName(doc.inpaintMasks, "Inpaint Mask") }),
        strokes: [cachedStroke("layer", cache, "add")],
      };
      return { inpaintMasks: [...doc.inpaintMasks, m], selectedId: m.id };
    });
  }, [edit]);

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

  /** The canvas is the flattened stack, so `frame` is a cache of it rather than
   *  something anyone writes to. Every consumer — export, upscale,
   *  post-process, the init for the next fill, what the wand samples — reads
   *  this, and changes the moment a layer is hidden, reordered or deleted. */
  useEffect(() => {
    let live = true;
    flattenLayers(rasterLayers, canvasSize)
      .then((img) => { if (live) setFrameState(img); })
      .catch(() => { if (live) setFrameState(null); });
    return () => { live = false; };
  }, [rasterLayers, canvasSize]);

  /** Replace the whole stack with one image. What "open this" and "restore
   *  that" mean: a different picture, not another layer on the same one. */
  const setDocumentImage = useCallback((img, card, name) => {
    setLivePreview(null);
    edit((doc) => ({
      rasterLayers: img ? [newRasterLayer({ name: name || "Background", image: img, card })] : [],
      inpaintMasks: doc.inpaintMasks.length ? doc.inpaintMasks : [newInpaintMask({ name: "Inpaint Mask 1" })],
    }), { redraw: false });
    setFrameCard(card || null);
  }, [edit]);

  const restoreHistory = useCallback((entry) => {
    if (!entry) return;
    setDocumentImage(entry.img, entry.card ?? null, "Restored");
  }, [setDocumentImage]);

  const removeHistory = useCallback((id) => {
    setHistory((h) => h.filter((x) => x.id !== id));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  /** Set (or drop) the init image.
   *
   *  This used to clear the mask, on the grounds that a mask belongs to the
   *  image it was drawn on. But a fill reads the *canvas*, not the init — an
   *  init only affects a full-canvas generation, so it cannot invalidate a mask
   *  drawn over the layers. Masks stay.
   */
  const applyInit = useCallback((src) => {
    setInitImage(src || null);
  }, []);

  const useAsInit = useCallback(() => {
    const src = showRaw ? frame : (postFrame || frame);
    if (!src) return;
    applyInit(src);
    toast("Canvas set as init", "success");
  }, [frame, postFrame, showRaw, applyInit, toast]);

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

  /** A new document: one Background layer at that size, one empty mask.
   *
   *  Deliberately *not* set as the init image. An untouched blank canvas is a
   *  surface to select on, not a picture to work from — with nothing masked,
   *  Generate should make a new image rather than img2img from empty pixels.
   *
   *  This is the one place masks go without the user saying so, and it is the
   *  right one: the canvas they belonged to no longer exists.
   */
  const newCanvas = useCallback((w, h, { quiet = false } = {}) => {
    const width = clampCanvasSide(w);
    const height = clampCanvasSide(h);
    const ground = blankImage(width, height);
    blankFrameRef.current = ground;
    const doc = newDocument(ground);
    setCanvasSizeState({ w: width, h: height });
    setFrameState(ground);
    setPostFrame(null);
    setFrameCard(null);
    setPendingCard(null);
    setLivePreview(null);
    applyInit(null);
    docRef.current = { ...doc, selectedId: doc.inpaintMasks[0].id };
    setRasterLayers(doc.rasterLayers);
    setInpaintMasks(doc.inpaintMasks);
    setSelectedId(doc.inpaintMasks[0].id);
    rasterizeMasks(doc.inpaintMasks, { w: width, h: height });
    clearUndo();
    if (!quiet) toast(`Blank canvas — ${width}x${height}`, "success");
  }, [applyInit, toast, rasterizeMasks, clearUndo]);

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
    setDocumentImage(url, card, "Background");
    applyInit(url);
    pushHistory(url, null, card);
    setPendingCard(card && card.params ? card : null);
    toast(card?.params ? "On canvas — Kiln settings found" : "On canvas", "success");
  }, [setDocumentImage, applyInit, pushHistory, toast]);

  /** The mask as the backend wants it: white where the fill should happen.
   *
   *  Reads the overlay, which is already the union of every enabled mask, and
   *  returns null when none of them contribute anything — which is how the eye
   *  toggles work without every caller having to check them.
   */
  const getMaskDataUrl = useCallback(() => {
    const c = maskRef.current;
    if (!c || !activeMasks(inpaintMasks).length) return null;
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
  }, [inpaintMasks, maskVersion]);

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
    edit((doc) => {
      let masks = doc.inpaintMasks;
      let id = paintTargetId(doc);
      if (!id) {
        const m = newInpaintMask({ name: uniqueName(masks, "Inpaint Mask") });
        masks = [...masks, m];
        id = m.id;
      }
      return {
        inpaintMasks: patchEntity(masks, id, (m) => ({
          enabled: true,
          strokes: [...m.strokes.filter((s) => s.type !== "split"), stroke],
        })),
        selectedId: id,
      };
    }, { redraw: false });
    rasterizeMasks(docRef.current.inpaintMasks, { w, h });
    return true;
  }, [edit, paintTargetId, rasterizeMasks]);

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
    // recomputation would move the mask under the user.
    const stroke = cachedStroke("wand", alphaToCache(alpha, w, h), erase ? "subtract" : "add");
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

  /** A finished fill becomes a layer, punched down to the region the mask
   *  allowed so it stacks over what is there rather than hiding it.
   *
   *  This is what replaced the staging area: the fill is added, not applied, so
   *  there is nothing to accept. Reject it by hiding the layer, deleting it, or
   *  pressing Undo — and compare two attempts by toggling between two layers.
   */
  const addFillLayer = useCallback(async (img, card, name) => {
    const punched = await punchToMask(img, maskRef.current);
    addRasterLayer({ image: punched, card, name });
    setLivePreview(null);
  }, [addRasterLayer]);

  const commitFrame = useCallback((img, raw, card, name) => {
    addRasterLayer({ image: img, card, name });
    pushHistory(img, raw, card);
    setFrameCard(card || null);
    setLivePreview(null);
  }, [addRasterLayer, pushHistory]);

  // Post-process sits between the flattened stack and the screen, so
  // Unprocessed shows the stack itself. A live preview outranks both.
  const canvasImage = livePreview || (showRaw ? frame : (postFrame || frame));
  // Still the untouched blank ground: nothing generated, dropped or restored
  // over it. A fill has to know, because a partial run started from empty
  // pixels gives back the empty pixels it started from.
  const canvasIsBlank = !!frame && frame === blankFrameRef.current;
  const activeSeed = frameCard?.params?.seed ?? null;
  const genRunning = !!(job && job.status === "running");
  const genPaused = genRunning && !!job?.detail?.paused;
  const canUndo = undoLen > 0;
  const maskPixels = useMemo(
    () => countMaskPixels(maskRef.current),
    // maskVersion is the signal; the pixels live on a ref.
    [maskVersion],
  );
  const liveMasks = activeMasks(inpaintMasks);
  const hasMask = liveMasks.length > 0 && maskPixels > 0;
  // The mask whose settings the panel edits, and whose name the labels use.
  const activeMask = inpaintMasks.find((m) => m.id === paintTargetId({ ...docRef.current, inpaintMasks, selectedId }))
    || null;

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
    tab, frame, postFrame, setPostFrame, showRaw, setShowRaw, canvasImage,
    initImage, setInitImage, applyInit, clearInit, clearCanvas,
    frameCard, setFrameCard, pendingCard, setPendingCard, applyCard, lockSeed, activeSeed,
    history, commitFrame, pushHistory, restoreHistory, removeHistory, clearHistory,
    useAsInit, useHistoryAsInit, loadFile, setDocumentImage,
    progress, setProgress,
    brushSize, setBrushSize, brushHard, setBrushHard, eraser, setEraser,
    maskTool, setMaskTool, wandTolerance, setWandTolerance,
    maskRef, heroRef,
    syncMaskOverlayRef, getMaskDataUrl, applyContrastMask, applyFloodFillMask,
    // Canvas entities
    rasterLayers, inpaintMasks, selectedId, setSelectedId, activeMask,
    maskPixels, hasMask, liveMasks, addStroke, rasterizeMasks,
    setEntityEnabled, setEntityLocked, renameEntity, setLayerOpacity,
    deleteEntity, moveEntity, addMask, clearMask, invertMask, setMaskParam,
    addRasterLayer, useLayerAsMask, addFillLayer,
    undo, canUndo, setLivePreview,
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
              className={`play-stage ${dragOver ? "drop-on" : ""} with-rail`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; if (f) loadFile(f); }}
            >
              <PlayCanvas brushable={!!canvasImage} />
              <div className="play-rail">
                <LayersPanel />
                <ResultsRail />
              </div>
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
    postFrame, showRaw, setShowRaw, initImage, progress, heroRef, canvasImage,
    frameCard, pendingCard, setPendingCard, applyCard, activeSeed, useAsInit,
    clearInit, clearCanvas, canvasSize, setCanvasSize, newCanvas, loadFile,
    activeMask, maskPixels, hasMask, liveMasks,
  } = usePlay();
  const shown = canvasImage;
  const [busy, setBusy] = useState(false);
  const openRef = useRef(null);
  const shownCard = frameCard;

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
            {postFrame && (
              <Tooltip text="Show the layer stack as it is, before post-processing. Useful for judging what the models actually produced.">
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
        {/* What the next run will do, stated beside the picture. The controls
            for it live in the Layers panel; this is only the consequence, which
            is the part that has to be legible at the instant you press the
            button. */}
        {hasMask && (
          <div className="selection-bar" role="status">
            <span className="sub">
              <strong>{liveMasks.length > 1
                ? `${liveMasks.length} masks on`
                : activeMask?.name || "Mask on"}</strong>
              {" "}— {maskPixels.toLocaleString()} px. The next run changes only this area;
              the rest of the canvas is kept.
            </span>
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
    syncMaskOverlayRef, maskTool, applyFloodFillMask, wandTolerance,
    addStroke, rasterizeMasks, inpaintMasks, liveMasks,
  } = usePlay();
  const drawing = useRef(false);
  const last = useRef(null);
  const wandBusy = useRef(false);
  // Points of the stroke in progress, normalised, recorded as they are painted.
  const points = useRef([]);
  const radiusRef = useRef(0);
  const masksRef = useRef(inpaintMasks);
  useEffect(() => { masksRef.current = inpaintMasks; }, [inpaintMasks]);

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
      // Setting width/height clears the canvas, so the masks have to be put
      // back. They used to be rescaled from the old raster; now the strokes are
      // replayed at the new size, which is both sharper and the reason stroke
      // coordinates are stored normalised. compositeMasks() sets the dimensions
      // itself, so the next syncSize sees a match and this cannot recurse.
      rasterizeMasks(masksRef.current, { w, h });
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

  // With every mask hidden the overlay still shows, faintly: otherwise "hidden"
  // and "deleted" look identical and the eye appears to have thrown work away.
  const dim = inpaintMasks.some((m) => m.strokes.length) && !liveMasks.length;

  return (
    <canvas
      className={`mask-overlay ${active ? "on" : ""} ${maskTool === "wand" ? "wand" : ""} ${dim ? "dim" : ""}`}
      ref={maskRef}
      role="img"
      aria-label={maskTool === "wand"
        ? "Inpaint mask — click to select a matching area"
        : "Inpaint mask — drag to paint"}
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

/** One row in the Layers panel. */
function EntityRow({ entity, extra }) {
  const {
    selectedId, setSelectedId, setEntityEnabled, setEntityLocked,
    renameEntity, deleteEntity, moveEntity,
  } = usePlay();
  const [renaming, setRenaming] = useState(false);
  const on = selectedId === entity.id;

  return (
    <div
      className={`layer-row ${on ? "on" : ""} ${entity.enabled ? "" : "off"}`}
      role="button"
      tabIndex={0}
      aria-pressed={on}
      onClick={() => setSelectedId(entity.id)}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSelectedId(entity.id); } }}
    >
      <div className="layer-head">
        <Tooltip text={entity.enabled ? "Hide — keeps it, ignores it" : "Show"}>
          <button
            type="button"
            className={`sel-eye ${entity.enabled ? "on" : ""}`}
            aria-label={entity.enabled ? `Hide ${entity.name}` : `Show ${entity.name}`}
            aria-pressed={entity.enabled}
            onClick={(e) => { e.stopPropagation(); setEntityEnabled(entity.id, !entity.enabled); }}
          >
            {entity.enabled ? "◉" : "○"}
          </button>
        </Tooltip>
        {renaming ? (
          <input
            type="text"
            className="sel-name-input"
            defaultValue={entity.name}
            autoFocus
            aria-label="Name"
            onClick={(e) => e.stopPropagation()}
            onBlur={(e) => { renameEntity(entity.id, e.target.value.trim()); setRenaming(false); }}
            onKeyDown={(e) => {
              if (e.key === "Enter") { renameEntity(entity.id, e.target.value.trim()); setRenaming(false); }
              if (e.key === "Escape") setRenaming(false);
            }}
          />
        ) : (
          <button
            type="button"
            className="sel-name grow"
            title="Rename"
            onClick={(e) => { e.stopPropagation(); setRenaming(true); }}
          >
            {entity.name}
          </button>
        )}
        <Tooltip text={entity.locked ? "Unlock" : "Lock — no edits until unlocked"}>
          <button
            type="button"
            className={`layer-lock ${entity.locked ? "on" : ""}`}
            aria-label={entity.locked ? `Unlock ${entity.name}` : `Lock ${entity.name}`}
            aria-pressed={entity.locked}
            onClick={(e) => { e.stopPropagation(); setEntityLocked(entity.id, !entity.locked); }}
          >
            {entity.locked ? "🔒" : "🔓"}
          </button>
        </Tooltip>
      </div>
      {extra}
      <div className="layer-actions">
        <button type="button" className="btn xs ghost" title="Move up"
          onClick={(e) => { e.stopPropagation(); moveEntity(entity.id, 1); }}>↑</button>
        <button type="button" className="btn xs ghost" title="Move down"
          onClick={(e) => { e.stopPropagation(); moveEntity(entity.id, -1); }}>↓</button>
        <div className="spacer" />
        <button type="button" className="btn xs ghost danger" title="Delete"
          onClick={(e) => { e.stopPropagation(); deleteEntity(entity.id); }}>Delete</button>
      </div>
    </div>
  );
}

/** Canvas entities, in the two groups InvokeAI splits them into.
 *
 *  Raster layers are the picture: hiding, reordering or deleting one changes
 *  what the canvas is, and therefore what the next run starts from. Inpaint
 *  masks are where a run is allowed to change things, and a mask is live for
 *  exactly as long as its row is shown — which is the answer to "should the
 *  mask survive a generation": it survives because nothing hid it.
 */
function LayersPanel() {
  const {
    rasterLayers, inpaintMasks, setLayerOpacity, useLayerAsMask,
    addMask, clearMask, invertMask, maskPixels, liveMasks, undo, canUndo,
  } = usePlay();

  return (
    <aside className="card layers-panel">
      <div className="row between center mb-2">
        <h3 className="mb-0">Layers</h3>
        <Tooltip text="Step back one edit — a fill, a painted stroke, a delete, a reorder. (Ctrl+Z)">
          <button type="button" className="btn ghost sm" onClick={undo} disabled={!canUndo}>Undo</button>
        </Tooltip>
      </div>

      <div className="layer-group-head">
        <span>Raster layers</span>
        <span className="sub">{rasterLayers.length}</span>
      </div>
      {rasterLayers.length === 0 && <p className="hint mb-2">Nothing yet.</p>}
      {/* Topmost first, which is how a stack reads. */}
      {[...rasterLayers].reverse().map((l) => (
        <EntityRow
          key={l.id}
          entity={l}
          extra={(
            <>
              <div className="layer-body">
                {l.image && <img className="layer-thumb" src={l.image} alt="" />}
                <div className="grow">
                  <Slider
                    label="Opacity"
                    value={l.opacity ?? 1}
                    min={0}
                    max={1}
                    step={0.05}
                    onChange={(v) => setLayerOpacity(l.id, v)}
                    fmt={(v) => `${Math.round(v * 100)}%`}
                  />
                </div>
              </div>
              <Tooltip text="Select the area this layer covers. A fill layer covers exactly the region it filled, so this is how you rework the same place again.">
                <button
                  type="button"
                  className="btn xs w-full"
                  onClick={(e) => { e.stopPropagation(); useLayerAsMask(l.id); }}
                  disabled={!l.image}
                >
                  Use as mask
                </button>
              </Tooltip>
            </>
          )}
        />
      ))}

      <div className="layer-group-head mt-2">
        <span>Inpaint masks</span>
        <span className="sub">{liveMasks.length ? `${maskPixels.toLocaleString()} px` : "none on"}</span>
      </div>
      {[...inpaintMasks].reverse().map((m) => (
        <EntityRow
          key={m.id}
          entity={m}
          extra={(
            <div className="layer-actions">
              <button type="button" className="btn xs ghost" title="Swap masked for unmasked"
                onClick={(e) => { e.stopPropagation(); invertMask(m.id); }}>Invert</button>
              <button type="button" className="btn xs ghost" title="Empty it, keep the row and its settings"
                onClick={(e) => { e.stopPropagation(); clearMask(m.id); }}
                disabled={!m.strokes.length}>Clear</button>
            </div>
          )}
        />
      ))}
      <button type="button" className="btn sm w-full mt-2" onClick={addMask}>Add mask</button>
    </aside>
  );
}
