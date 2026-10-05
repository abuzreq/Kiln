import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CreatePanel from "../components/CreatePanel.jsx";
import { BendWorkspace } from "./Craft.jsx";
import SweepPanel from "./Sweep.jsx";
import Merge from "./Merge.jsx";
import { useApp } from "../state.jsx";
import { api, downloadPost, mediaUrl, thumbUrl } from "../api.js";
import {
  EyeIcon, InvertIcon, ClearIcon, UpIcon, DownIcon, TrashIcon, DownloadIcon, CaptureIcon,
  UndoIcon, RedoIcon, ChevronDownIcon, AddIcon,
} from "../components/icons.jsx";
import { Progress, Slider, Tooltip, TipLabel, Popover } from "../components/ui.jsx";
import { ToolRail, ToolOptions } from "../components/CanvasTools.jsx";
import { PlayCtx, usePlay, fileToDataUrl } from "./playContext.jsx";
import {
  loadSampleParams, saveSampleParams, SampleSettingsPanel, paramsFromCard, cardLabel,
  LIVE_PARAM_KEYS,
} from "../sampleSettings.jsx";
import ModelPicker from "../components/ModelPicker.jsx";
import PlaySetup from "../components/PlaySetup.jsx";
import { buildContrastMask, floodFillMask, measureMask } from "../contrastMask.js";
import {
  brushStroke, cachedStroke, invertStroke, moveStroke, alphaToCache, overlayToCache,
  paintBrushPoint, rasterize, hatchTile, shapeStroke, replayShape, MASK_RGB,
} from "../selection.js";
import {
  newRasterLayer, newInpaintMask, newDocument, blankImage, uniqueName, uniqueFrom, activeMasks,
  maskShown, flattenLayers, compositeMasks, compositeOnto, fitIntoCanvas, imageSize, punchToMask,
  maskDataUrlFrom, layerAlphaToCache, reorder,
} from "../layers.js";

const HISTORY_MAX = 24;
const TERMINAL = new Set(["done", "error", "cancelled"]);

/** queue_YYYYmmdd-HHMMSS, local time: one captures folder per queue. */
function queueFolderName(d = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  return `queue_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}`
    + `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
// The watched run is polled in full at the browser's preview rate; the rest of
// the queue only needs progress and place in line, from the light job list.
const WATCH_POLL_MS = 300;
const LIST_EVERY = 3;   // ticks, so about once a second
// One entry is a whole document -- both entity arrays and the selected row.
// That is only stroke lists and image references, so it is cheap enough to keep
// plenty of, and it makes Undo mean the same thing everywhere.
const UNDO_MAX = 40;
const CANVAS_TABS = new Set(["create"]);

// Canvas bounds. The floor is a size a brush can still be aimed inside; the
// ceiling is well past what these models sample at, and a fill is scaled down
// to MAX_FILL_SIDE anyway, so nothing is gained by going bigger.
// Which tab the Layers | Assets panel shows, remembered per browser.
const MANAGE_KEY = "kiln.canvasManageTab";
const CANVAS_MIN = 64;
const CANVAS_MAX = 2048;
// An opened image sets the canvas to its own size. Bigger than the typed-in
// ceiling because an upscale is meant to land bigger than you could type.
const OPEN_MAX = 4096;

function clampCanvasSide(v) {
  const n = Math.round(Number(v) || 0);
  if (!n) return CANVAS_MIN;
  return Math.max(CANVAS_MIN, Math.min(CANVAS_MAX, n));
}

/** A canvas size for an opened image: its own, unless that is absurd. */
function openSize({ w, h }) {
  const k = Math.min(1, OPEN_MAX / Math.max(w, h, 1));
  return {
    w: Math.max(CANVAS_MIN, Math.round(w * k)),
    h: Math.max(CANVAS_MIN, Math.round(h * k)),
  };
}

const fileStem = (name) => (name || "").replace(/\.[^.]+$/, "");

export default function Play() {
  const {
    toast, playTab, bendStack, setBendStack, modelPath, setModelPath, setTabBusy, models,
    modelsPartial,
  } = useApp();
  const tab = playTab;
  // The settings panel needs the selected model, not just its path: whether the
  // EMA toggle does anything, and what image sizes this model can actually
  // produce, are properties of the model rather than of the sampler. While the
  // list is still streaming in, the picked model counts as soon as it arrives.
  const activeModel = useMemo(
    () => (models || modelsPartial || []).find((m) => m.path === modelPath) || null,
    [models, modelsPartial, modelPath],
  );
  // `frame` is the flattened stack — a cache, never written to directly.
  const [frame, setFrameState] = useState(null);
  // Post-process is a display stage over the flattened stack rather than an
  // edit to it, so there is no longer a separate "raw" frame to keep: the
  // flattened stack *is* the unprocessed image, and Unprocessed just shows it.
  const [postFrame, setPostFrame] = useState(null);
  // Per-tab state that has to outlive the tab's own component (see usePlayState).
  const [tabState, setTabState] = useState({});
  const [pendingCard, setPendingCard] = useState(null);
  const [showRaw, setShowRaw] = useState(false);
  const [history, setHistory] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [brushSize, setBrushSize] = useState(48);
  const [brushHard, setBrushHard] = useState(false);
  const [eraser, setEraser] = useState(false);
  // Move by default: a stray click on the picture then never paints.
  const [maskTool, setMaskTool] = useState("move");
  // The Shape tool's kind, and the Generate panel's settings; both live here
  // so a tab switch does not reset them.
  const [shapeKind, setShapeKind] = useState("rect");
  const [genShape, setGenShape] = useState(() => ({
    kind: "blobs", seed: Math.floor(Math.random() * 2 ** 31), coverage: 0.3, soften: 4, params: {},
  }));
  // A polygon in progress lives in the overlay; it registers these so the
  // panel's keys (Enter, Esc, Backspace) can reach it, and reports how many
  // corners are placed so the hint can say so.
  const polygonRef = useRef({ close: () => {}, cancel: () => {}, pop: () => {} });
  const [polyCount, setPolyCount] = useState(0);
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
  // The layer runs write into. Set by selecting a layer row, and kept when a
  // mask row is selected afterwards, so painting a mask does not change where
  // the fill lands. There is always one: see pickActiveLayer.
  const [activeLayerId, setActiveLayerId] = useState(null);
  const [undoLen, setUndoLen] = useState(0);
  const [redoLen, setRedoLen] = useState(0);
  // Images the user keeps at hand (the workspace assets folder).
  const [assets, setAssets] = useState([]);
  // In-progress frames from a running job, kept off the layer stack so a live
  // preview never becomes a layer of its own.
  const [livePreview, setLivePreview] = useState(null);
  // Whether a run shows its in-progress picture. A viewing preference, kept per
  // browser; with it off the canvas keeps the old picture until the result lands.
  const [livePreviewOn, setLivePreviewOnState] = useState(() => {
    try { return localStorage.getItem("kiln.livePreview") !== "off"; } catch { return true; }
  });
  const setLivePreviewOn = useCallback((on) => {
    setLivePreviewOnState(on);
    try { localStorage.setItem("kiln.livePreview", on ? "on" : "off"); } catch { /* not fatal */ }
  }, []);

  // The Create panel's run, so the blank canvas's own Generate button can
  // start the same run the panel would. Set by CreatePanel on every render.
  const generateRef = useRef(null);
  // Which tab the Layers | Assets panel shows. Here rather than in the panel
  // so the blank canvas can point at Assets. Remembered per browser.
  const [manageTab, setManageTabState] = useState(() => {
    try { return localStorage.getItem(MANAGE_KEY) === "assets" ? "assets" : "layers"; } catch { return "layers"; }
  });
  const setManageTab = useCallback((t) => {
    setManageTabState(t);
    try { localStorage.setItem(MANAGE_KEY, t); } catch { /* not fatal */ }
  }, []);

  // --- Generation queue --------------------------------------------------
  // Every Create run the server holds for us, oldest first, until it lands.
  // One is *watched*: its preview is on the canvas, its progress in the
  // header, and Pause / Stop & save act on it. The rest are followed through
  // the light job list. Kept here, not in CreatePanel, because Play stays
  // mounted: a queue keeps landing while you are on another tab.
  //   run = { id, job, fill, toResults, discard }
  const [runs, setRuns] = useState([]);
  const runsRef = useRef([]);
  const [watchedId, setWatchedId] = useState(null);
  const watchedRef = useRef(null);
  // id -> what to do with the finished job; registered by whoever submitted it.
  const settlers = useRef({});
  // The captures folder for this queue's results: named when a queue forms,
  // shared by every run sent to Results while it lasts, forgotten once it empties.
  const queueFolderRef = useRef(null);
  const [queueFolder, setQueueFolder] = useState(null);
  const commitRuns = useCallback((fn) => {
    runsRef.current = fn(runsRef.current);
    setRuns(runsRef.current);
    if (!runsRef.current.length && queueFolderRef.current) {
      queueFolderRef.current = null;
      setQueueFolder(null);
    }
  }, []);
  const watch = useCallback((id) => {
    if (watchedRef.current === id) return;
    watchedRef.current = id;
    setWatchedId(id);
    // The picture on the canvas belongs to the run you are watching.
    setLivePreview(null);
  }, []);
  /** Follow freshly submitted jobs until they land.
   *
   *  The landing rule lives here: a run goes into the layer only if it was
   *  alone for its whole life. Submitting while anything is live sends the new
   *  run *and* every live one to Results, so ten queued runs never take turns
   *  replacing the layer, and which one ended up there never depends on
   *  finishing order. Fills are exempt -- each is punched to its own mask, and
   *  the panel only starts one when nothing else is live.
   *
   *  Returns one promise per job, settled once its `settle` has run. */
  const trackRuns = useCallback((jobs, { settle, fill = false, toResults = false }) => {
    const queueing = toResults || jobs.length > 1 || runsRef.current.length > 0;
    if (queueing && !queueFolderRef.current) {
      queueFolderRef.current = queueFolderName();
      setQueueFolder(queueFolderRef.current);
    }
    const folder = queueFolderRef.current;
    const done = jobs.map((j) => new Promise((resolve) => {
      settlers.current[j.id] = async (finished, run) => {
        try { await settle(finished, run); } finally { resolve(finished); }
      };
    }));
    commitRuns((rs) => [
      ...rs.map((r) => (queueing && !r.fill ? { ...r, toResults: true, folder } : r)),
      ...jobs.map((j) => ({
        id: j.id, job: j, fill, toResults: queueing && !fill, folder: queueing ? folder : null,
        discard: false,
      })),
    ]);
    if (!watchedRef.current && jobs.length) watch(jobs[0].id);
    return done;
  }, [commitRuns, watch]);
  /** Cancel from the queue strip: the run is dropped, not kept. */
  const cancelRun = useCallback(async (id) => {
    commitRuns((rs) => rs.map((r) => (r.id === id ? { ...r, discard: true } : r)));
    await api.post(`/jobs/${id}/cancel`);
  }, [commitRuns]);

  // One poll loop for the whole queue, alive while anything is in it.
  const anyRuns = runs.length > 0;
  useEffect(() => {
    if (!anyRuns) return undefined;
    let stopped = false;
    let timer = null;
    let n = 0;
    const nextWatched = () => {
      const rs = runsRef.current;
      return (rs.find((r) => r.job.status === "running") || rs[0])?.id ?? null;
    };
    const absorb = async (id, j) => {
      const run = runsRef.current.find((r) => r.id === id);
      if (!run) return;               // already landed via the other poll
      const job = j || { id, status: "error", message: "the job disappeared", detail: {} };
      if (!TERMINAL.has(job.status)) {
        commitRuns((rs) => rs.map((r) => (r.id === id ? { ...r, job } : r)));
        if (id === watchedRef.current && job.detail?.frame) setLivePreview(job.detail.frame);
        return;
      }
      commitRuns((rs) => rs.filter((r) => r.id !== id));
      if (id === watchedRef.current) watch(nextWatched());
      const settle = settlers.current[id];
      delete settlers.current[id];
      if (settle) await settle(job, run);
    };
    const tick = async () => {
      n += 1;
      try {
        const wid = watchedRef.current;
        if (wid) {
          const j = await api.get(`/jobs/${wid}`);
          if (stopped) return;
          await absorb(wid, j);
        }
        const others = runsRef.current.filter((r) => r.id !== watchedRef.current);
        if (others.length && (n % LIST_EVERY === 1 || !wid)) {
          const list = await api.get("/jobs?kinds=sample,inpaint&light=1");
          if (stopped) return;
          const byId = new Map(list.map((j) => [j.id, j]));
          for (const r of others) {
            const j = byId.get(r.id);
            // A finished one is fetched in full once: its images are what lands.
            if (j && TERMINAL.has(j.status)) await absorb(r.id, await api.get(`/jobs/${r.id}`));
            else await absorb(r.id, j || null);
          }
        }
      } catch { /* a missed poll is retried on the next tick */ }
      if (!stopped && runsRef.current.length) timer = setTimeout(tick, WATCH_POLL_MS);
    };
    tick();
    return () => { stopped = true; clearTimeout(timer); };
  }, [anyRuns, commitRuns, watch]);
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
  const redoRef = useRef([]);
  // The coalesce key of the last undo entry, so a slider drag folds into one.
  const undoKeyRef = useRef(null);
  // A mirror of the entity state, so the pointer handlers that add strokes can
  // read the current value without going stale between renders.
  const docRef = useRef({ rasterLayers: [], inpaintMasks: [], selectedId: null });
  const activeLayerRef = useRef(null);
  const canvasSizeRef = useRef(null);
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
  useEffect(() => { activeLayerRef.current = activeLayerId; }, [activeLayerId]);
  useEffect(() => { canvasSizeRef.current = canvasSize; }, [canvasSize]);

  /** The layer a run writes into. Always resolves to one: the chosen layer if
   *  it still exists, else the topmost shown layer, else the top of the stack.
   *  A document always has at least one layer (deleteEntity keeps the last),
   *  so a run never has to invent a layer of its own. */
  const pickActiveLayer = useCallback((doc, id) => (
    doc.rasterLayers.find((l) => l.id === id)
    || [...doc.rasterLayers].reverse().find((l) => l.enabled)
    || doc.rasterLayers[doc.rasterLayers.length - 1]
    || null
  ), []);

  /** Highlight a row; a layer row also becomes the active layer. */
  const selectEntity = useCallback((id) => {
    setSelectedId(id);
    if (docRef.current.rasterLayers.some((l) => l.id === id)) {
      activeLayerRef.current = id;
      setActiveLayerId(id);
    }
  }, []);

  /** Redraw the mask store: the union of every mask that is on. What is
   *  displayed is the overlay's business (see MaskOverlay), and follows a
   *  different switch. */
  const rasterizeMasks = useCallback((masks, size) => {
    const c = maskRef.current;
    if (!c) return;
    const w = size?.w || c.width;
    const h = size?.h || c.height;
    if (!w || !h) return;
    compositeMasks((masks || []).filter((m) => m.enabled), c, w, h);
    syncMaskOverlayRef.current();
    bumpMask();
  }, [bumpMask]);

  /** Push an undo snapshot. Consecutive pushes with the same `key` collapse
   *  into one entry: an opacity drag fires per slider tick, and forty ticks
   *  used to evict the fill you were trying to undo. */
  const pushUndo = useCallback((snap, key = null) => {
    if (key && key === undoKeyRef.current && undoRef.current.length) return;
    undoKeyRef.current = key;
    undoRef.current = [snap, ...undoRef.current].slice(0, UNDO_MAX);
    setUndoLen(undoRef.current.length);
  }, []);

  const clearUndo = useCallback(() => {
    undoRef.current = [];
    redoRef.current = [];
    undoKeyRef.current = null;
    setUndoLen(0);
    setRedoLen(0);
  }, []);

  /** Make `doc` the current document, on screen and in the mirror. */
  const showDoc = useCallback((doc) => {
    docRef.current = doc;
    setRasterLayers(doc.rasterLayers);
    setInpaintMasks(doc.inpaintMasks);
    setSelectedId(doc.selectedId);
    rasterizeMasks(doc.inpaintMasks);
  }, [rasterizeMasks]);

  /** The one mutator every entity edit goes through.
   *
   *  `fn` gets the current document and returns the parts of it that change.
   *  Snapshotting the whole document here is what makes one Undo cover a
   *  painted stroke, a deleted layer, a reorder and an accepted fill alike --
   *  which is the behaviour that replaced the staging area's Accept/Discard.
   */
  const edit = useCallback((fn, { redraw = true, undo = true, coalesce = null } = {}) => {
    const prev = docRef.current;
    const patch = typeof fn === "function" ? fn(prev) : fn;
    if (!patch) return;
    const next = { ...prev, ...patch };
    if (undo) {
      pushUndo(prev, coalesce);
      // A new edit forks history; what was undone is no longer reachable.
      redoRef.current = [];
      setRedoLen(0);
    }
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
    undoKeyRef.current = null;
    setUndoLen(rest.length);
    // Redo exists because Undo is document-wide: one stray Ctrl+Z after a fill
    // would otherwise cost a GPU run, and Results can only put it back by
    // replacing the whole stack.
    redoRef.current = [docRef.current, ...redoRef.current].slice(0, UNDO_MAX);
    setRedoLen(redoRef.current.length);
    showDoc(prev);
  }, [showDoc]);

  const redo = useCallback(() => {
    const stack = redoRef.current;
    if (!stack.length) return;
    const [next, ...rest] = stack;
    redoRef.current = rest;
    setRedoLen(rest.length);
    undoRef.current = [docRef.current, ...undoRef.current].slice(0, UNDO_MAX);
    undoKeyRef.current = null;
    setUndoLen(undoRef.current.length);
    showDoc(next);
  }, [showDoc]);

  /** Which mask the brush paints into.
   *
   *  The selected row if it is a mask, else the topmost enabled one. Falling
   *  back means the tools always have a target, so painting on a fresh canvas
   *  works without first understanding the panel.
   */
  const paintTargetId = useCallback((doc) => {
    const sel = doc.inpaintMasks.find((m) => m.id === doc.selectedId);
    if (sel) return sel.id;
    const open = [...doc.inpaintMasks].reverse().find((m) => m.enabled);
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
          visible: true,
          strokes: [...m.strokes, stroke],
        })),
        selectedId: id,
      };
    }, { redraw });
  }, [edit, paintTargetId]);

  /** A layer's shown/hidden, or a mask's on/off. */
  const setEntityEnabled = useCallback((id, on) => {
    edit((doc) => ({
      rasterLayers: patchEntity(doc.rasterLayers, id, { enabled: !!on }),
      inpaintMasks: patchEntity(doc.inpaintMasks, id, { enabled: !!on }),
    }));
  }, [edit]);

  /** A mask's shown/hidden: only what is drawn, never what the run gets. */
  const setMaskVisible = useCallback((id, on) => {
    edit((doc) => ({ inpaintMasks: patchEntity(doc.inpaintMasks, id, { visible: !!on }) }));
  }, [edit]);

  /** Painting into a mask turns it on and shows it, so a stroke never lands
   *  somewhere it cannot be seen or will not count. Done before the stroke
   *  starts, so the live brush draws onto a display that already has the
   *  rest of that mask in it. */
  const wakeMask = useCallback((id) => {
    const m = docRef.current.inpaintMasks.find((x) => x.id === id);
    if (!m || (m.enabled && maskShown(m))) return;
    edit((doc) => ({ inpaintMasks: patchEntity(doc.inpaintMasks, id, { enabled: true, visible: true }) }));
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
    }), { redraw: false, coalesce: `opacity:${id}` });
  }, [edit]);

  /** Remove a row. The last layer stays: there is always one to write into. */
  const deleteEntity = useCallback((id) => {
    edit((doc) => {
      const isLayer = doc.rasterLayers.some((l) => l.id === id);
      if (isLayer && doc.rasterLayers.length <= 1) return null;
      return {
        rasterLayers: doc.rasterLayers.filter((e) => e.id !== id),
        inpaintMasks: doc.inpaintMasks.filter((e) => e.id !== id),
        selectedId: doc.selectedId === id ? null : doc.selectedId,
      };
    });
  }, [edit]);

  /** A new empty layer, on top, selected and active. */
  const addLayer = useCallback(() => {
    const { w, h } = canvasSizeRef.current;
    let id = null;
    edit((doc) => {
      const l = newRasterLayer({ name: uniqueName(doc.rasterLayers, "Layer"), image: blankImage(w, h) });
      id = l.id;
      return { rasterLayers: [...doc.rasterLayers, l], selectedId: l.id };
    }, { redraw: false });
    if (id) { activeLayerRef.current = id; setActiveLayerId(id); }
  }, [edit]);

  /** A copy of a layer directly above it, selected and active. The way to try
   *  something else on the same ground without losing what is there. */
  const duplicateLayer = useCallback((id) => {
    let newId = null;
    edit((doc) => {
      const i = doc.rasterLayers.findIndex((l) => l.id === id);
      if (i < 0) return null;
      const src = doc.rasterLayers[i];
      const copy = newRasterLayer({
        name: uniqueFrom(doc.rasterLayers, `${src.name} copy`),
        image: src.image,
        card: src.card,
      });
      copy.opacity = src.opacity;
      newId = copy.id;
      const out = [...doc.rasterLayers];
      out.splice(i + 1, 0, copy);
      return { rasterLayers: out, selectedId: copy.id };
    }, { redraw: false });
    if (newId) { activeLayerRef.current = newId; setActiveLayerId(newId); }
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
          visible: true,
          strokes: [...m.strokes, invertStroke()],
        })),
      };
    });
  }, [edit, paintTargetId]);

  /** Translate a mask by a fraction of the canvas. Consecutive moves fold into
   *  one stroke, so a drag off the canvas and back loses nothing, and the
   *  stroke list does not grow by one per nudge. */
  const moveMask = useCallback((id, dx, dy, { coalesce = null } = {}) => {
    if (!id || (!dx && !dy)) return;
    edit((doc) => ({
      inpaintMasks: patchEntity(doc.inpaintMasks, id, (m) => {
        const last = m.strokes[m.strokes.length - 1];
        if (last?.type !== "move") return { strokes: [...m.strokes, moveStroke(dx, dy)] };
        const nx = last.dx + dx;
        const ny = last.dy + dy;
        const rest = m.strokes.slice(0, -1);
        const still = Math.abs(nx) < 1e-9 && Math.abs(ny) < 1e-9;
        return { strokes: still ? rest : [...rest, moveStroke(nx, ny)] };
      }),
    }), { coalesce });
  }, [edit]);

  /** Arrow-key move, in canvas pixels. A held key is one undo entry. */
  const nudgeMask = useCallback((id, px, py) => {
    const c = maskRef.current;
    if (!c?.width || !c?.height) return;
    moveMask(id, px / c.width, py / c.height, { coalesce: `move:${id}` });
  }, [moveMask]);

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

  /** A new layer holding an image, on top of the stack. User-initiated only:
   *  the Results "Layer" button. Runs never call this. */
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

  /** Put an image into a layer, replacing what it had. Turns the layer on:
   *  a run landing in a hidden layer would look like it produced nothing. */
  const writeLayer = useCallback((id, image, card) => {
    edit((doc) => ({
      rasterLayers: patchEntity(doc.rasterLayers, id, { image, card: card ?? null, enabled: true }),
    }), { redraw: false });
  }, [edit]);

  /** Select the area a layer covers, and nothing else.
   *
   *  A layer that has only taken fills is transparent everywhere else, so its
   *  alpha is exactly the ground those fills covered. Every other mask is
   *  turned off: adding the new mask alongside the old one, which was usually
   *  still on, made the bar read "2 masks on" and the next fill cover the union.
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
      return {
        inpaintMasks: [...doc.inpaintMasks.map((x) => ({ ...x, enabled: false })), m],
        selectedId: m.id,
      };
    });
  }, [edit]);

  /** A mask of its own for one stroke: what "New mask" makes from a generated
   *  shape or a contrast side. Other masks are left as they are: a composition
   *  is several masks on at once, and the bar and the union fill already say so.
   *  (useLayerAsMask turns the others off on purpose; that flow re-selects an
   *  area a fill just covered, and the old mask would double it.) */
  const addMaskWithStroke = useCallback((stroke, name) => {
    edit((doc) => {
      const m = {
        ...newInpaintMask({ name: uniqueName(doc.inpaintMasks, name || "Inpaint Mask") }),
        strokes: [stroke],
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
      // Every layer hidden is still a canvas -- a blank one to paint a mask on
      // or generate into -- not an empty box.
      .then((img) => { if (live) setFrameState(img || blankImage(canvasSize.w, canvasSize.h)); })
      .catch(() => { if (live) setFrameState(null); });
    return () => { live = false; };
  }, [rasterLayers, canvasSize]);

  /** A new document around one image, at the image's own size.
   *
   *  What "open this" means: a different picture, not another layer on the
   *  same one. The canvas takes the image's size, because flatten stretches
   *  every layer to the canvas and an opened image should look like itself.
   *  Masks are kept only when asked (upscale): their strokes are normalised,
   *  so they replay at the new size. */
  const openDocument = useCallback(async (img, card, name, { keepMasks = false } = {}) => {
    const size = openSize(await imageSize(img));
    setLivePreview(null);
    setPostFrame(null);
    blankFrameRef.current = null;
    canvasSizeRef.current = size;
    setCanvasSizeState(size);
    let id = null;
    edit((doc) => {
      const l = newRasterLayer({ name: name || "Layer 1", image: img, card });
      id = l.id;
      return {
        rasterLayers: [l],
        inpaintMasks: keepMasks && doc.inpaintMasks.length
          ? doc.inpaintMasks
          : [newInpaintMask({ name: "Inpaint Mask 1" })],
        selectedId: l.id,
      };
    });
    activeLayerRef.current = id;
    setActiveLayerId(id);
  }, [edit]);

  /** Put a result into the active layer. Results are canvas-sized, so this is
   *  the honest "show me that one again" -- and it used to replace the whole
   *  stack, which read as layers being deleted by a click meant to look. */
  const placeHistory = useCallback((entry) => {
    if (!entry?.img) return;
    const target = pickActiveLayer(docRef.current, activeLayerRef.current);
    if (!target) return;
    writeLayer(target.id, entry.img, entry.card ?? null);
  }, [pickActiveLayer, writeLayer]);

  /** Bring a result back as a new layer over the stack. */
  const addHistoryAsLayer = useCallback((entry) => {
    if (!entry?.img) return;
    const seed = entry.card?.params?.seed;
    addRasterLayer({
      image: entry.img,
      card: entry.card ?? null,
      name: seed != null ? `Result · seed ${seed}` : "Result",
    });
    toast("Added as a layer", "success");
  }, [addRasterLayer, toast]);

  const removeHistory = useCallback((id) => {
    setHistory((h) => h.filter((x) => x.id !== id));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  /** A new document: one Background layer at that size, one empty mask.
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
    setPendingCard(null);
    setLivePreview(null);
    docRef.current = { ...doc, selectedId: doc.inpaintMasks[0].id };
    canvasSizeRef.current = { w: width, h: height };
    setRasterLayers(doc.rasterLayers);
    setInpaintMasks(doc.inpaintMasks);
    setSelectedId(doc.inpaintMasks[0].id);
    activeLayerRef.current = doc.rasterLayers[0].id;
    setActiveLayerId(doc.rasterLayers[0].id);
    rasterizeMasks(doc.inpaintMasks, { w: width, h: height });
    clearUndo();
    if (!quiet) toast(`Blank canvas — ${width}x${height}`, "success");
  }, [toast, rasterizeMasks, clearUndo]);

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

  // --- Assets: images the user keeps at hand ---------------------------------

  const loadAssets = useCallback(async () => {
    try {
      const r = await api.get("/assets");
      setAssets(r.assets || []);
    } catch { /* the panel just stays empty */ }
  }, []);
  useEffect(() => { loadAssets(); }, [loadAssets]);

  /** Keep an image as an asset. Says so when it cannot: a drop that produces
   *  nothing looks like the panel ignored it. */
  const keepAsset = useCallback(async (image, name, card) => {
    try {
      const r = await api.post("/assets", { image, name: name || undefined, card: card || undefined });
      if (r.asset) setAssets((a) => [r.asset, ...a.filter((x) => x.path !== r.asset.path)]);
      return r.asset || null;
    } catch (e) {
      toast(`Could not keep ${name || "the image"} as an asset: ${e.message}`, "error");
      return null;
    }
  }, [toast]);

  const deleteAsset = useCallback(async (asset) => {
    try {
      await api.del("/assets", { path: asset.path });
      setAssets((a) => a.filter((x) => x.path !== asset.path));
    } catch (e) { toast(e.message, "error"); }
  }, [toast]);

  /** The asset's bytes as a data URL, which is what layers hold. */
  const assetDataUrl = useCallback(async (asset) => {
    const blob = await fetch(mediaUrl(asset.path)).then((r) => {
      if (!r.ok) throw new Error("Could not read that asset");
      return r.blob();
    });
    return fileToDataUrl(blob);
  }, []);

  /** Place an asset into the active layer, fitted to the canvas. */
  const placeAsset = useCallback(async (asset) => {
    try {
      const url = await assetDataUrl(asset);
      const { w, h } = canvasSizeRef.current;
      const fitted = await fitIntoCanvas(url, w, h);
      const target = pickActiveLayer(docRef.current, activeLayerRef.current);
      if (!target) return;
      let card = null;
      try { card = (await api.post("/perform/read-params", { image: url })).card || null; } catch { /* none */ }
      writeLayer(target.id, fitted, card);
      toast(`Placed into ${target.name}`, "success");
    } catch (e) { toast(e.message, "error"); }
  }, [assetDataUrl, pickActiveLayer, writeLayer, toast]);

  /** Open an asset as a new document at its own size. */
  const openAsset = useCallback(async (asset) => {
    try {
      const url = await assetDataUrl(asset);
      let card = null;
      try { card = (await api.post("/perform/read-params", { image: url })).card || null; } catch { /* none */ }
      await openDocument(url, card, asset.name);
      setPendingCard(card && card.params ? card : null);
      toast("Opened", "success");
    } catch (e) { toast(e.message, "error"); }
  }, [assetDataUrl, openDocument, toast]);

  /** Add files to the assets without touching the canvas. */
  const addAssetFiles = useCallback(async (files) => {
    const all = [...(files || [])];
    const imgs = all.filter((f) => f.type.startsWith("image/"));
    if (all.length && !imgs.length) { toast("Only images can be assets", "error"); return; }
    let n = 0;
    for (const f of imgs) {
      const url = await fileToDataUrl(f);
      if (await keepAsset(url, fileStem(f.name))) n += 1;
    }
    if (n) toast(n === 1 ? "Added to assets" : `${n} added to assets`, "success");
  }, [keepAsset, toast]);

  /** Open an image file as the document, and keep it as an asset so it can be
   *  placed again later without finding the file again. */
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
    await openDocument(url, card, fileStem(file.name) || "Layer 1");
    pushHistory(url, null, card);
    setPendingCard(card && card.params ? card : null);
    keepAsset(url, fileStem(file.name), card);
    toast(card?.params ? "On canvas — Kiln settings found" : "On canvas", "success");
  }, [openDocument, pushHistory, keepAsset, toast]);

  /** The mask as the backend wants it: white where the fill should happen.
   *
   *  Reads the overlay, which is already the union of every enabled mask, and
   *  returns null when none of them contribute anything — which is how the eye
   *  toggles work without every caller having to check them.
   */
  const getMaskDataUrl = useCallback(() => {
    if (!activeMasks(inpaintMasks).length) return null;
    return maskDataUrlFrom(maskRef.current);
  }, [inpaintMasks, maskVersion]);

  /** A contrast split becomes one stroke, replacing whatever came before it.
   *
   *  Replacing rather than adding is what the two preview tiles imply: picking
   *  the other side should give you the other side, not both sides at once.
   */
  const applyContrastMask = useCallback(async (imageSrc, options, { target = "active", name = "Contrast" } = {}) => {
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
      if (target === "new") {
        const m = { ...newInpaintMask({ name: uniqueName(doc.inpaintMasks, name) }), strokes: [stroke] };
        return { inpaintMasks: [...doc.inpaintMasks, m], selectedId: m.id };
      }
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
          visible: true,
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
    create: runs.length > 0,
    // the compare and the parameter sweep are both long Bend runs
    bend: !!tabState["bend.busy"] || !!tabState["bend.sweep"]?.busy,
    merge: ["queued", "running"].includes(tabState["merge.job"]?.status) || !!tabState["merge.writing"],
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

  /** A finished fill lands in the active layer: punched down to the region the
   *  mask allowed, then drawn over what the layer already had there.
   *
   *  So the layer accumulates -- fill one area, then the next, and the layer
   *  holds both -- and a second attempt at the same area replaces the first
   *  in that layer. To keep the first, duplicate the layer before filling
   *  again; to drop the second, Undo. Kiln never adds a layer of its own.
   */
  const fillIntoLayer = useCallback(async (img, card, maskSrc, { maskIds = [] } = {}) => {
    const punched = await punchToMask(img, maskSrc);
    const target = pickActiveLayer(docRef.current, activeLayerRef.current);
    if (!target) return;
    const { w, h } = canvasSizeRef.current;
    const merged = await compositeOnto(target.image, punched, w, h);
    // The layer write and the masks turning off are one edit, so one Undo
    // takes back both: the fill, and the mask that made it going quiet.
    // A mask that has done its fill turns off rather than staying armed, so
    // the next Generate is the whole canvas again unless you turn it back on.
    const off = new Set(maskIds);
    edit((doc) => ({
      rasterLayers: patchEntity(doc.rasterLayers, target.id, { image: merged, card: card ?? null, enabled: true }),
      inpaintMasks: doc.inpaintMasks.map((m) => (off.has(m.id) ? { ...m, enabled: false } : m)),
    }), { redraw: off.size > 0 });
    setLivePreview(null);
  }, [pickActiveLayer, edit]);

  /** A whole-canvas generation replaces the active layer's image. */
  const generateIntoLayer = useCallback((img, raw, card) => {
    const target = pickActiveLayer(docRef.current, activeLayerRef.current);
    if (target) writeLayer(target.id, img, card);
    pushHistory(img, raw, card);
    setLivePreview(null);
  }, [pickActiveLayer, writeLayer, pushHistory]);

  // Post-process sits between the flattened stack and the screen, so
  // Unprocessed shows the stack itself. A live preview outranks both.
  const canvasImage = livePreview || (showRaw ? frame : (postFrame || frame));
  // Still the untouched blank ground: nothing generated, dropped or restored
  // over it. A fill has to know, because a partial run started from empty
  // pixels gives back the empty pixels it started from.
  const canvasIsBlank = (!!frame && frame === blankFrameRef.current)
    || !rasterLayers.some((l) => l.enabled && l.image);

  /** A picture made elsewhere (a Sweep cell, a bend comparison) brought to
   *  the canvas to keep working on. A blank canvas becomes that picture, at
   *  its size, the way a dropped file does; anything already there is kept
   *  and the picture goes on top as a layer of its own. It joins Results
   *  either way. Callers switch to the tab and apply the recipe themselves. */
  const commitFrame = useCallback(async (img, raw, card, name) => {
    if (!img) return;
    const seed = card?.params?.seed;
    const label = name || (seed != null ? `Seed ${seed}` : "Opened");
    if (canvasIsBlank) {
      await openDocument(img, card ?? null, label, { keepMasks: true });
    } else {
      addRasterLayer({ image: img, card: card ?? null, name: label });
      toast("Added as a layer on top", "success");
    }
    pushHistory(img, raw, card);
    setLivePreview(null);
  }, [canvasIsBlank, openDocument, addRasterLayer, pushHistory, toast]);
  // The recipe the canvas carries: that of the topmost shown layer that has
  // one. A stack has no single recipe, and this is the layer whose pixels are
  // most of what you see. It used to be a separate state that fills never
  // updated, so the seed pill and the exported PNG kept naming the layer
  // underneath.
  const frameCard = useMemo(
    () => [...rasterLayers].reverse().find((l) => l.enabled && l.image && l.card)?.card ?? null,
    [rasterLayers],
  );
  const activeLayer = useMemo(
    () => pickActiveLayer({ rasterLayers }, activeLayerId),
    [rasterLayers, activeLayerId, pickActiveLayer],
  );
  const activeSeed = frameCard?.params?.seed ?? null;
  const job = runs.find((r) => r.id === watchedId)?.job ?? null;
  const genRunning = !!(job && job.status === "running");
  const genPaused = genRunning && !!job?.detail?.paused;
  const progress = genRunning ? { value: job.progress, message: job.message } : null;
  const canUndo = undoLen > 0;
  const canRedo = redoLen > 0;
  const maskMeasure = useMemo(
    () => measureMask(maskRef.current),
    // maskVersion is the signal; the pixels live on a ref.
    [maskVersion],
  );
  const maskPixels = maskMeasure.count;
  const liveMasks = activeMasks(inpaintMasks);
  const hasMask = liveMasks.length > 0 && maskPixels > 0;
  // The mask whose settings the panel edits, and whose name the labels use.
  const activeMask = inpaintMasks.find((m) => m.id === paintTargetId({ ...docRef.current, inpaintMasks, selectedId }))
    || null;
  // The box drawn around the active mask, in canvas pixels. When it is the
  // only mask on, the union scan above already measured it; otherwise its own
  // raster is measured apart from the others.
  const measureScratch = useRef(null);
  const liveKey = liveMasks.map((m) => m.id).join(",");
  const activeBox = useMemo(() => {
    const c = maskRef.current;
    if (!c?.width || !activeMask?.enabled || !maskShown(activeMask) || !activeMask.strokes.length) return null;
    let m = maskMeasure;
    if (liveKey !== activeMask.id) {
      if (!measureScratch.current) measureScratch.current = document.createElement("canvas");
      rasterize(activeMask, measureScratch.current, c.width, c.height);
      m = measureMask(measureScratch.current);
    }
    return m.count ? { count: m.count, bbox: m.bbox, w: c.width, h: c.height } : null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [maskVersion, maskMeasure, activeMask, liveKey]);

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
    generateRef, manageTab, setManageTab,
    tab, frame, postFrame, setPostFrame, showRaw, setShowRaw, canvasImage,
    clearCanvas,
    frameCard, pendingCard, setPendingCard, applyCard, lockSeed, activeSeed,
    history, generateIntoLayer, fillIntoLayer, pushHistory, placeHistory, addHistoryAsLayer, commitFrame,
    removeHistory, clearHistory, loadFile, openDocument,
    progress,
    brushSize, setBrushSize, brushHard, setBrushHard, eraser, setEraser,
    maskTool, setMaskTool, wandTolerance, setWandTolerance,
    shapeKind, setShapeKind, genShape, setGenShape, polygonRef, polyCount, setPolyCount,
    maskRef, heroRef,
    syncMaskOverlayRef, getMaskDataUrl, applyContrastMask, applyFloodFillMask,
    // Canvas entities
    rasterLayers, inpaintMasks, selectedId, selectEntity, activeMask, activeLayer, activeBox,
    maskPixels, hasMask, liveMasks, addStroke, rasterizeMasks,
    setEntityEnabled, setMaskVisible, wakeMask, renameEntity, setLayerOpacity,
    deleteEntity, moveEntity, addLayer, duplicateLayer,
    addMask, clearMask, invertMask, setMaskParam, moveMask, nudgeMask,
    addRasterLayer, useLayerAsMask, addMaskWithStroke,
    undo, canUndo, redo, canRedo, setLivePreview, livePreviewOn, setLivePreviewOn,
    // Assets
    assets, addAssetFiles, deleteAsset, placeAsset, openAsset,
    sampleParams, setSampleParam, mergeSampleParams, maskVersion, bumpMask,
    canvasSize, setCanvasSize, newCanvas, canvasIsBlank,
    tabState, setTabStateKey,
    // The document as of the last edit, updated synchronously. The overlay
    // rebuilds its display inside rasterizeMasks, before React has rendered
    // the new state, so it reads this rather than a prop.
    docRef,
    job, genRunning, genPaused,
    runs, watchedId, watch, trackRuns, cancelRun, queueFolder,
  };

  // Create maps its Change slider onto skip + noise_level and passes them as
  // overrides, so whatever the panel shows for noise level is discarded on that
  // tab. Saying so beats a slider that quietly does nothing.
  const overriddenBy = tab === "create" ? { noise_level: "The Change slider in Create" } : null;
  const sweptBy = tab === "sweep" ? sweptParams : null;
  // While a run is paused, only the live keys stay editable; the rest dim, so
  // what Resume will apply is what stands out. While one is running the whole
  // panel stays open: whatever you change is what the next queued run is made with.
  const editableKeys = genPaused ? LIVE_PARAM_KEYS : null;
  const stepsMin = genPaused ? (job?.detail?.step || 1) : undefined;

  return (
    <PlayCtx.Provider value={value}>
      <div className={`play ${CANVAS_TABS.has(tab) ? "fill" : ""}`.trim()}>
        <PlaySetup
          model={activeModel}
          params={sampleParams}
          setParam={setSampleParam}
          sweptBy={sweptBy}
          overriddenBy={overriddenBy}
          editable={editableKeys}
          stepsMin={stepsMin}
        >
          <div className="play-top">
            <div className="card play-model-picker">
              <ModelPicker />
            </div>
            <div className="play-sample-settings">
              <SampleSettingsPanel
                params={sampleParams}
                setParam={setSampleParam}
                model={activeModel}
                overriddenBy={overriddenBy}
                sweptBy={sweptBy}
                editable={editableKeys}
                running={false}
                stepsMin={stepsMin}
              />
            </div>
          </div>
        </PlaySetup>
        {/* Canvas: the picture with its results strip under it, and everything
            you make and manage with in one column on the right. The tab fills
            the window instead of scrolling, so the picture takes whatever
            height is left and the results never fall below the fold. */}
        <div className={`play-body ${CANVAS_TABS.has(tab) ? "canvas-layout" : "solo"}`}>
          {CANVAS_TABS.has(tab) && (
            <>
              <ToolRail />
              <div
                className={`play-stage canvas-col ${dragOver ? "drop-on" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; if (f) loadFile(f); }}
              >
                <PlayCanvas brushable={!!canvasImage} />
                <ResultsStrip />
              </div>
              <aside className="canvas-inspector" aria-label="Generate, layers and assets">
                <CreatePanel />
                <ManagePanel />
              </aside>
            </>
          )}
          {tab === "bend" && <div className="play-solo"><BendWorkspace stack={bendStack} setStack={setBendStack} /></div>}
          {tab === "merge" && <div className="play-solo"><Merge /></div>}
          {tab === "sweep" && <div className="play-solo"><SweepPanel /></div>}
        </div>
      </div>
    </PlayCtx.Provider>
  );
}

const ZOOM_MIN = 0.25;
const ZOOM_MAX = 8;

function PlayCanvas({ brushable }) {
  const { toast, modelPath } = useApp();
  const {
    postFrame, showRaw, setShowRaw, progress, heroRef, canvasImage, syncMaskOverlayRef, tab,
    livePreviewOn, setLivePreviewOn,
    frameCard, pendingCard, setPendingCard, applyCard, activeSeed,
    clearCanvas, canvasSize, setCanvasSize, newCanvas, loadFile,
    activeMask, maskPixels, hasMask, liveMasks, activeLayer,
    undo, canUndo, redo, canRedo, canvasIsBlank, runs, generateRef, setManageTab,
  } = usePlay();
  const shown = canvasImage;
  const [busy, setBusy] = useState(false);
  const openRef = useRef(null);
  const shownCard = frameCard;


  // Zoom and pan. A CSS transform on the wrapper around the picture and its
  // mask overlay: the overlay's pointer maths reads client rects, which
  // already include the transform, so painting needs no changes to follow.
  const viewRef = useRef(null);
  const [view, setView] = useState({ s: 1, x: 0, y: 0 });
  const pan = useRef(null);
  const [panning, setPanning] = useState(false);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const spaceRef = useRef(false);

  // A new document is a new picture; start it at 100%.
  useEffect(() => { setView({ s: 1, x: 0, y: 0 }); }, [canvasSize.w, canvasSize.h]);
  // The hatch period is fixed in screen pixels, so a zoom changes it on the
  // canvas; redraw the overlay once the transform has settled.
  useEffect(() => { syncMaskOverlayRef.current(); }, [view.s, syncMaskOverlayRef]);

  /** Zoom by `k` about a point given in the wrapper's untransformed space. */
  const zoomAt = useCallback((k, cx, cy) => {
    setView((v) => {
      const s = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, v.s * k));
      const f = s / v.s;
      return { s, x: cx - (cx - v.x) * f, y: cy - (cy - v.y) * f };
    });
  }, []);

  /** Zoom about what is currently at the middle of the picture. */
  const zoomStep = (k) => {
    const w = viewRef.current;
    if (!w) return;
    const v = view;
    zoomAt(k, v.x + (v.s * w.offsetWidth) / 2, v.y + (v.s * w.offsetHeight) / 2);
  };

  // Wheel zooms about the pointer. A native listener, because React's wheel
  // handler is passive and cannot stop the page from scrolling instead.
  useEffect(() => {
    const hero = heroRef.current;
    if (!hero) return undefined;
    const onWheel = (e) => {
      const w = viewRef.current;
      if (!w) return;
      e.preventDefault();
      const r = hero.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left - w.offsetLeft, e.clientY - r.top - w.offsetTop);
    };
    hero.addEventListener("wheel", onWheel, { passive: false });
    return () => hero.removeEventListener("wheel", onWheel);
  }, [heroRef, zoomAt, shown]);

  // Space held turns a drag anywhere on the picture into a pan.
  useEffect(() => {
    if (tab !== "create") return undefined;
    const isTyping = (e) => {
      const tag = e.target?.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target?.isContentEditable;
    };
    const down = (e) => {
      if (e.code !== "Space" || isTyping(e) || e.repeat) return;
      e.preventDefault();
      spaceRef.current = true;
      setSpaceHeld(true);
    };
    const up = (e) => {
      if (e.code !== "Space") return;
      spaceRef.current = false;
      setSpaceHeld(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [tab]);

  /** Pan on the middle button, with Space held, or by dragging the empty
   *  ground around the picture. Runs in the capture phase so the overlay
   *  never sees the press as a stroke. */
  const onHeroPointerDown = (e) => {
    if (e.target.closest?.(".hero-chips, .hero-blank-card")) return;
    const onGround = e.target === heroRef.current || e.target === viewRef.current;
    if (!(e.button === 1 || spaceRef.current || onGround)) return;
    e.preventDefault();
    e.stopPropagation();
    try { heroRef.current.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
    pan.current = { px: e.clientX, py: e.clientY, x: view.x, y: view.y };
    setPanning(true);
  };
  const onHeroPointerMove = (e) => {
    const p = pan.current;
    if (!p) return;
    setView((v) => ({ ...v, x: p.x + (e.clientX - p.px), y: p.y + (e.clientY - p.py) }));
  };
  const endPan = () => { pan.current = null; setPanning(false); };

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

  // The picture's size on screen at 100% of the view, as a share of its real
  // pixels. The zoom read-out is in real pixels, so "100%" means one image
  // pixel per screen pixel, and Fit says how far the stage has shrunk it.
  const imgRef = useRef(null);
  const [fit, setFit] = useState(1);
  const [natural, setNatural] = useState(null);
  const measureFit = useCallback(() => {
    const im = imgRef.current;
    if (!im || !im.naturalWidth) return;
    setFit(im.offsetWidth / im.naturalWidth);
    setNatural({ w: im.naturalWidth, h: im.naturalHeight });
  }, []);
  useEffect(() => {
    const hero = heroRef.current;
    if (!hero || typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(measureFit);
    ro.observe(hero);
    return () => ro.disconnect();
  }, [heroRef, measureFit]);
  const atFit = view.s === 1 && view.x === 0 && view.y === 0;
  const pct = Math.round(view.s * fit * 100);
  const zoomText = atFit ? `Fit · ${pct}%` : `${pct}%`;
  const zoomTo = (p) => zoomStep((p / 100 / fit) / view.s);
  const size = natural || canvasSize;

  return (
    <div className="canvas-card canvas-wrap">
      {/* One bar over the picture: the document on the left, the active tool's
          settings in the middle, and what acts on the whole picture on the
          right -- undo, zoom, and getting it out. */}
      <div className="stage-bar" role="toolbar" aria-label="Canvas">
        <Popover
          label="Canvas"
          triggerClass="btn ghost sm doc-menu-btn"
          align="start"
          triggerLabel={`Canvas, ${size.w} × ${size.h}: new, open, clear`}
          trigger={<><span className="tnum">{size.w} × {size.h}</span><ChevronDownIcon /></>}
        >
          {(close) => (
            <div className="bar-menu doc-menu">
              <div className="bar-menu-title">New canvas</div>
              <div className="row center gap-2">
                <span className="canvas-size-pair">
                  <input
                    type="number" className="canvas-size-input" aria-label="Canvas width"
                    value={canvasSize.w} min={CANVAS_MIN} max={CANVAS_MAX} step={64}
                    onChange={(e) => setCanvasSize({ w: e.target.value })}
                  />
                  <span className="sub">×</span>
                  <input
                    type="number" className="canvas-size-input" aria-label="Canvas height"
                    value={canvasSize.h} min={CANVAS_MIN} max={CANVAS_MAX} step={64}
                    onChange={(e) => setCanvasSize({ h: e.target.value })}
                  />
                </span>
                <button type="button" className="btn sm primary" onClick={() => { newCanvas(canvasSize.w, canvasSize.h); close(); }}>
                  New
                </button>
              </div>
              <p className="sub mb-0">A blank canvas at this size. What is on it now stays in Results.</p>
              <span className="bar-menu-sep" />
              <button type="button" className="bar-menu-item" onClick={() => { openRef.current?.click(); close(); }}>
                <span className="grow">Open an image…</span><span className="sub">or drop, or paste</span>
              </button>
              {shown && (
                <button type="button" className="bar-menu-item danger" onClick={() => { clearCanvas(); close(); }}
                  title="Back to blank at the current size, dropping every layer and mask. The image itself stays in Results.">
                  <TrashIcon /> <span className="grow">Clear canvas</span>
                </button>
              )}
            </div>
          )}
        </Popover>
        {/* Outside the menu, which unmounts on close: the file picker outlives it. */}
        <input
          ref={openRef}
          type="file"
          accept="image/*"
          className="hidden-file"
          onChange={(e) => { loadFile(e.target.files?.[0]); e.target.value = ""; }}
        />
        <span className="bar-sep" aria-hidden="true" />
        <ToolOptions />
        <div className="stage-bar-end">
          <Tooltip text="Step back one edit: a fill, a stroke, a delete, a reorder (Ctrl+Z)">
            <button type="button" className="btn ghost sm icon" onClick={undo} disabled={!canUndo} aria-label="Undo">
              <UndoIcon />
            </button>
          </Tooltip>
          <Tooltip text="Put back what Undo took away (Ctrl+Shift+Z)">
            <button type="button" className="btn ghost sm icon" onClick={redo} disabled={!canRedo} aria-label="Redo">
              <RedoIcon />
            </button>
          </Tooltip>
          {shown && (
            <>
              <span className="bar-sep" aria-hidden="true" />
              <div className="zoom-group" role="group" aria-label="Zoom">
                <button type="button" className="btn ghost sm icon" onClick={() => zoomStep(1 / 1.25)} aria-label="Zoom out" title="Zoom out">−</button>
                <Popover label="Zoom" triggerClass="btn ghost sm zoom-btn tnum" trigger={zoomText}
                  triggerLabel={`Zoom: ${zoomText}. The wheel over the picture zooms too; drag the ground, the middle button, or Space to pan.`}>
                  {(close) => (
                    <div className="bar-menu">
                      <button type="button" className={`bar-menu-item ${atFit ? "on" : ""}`}
                        onClick={() => { setView({ s: 1, x: 0, y: 0 }); close(); }}>
                        <span className="grow">Fit</span><span className="sub tnum">{Math.round(fit * 100)}%</span>
                      </button>
                      {[50, 100, 200, 400].map((p) => (
                        <button key={p} type="button" className={`bar-menu-item ${!atFit && pct === p ? "on" : ""}`}
                          onClick={() => { zoomTo(p); close(); }}>
                          <span className="grow tnum">{p}%</span>
                          {p === 100 && <span className="sub">actual pixels</span>}
                        </button>
                      ))}
                    </div>
                  )}
                </Popover>
                <button type="button" className="btn ghost sm icon" onClick={() => zoomStep(1.25)} aria-label="Zoom in" title="Zoom in">+</button>
              </div>
              {/* Nothing to keep from a blank canvas. */}
              {!canvasIsBlank && (
                <>
                  <span className="bar-sep" aria-hidden="true" />
                  <CaptureButton image={shown} card={frameCard} label={<><CaptureIcon /> Capture</>} />
                  <Tooltip text="Download the canvas as a PNG. The settings that made it are written into the file, so dropping it back into Kiln restores them.">
                    <button type="button" className="btn sm icon" onClick={download} disabled={busy} aria-label="Download">
                      {busy ? "…" : <DownloadIcon />}
                    </button>
                  </Tooltip>
                </>
              )}
            </>
          )}
        </div>
      </div>
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
      <div
        className={`hero ${brushable ? "brushable" : ""} ${spaceHeld ? "pan-ready" : ""} ${panning ? "panning" : ""}`}
        ref={heroRef}
        onPointerDownCapture={onHeroPointerDown}
        onPointerMove={onHeroPointerMove}
        onPointerUp={endPan}
        onPointerCancel={endPan}
      >
        {progress && <div className="hero-progress"><Progress value={progress.value} /></div>}
        {shown ? (
          <div
            className="hero-view"
            ref={viewRef}
            style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}
          >
            <img ref={imgRef} src={shown} alt="canvas" draggable={false} onLoad={measureFit} />
            <MaskOverlay active={brushable} />
          </div>
        ) : (
          <span className="sub">Generate, drop, or paste an image.</span>
        )}
        {/* A blank canvas says what it is and what to do with it, rather than
            showing an empty checkerboard. Gone as soon as there is anything:
            a run, a mask, a picture. */}
        {canvasIsBlank && !progress && !hasMask && !runs.length && (
          <div className="hero-blank">
            <div className="hero-blank-card">
              <b>A blank {size.w} × {size.h} canvas</b>
              <p className="sub mb-0">
                Generate a first image, or bring one in to rework. Drop or paste an image anywhere on this page.
              </p>
              <div className="row center gap-2">
                <button type="button" className="btn primary" onClick={() => generateRef.current?.()}
                  disabled={!modelPath}>
                  Generate
                </button>
                <button type="button" className="btn" onClick={() => openRef.current?.click()}>
                  Open an image…
                </button>
              </div>
              <button type="button" className="btn ghost sm" onClick={() => setManageTab("assets")}>
                or pick one from Assets →
              </button>
            </div>
          </div>
        )}
        {/* What the picture is, and how it is being shown, on the picture
            itself rather than in a header above it. */}
        <div className="hero-chips start">
          {progress ? (
            <span className="hero-chip">{progress.message}</span>
          ) : activeSeed != null ? (
            <span className="hero-chip mono" title="Seed that produced this image">
              seed {activeSeed}{activeLayer ? ` · ${activeLayer.name}` : ""}
            </span>
          ) : null}
        </div>
        <div className="hero-chips end">
          <Tooltip text="Show the picture while it is being made. The preview is the raw image; Finish is applied to the result only. Turn it off to keep the current picture on the canvas until the result lands, which saves a little time per run.">
            <label className="hero-chip hero-switch">
              <input type="checkbox" checked={livePreviewOn} onChange={(e) => setLivePreviewOn(e.target.checked)} />
              Live preview
            </label>
          </Tooltip>
          {postFrame && (
            <Tooltip text="Show the layer stack as it is, before post-processing. Useful for judging what the models actually produced.">
              <label className="hero-chip hero-switch">
                <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} />
                Unprocessed
              </label>
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

/** Results and the queue, in one strip under the picture.
 *
 *  Fixed height and always on screen: as a grid that grew under the canvas,
 *  results sat below the fold at 900px, so every Generate ended in a scroll.
 *  Runs in flight come first (a progress ring, or a place in line), then every
 *  image this session made, newest first. The strip scrolls sideways; the page
 *  does not.
 */
function ResultsStrip() {
  const {
    history, frame, placeHistory, removeHistory, clearHistory,
    addHistoryAsLayer, applyCard, activeLayer,
    runs, watchedId, watch, cancelRun, queueFolder,
  } = usePlay();
  const live = runs.filter((r) => r.job.status === "running").length;
  const waiting = runs.length - live;
  // Where they land, as the landing rule decided: a lone run still goes to its
  // layer, and a queue (or a fill, which is never queued) says so too.
  const toResults = runs.filter((r) => r.toResults).length;
  const lands = toResults === runs.length ? `Lands here, saved to captures/${queueFolder}`
    : toResults === 0 ? (runs.some((r) => r.fill) ? "Lands over the mask" : "Lands in the layer")
      : "Fill lands over the mask, the rest here";

  return (
    <section className="card results-strip" aria-label="Results">
      <div className="results-strip-head">
        <h3 className="mb-0">Results <span className="sub">· {history.length}</span></h3>
        {runs.length > 0 ? (
          <span className="sub">
            {live} running{waiting ? `, ${waiting} waiting` : ""}. {lands}.
          </span>
        ) : history.length > 0 ? (
          <span className="sub">Click one to put it into {activeLayer?.name || "the active layer"}.</span>
        ) : null}
        {history.length > 0 && (
          <button type="button" className="btn ghost xs results-strip-clear" onClick={clearHistory}>Clear</button>
        )}
      </div>
      <div className="results-strip-row">
        {runs.map((r) => (
          <QueueTile
            key={r.id}
            run={r}
            watched={r.id === watchedId}
            onWatch={watch}
            onCancel={() => cancelRun(r.id)}
          />
        ))}
        {history.map((h, i) => {
          const seed = h.card?.params?.seed;
          return (
            <div key={h.id} className={`strip-tile ${h.img === frame ? "on" : ""}`}>
              <button
                type="button"
                className="strip-thumb"
                onClick={() => placeHistory(h)}
                aria-label={`Result ${i + 1}${seed != null ? `, seed ${seed}` : ""}`}
                title={`${cardLabel(h.card) || `Result ${i + 1}`} — click to put it into ${activeLayer?.name || "the active layer"}`}
              >
                <img src={h.img} alt="" />
                <span className="play-result-idx">#{i + 1}</span>
              </button>
              <span className="strip-cap mono">{seed != null ? seed : "—"}</span>
              <div className="strip-actions">
                <Tooltip text="Add on top of the layer stack, keeping what is there">
                  <button type="button" className="btn xs" onClick={() => addHistoryAsLayer(h)}>Layer</button>
                </Tooltip>
                <CaptureButton image={h.img} card={h.card} label="Save" className="btn xs" />
                <Tooltip text={h.card?.params ? "Load this image's settings into the sampler" : "No settings recorded for this image"}>
                  <button type="button" className="btn xs" onClick={() => applyCard(h.card)} disabled={!h.card?.params}>
                    Set
                  </button>
                </Tooltip>
              </div>
              <button
                type="button"
                className="strip-x"
                onClick={() => removeHistory(h.id)}
                aria-label={`Remove result ${i + 1}`}
                title="Remove from results"
              >×</button>
            </div>
          );
        })}
        {!runs.length && !history.length && (
          <p className="sub results-strip-empty">
            Everything you generate lands here too, so nothing is lost when the canvas changes.
          </p>
        )}
      </div>
    </section>
  );
}

function seedsOf(job) {
  const d = job.detail || {};
  return d.seeds || (d.seed != null ? [d.seed] : []);
}

/** A run in flight. Clicking it watches it (its preview goes on the canvas);
 *  the x cancels it, and what it had is dropped. */
function QueueTile({ run, watched, onWatch, onCancel }) {
  const j = run.job;
  const running = j.status === "running";
  const paused = running && !!j.detail?.paused;
  const model = j.detail?.card?.model || "model";
  const seeds = seedsOf(j);
  const seedText = seeds.length
    ? (seeds.length > 1 ? `seeds ${Math.min(...seeds)}–${Math.max(...seeds)}` : `seed ${seeds[0]}`)
    : "";
  const place = j.detail?.queue?.position;
  const state = running
    ? (paused ? "paused" : `${Math.round((j.progress || 0) * 100)}%`)
    : "waiting";
  const label = [j.kind === "inpaint" ? "fill" : null, model, seedText].filter(Boolean).join(" · ");
  return (
    <div className={`strip-tile queue-tile ${watched ? "on" : ""} ${running ? "running" : "waiting"}`}>
      <button
        type="button"
        className="strip-thumb"
        onClick={() => onWatch(run.id)}
        aria-label={`${label}, ${state}${watched ? ", watching" : ""}`}
        title={watched ? `Watching: ${label}` : `Watch this run: its preview goes on the canvas. ${label}`}
      >
        {running
          ? <span className="queue-ring big" style={{ "--p": j.progress || 0 }} aria-hidden="true" />
          : <span className="queue-place mono" aria-hidden="true">{place ? `#${place}` : "…"}</span>}
        <span className="queue-tile-state mono">{state}</span>
      </button>
      <span className="strip-cap" title={label}>{j.kind === "inpaint" ? "fill" : seedText || "run"}</span>
      <button type="button" className="strip-x" onClick={onCancel} aria-label="Cancel this run" title="Cancel this run">×</button>
    </div>
  );
}

/** A canvas kept on a ref, made on first use. */
function scratchOf(ref) {
  if (!ref.current) ref.current = document.createElement("canvas");
  return ref.current;
}

function copyCanvas(src, dstRef) {
  const dst = scratchOf(dstRef);
  dst.width = src.width;
  dst.height = src.height;
  dst.getContext("2d").drawImage(src, 0, 0);
  return dst;
}

/** The mask on the canvas.
 *
 *  Two canvases over the picture. `maskRef` is the data store: the union of
 *  every mask that is on, whose alpha is the per-pixel strength the backend,
 *  the fill punch and the pixel count all read. The brush paints onto it
 *  directly. It is invisible.
 *
 *  `displayRef` shows the masks that are *shown* -- a different switch. Two
 *  rasters feed it: the shown masks that are on, hatched in full, and the
 *  shown masks that are off, hatched faint. Each is the raster drawn through
 *  a hatch tile with source-in, so the hatch fades with mask strength.
 *  Keeping store and display apart is what lets the display follow its own
 *  switch without touching the numbers the run depends on.
 *
 *  Around the active mask sits a bounding box with a label that doubles as a
 *  grip; with the Move tool the whole overlay drags. A drag translates cached
 *  rasters of the active mask and the rest, and commits one `move` stroke on
 *  release -- replaying every stroke per pointermove would not keep up.
 */
function MaskOverlay({ active }) {
  const {
    maskRef, heroRef, brushSize, brushHard, eraser, canvasImage, frame,
    syncMaskOverlayRef, maskTool, applyFloodFillMask, wandTolerance,
    addStroke, rasterizeMasks, activeMask, activeBox, moveMask, wakeMask, docRef,
    shapeKind, polygonRef, setPolyCount,
  } = usePlay();
  const drawing = useRef(false);
  // A rectangle or ellipse being dragged, and the corners of a polygon being
  // placed (canvas pixels). Both preview over snapshots of the rasters and
  // commit one shape stroke at the end.
  const shaping = useRef(null);
  const poly = useRef([]);
  const snapShape = useRef(null);
  const snapShapeVis = useRef(null);
  const last = useRef(null);
  const wandBusy = useRef(false);
  // Points of the stroke in progress, normalised, recorded as they are painted.
  const points = useRef([]);
  const radiusRef = useRef(0);
  // The masks as of the last edit -- not the rendered prop, which is one
  // render behind at the moment the display is rebuilt.
  const masksNow = () => docRef.current.inpaintMasks;
  const displayRef = useRef(null);
  const stageRef = useRef(null);
  const boxRef = useRef(null);
  const hatch = useRef({ scale: 0, pattern: null });
  // What the display is made of: the masks that are on and shown. The live
  // brush paints into it alongside the store.
  const visOnRef = useRef(null);
  const hatchScratch = useRef(null);
  // A drag in progress: where it started, and the rasters it slides around.
  const moving = useRef(null);
  const snapOwn = useRef(null);
  const snapStore = useRef(null);
  const snapVisOn = useRef(null);

  const hatchFor = (ctx, scale) => {
    if (!hatch.current.pattern || Math.abs(hatch.current.scale - scale) > 0.01) {
      hatch.current = { scale, pattern: ctx.createPattern(hatchTile(scale), "repeat") };
    }
    return hatch.current.pattern;
  };

  /** Rebuild the display raster from the mask list: masks that are on and
   *  shown. An off mask is not drawn whatever its eye says -- off means out
   *  of the way -- and the eye keeps its state for when it is on again.
   *  `excludeId` leaves one mask out, for a drag that draws it separately. */
  const drawVisible = (excludeId = null) => {
    const c = maskRef.current;
    if (!c?.width) return;
    const shown = masksNow().filter((m) => m.enabled && maskShown(m) && m.id !== excludeId);
    compositeMasks(shown, scratchOf(visOnRef), c.width, c.height);
  };

  /** `src` as hatching, into `scratch`. */
  const hatchInto = (src, scratch, scale) => {
    if (scratch.width !== src.width || scratch.height !== src.height) {
      scratch.width = src.width;
      scratch.height = src.height;
    }
    const ctx = scratch.getContext("2d");
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, scratch.width, scratch.height);
    ctx.drawImage(src, 0, 0);
    ctx.globalCompositeOperation = "source-in";
    ctx.fillStyle = hatchFor(ctx, scale);
    ctx.fillRect(0, 0, scratch.width, scratch.height);
    ctx.globalCompositeOperation = "source-over";
  };

  /** Redraw the display: the hatch, where the shown masks are. */
  const paintDisplay = () => {
    const c = maskRef.current;
    const d = displayRef.current;
    if (!c || !d || !c.width || !c.height || !visOnRef.current) return;
    if (d.width !== c.width || d.height !== c.height) {
      d.width = c.width;
      d.height = c.height;
    }
    const r = c.getBoundingClientRect();
    const scale = c.width / Math.max(r.width, 1);
    const scratch = scratchOf(hatchScratch);
    hatchInto(visOnRef.current, scratch, scale);
    const ctx = d.getContext("2d");
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, d.width, d.height);
    ctx.drawImage(scratch, 0, 0);
  };

  const syncSize = () => {
    const c = maskRef.current;
    const stage = stageRef.current;
    const hero = heroRef.current;
    const img = hero?.querySelector("img");
    if (!c || !stage || !img) return;
    // The stage takes the picture's layout box inside the view wrapper; both
    // canvases and the box fill it, so the box can be placed in percentages
    // and never needs re-laying. Offsets, not client rects: the wrapper is
    // transformed for zoom and pan, and the stage is transformed with it.
    stage.style.left = `${img.offsetLeft}px`;
    stage.style.top = `${img.offsetTop}px`;
    stage.style.width = `${img.offsetWidth}px`;
    stage.style.height = `${img.offsetHeight}px`;
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
      rasterizeMasks(masksNow(), { w, h });
      return;
    }
    drawVisible();
    paintDisplay();
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
    const radius = (brushSize / 2) * pt.scale;
    radiusRef.current = radius;
    const opts = { hard: brushHard, erase: eraser };
    // Into the store, and into the display's "shown and on" raster: the mask
    // being painted is both, because pointerdown woke it.
    for (const target of [c, visOnRef.current]) {
      if (!target) continue;
      const ctx = target.getContext("2d");
      paintBrushPoint(ctx, last.current, pt, radius, opts);
      ctx.globalCompositeOperation = "source-over";
    }
    last.current = { x: pt.x, y: pt.y };
    points.current.push({ x: pt.x / c.width, y: pt.y / c.height });
    paintDisplay();
  };

  /** Start dragging the active mask: snapshot it, the store without it, and
   *  the display rasters without it. */
  const beginMove = (e) => {
    const c = maskRef.current;
    if (!active || !c?.width || !activeMask?.enabled || !maskShown(activeMask) || !activeMask.strokes.length) return;
    e.stopPropagation();
    e.preventDefault();
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
    rasterize(activeMask, scratchOf(snapOwn), c.width, c.height);
    compositeMasks(
      masksNow().filter((m) => m.enabled && m.id !== activeMask.id),
      scratchOf(snapStore), c.width, c.height,
    );
    drawVisible(activeMask.id);
    copyCanvas(visOnRef.current, snapVisOn);
    const pt = pointAt(e);
    moving.current = { x: pt.x, y: pt.y, scale: pt.scale, dx: 0, dy: 0 };
  };

  /** Slide the snapshots; nothing is committed until release. */
  const moveTo = (e) => {
    const mv = moving.current;
    const c = maskRef.current;
    if (!mv || !c) return;
    const pt = pointAt(e);
    mv.dx = Math.round(pt.x - mv.x);
    mv.dy = Math.round(pt.y - mv.y);
    const redraw = (target, base) => {
      const ctx = target.getContext("2d");
      ctx.globalCompositeOperation = "source-over";
      ctx.clearRect(0, 0, target.width, target.height);
      ctx.drawImage(base, 0, 0);
      ctx.drawImage(snapOwn.current, mv.dx, mv.dy);
    };
    redraw(c, snapStore.current);
    redraw(visOnRef.current, snapVisOn.current);
    paintDisplay();
    if (boxRef.current) {
      boxRef.current.style.transform = `translate(${mv.dx / mv.scale}px, ${mv.dy / mv.scale}px)`;
    }
  };

  /** Commit the drag as one move stroke, one undo entry. */
  const endMove = () => {
    const mv = moving.current;
    const c = maskRef.current;
    moving.current = null;
    if (boxRef.current) boxRef.current.style.transform = "";
    if (!mv || !c?.width) return;
    if (mv.dx || mv.dy) moveMask(activeMask.id, mv.dx / c.width, mv.dy / c.height);
    else rasterizeMasks(masksNow());
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

  // --- shapes --------------------------------------------------------
  const shapeMode = eraser ? "subtract" : "add";
  const norm = (pt) => {
    const c = maskRef.current;
    return { x: pt.x / c.width, y: pt.y / c.height };
  };
  /** Snapshot the store and the display raster, so a shape can be previewed
   *  over them and the preview thrown away between pointer moves. */
  const snapForShape = () => {
    const c = maskRef.current;
    if (!c) return;
    copyCanvas(c, snapShape);
    copyCanvas(visOnRef.current, snapShapeVis);
  };
  /** Draw the snapshots back, then the stroke over them, into store and display. */
  const previewShape = (stroke) => {
    const c = maskRef.current;
    if (!c) return;
    const draw = (target, base) => {
      if (!target || !base) return;
      const ctx = target.getContext("2d");
      ctx.globalCompositeOperation = "source-over";
      ctx.clearRect(0, 0, target.width, target.height);
      ctx.drawImage(base, 0, 0);
      ctx.globalCompositeOperation = stroke.mode === "subtract" ? "destination-out" : "source-over";
      replayShape(ctx, stroke, target.width, target.height);
      ctx.globalCompositeOperation = "source-over";
    };
    draw(c, snapShape.current);
    draw(visOnRef.current, snapShapeVis.current);
    paintDisplay();
  };
  const beginShape = (e) => {
    const c = maskRef.current;
    if (!active || !c?.width) return;
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
    if (activeMask) wakeMask(activeMask.id);
    syncSize();
    snapForShape();
    const pt = pointAt(e);
    shaping.current = { x: pt.x, y: pt.y, x1: null, y1: null };
  };
  const shapeTo = (e) => {
    const sh = shaping.current;
    if (!sh) return;
    const pt = pointAt(e);
    let { x: x1, y: y1 } = pt;
    if (e.shiftKey) {
      // Shift: a square or a circle, growing toward the pointer.
      const d = Math.max(Math.abs(x1 - sh.x), Math.abs(y1 - sh.y));
      x1 = sh.x + Math.sign(x1 - sh.x || 1) * d;
      y1 = sh.y + Math.sign(y1 - sh.y || 1) * d;
    }
    sh.x1 = x1;
    sh.y1 = y1;
    previewShape(shapeStroke({ shape: shapeKind, points: [norm(sh), norm({ x: x1, y: y1 })], mode: shapeMode }));
  };
  const endShape = () => {
    const sh = shaping.current;
    shaping.current = null;
    if (!sh) return;
    if (sh.x1 == null || Math.abs(sh.x1 - sh.x) < 2 || Math.abs(sh.y1 - sh.y) < 2) {
      rasterizeMasks(masksNow());
      return;
    }
    addStroke(shapeStroke({ shape: shapeKind, points: [norm(sh), norm({ x: sh.x1, y: sh.y1 })], mode: shapeMode }));
  };

  /** The polygon so far: its fill once it has three corners, and always its
   *  outline and corners on the display, so the first two clicks show. */
  const drawPolyGuides = (all) => {
    const d = displayRef.current;
    const c = maskRef.current;
    if (!d || !c || !all.length) return;
    const r = c.getBoundingClientRect();
    const scale = c.width / Math.max(r.width, 1);
    const ctx = d.getContext("2d");
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = `rgb(${MASK_RGB.join(",")})`;
    ctx.lineWidth = Math.max(1, 1.5 * scale);
    ctx.setLineDash([6 * scale, 4 * scale]);
    ctx.beginPath();
    ctx.moveTo(all[0].x, all[0].y);
    for (let i = 1; i < all.length; i += 1) ctx.lineTo(all[i].x, all[i].y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "rgba(0,0,0,0.6)";
    ctx.lineWidth = Math.max(1, scale);
    for (const q of all) {
      ctx.beginPath();
      ctx.arc(q.x, q.y, 4 * scale, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  };
  const polyPreview = (cursor) => {
    const all = cursor ? [...poly.current, cursor] : [...poly.current];
    previewShape(shapeStroke({ shape: "polygon", points: all.map(norm), mode: shapeMode }));
    drawPolyGuides(all);
  };
  const polyClose = () => {
    const pts = poly.current;
    poly.current = [];
    setPolyCount(0);
    if (pts.length >= 3) addStroke(shapeStroke({ shape: "polygon", points: pts.map(norm), mode: shapeMode }));
    else rasterizeMasks(masksNow());
  };
  const polyCancel = () => {
    poly.current = [];
    setPolyCount(0);
    rasterizeMasks(masksNow());
  };
  const polyPop = () => {
    poly.current.pop();
    setPolyCount(poly.current.length);
    if (!poly.current.length) rasterizeMasks(masksNow());
    else polyPreview(null);
  };
  const polyAdd = (e) => {
    const c = maskRef.current;
    if (!active || !c?.width) return;
    const pt = pointAt(e);
    const pts = poly.current;
    if (!pts.length) {
      if (activeMask) wakeMask(activeMask.id);
      syncSize();
      snapForShape();
    } else {
      const first = pts[0];
      const lastPt = pts[pts.length - 1];
      // Clicking the first corner closes; a click on top of the last corner
      // (the second half of a double-click) adds nothing.
      if (pts.length >= 3 && Math.hypot(pt.x - first.x, pt.y - first.y) < 10 * pt.scale) { polyClose(); return; }
      if (Math.hypot(pt.x - lastPt.x, pt.y - lastPt.y) < 3 * pt.scale) return;
    }
    pts.push({ x: pt.x, y: pt.y });
    setPolyCount(pts.length);
    polyPreview(null);
  };
  useEffect(() => {
    polygonRef.current = { close: polyClose, cancel: polyCancel, pop: polyPop };
  });
  // Switching tool or kind abandons a polygon in progress.
  useEffect(() => {
    if (poly.current.length && !(maskTool === "shape" && shapeKind === "polygon")) polyCancel();
  }, [maskTool, shapeKind]);

  // The wand samples the stack itself, not what is on screen: with post-process
  // on, the screen is a display stage, and the fill it is selecting for reads
  // the stack. Selecting on one and filling on the other made the region and
  // the fill disagree at every post-processed edge.
  const wandClick = async (e) => {
    if (!active || !frame || wandBusy.current) return;
    const c = maskRef.current;
    if (!c) return;
    const pt = pointAt(e);
    wandBusy.current = true;
    try {
      await applyFloodFillMask(frame, pt.x, pt.y, { tolerance: wandTolerance, eraser });
    } finally {
      wandBusy.current = false;
    }
  };

  const pct = (v, of) => `${(v / of) * 100}%`;

  return (
    <div className="mask-stage" ref={stageRef}>
      <canvas className="mask-display" ref={displayRef} aria-hidden="true" />
      <canvas
        className={`mask-overlay ${active ? "on" : ""} ${maskTool === "wand" ? "wand" : ""} ${maskTool === "move" ? "move" : ""} ${maskTool === "shape" && shapeKind !== "pattern" ? "shape" : ""}`}
        ref={maskRef}
        role="img"
        aria-label={maskTool === "wand"
          ? "Inpaint mask — click to select a matching area"
          : maskTool === "move"
            ? "Inpaint mask — drag to move"
            : maskTool === "shape"
              ? (shapeKind === "polygon" ? "Inpaint mask — click to place corners" : "Inpaint mask — drag a shape")
              : "Inpaint mask — drag to paint"}
        onPointerDown={(e) => {
          if (maskTool === "wand") {
            wandClick(e);
            return;
          }
          if (maskTool === "move") {
            beginMove(e);
            return;
          }
          if (maskTool === "shape") {
            if (shapeKind === "polygon") polyAdd(e);
            else if (shapeKind !== "pattern") beginShape(e);
            return;
          }
          // Throws if the pointer id is not an active pointer, which is the case
          // for synthetic events. Capture is a nicety -- losing it should not
          // cost the stroke.
          try { e.target.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
          if (activeMask) wakeMask(activeMask.id);
          syncSize();
          drawing.current = true;
          last.current = null;
          points.current = [];
          paint(e);
        }}
        onPointerMove={(e) => {
          if (moving.current) moveTo(e);
          else if (shaping.current) shapeTo(e);
          else if (maskTool === "shape" && shapeKind === "polygon" && poly.current.length) polyPreview(pointAt(e));
          else if (maskTool === "brush" && drawing.current) paint(e);
        }}
        onPointerUp={() => (moving.current ? endMove() : shaping.current ? endShape() : endStroke())}
        onPointerCancel={() => (moving.current ? endMove() : shaping.current ? endShape() : endStroke())}
        onDoubleClick={() => { if (maskTool === "shape" && shapeKind === "polygon") polyClose(); }}
      />
      {activeBox && (
        <div
          className="mask-bbox"
          ref={boxRef}
          style={{
            left: pct(activeBox.bbox.x0, activeBox.w),
            top: pct(activeBox.bbox.y0, activeBox.h),
            width: pct(activeBox.bbox.x1 - activeBox.bbox.x0, activeBox.w),
            height: pct(activeBox.bbox.y1 - activeBox.bbox.y0, activeBox.h),
          }}
        >
          {/* The label is the grip: draggable with any tool, while the box
              itself lets the pointer through so painting inside it still works. */}
          <span
            className="mask-bbox-label"
            title="Drag to move the mask"
            onPointerDown={beginMove}
            onPointerMove={moveTo}
            onPointerUp={endMove}
            onPointerCancel={endMove}
          >
            {activeMask?.name}
          </span>
        </div>
      )}
    </div>
  );
}

/** An eye, open or crossed. Visibility used to be a filled/hollow dot, which
 *  read as a radio button -- as if only one row could be on at a time. */
/** One row in the Layers panel: one line, and its controls only when selected.
 *
 *  Three independent states, each with its own cue so none reads as another:
 *
 *    shown / hidden   the eye, and a dimmed row when hidden -- any number of
 *                     rows can be on, it is not a choice between them
 *    selected         the highlighted row; what the details unfold under, and
 *                     for a mask, what the brush paints into
 *    active (layers)  the ACTIVE tag; where a run lands. Follows selection of
 *                     a layer row and stays when a mask row is selected next
 *
 *  Every row used to show all of its controls all the time, which made three
 *  layers enough to push the rest of the panel below the fold.
 */
/** A mask's shape at row size: its strokes replayed small, blue on a checker.
 *  Redraws when the strokes change, which every edit makes them do. */
function MaskThumb({ mask, w, h, version }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ar = w && h ? w / h : 1;
    const rw = ar >= 1 ? 96 : Math.max(8, Math.round(96 * ar));
    const rh = ar >= 1 ? Math.max(8, Math.round(96 / ar)) : 96;
    const scratch = document.createElement("canvas");
    rasterize(mask, scratch, rw, rh);
    const size = 26;
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, size);
    const fit = Math.min(size / rw, size / rh);
    const dw = rw * fit;
    const dh = rh * fit;
    ctx.drawImage(scratch, (size - dw) / 2, (size - dh) / 2, dw, dh);
  }, [mask.strokes, version, w, h]);
  return <canvas ref={ref} className="layer-thumb mask-thumb" aria-hidden="true" />;
}

function EntityRow({
  entity, thumb, details, actions, headExtra, active = false, canDelete = true, shown, onShow, eyeTip,
}) {
  const {
    selectedId, selectEntity, setEntityEnabled, renameEntity, deleteEntity, moveEntity,
  } = usePlay();
  const [renaming, setRenaming] = useState(false);
  const on = selectedId === entity.id;
  // The eye is shown/hidden. For a layer that is its one switch (`enabled`);
  // a mask has a separate one (`visible`) and the caller passes it in.
  const isShown = shown ?? entity.enabled;
  const toggleShown = onShow ?? ((v) => setEntityEnabled(entity.id, v));

  return (
    <div
      className={`layer-row ${on ? "on" : ""} ${isShown ? "" : "off"} ${active ? "active" : ""}`}
      role="button"
      tabIndex={0}
      aria-pressed={on}
      title={active ? `${entity.name} — runs land here` : undefined}
      onClick={() => selectEntity(entity.id)}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectEntity(entity.id); } }}
    >
      <div className="layer-head">
        <Tooltip text={eyeTip || (isShown ? "Hide — kept, but not part of the canvas" : "Show")}>
          <button
            type="button"
            className={`sel-eye ${isShown ? "on" : ""}`}
            aria-label={isShown ? `Hide ${entity.name}` : `Show ${entity.name}`}
            aria-pressed={isShown}
            onClick={(e) => { e.stopPropagation(); toggleShown(!isShown); }}
          >
            <EyeIcon off={!isShown} />
          </button>
        </Tooltip>
        {thumb}
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
            title="Double-click to rename"
            onClick={(e) => { e.stopPropagation(); selectEntity(entity.id); }}
            onDoubleClick={(e) => { e.stopPropagation(); setRenaming(true); }}
          >
            {entity.name}
          </button>
        )}
        {headExtra}
        {active && <span className="layer-active-tag" aria-label="Active layer">active</span>}
      </div>
      {on && (
        <div className="layer-details">
          {details}
          {/* One row of icon buttons: the row's own actions, then order, then
              delete. Each says what it is on hover and to a reader. */}
          <div className="layer-actions">
            {actions}
            <Tooltip text="Move up">
              <button type="button" className="btn xs ghost icon" aria-label="Move up"
                onClick={(e) => { e.stopPropagation(); moveEntity(entity.id, 1); }}><UpIcon /></button>
            </Tooltip>
            <Tooltip text="Move down">
              <button type="button" className="btn xs ghost icon" aria-label="Move down"
                onClick={(e) => { e.stopPropagation(); moveEntity(entity.id, -1); }}><DownIcon /></button>
            </Tooltip>
            <div className="spacer" />
            <Tooltip text={canDelete ? "Delete" : "The last layer stays — there is always one to work in"}>
              <button type="button" className="btn xs ghost danger icon" aria-label="Delete"
                disabled={!canDelete}
                onClick={(e) => { e.stopPropagation(); deleteEntity(entity.id); }}><TrashIcon /></button>
            </Tooltip>
          </div>
        </div>
      )}
    </div>
  );
}

/** What the canvas is made of, and the images kept at hand: Layers | Assets.
 *
 *  One panel with two tabs, where there were three cards (Assets, Layers,
 *  Inpaint Masks) that took turns pushing each other below the fold. Files
 *  dropped anywhere on it are kept as assets; the stage underneath would
 *  otherwise open them as the canvas.
 */
function ManagePanel() {
  const { rasterLayers, assets, addAssetFiles, manageTab: tab, setManageTab: setTab } = usePlay();
  const [over, setOver] = useState(false);
  const tabs = [
    { id: "layers", label: <>Layers <span className="sub">· {rasterLayers.length}</span></> },
    { id: "assets", label: <>Assets <span className="sub">· {assets.length}</span></> },
  ];

  return (
    <section
      className={`card inspector-panel manage-panel ${over ? "drop-on" : ""}`.trim()}
      aria-label="Layers and assets"
      onDragOver={(e) => {
        if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
        e.preventDefault(); e.stopPropagation(); setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        if (!e.dataTransfer?.files?.length) return;
        e.preventDefault();
        e.stopPropagation();
        setOver(false);
        addAssetFiles(e.dataTransfer.files);
        setTab("assets");
      }}
    >
      <div className="panel-tabs" role="tablist" aria-label="Layers and assets">
        {tabs.map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={tab === t.id}
            className={tab === t.id ? "on" : ""} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="panel-body" role="tabpanel">
        {tab === "layers" ? <LayersList /> : <AssetsGrid />}
      </div>
    </section>
  );
}

/** Images the user keeps at hand: the workspace assets folder.
 *
 *  Anything opened onto the canvas lands here too, so a picture only ever has
 *  to be found in the file browser once. Click places into the active layer,
 *  fitted to the canvas; Open makes it the document at its own size.
 */
function AssetsGrid() {
  const { assets, addAssetFiles, deleteAsset, placeAsset, openAsset, activeLayer } = usePlay();
  const fileRef = useRef(null);
  return (
    <>
      <div className="assets-grid">
        {assets.map((a) => (
          <div key={a.path} className="asset">
            <button
              type="button"
              className="asset-thumb"
              title={`${a.name}${a.size ? ` · ${a.size[0]}×${a.size[1]}` : ""} — click to place into ${activeLayer?.name || "the active layer"}`}
              onClick={() => placeAsset(a)}
            >
              <img src={thumbUrl(a.path)} alt={a.name} loading="lazy" />
            </button>
            <div className="asset-actions">
              <button type="button" className="btn xs ghost" title="Open as a new canvas at its own size"
                onClick={() => openAsset(a)}>Open</button>
              <button type="button" className="btn xs ghost" title="Remove from assets" aria-label={`Remove ${a.name}`}
                onClick={() => deleteAsset(a)}>×</button>
            </div>
          </div>
        ))}
        <button type="button" className="asset-add" onClick={() => fileRef.current?.click()}
          title="Keep images here to place on a layer later, without finding the file again">
          <AddIcon size={16} />
          Add…
        </button>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden-file"
        onChange={(e) => { addAssetFiles(e.target.files); e.target.value = ""; }}
      />
      <p className="hint mb-0">
        Drop images on this panel to keep them. Click one to place it in {activeLayer?.name || "the active layer"};
        Open makes it the canvas.
      </p>
    </>
  );
}

/** Canvas entities, in the two groups InvokeAI splits them into, in one list.
 *
 *  Masks come first because they decide what the next run does, so they have
 *  to stay in view. Raster layers are the picture: hiding, reordering or
 *  deleting one changes what the canvas is, and therefore what the next run
 *  starts from — and one of them is active, which is where the run lands.
 *  Layers are made by the user, never by a run. A mask is live for exactly as
 *  long as it is used — which is the answer to "should the mask survive a
 *  generation": it survives because nothing turned it off.
 */
function LayersList() {
  const {
    rasterLayers, inpaintMasks, setLayerOpacity, useLayerAsMask, activeLayer,
    addLayer, duplicateLayer, setEntityEnabled, setMaskVisible,
    addMask, clearMask, invertMask, liveMasks, canvasSize, maskVersion,
  } = usePlay();

  return (
    <div className="manage-list">
      <div className="manage-group-head">
        <span>Masks <span className="sub">· {liveMasks.length ? `${liveMasks.length} used` : "none used"}</span></span>
        <Tooltip text="A new empty mask, picked so strokes go into it. Every mask that is used counts for the next run.">
          <button type="button" className="btn ghost xs" onClick={addMask}><AddIcon /> Mask</button>
        </Tooltip>
      </div>
      {inpaintMasks.length === 0 && (
        <p className="hint mb-0">None yet. Paint one with the Brush (B), or take an area with Wand, Shape or Split.</p>
      )}
      {[...inpaintMasks].reverse().map((m) => (
        <EntityRow
          key={m.id}
          entity={m}
          thumb={<MaskThumb mask={m} w={canvasSize.w} h={canvasSize.h} version={maskVersion} />}
          shown={maskShown(m)}
          onShow={(v) => setMaskVisible(m.id, v)}
          eyeTip={maskShown(m)
            ? "Hide the hatching. Whether the next run uses the mask is Used / Off, not this."
            : "Show the hatching"}
          headExtra={(
            <Tooltip text={!m.enabled
              ? "Off: the next run ignores this area and it is not drawn. Click to use it."
              : m.strokes.length
                ? "Used: the next run changes this area. Click to turn it off; it is then not drawn either."
                : "On, but empty: paint into it and the next run will change that area. Click to turn it off."}
            >
              <button
                type="button"
                className={`layer-onpill ${m.enabled ? "on" : ""}`}
                aria-pressed={m.enabled}
                aria-label={m.enabled ? `Stop using ${m.name}` : `Use ${m.name}`}
                onClick={(e) => { e.stopPropagation(); setEntityEnabled(m.id, !m.enabled); }}
              >
                {!m.enabled ? "off" : m.strokes.length ? "used" : "empty"}
              </button>
            </Tooltip>
          )}
          actions={(
            <>
              <Tooltip text="Swap masked for unmasked (Ctrl+Shift+I)">
                <button type="button" className="btn xs ghost"
                  onClick={(e) => { e.stopPropagation(); invertMask(m.id); }}><InvertIcon /> Invert</button>
              </Tooltip>
              <Tooltip text="Empty it, keeping the row and its settings (Ctrl+D)">
                <button type="button" className="btn xs ghost"
                  onClick={(e) => { e.stopPropagation(); clearMask(m.id); }}
                  disabled={!m.strokes.length}><ClearIcon /> Clear</button>
              </Tooltip>
            </>
          )}
        />
      ))}

      <span className="manage-sep" aria-hidden="true" />
      <div className="manage-group-head">
        <span>Layers</span>
        <Tooltip text="A new empty layer on top. Runs land in the active layer, so make one to keep the next attempt apart from what is here.">
          <button type="button" className="btn ghost xs" onClick={addLayer}><AddIcon /> Layer</button>
        </Tooltip>
      </div>
      {/* Topmost first, which is how a stack reads. */}
      {[...rasterLayers].reverse().map((l) => (
        <EntityRow
          key={l.id}
          entity={l}
          active={activeLayer?.id === l.id}
          canDelete={rasterLayers.length > 1}
          thumb={l.image ? <img className="layer-thumb" src={l.image} alt="" /> : null}
          details={(
            <Slider
              label="Opacity"
              value={l.opacity ?? 1}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => setLayerOpacity(l.id, v)}
              fmt={(v) => `${Math.round(v * 100)}%`}
            />
          )}
          actions={(
            <>
              <Tooltip text="A copy of this layer above it, made active — try something else on the same ground and keep this one.">
                <button type="button" className="btn xs ghost"
                  onClick={(e) => { e.stopPropagation(); duplicateLayer(l.id); }}>Duplicate</button>
              </Tooltip>
              <Tooltip text="Select exactly the area this layer covers, and nothing else. On a layer that has only taken fills, that is the ground those fills covered.">
                <button type="button" className="btn xs ghost"
                  onClick={(e) => { e.stopPropagation(); useLayerAsMask(l.id); }}
                  disabled={!l.image}>Select area</button>
              </Tooltip>
            </>
          )}
        />
      ))}
    </div>
  );
}
