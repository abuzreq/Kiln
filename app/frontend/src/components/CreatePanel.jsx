import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay } from "../screens/playContext.jsx";
import { Slider, Select, Num, Seg, TipLabel, Tooltip } from "./ui.jsx";
import { useFrameSize } from "./CanvasTools.jsx";
import {
  buildSamplePayload, buildInpaintPayload, changeToParams, effectiveSteps, skippedSteps,
  fillSizeFor, solverCanResample, LIVE_PARAM_KEYS,
  liveEditLabels, joinLabels } from "../sampleSettings.jsx";
import { compositePostprocWithMask } from "../contrastMask.js";
import { bendPresetSynopsis, bendPresetSummary } from "../bendSynopsis.js";

const PANEL_KEY = "kiln.createPanelTab";

const DEFAULT_PP = { contrast: 1, gamma: 1, saturation: 1, eqhist: 0, unsharp: 0, noise: 0 };

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
    job, genRunning, genPaused, canvasIsBlank, runs, trackRuns, setSampleParam, generateRef,
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
  // Generate or Finish. Remembered, so someone who is finishing a picture
  // comes back to Finish.
  const [panelTab, setPanelTabState] = useState(() => (loadStored(PANEL_KEY) === "finish" ? "finish" : "generate"));
  const setPanelTab = (t) => {
    setPanelTabState(t);
    try { localStorage.setItem(PANEL_KEY, t); } catch { /* ignore */ }
  };

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
  // Job id -> the live settings it was started (or last resumed) with. Resume
  // sends only what differs, and with a queue each run has its own baseline.
  const snapshots = useRef({});
  // The current run(), for the keyboard handler, which is bound once per tab.
  const runRef = useRef(null);
  const ppSource = frame;
  const ppGen = useRef(0);
  // Region fill runs at the canvas's own resolution, so the user needs to see it.
  const canvasSize = useFrameSize(frame);

  useEffect(() => {
    api.get("/library/bends").then(setBendPresets).catch(() => {});
  }, []);

  useEffect(() => {
    if (tab !== "create") return undefined;
    const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const onKey = (e) => {
      // Ctrl+Enter runs, from anywhere on the tab -- the prompt field included,
      // so words can be typed and sent without reaching for the mouse.
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !e.altKey && !e.repeat) {
        e.preventDefault();
        runRef.current?.();
        return;
      }
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

  runRef.current = () => {
    if (hasMask && runs.length) { toast("A fill starts when nothing else is running", "error"); return; }
    run();
  };
  generateRef.current = runRef.current;

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

  // What the run will do and where it lands, in words, next to the button.
  // This replaced a notice over the picture ("Inpaint Mask 1"), which said only
  // half of it and pushed the picture down whenever a mask was on.
  const what = genPaused
    ? <>Paused · step {job?.detail?.step || "?"} / {job?.detail?.total || "?"}</>
    : hasMask
      ? <>Fills <strong>{liveMasks.length > 1 ? `${liveMasks.length} masks` : maskName}</strong> · {stepText}</>
      : reworkCanvas
        ? <>Reworks the <strong>whole canvas</strong>, keeping its layout · {stepText}</>
        : <>A <strong>new image</strong> · {sampleParams.image_size}px · {stepText}</>;
  const whatTip = hasMask
    ? `Only the masked area changes; the rest of the canvas is kept exactly. When the fill lands the mask turns off, so the next run is the whole canvas again; turn it back on in Layers to try another setting on the same area.`
    : canvasIsBlank
      ? "Makes a new image. Mask an area to rework only that part instead."
      : "Reworks the whole canvas by the amount of Change; at full Change it makes a new image. Mask an area to rework only that part instead.";

  return (
    <section className="card inspector-panel create-panel" aria-label="Make">
      <div className="panel-tabs" role="tablist" aria-label="Make">
        {[
          { id: "generate", label: "Generate" },
          { id: "finish", label: "Finish", extra: ppOn ? <span className="panel-tab-dot" title="Post-process is on" /> : null },
        ].map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={panelTab === t.id}
            className={panelTab === t.id ? "on" : ""}
            onClick={() => setPanelTab(t.id)}
          >
            {t.label}{t.extra}
          </button>
        ))}
      </div>

      {panelTab === "generate" ? (
        <div className="panel-body" role="tabpanel" aria-label="Generate">
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

          {/* The prompt, where it is used. It is the same stored value as the
              Prompt in Setup ▸ Advanced, which Bend, Merge and Sweep sample
              with, so the two can never disagree. */}
          <label className="field steer-words">
            <TipLabel tip={"Steers the picture towards these words with CLIP guidance while it samples. "
              + "Kiln's models are unconditional: the words nudge, they do not describe. Empty means no guidance."
              + "\n\nThe same prompt as in Setup ▸ Advanced, where its strength and ramp live; Bend, Merge and Sweep use it too."}
            >
              <span>Steer with words</span>
            </TipLabel>
            <input
              type="text"
              value={sampleParams.text || ""}
              placeholder="e.g. a rust-red coastline, aerial"
              onChange={(e) => setSampleParam("text", e.target.value)}
            />
          </label>

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
            <p className="hint mb-0 mono-hint">
              {bendPresetSynopsis(presetByName(bendValue), ops)}
            </p>
          )}

          {/* The fill's edge belongs to the mask, but it is decided at the moment
              of filling, so it sits with the button that fills. */}
          {hasMask && !genRunning && (
            <div className="fill-edge">
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
                  Switch the sampler to DDIM or DPM-Solver++ in Setup to enable this.
                </p>
              )}
            </div>
          )}

        </div>
      ) : null}
      {/* The outcome and the button sit under the scrolling settings, never
          inside them: a long Fill edge must not push the button out of view. */}
      {panelTab === "generate" && (
        <div className="panel-foot">
          <div className="create-generate-box outcome">
            <span className="outcome-what">
              <TipLabel tip={whatTip}>
                {what}{!genPaused && variations > 1 ? ` · ${variations} variations` : ""}
              </TipLabel>
            </span>
            {/* Where the result lands, and at what size. The layer is chosen in
                the Layers panel; this is only the consequence, said next to the
                button, with the reasons a hover away. */}
            <span className="sub">
              {queueing ? (
                <TipLabel tip={
                  "While a queue exists, every run lands in Results and the layer is left alone: "
                  + "otherwise each run would replace the layer in turn, and which one stayed would "
                  + "depend on which finished last. Put any of them into a layer from Results."
                  + "\n\nEach image is also saved, with its recipe, to one folder per queue under "
                  + "captures, so a long queue cannot push its first images out of Results."
                }
                >
                  Lands in <strong>Results</strong> · saved to captures
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
                  {hasMask ? "Outside stays as it is · lands in " : "Lands in "}
                  <strong>{layerName}</strong>
                  {fill && canvasSize ? ` · ${fill.w}×${fill.h}` : ""}
                </TipLabel>
              )}
            </span>
            {/* Generate never locks: while anything is in flight it queues the
                settings as they are at the click. Fills are the exception, see run(). */}
            <button
              type="button"
              className="btn primary w-full mt-1 generate-btn"
              onClick={run}
              disabled={!modelPath || (hasMask && inFlight)}
              title={hasMask && inFlight ? "A fill starts when nothing else is running" : undefined}
            >
              {generateLabel}
            </button>
            {genRunning && (
              <div className="row gap-2">
                {genPaused ? (
                  <button type="button" className="btn primary grow" onClick={resume}>Resume</button>
                ) : (
                  <button type="button" className="btn grow" onClick={pause}>Pause</button>
                )}
                <Tooltip text="Stop now and keep what it has so far, in the layer it was going to">
                  <button type="button" className="btn danger grow" onClick={stopAndSave}>Stop &amp; keep</button>
                </Tooltip>
              </div>
            )}
            <span className="sub generate-keys">
              Ctrl+Enter {hasMask ? "fills" : "generates"}{!hasMask ? " · clicking while one runs queues it" : ""}
            </span>
          </div>
        </div>
      )}
      {panelTab === "finish" && (
        <div className="panel-body" role="tabpanel" aria-label="Finish">
          <div className="row between center gap-2">
            <span className="sub">
              <TipLabel tip={hasMask
                ? "With a mask on, the adjustments apply inside it only. They are a display stage over the layers, not an edit to them, so Download and Capture bake them in but the layers stay as they are."
                : "Applied to the whole canvas as a display stage over the layers, not an edit to them: Download and Capture bake it in, the layers stay as they are."}
              >
                Applies to <strong>{hasMask ? `inside ${maskName}` : "the whole canvas"}</strong>
              </TipLabel>
            </span>
          </div>

          <div className="finish-block">
            <div className="row between center">
              <span className="finish-block-title">Post-process</span>
              <button
                type="button"
                role="switch"
                aria-checked={ppOn}
                aria-label="Post-process"
                className={`switch ${ppOn ? "on" : ""}`.trim()}
                onClick={() => setPpOn(!ppOn)}
              >
                <span />
              </button>
            </div>
            {ppOn ? (
              <>
                <Slider label="Contrast" value={pp.contrast} min={0.5} max={2} step={0.05}
                  onChange={(v) => setPpField("contrast", v)}
                  tip="Boost or flatten contrast after sampling." />
                <Slider label="Gamma" value={pp.gamma} min={0.5} max={2} step={0.05}
                  onChange={(v) => setPpField("gamma", v)}
                  tip="Brighten (lower) or darken (higher) midtones." />
                <Slider label="Sharpen" value={pp.unsharp} min={0} max={4} step={0.1}
                  onChange={(v) => setPpField("unsharp", v)}
                  tip="Unsharp-mask strength on the finished image." />
                <p className="hint mb-0">Unprocessed, on the picture, shows the layers without it.</p>
              </>
            ) : (
              <p className="hint mb-0">Contrast, gamma and sharpening over the finished picture. The layers are never changed.</p>
            )}
          </div>

          <div className="finish-block">
            <div className="row between center gap-2">
              <span className="finish-block-title">
                <TipLabel tip="Enlarges the finished image with Lanczos resampling. Ignores masks, and flattens the layer stack into one layer at the new size; the masks are kept.">
                  Upscale
                </TipLabel>
              </span>
              <Seg
                ariaLabel="Upscale factor"
                size="sm"
                value={String(srFactor)}
                onChange={(v) => setSrFactor(Number(v))}
                tabs={[2, 3, 4].map((f) => ({ id: String(f), label: `${f}×`, tip: `${f} times the size` }))}
              />
            </div>
            <Slider label="Sharpen" value={srSharpen} min={0} max={3} step={0.1} onChange={setSrSharpen}
              tip="Unsharp-mask strength applied after enlarging." />
            <button
              type="button"
              className="btn sm w-full"
              onClick={upscale}
              disabled={srBusy || !frame}
            >
              {srBusy
                ? "Upscaling…"
                : canvasSize
                  ? `Upscale to ${canvasSize.w * srFactor} × ${canvasSize.h * srFactor}`
                  : "Upscale"}
            </button>
            <p className="hint mb-0">Flattens the layers into one; the masks are kept. The result also lands in Results.</p>
          </div>
        </div>
      )}
    </section>
  );
}
