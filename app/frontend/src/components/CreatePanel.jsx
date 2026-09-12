import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay } from "../screens/playContext.jsx";
import { Slider, Select, Num, Disclose, TipLabel } from "./ui.jsx";
import {
  buildSamplePayload, buildInpaintPayload, changeToParams, effectiveSteps, skippedSteps,
  fillSizeFor, solverCanResample, LIVE_PARAM_KEYS,
  liveEditLabels, joinLabels } from "../sampleSettings.jsx";
import { contrastPreview, compositePostprocWithMask } from "../contrastMask.js";
import { bendPresetSynopsis, bendPresetSummary } from "../bendSynopsis.js";

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
    generateIntoLayer, fillIntoLayer, pushHistory, setProgress, frame, setPostFrame,
    sampleParams, openDocument,
    brushSize, setBrushSize, brushHard, setBrushHard, eraser, setEraser,
    maskTool, setMaskTool, wandTolerance, setWandTolerance,
    getMaskDataUrl, applyContrastMask, frameCard, activeLayer,
    activeMask, maskPixels, hasMask, invertMask, clearMask, setMaskParam, nudgeMask,
    undo, canUndo, redo, canRedo, tab, setLivePreview,
    job, setJob, genRunning, genPaused, canvasIsBlank,
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
  const pausedSnapshot = useRef(null);
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
  }, [tab, canUndo, undo, canRedo, redo, clearMask, invertMask, activeMask, maskTool, nudgeMask]);

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

  // Live frames go to a preview slot rather than into the stack: a half-drawn
  // intermediate is not a layer, and only the finished result should be one.
  const onJob = (j) => {
    setJob(j);
    if (j.detail?.frame) setLivePreview(j.detail.frame);
    setProgress(j.status === "running" ? { value: j.progress, message: j.message } : null);
  };

  const regionForSide = (side) => {
    if (splitMethod === "contrast") {
      return side === "foreground" ? "high" : "low";
    }
    return side === "foreground" ? "dark" : "light";
  };

  // The split reads the stack, not the screen: with post-process on the two
  // differ, and the fill this selects for reads the stack.
  const applySplit = async (side = maskSide) => {
    if (!frame) { toast("Put an image on the canvas first", "error"); return; }
    if (!splitPreview) { toast("Calculate contrast regions first", "error"); return; }
    setSplitBusy(true);
    try {
      const ok = await applyContrastMask(frame, { ...splitOpts, region: regionForSide(side) });
      if (ok) {
        setMaskSide(side);
        const def = SPLIT_SIDES.find((x) => x.id === side);
        const label = splitMethod === "luminance" ? def?.luminance : def?.contrast;
        toast(`Selected the ${(label || side).toLowerCase()} side`, "success");
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
   *  Results, from where any of them can be put into a layer. */
  const finishGeneration = (done) => {
    setProgress(null);
    setLivePreview(null);
    if (done.status === "error") toast(done.message, "error");
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
    setJob(null);
    pausedSnapshot.current = null;
  };

  /** A fill lands in the active layer, over what the layer had in that area.
   *
   *  Undo takes it back. To keep two attempts side by side, duplicate the
   *  layer first, or make a new one: toggling between them is the comparison.
   */
  const finishFill = async (done, maskSrc) => {
    setProgress(null);
    if (done.status === "error") toast(done.message, "error");
    if (done.status === "done") setSplitPending(true);
    if (done.status === "done" || done.status === "cancelled") {
      const out = framesOf(done);
      for (let i = 0; i < out.length - 1; i += 1) pushHistory(out[i].img, out[i].raw, out[i].card);
      const pick = out[out.length - 1];
      if (pick) {
        await fillIntoLayer(pick.img, pick.card, maskSrc);
        pushHistory(pick.img, pick.raw, pick.card);
      }
      if (out.length > 1) {
        toast(`${out.length} variations — the last is in ${layerName}, the rest in Results`, "success");
      }
    }
    setLivePreview(null);
    setJob(null);
    pausedSnapshot.current = null;
  };

  /** One masked run. Returns when the job settles.
   *
   *  `init` is the base the fill works from and `mask` the exact mask sent,
   *  both passed in rather than read here: the mask is what the result is
   *  punched to afterwards, and it has to be the one the backend blended with,
   *  not whatever the overlay holds by the time the job finishes.
   */
  const runFill = async ({ init, mask, batchSize = 1 }) => {
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
    try {
      const { job: j } = await api.post("/perform/inpaint", body);
      onJob(j);
      const done = await pollJob(j.id, onJob, 300);
      await finishFill(done, mask);
      return done;
    } catch (e) {
      toast(e.message, "error"); setProgress(null); setLivePreview(null); setJob(null);
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
      const { job: j } = await api.post("/perform/sample", body);
      onJob(j);
      const done = await pollJob(j.id, onJob, 300);
      finishGeneration(done);
    } catch (e) { toast(e.message, "error"); setProgress(null); setLivePreview(null); setJob(null); }
  };

  // What the live settings were when the run was paused. Resume sends only what
  // actually changed, so this has to cover every key the panel lets you edit --
  // including the guidance ones, which is how a prompt swap reaches the sampler.
  const snapshotLive = () => Object.fromEntries(
    LIVE_PARAM_KEYS.map((k) => [k, sampleParams[k]]),
  );

  const pause = async () => {
    if (!job) return;
    pausedSnapshot.current = snapshotLive();
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
    const snap = pausedSnapshot.current || {};
    const updates = {};
    for (const k of LIVE_PARAM_KEYS) {
      if (sampleParams[k] !== snap[k]) updates[k] = sampleParams[k];
    }
    if (updates.seed === "") updates.seed = null;
    try {
      await api.post(`/jobs/${job.id}/resume`, updates);
      pausedSnapshot.current = snapshotLive();
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
    ? `${maskName} · ${maskPixels.toLocaleString()} px · ${stepText}`
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
              + "The mask stays until you turn it off, so you can try another model or setting on the same area."
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
                    ? `Optional UNet tweaks saved in Play ▸ Bend, applied when filling ${maskName}. Remembered with it.`
                    : "Optional layer tweaks saved in Play ▸ Bend, applied to the whole canvas.",
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
          </span>
          {genRunning ? (
            <div className="row gap-2 mt-2">
              {genPaused ? (
                <button type="button" className="btn primary grow" onClick={resume}>Resume</button>
              ) : (
                <button type="button" className="btn grow" onClick={pause}>Pause</button>
              )}
              <button type="button" className="btn danger grow" onClick={stopAndSave}>Stop &amp; save</button>
            </div>
          ) : (
            <button type="button" className="btn primary w-full mt-2" onClick={run} disabled={!modelPath}>
              {hasMask ? "Fill mask" : reworkCanvas ? "Rework canvas" : "Generate"}
            </button>
          )}
        </div>
      </div>

      {/* ——— 2. Mask —————————————————————————————————— */}
      <Disclose
        title="Mask"
        defaultOpen
        className="create-section"
        extra={hasMask ? <span className="pill on">{maskPixels.toLocaleString()} px</span> : null}
        tip="Paint where a run may change things. Masks are rows in the Layers panel and stay until you hide or delete them."
      >
        <div className="row between center wrap gap-2 mb-2">
          <span className="sub grow">
            <TipLabel tip="Strokes go into this mask. Pick another mask row in the Layers panel to paint into that one instead; with no row picked, the topmost mask that is on takes them. Painting into a mask turns it on and shows it.">
              Into <strong>{maskName}</strong>
            </TipLabel>
          </span>
          <div className="row center gap-2">
            <button
              type="button"
              className="btn ghost sm"
              onClick={undo}
              disabled={!canUndo}
              title="Step back one edit (Ctrl+Z)"
            >
              Undo
            </button>
            <button
              type="button"
              className="btn ghost sm"
              onClick={() => invertMask(activeMask?.id)}
              title="Swap masked for unmasked (Ctrl+Shift+I)"
            >
              Invert
            </button>
            <button
              type="button"
              className="btn ghost sm"
              onClick={() => activeMask && clearMask(activeMask.id)}
              disabled={!activeMask?.strokes.length}
              title="Empty the mask, keeping its row and settings (Ctrl+D)"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="section-title">Tool</div>
        <div className="row gap-2 mb-2 wrap center">
          <div className="seg" role="group" aria-label="Mask tool">
            <button type="button" className={maskTool === "brush" ? "on" : ""} onClick={() => setMaskTool("brush")}>Brush</button>
            <button type="button" className={maskTool === "wand" ? "on" : ""} onClick={() => setMaskTool("wand")}>Wand</button>
            <button type="button" className={maskTool === "move" ? "on" : ""} onClick={() => setMaskTool("move")}>Move</button>
          </div>
          {maskTool !== "move" && (
            <div className="seg" role="group" aria-label="Paint or erase">
              <button type="button" className={!eraser ? "on" : ""} onClick={() => setEraser(false)}>Paint</button>
              <button type="button" className={eraser ? "on" : ""} onClick={() => setEraser(true)}>Erase</button>
            </div>
          )}
          {maskTool === "brush" && (
            <div className="seg" role="group" aria-label="Brush edge">
              <button type="button" className={!brushHard ? "on" : ""} onClick={() => setBrushHard(false)}>Soft</button>
              <button type="button" className={brushHard ? "on" : ""} onClick={() => setBrushHard(true)}>Hard</button>
            </div>
          )}
        </div>

        {maskTool === "wand" ? (
          <Slider
            label="Tolerance"
            value={wandTolerance}
            min={0}
            max={100}
            step={1}
            onChange={setWandTolerance}
            tip="How similar a neighboring pixel's color must be to join the mask. Click the canvas to add it."
          />
        ) : maskTool === "move" ? (
          <p className="hint mb-2">
            <TipLabel tip={`Drag anywhere on the canvas to move ${maskName}, or press the arrow keys: 1 px, or 10 with Shift. With any tool, the label on the mask's box drags it too.`}>
              Drag the mask, or use the arrow keys
            </TipLabel>
          </p>
        ) : (
          <Slider label="Size" value={brushSize} min={8} max={160} step={2} onChange={setBrushSize}
            tip="Brush diameter in canvas pixels." />
        )}

        {/* Change moved up to Generate, where it is one slider instead of two.
            What is left here is the geometry of the fill, which is a property
            of the mask and belongs beside the tools that make it. Brush
            hardness shapes strokes only; it no longer reaches into the run. */}
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

        <div className="section-title mt-2">
          <TipLabel tip="Split the canvas in two by brightness or by local contrast: set the split, press Calculate, then apply one side to the mask. Replaces any earlier split in that mask.">
            Select by contrast
          </TipLabel>
        </div>
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
        <button
          type="button"
          className="btn sm primary mt-2"
          onClick={() => applySplit(maskSide)}
          disabled={!splitPreview || splitBusy}
        >
          {splitBusy && splitPreview ? "Applying…" : "Apply to mask"}
        </button>

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
