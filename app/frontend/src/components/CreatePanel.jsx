import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay } from "../screens/playContext.jsx";
import { Slider, Select, Num, Disclose } from "./ui.jsx";
import {
  buildSamplePayload, buildInpaintPayload, changeToParams, effectiveSteps, skippedSteps,
  fillSizeFor, solverCanResample, LIVE_PARAM_KEYS,
  liveEditLabels, joinLabels } from "../sampleSettings.jsx";
import {
  contrastPreview, compositePostprocWithMask, countMaskPixels,
} from "../contrastMask.js";
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
    initImage, setInitImage, commitFrame, pushHistory, setFrame, setProgress, frame, frameRaw,
    canvasImage, sampleParams,
    brushSize, setBrushSize, brushHard, setBrushHard, eraser, setEraser,
    maskTool, setMaskTool, wandTolerance, setWandTolerance,
    clearMask, getMaskDataUrl, applyContrastMask, maskRef, maskVersion, frameCard,
    undoMask, clearMaskUndo, canUndoMask, tab,
    job, setJob, genRunning, genPaused, canvasIsBlank,
  } = usePlay();

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
  const [regionBendPreset, setRegionBendPreset] = useState(() => loadStored("kiln.regionBendPreset"));
  const [srFactor, setSrFactor] = useState(2);
  const [srSharpen, setSrSharpen] = useState(0.5);
  const [srBusy, setSrBusy] = useState(false);
  const [genChange, setGenChange] = useState(0.7);
  const [regionChange, setRegionChange] = useState(0.65);

  // A fill on a blank canvas has to run the whole schedule. Change works by
  // skipping the early, high-noise steps to preserve what is already there --
  // and on empty pixels there is nothing worth preserving, so a partial run
  // just hands back the blank it started from. Snapped to full whenever the
  // canvas is blank; it stays editable, with a warning below if it is turned
  // down. Keyed on the frame as well, so pressing New canvas restores full
  // Change rather than carrying a turned-down value onto a fresh blank.
  useEffect(() => {
    if (canvasIsBlank) setRegionChange(1);
  }, [canvasIsBlank, frame]);
  // RePaint resampling, on at 2 passes by default: the fill agrees with what
  // surrounds it far better than in a single pass, and most of that gain is
  // already there at two. 1 is off, which is how fills used to work.
  const [resample, setResample] = useState(2);
  const canResample = solverCanResample(sampleParams.sampler);
  const [feather, setFeather] = useState(8);
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
  const ppSource = frameRaw || frame;
  const ppGen = useRef(0);
  const [canvasSize, setCanvasSize] = useState(null);

  // Region fill runs at the canvas's own resolution, so the user needs to see it.
  useEffect(() => {
    if (!canvasImage) { setCanvasSize(null); return undefined; }
    let live = true;
    const im = new Image();
    im.onload = () => { if (live) setCanvasSize({ w: im.naturalWidth, h: im.naturalHeight }); };
    im.src = canvasImage;
    return () => { live = false; };
  }, [canvasImage]);

  useEffect(() => {
    api.get("/library/bends").then(setBendPresets).catch(() => {});
  }, []);

  useEffect(() => {
    if (tab !== "create") return undefined;
    const onKey = (e) => {
      const z = e.key === "z" || e.key === "Z";
      if (!z || !(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey) return;
      const t = e.target;
      const tag = t?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t?.isContentEditable) return;
      if (!canUndoMask) return;
      e.preventDefault();
      undoMask();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tab, canUndoMask, undoMask]);

  useEffect(() => {
    localStorage.setItem("kiln.genBendPreset", genBendPreset);
  }, [genBendPreset]);

  useEffect(() => {
    localStorage.setItem("kiln.regionBendPreset", regionBendPreset);
  }, [regionBendPreset]);

  const splitOpts = useMemo(() => {
    const base = splitMethod === "luminance"
      ? { method: "luminance", autoThreshold: false, threshold: brightnessThreshold, soften: splitSoften }
      : { method: "contrast", autoThreshold: true, thresholdBias: contrastThreshold, soften: splitSoften };
    return { ...base, presmooth: smoothOn ? smoothRadius : 0 };
  }, [splitMethod, brightnessThreshold, contrastThreshold, splitSoften, smoothOn, smoothRadius]);

  // Invalidate calculated preview when inputs change so Apply requires Calculate again.
  useEffect(() => {
    setSplitPreview(null);
  }, [canvasImage, splitOpts]);

  const maskPixels = useMemo(
    () => countMaskPixels(maskRef.current),
    [maskVersion, maskRef],
  );
  const hasMask = maskPixels > 0;

  useEffect(() => {
    if (!ppSource) return;
    // Off, or defaults that leave the image unchanged — never hit the API.
    if (!ppOn || isIdentityPostproc(pp)) {
      setFrame(ppSource, frameRaw);
      return;
    }
    const gen = ++ppGen.current;
    // Preserve the unprocessed source so live postproc cannot feed on itself.
    const raw = frameRaw || ppSource;
    const t = setTimeout(async () => {
      try {
        const mask = getMaskDataUrl();
        const r = await api.post("/perform/postproc", { image: raw, postproc: pp });
        let image = r.image;
        if (mask) {
          image = await compositePostprocWithMask(raw, r.image, mask);
        }
        if (gen === ppGen.current) setFrame(image, raw);
      } catch { /* ignore transient errors while dragging sliders */ }
    }, 220);
    return () => clearTimeout(t);
  }, [pp, ppOn, ppSource, frameRaw, setFrame, getMaskDataUrl, maskVersion]);

  const onJob = (j) => {
    setJob(j);
    if (j.detail?.frame) setFrame(j.detail.frame, j.detail.frame_raw, j.detail.card);
    setProgress(j.status === "running" ? { value: j.progress, message: j.message } : null);
  };

  const regionForSide = (side) => {
    if (splitMethod === "contrast") {
      return side === "foreground" ? "high" : "low";
    }
    return side === "foreground" ? "dark" : "light";
  };

  const applySplit = async (side = maskSide) => {
    if (!canvasImage) { toast("Put an image on the canvas first", "error"); return; }
    if (!splitPreview) { toast("Calculate contrast regions first", "error"); return; }
    setSplitBusy(true);
    try {
      const ok = await applyContrastMask(canvasImage, { ...splitOpts, region: regionForSide(side) });
      if (ok) {
        setMaskSide(side);
        const def = SPLIT_SIDES.find((x) => x.id === side);
        const label = splitMethod === "luminance" ? def?.luminance : def?.contrast;
        toast(`Masked the ${(label || side).toLowerCase()} side`, "success");
      } else {
        toast("Could not apply mask", "error");
      }
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setSplitBusy(false);
    }
  };

  // Setting or swapping the init image changes what is being worked on, the
  // same way a generation does. The ref starts at the current value so simply
  // returning to this tab with an init already set does not re-trigger.
  const lastInit = useRef(initImage);
  useEffect(() => {
    if (initImage && initImage !== lastInit.current) setSplitPending(true);
    lastInit.current = initImage;
  }, [initImage]);

  // A generation replaces the canvas, which invalidates any split that was on
  // screen. Recomputing it here means the contrast tiles are ready to use
  // straight away instead of needing a manual Calculate after every run.
  useEffect(() => {
    if (!splitPending || !canvasImage || splitBusy) return;
    setSplitPending(false);
    calculateSplit();
    // calculateSplit is recreated every render; the flag is cleared above, so
    // this cannot re-enter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splitPending, canvasImage]);

  const calculateSplit = async () => {
    if (!canvasImage) { toast("Put an image on the canvas first", "error"); return; }
    setSplitBusy(true);
    try {
      const p = await contrastPreview(canvasImage, splitOpts);
      setSplitPreview(p);
    } catch (e) {
      setSplitPreview(null);
      toast(e.message || "Contrast calculate failed", "error");
    } finally {
      setSplitBusy(false);
    }
  };

  const clearInit = () => {
    setInitImage(null);
    setFrame(null, null);
    clearMask({ skipUndo: true });
    clearMaskUndo();
    setSplitPreview(null);
  };

  // A selection outlives the fill it drove. Trying a second model, or a
  // different setting, on the same area is the whole point of having selected
  // it -- and a wand selection cannot be redrawn by hand anyway, least of all
  // once the fill has changed the pixels it was derived from. Deselecting is
  // the user's call, never a side effect of generating.
  const finishJob = (done, region) => {
    setProgress(null);
    if (done.status === "error") toast(done.message, "error");
    if (done.status === "done") setSplitPending(true);
    if (done.status === "done" || done.status === "cancelled") {
      const frames = done.detail?.frames;
      const framesRaw = done.detail?.frames_raw;
      const cards = done.detail?.cards;
      const cardFor = (i) => cards?.[i] ?? done.detail?.card ?? null;
      const kept = region ? " — selection kept" : "";
      if (frames?.length > 1) {
        frames.forEach((img, i) => {
          const raw = framesRaw?.[i] ?? null;
          if (i === frames.length - 1) commitFrame(img, raw, cardFor(i));
          else pushHistory(img, raw, cardFor(i));
        });
        if (done.status === "done") toast(`${frames.length} variations saved to Results${kept}`, "success");
        else toast(`Stopped — ${frames.length} variation(s) saved${kept}`, "success");
      } else if (done.detail?.frame) {
        commitFrame(done.detail.frame, done.detail.frame_raw, cardFor(0));
        if (done.status === "cancelled") toast(`Stopped — saved to Results${kept}`, "success");
        else if (region) toast("Filled — selection kept, try another model or setting", "success");
      }
    }
    setJob(null);
    pausedSnapshot.current = null;
  };

  const run = async () => {
    if (!modelPath) { toast("Pick a model first", "error"); return; }
    const mask = getMaskDataUrl();

    if (mask && !frame) {
      toast("Put an image on the canvas to fill a region", "error");
      return;
    }

    const batchSize = Math.max(1, Math.min(4, Math.round(variations) || 1));

    try {
      if (mask) {
        const genBends = resolveBends(bendPresets, regionBendPreset)
          || resolveBends(bendPresets, genBendPreset);
        const mapped = changeToParams(brushHard ? 1 : regionChange, sampleParams.steps, true);
        const body = buildInpaintPayload(sampleParams, {
          model_path: modelPath,
          init_image: frame,
          mask,
          bends: genBends,
          bend_preset: regionBendPreset || genBendPreset || "",
          feather: brushHard ? 0 : feather,
          overrides: { ...mapped, resample: canResample ? resample : 1 },
          batch_size: batchSize,
        });
        const { job: j } = await api.post("/perform/inpaint", body);
        onJob(j);
        const done = await pollJob(j.id, onJob, 300);
        finishJob(done, true);
      } else {
        const genBends = resolveBends(bendPresets, genBendPreset);
        const mapped = changeToParams(genChange, sampleParams.steps, !!initImage);
        const body = buildSamplePayload(sampleParams, {
          model_path: modelPath,
          bends: genBends,
          bend_preset: genBendPreset || "",
          init_image: initImage,
          postproc: ppOn ? pp : {},
          overrides: mapped,
          batch_size: batchSize,
        });
        const { job: j } = await api.post("/perform/sample", body);
        onJob(j);
        const done = await pollJob(j.id, onJob, 300);
        finishJob(done, false);
      }
    } catch (e) { toast(e.message, "error"); setProgress(null); setJob(null); }
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

  const upscale = async () => {
    const src = frame;
    if (!src) return;
    setSrBusy(true);
    try {
      const r = await api.post("/tools/superres", { image: src, factor: srFactor, sharpen: srSharpen });
      commitFrame(r.image, null, frameCard ? { ...frameCard, upscaled: srFactor } : null);
      toast(`Upscaled to ${r.size[0]}×${r.size[1]}`, "success");
    } catch (e) { toast(e.message, "error"); }
    setSrBusy(false);
  };

  const setPpField = (k, v) => setPp((s) => ({ ...s, [k]: v }));

  const runSteps = hasMask
    ? effectiveSteps(brushHard ? 1 : regionChange, sampleParams.steps, true)
    : effectiveSteps(genChange, sampleParams.steps, !!initImage);
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
    ? `Painted region · ${maskPixels.toLocaleString()} px · ${stepText}`
    : initImage
      ? `Full canvas · from init · ${sampleParams.image_size}px · ${stepText}`
      : `Full canvas · new generation · ${sampleParams.image_size}px · ${stepText}`;

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
        <h3>Generate</h3>
        <p className="hint mb-2">
          {hasMask
            ? "A selection is active — this reworks only that area and leaves the rest of the canvas alone. "
              + "The selection stays after each fill, so you can try another model or setting on the same area."
            : "Makes a new image, or reworks an init image. Select an area below to rework only that part instead."}
        </p>

        {initImage && !hasMask && (
          <>
            <Slider
              label="Change · whole image"
              value={genChange}
              min={0}
              max={1}
              step={0.05}
              onChange={setGenChange}
              fmt={(v) => `${v < 0.3 ? "subtle" : v < 0.7 ? "medium" : "strong"} · ${effectiveSteps(v, sampleParams.steps, true)} steps`}
              tip={"How far to move from the init image. Subtle keeps more of it; strong invents more."
                + "\n\nWhat it does is skip steps. The first steps of the schedule are the high-noise "
                + "ones that would wipe the init image out, so the run starts partway down instead: the "
                + "lower the Change, the more of those steps are skipped and the less the init image is "
                + "altered. Right now it skips "
                + `${skippedSteps(genChange, sampleParams.steps, true)} of ${sampleParams.steps} steps`
                + ", which is why the run is shorter than the Steps box says. It also sets how much "
                + "extra noise is mixed in at each remaining step."
                + "\n\nThis one governs the whole image. Region has its own Change for a painted area — "
                + "only one applies, depending on whether a mask is painted."}
            />
            <div className="row center gap-2 mb-2">
              <div className="thumb-sm"><img src={initImage} alt="init" /></div>
              <span className="sub grow">Init image set</span>
              <button type="button" className="btn ghost sm" onClick={clearInit}>Clear init</button>
            </div>
          </>
        )}
        {hasMask && initImage && (
          <p className="hint mb-2">Filling a selection uses the canvas. Deselect to generate from the init image instead.</p>
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
              tip="How many seeds to sample in one GPU run (1–4). With a fixed Seed, uses seed, seed+1, …. All land in Results."
            />
          </div>
          {bendPresets.length > 0 && (
            <div className="grow">
              <Select
                label="Bend preset"
                value={genBendPreset}
                onChange={setGenBendPreset}
                options={bendOptions}
                tip={bendTip(
                  genBendPreset,
                  "Optional layer tweaks saved in Play ▸ Bend, applied to the whole canvas.",
                )}
              />
            </div>
          )}
        </div>
        {genBendPreset && presetByName(genBendPreset) && (
          <p className="hint mb-2 mono-hint">
            {bendPresetSynopsis(presetByName(genBendPreset), ops)}
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
          {stepsTrimmed && (
            <span className="sub block mt-1">
              Keeping part of the image means starting partway down the schedule, so
              {" "}{sampleParams.steps - runSteps} early steps are skipped. Raise Change to use more.
            </span>
          )}
          {fill && canvasSize && (
            <span className="sub block mt-1">
              Fills at {fill.w}×{fill.h}
              {scaledFill
                ? ` (canvas ${canvasSize.w}×${canvasSize.h}, scaled back on return)`
                : " — your canvas's own size"}
            </span>
          )}
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
              {hasMask ? "Fill selection" : "Generate"}
            </button>
          )}
        </div>
      </div>

      {/* ——— 2. Selection —————————————————————————————— */}
      <Disclose
        title="Selection"
        defaultOpen
        className="create-section"
        extra={hasMask ? <span className="pill on">{maskPixels.toLocaleString()} px</span> : null}
        tip="Select part of the canvas so Generate reworks only that area. The selection stays until you deselect it."
      >
        <div className="row between center wrap gap-2 mb-2">
          <p className="hint mb-0 grow">
            Brush, wand, or split by contrast. Fill selection then reworks just that area.
          </p>
          <div className="row center gap-2">
            <button
              type="button"
              className="btn ghost sm"
              onClick={undoMask}
              disabled={!canUndoMask}
              title="Undo last mask edit (Ctrl+Z)"
            >
              Undo
            </button>
            <button type="button" className="btn ghost sm" onClick={() => clearMask()} disabled={!hasMask}>
              Deselect
            </button>
          </div>
        </div>

        <div className="section-title">Tool</div>
        <div className="row gap-2 mb-2 wrap center">
          <div className="seg" role="group" aria-label="Mask tool">
            <button type="button" className={maskTool === "brush" ? "on" : ""} onClick={() => setMaskTool("brush")}>Brush</button>
            <button type="button" className={maskTool === "wand" ? "on" : ""} onClick={() => setMaskTool("wand")}>Wand</button>
          </div>
          <div className="seg" role="group" aria-label="Paint or erase">
            <button type="button" className={!eraser ? "on" : ""} onClick={() => setEraser(false)}>Paint</button>
            <button type="button" className={eraser ? "on" : ""} onClick={() => setEraser(true)}>Erase</button>
          </div>
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
            tip="How similar a neighboring pixel's color must be to join the selection. Click the canvas to select."
          />
        ) : (
          <Slider label="Size" value={brushSize} min={8} max={160} step={2} onChange={setBrushSize}
            tip="Brush diameter in canvas pixels." />
        )}

        {!brushHard && (
          <div className="row gap-2">
            <div className="grow">
              <Slider
                label={hasMask ? "Change · painted region" : "Change · painted region (paint one first)"}
                value={regionChange}
                min={0}
                max={1}
                step={0.05}
                onChange={setRegionChange}
                fmt={(v) => `${v < 0.3 ? "subtle" : v < 0.7 ? "medium" : "strong"} · ${effectiveSteps(v, sampleParams.steps, true)} steps`}
                tip={"How strongly the model restyles the painted region."
                  + "\n\nWhat it does is skip steps. The first steps of the schedule are the high-noise "
                  + "ones that would wipe out what is already there, so the fill starts partway down "
                  + "instead: the lower the Change, the more of those steps are skipped and the less the "
                  + "region is altered. Right now it skips "
                  + `${skippedSteps(brushHard ? 1 : regionChange, sampleParams.steps, true)} of ${sampleParams.steps} steps`
                  + ", which is why the fill is shorter than the Steps box says. It also sets how much "
                  + "extra noise is mixed in at each remaining step."
                  + "\n\nThis one governs the painted area only. Generate has its own Change for the whole "
                  + "image — only one applies, depending on whether a mask is painted."
                  + (hasMask ? "" : "\n\nNothing is painted yet, so this has no effect right now.")}
              />
            </div>
            <div className="grow">
              <Slider label="Feather" value={feather} min={0} max={32} step={1} onChange={setFeather}
                tip="Soft edge blend for the inpaint mask." />
            </div>
          </div>
        )}
        {canvasIsBlank && !brushHard && regionChange < 1 && (
          <p className="callout mb-2">
            The canvas is blank, so anything below full Change keeps most of the empty
            pixels — a fill needs the whole schedule when there is nothing underneath
            to preserve.
          </p>
        )}

        <Slider
          label={canResample ? "Harmonize" : "Harmonize — needs DDIM or DPM-Solver++"}
          value={canResample ? resample : 1}
          min={1}
          max={8}
          step={1}
          onChange={(v) => setResample(Math.round(v))}
          disabled={!canResample}
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

        <div className="section-title mt-2">Select by contrast</div>
        <p className="hint mb-2">Set the split, press Calculate, then apply one side to the mask.</p>
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
            disabled={!canvasImage || splitBusy || !!splitPreview}
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

        {bendPresets.length > 0 && (
          <div className="mt-2">
            <Select
              label="Bend preset for the region"
              value={regionBendPreset}
              onChange={setRegionBendPreset}
              options={[
                { value: "", label: "Same as generation", title: "Reuse the bend chosen above." },
                ...bendOptions.slice(1),
              ]}
              tip={bendTip(
                regionBendPreset,
                "Optional bend used only when a mask is painted. “Same as generation” reuses the one above.",
              )}
            />
            {regionBendPreset && presetByName(regionBendPreset) && (
              <p className="hint mb-0 mt-1 mono-hint">
                {bendPresetSynopsis(presetByName(regionBendPreset), ops)}
              </p>
            )}
          </div>
        )}
      </Disclose>

      {/* ——— 3. Finish ————————————————————————————————— */}
      <Disclose
        title="Finish"
        defaultOpen
        className="create-section"
        extra={ppOn ? <span className="pill on">post-process on</span> : null}
        tip="Adjustments applied to the finished image, not to sampling."
      >
        <div className="row between center wrap gap-2 mb-2">
          <p className="hint mb-0 grow">
            {hasMask
              ? "Post-process follows the selection; upscale always uses the whole canvas."
              : "Applied to the whole canvas."}
          </p>
          <label className="row center gap-2">
            <input type="checkbox" checked={ppOn} onChange={(e) => setPpOn(e.target.checked)} />
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

        <div className="section-title mt-2">Upscale</div>
        <p className="hint mb-2">Enlarges the finished image with Lanczos resampling. Ignores the mask.</p>
        <div className="row gap-2 center">
          <div className="w-100">
            <Num label="Scale" value={srFactor} onChange={setSrFactor} min={2} max={4} step={1}
              tip="Upscale factor for the canvas image." />
          </div>
          <div className="grow">
            <Slider label="Sharpen" value={srSharpen} min={0} max={3} step={0.1} onChange={setSrSharpen}
              tip="Unsharp-mask strength applied after enlarging." />
          </div>
          <button type="button" className="btn sm self-end mb-2" onClick={upscale} disabled={srBusy || !frame}>
            {srBusy ? "…" : "Upscale"}
          </button>
        </div>
      </Disclose>
    </div>
  );
}
