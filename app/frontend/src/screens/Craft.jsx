import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, downloadPost, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay, usePlayState } from "./playContext.jsx";
import { Select, Num, Disclose, Progress, Modal, Seg, Popover } from "../components/ui.jsx";
import { DownloadIcon, ExpandIcon, TilesIcon, TransferIcon } from "../components/icons.jsx";
import UnetVisualizer from "../components/UnetVisualizer.jsx";
import BendPipeline from "../components/BendPipeline.jsx";
import BendEditor, { BendInspector } from "../components/BendEditor.jsx";
import BendPresets from "../components/BendPresets.jsx";
import {
  CompareHistory, CompareMeta, CompareViewer, MODES, MODES_LARGE, StackChips,
} from "../components/BendCompare.jsx";
import { buildSamplePayload } from "../sampleSettings.jsx";
import {
  expandGroup, holdersOf, isGroup, resolveTargets, toggleNodeTargets,
} from "../bendTargets.js";

const newBendId = () => Math.random().toString(36).slice(2);

// How the model is shown while you pick what to bend. Simple is the signal
// path in words; Structure is the map, at two grains. The geometry under the
// two map grains is the same, so switching does not move anything.
const VIEWS = [
  { id: "simple", label: "Simple", tip: "The model as a signal path: encoder, bottleneck, decoder" },
  { id: "structure", label: "Structure", tip: "The map of the network itself" },
];
// "overview" is the stored id from before the label became Levels; keeping it
// keeps everyone's saved choice.
const GRAINS = [
  { id: "overview", label: "Levels", tip: "One capsule per resolution level" },
  { id: "layers", label: "Layers", tip: "Every layer you can bend" },
];
const VIEW_KEY = "kiln.bendMapView";
const COMPARE_MODE_KEY = "kiln.bendCompareMode";
// Earlier tries kept for the session. Each holds two small images and a stack.
const HISTORY_MAX = 6;
const GRAIN_KEY = "kiln.bendMapDensity";
const INNER_KEY = "kiln.bendMapInner";

function loadPref(key, list, fallback) {
  try {
    const v = localStorage.getItem(key);
    return list.some((d) => d.id === v) ? v : fallback;
  } catch { return fallback; }
}

const MAP_HINT = {
  simple: {
    focus: "Click a stage to bend all of it, or pick single layers inside it.",
    empty: "Click a stage to start a bend on it.",
  },
  overview: {
    focus: "Click a level to target its layers, + to open it up, a dashed arc to bend that skip alone. Solid rings: this bend. Numbers: the others.",
    empty: "Click a level to start a bend on it.",
  },
  layers: {
    focus: "Click to target, shift+click for a range, drag a box (alt+drag removes), click a dashed arc to bend that skip alone. Solid rings: this bend. Numbers: the others.",
    empty: "Click a layer to start a bend on it.",
  },
};

const BEND_SWEEP_EMPTY = {
  bend: 0, param: "", from: 0, to: 1, count: 5,
  frames: null, busy: false, job: null, fps: 8, pingpong: true,
};

/** A stack as it was, without the ids that only mean something on screen. */
const stackKey = (bends) => JSON.stringify((bends || []).map(({ id, ...b }) => b));

/** Kiln's zip route wants data URLs; a pair handed over from Discoveries is
 *  served files. */
async function asDataUrl(src) {
  if (!src || src.startsWith("data:")) return src;
  const blob = await fetch(src).then((r) => r.blob());
  return new Promise((resolve) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.readAsDataURL(blob);
  });
}

export function BendWorkspace({ stack, setStack }) {
  const { toast, modelPath, ops: sharedOps, setPlayTab, bendCompare, setBendCompare } = useApp();
  const { sampleParams, commitFrame, applyCard } = usePlay();
  const [graph, setGraph] = useState(null);
  // The focused bend is the one the map edits. Focus and expansion are the same
  // thing, so there is always exactly one bend the map is talking about.
  const [focusedBendId, setFocusedBendId] = useState(null);
  const [note, setNote] = useState(null);
  const [presets, setPresets] = useState([]);
  const [saveName, setSaveName] = useState("");
  const importRef = useRef(null);
  // the before/after compare is two full samples — keep it across tab switches
  const [genBusy, setGenBusy] = usePlayState("bend.busy", false);
  const [genJob, setGenJob] = usePlayState("bend.job", null);
  const [genPlain, setGenPlain] = usePlayState("bend.plain", null);
  const [genBent, setGenBent] = usePlayState("bend.bent", null);
  const [sweep, setSweep] = usePlayState("bend.sweep", BEND_SWEEP_EMPTY);
  const [gif, setGif] = usePlayState("bend.gif", null);
  // What the cached unbent image (bend.plain) was sampled from. Kept in Play
  // state alongside the image itself so a tab switch does not silently
  // invalidate one without the other.
  const [cachedPlainKey, setCachedPlainKey] = usePlayState("bend.plainKey", null);
  // Every compare this session, newest first: { id, plain, bent, card, stack,
  // meta }. `shownTry` is the one in the viewer; null shows bend.plain/bent
  // as they are (a pair handed over from Discoveries has no entry).
  const [history, setHistory] = usePlayState("bend.history", []);
  const [shownTry, setShownTry] = usePlayState("bend.shownTry", null);
  const [compareMode, setCompareModeState] = useState(() => loadPref(COMPARE_MODE_KEY, MODES, "wipe"));
  const [bigMode, setBigMode] = useState("split");
  const [compareOpen, setCompareOpen] = useState(false);
  const [presetsOpen, setPresetsOpen] = useState(false);
  const [view, setViewState] = useState(() => loadPref(VIEW_KEY, VIEWS, "simple"));
  const [grain, setGrainState] = useState(() => loadPref(GRAIN_KEY, GRAINS, "overview"));
  const [gifBusy, setGifBusy] = useState(false);

  useEffect(() => {
    api.get("/craft/bends").then(setPresets).catch(() => {});
  }, []);

  const remember = (key, value) => {
    try { localStorage.setItem(key, value); } catch { /* private window: the session keeps it */ }
  };
  const setView = (v) => { setViewState(v); remember(VIEW_KEY, v); };
  const setGrain = (g) => { setGrainState(g); remember(GRAIN_KEY, g); };
  // The layers inside blocks, for those who want them: off unless asked for.
  const [showInner, setShowInnerState] = useState(() => {
    try { return localStorage.getItem(INNER_KEY) === "1"; } catch { return false; }
  });
  const setShowInner = (v) => { setShowInnerState(v); remember(INNER_KEY, v ? "1" : "0"); };
  const setCompareMode = (m) => { setCompareModeState(m); remember(COMPARE_MODE_KEY, m); };
  const shown = view === "simple" ? "simple" : grain;

  const ops = sharedOps || [];
  // Kiln's own recipes are listed apart from the user's: they are the answer to
  // "what does bending even do", which a list of one's own saved stacks is not.
  const starterPresets = presets.filter((p) => p.builtin);
  const savedPresets = presets.filter((p) => !p.builtin);


  useEffect(() => {
    if (!modelPath) return;
    setGraph(null);
    setGenPlain(null);
    setGenBent(null);
    // tries of another model say nothing about this one
    setHistory([]);
    setShownTry(null);
    api.post("/craft/introspect", { model_path: modelPath })
      .then(setGraph)
      .catch((e) => toast(e.message, "error"));
  }, [modelPath]);

  // A before/after handed over from the Discoveries drawer. Declared after the
  // reset above on purpose: opening a discovery usually switches model too, and
  // effects run in declaration order, so this one has the last word in the
  // commit that does both. Consumed once -- a later model switch is the user
  // moving on, and should clear the pair rather than restore it.
  useEffect(() => {
    if (!bendCompare) return;
    setGenPlain(bendCompare.plain || null);
    setGenBent(bendCompare.bent || null);
    // Not a sampled image of these settings, so the unbent cache must not claim
    // it: leaving a stale key here would let Compare skip rendering and show
    // the discovery's "before" as if it were this model's.
    setCachedPlainKey(null);
    setShownTry(null);
    setBendCompare(null);
  }, [bendCompare]);

  // Keep focus pointing at a bend that still exists.
  useEffect(() => {
    if (focusedBendId && stack.some((b) => b.id === focusedBendId)) return;
    setFocusedBendId(stack[0]?.id || null);
  }, [stack, focusedBendId]);

  const nodes = graph?.nodes || [];
  // Everything a bend can target: the main points, plus each skip connection
  // and the layers inside blocks (`extra`). Lookups and highlights use all of
  // them; counts, groups and the map's geometry stay on the main points.
  const points = useMemo(() => [
    ...nodes,
    ...(graph?.skips || []).map((s) => ({ ...s, extra: "skip" })),
    ...(graph?.inner || []).map((n) => ({ ...n, extra: "inner" })),
  ], [graph]);
  const focusedBend = stack.find((b) => b.id === focusedBendId) || null;
  const focusIndex = stack.findIndex((b) => b.id === focusedBendId);

  const focusTargets = useMemo(
    () => resolveTargets(focusedBend?.targets, points),
    [focusedBend, points],
  );
  // Which other bend holds each layer, by its number in the stack, so editing
  // one bend never hides the others -- and the map can say which one it is.
  // Bends that are off are included and flagged, and drawn faded.
  const holders = useMemo(
    () => holdersOf(stack, points, focusedBendId),
    [stack, focusedBendId, points],
  );

  const updateBend = (id, patch) => setStack(stack.map((b) => (b.id === id ? { ...b, ...patch } : b)));
  // Focus moves on by itself: the effect above re-points it at the first bend.
  const removeBend = (id) => setStack(stack.filter((b) => b.id !== id));
  // A copy straight after the original, focused, so the variation is the one
  // being edited and the original is a click (or a switch-off) away.
  const duplicateBend = (id) => {
    const i = stack.findIndex((b) => b.id === id);
    if (i < 0) return;
    const src = stack[i];
    const copy = {
      ...src, id: newBendId(), params: { ...src.params }, targets: [...(src.targets || [])],
    };
    setStack([...stack.slice(0, i + 1), copy, ...stack.slice(i + 1)]);
    setFocusedBendId(copy.id);
    setNote(null);
  };

  const addBend = (targets) => {
    const op = ops[0];
    if (!op) return null;
    const s = op.schedule || { start: 0, end: 1 };
    const params = {};
    (op.params || []).forEach((p) => { params[p.name] = p.default; });
    const bend = {
      id: newBendId(),
      op: op.name,
      params,
      // No layers by default. Adding a bend should not silently reach into
      // every layer of the network -- picking where it applies is the point of
      // the map, and "all" is a deliberate choice, not a starting position.
      targets: targets || [],
      step_start: s.start ?? 0,
      step_end: s.end ?? 1,
      active: true,
    };
    setStack([...stack, bend]);
    setFocusedBendId(bend.id);
    setNote(null);
    return bend;
  };

  // The map is the picker: clicking layers writes straight into the focused bend.
  const onMapToggle = (ids, opts) => {
    if (!focusedBend) return;
    const { targets, expanded } = toggleNodeTargets(focusedBend, ids, points, opts);
    updateBend(focusedBend.id, { targets });
    setNote(expanded.length ? { bendId: focusedBend.id, groups: expanded } : null);
  };

  const onCreateFromNode = (ids) => {
    const list = (Array.isArray(ids) ? ids : [ids]).filter(Boolean);
    if (!list.length || !addBend(list)) return;
    toast(list.length > 1 ? `Started a bend on ${list.length} layers` : "Started a bend on that layer", "success");
  };

  // Turning a group on absorbs the individual layers it already covers, so
  // re-collapsing after a map edit leaves a clean chip instead of a duplicate set.
  const toggleGroup = (g) => {
    if (!focusedBend) { addBend([g]); return; }
    const has = focusedBend.targets.includes(g);
    const covered = new Set(expandGroup(g, nodes));
    updateBend(focusedBend.id, {
      targets: has
        ? focusedBend.targets.filter((x) => x !== g)
        : [...focusedBend.targets.filter((x) => isGroup(x) || !covered.has(x)), g],
    });
    setNote(null);
  };

  const savePreset = async () => {
    const name = saveName.trim();
    if (!name) { toast("Name this bend setup first", "error"); return; }
    try {
      await api.post("/craft/bends", { name, bends: stack, model_hint: modelPath });
      toast(`Saved “${name}” — it will show up in Create`, "success");
      setSaveName("");
      setPresetsOpen(false);
      api.get("/craft/bends").then(setPresets);
    } catch (e) { toast(e.message, "error"); }
  };

  // Keep the stack and put the preset's bends after it: the way to combine a
  // starter with what you already have, which Load (a replacement) cannot do.
  const addPreset = (name) => {
    const p = presets.find((x) => x.name === name);
    if (!p?.bends?.length) return;
    const added = p.bends.map((b) => ({ ...b, id: newBendId() }));
    setStack([...stack, ...added]);
    setFocusedBendId(added[0].id);
    setNote(null);
    setPresetsOpen(false);
    toast(`Added “${name.replace(/^starter-/, "")}” — ${added.length} bend${added.length === 1 ? "" : "s"} after yours`, "success");
  };

  const loadPreset = (name) => {
    const p = presets.find((x) => x.name === name);
    if (p) {
      const loaded = (p.bends || []).map((b) => ({ ...b, id: newBendId() }));
      setStack(loaded);
      setFocusedBendId(loaded[0]?.id || null);
      setNote(null);
      setPresetsOpen(false);
      toast(`Loaded “${name}”`, "success");
    }
  };

  // Interchange with other network-bending tools. Groups are resolved to real
  // layer names first: "encoder" means nothing outside Kiln.
  const exportBends = async () => {
    if (!stack.length) { toast("Nothing to export", "error"); return; }
    const resolved = stack.filter((b) => b.active).map((b) => ({
      ...b, targets: [...resolveTargets(b.targets, points)],
    }));
    if (!resolved.length) { toast("No active bends to export", "error"); return; }
    try {
      const name = saveName.trim() || "kiln-bends";
      const headers = await downloadPost("/craft/bends/export", {
        bends: resolved,
        name,
        max_denoising_steps: sampleParams.steps,
      }, `${name}.json`);
      let report = null;
      try { report = JSON.parse(headers?.get("X-Kiln-Export") || "null"); } catch { /* optional */ }
      // Skips and q/k/v parts are Kiln's own: no other tool has a path for them.
      const left = report?.kiln_only_targets?.length || 0;
      const leftOut = left ? ` — left out ${left} skip/q·k·v target${left === 1 ? "" : "s"} no other tool can address` : "";
      if (report?.schedule_flattened) {
        toast(`Exported ${report.bends_written} layer bends — the format has one schedule for the whole file, so the per-bend windows were merged${leftOut}`, "warn");
      } else if (left) {
        toast(`Exported ${report.bends_written} layer bends${leftOut}`, "warn");
      } else {
        toast(`Exported ${report?.bends_written ?? resolved.length} layer bends`, "success");
      }
    } catch (e) { toast(e.message || "Export failed", "error"); }
  };

  const importBends = async (file) => {
    if (!file) return;
    try {
      const doc = JSON.parse(await file.text());
      const { bends, report } = await api.post("/craft/bends/import", {
        doc, layers: points.map((n) => n.id),
      });
      if (!bends.length) { toast("That file had no bends this build understands", "error"); return; }
      const loaded = bends.map((b) => ({ ...b, id: newBendId() }));
      setStack(loaded);
      setFocusedBendId(loaded[0]?.id || null);
      setNote(null);
      // A file from another tool names layers from a different architecture, so
      // say plainly how many landed rather than silently loading dead bends.
      if (report.checked_layers && report.layers_unmatched) {
        toast(`Loaded ${loaded.length} bends, but ${report.layers_unmatched} of ${report.layers_matched + report.layers_unmatched} layer paths do not exist in this model — retarget them on the map`, "warn");
      } else if (report.unknown_ops.length) {
        toast(`Loaded ${loaded.length} bends; skipped unknown ops: ${report.unknown_ops.join(", ")}`, "warn");
      } else {
        toast(`Loaded ${loaded.length} bends from file`, "success");
      }
    } catch (e) {
      toast(e.message?.includes("JSON") ? "That is not a valid JSON file" : (e.message || "Import failed"), "error");
    }
  };

  const generateCompare = async () => {
    if (!modelPath) { toast("Pick a model", "error"); return; }
    if (!stack.some((b) => b.active)) { toast("Add and enable at least one bend", "error"); return; }
    setGenBusy(true);
    setGenBent(null);
    try {
      const plainBody = buildSamplePayload(sampleParams, { model_path: modelPath, postproc: {} });
      // The unbent side does not depend on the bend stack at all, so editing
      // bends and comparing again should not pay for it twice. It is keyed on
      // everything that *does* decide it -- model and sampler settings -- and
      // only when the seed is pinned: a blank seed is redrawn every run, so a
      // cached baseline would be an image of a different thing entirely.
      const plainKey = plainBody.seed == null ? null : JSON.stringify(plainBody);
      let plainFrame = (plainKey && plainKey === cachedPlainKey) ? genPlain : null;
      const reusedPlain = !!plainFrame;
      if (!plainFrame) {
        setGenPlain(null);
        const { job: j0 } = await api.post("/perform/sample", { ...plainBody, bends: null });
        setGenJob(j0);
        const plain = await pollJob(j0.id, setGenJob, 300);
        if (plain.status === "error") throw new Error(plain.message || "Sample without bends failed");
        plainFrame = plain.detail?.frame || null;
        setCachedPlainKey(plainKey);
      }
      setGenPlain(plainFrame);

      const bentBody = buildSamplePayload(sampleParams, {
        model_path: modelPath,
        bends: stack.filter((b) => b.active),
        postproc: {},
      });
      const { job: j1 } = await api.post("/perform/sample", bentBody);
      setGenJob(j1);
      const bent = await pollJob(j1.id, setGenJob, 300);
      if (bent.status === "error") throw new Error(bent.message || "Sample with bends failed");
      const bentFrame = bent.detail?.frame || null;
      setGenBent(bentFrame);
      const entry = {
        id: newBendId(),
        plain: plainFrame,
        bent: bentFrame,
        card: bent.detail?.card || null,
        stack: stack.map((b) => ({ ...b })),
        meta: {
          seed: bentBody.seed, steps: bentBody.steps, size: bentBody.image_size, reused: reusedPlain,
        },
      };
      setHistory((h) => [entry, ...(h || [])].slice(0, HISTORY_MAX));
      setShownTry(entry.id);
      toast(reusedPlain
        ? "Compared — reused the unbent image, only the bent side was sampled"
        : "Compared with vs without bends", "success");
    } catch (e) { toast(e.message, "error"); }
    setGenBusy(false);
    setGenJob(null);
  };

  const pair = (shownTry && history?.find((h) => h.id === shownTry))
    || { plain: genPlain, bent: genBent, card: null, stack: null, meta: null };
  const hasPair = !!(pair.plain || pair.bent);
  const restorable = !!pair.stack && stackKey(pair.stack) !== stackKey(stack);

  const restoreTry = (entry) => {
    const loaded = entry.stack.map((b) => ({ ...b, id: newBendId() }));
    setStack(loaded);
    setFocusedBendId(loaded[0]?.id || null);
    setNote(null);
    toast("Restored the bends from that try — they replaced your stack", "success");
  };

  const downloadPair = async () => {
    try {
      const [plain, bent] = await Promise.all([asDataUrl(pair.plain), asDataUrl(pair.bent)]);
      await downloadPost("/tools/zip", {
        images: [
          { image: plain, value: "without", card: null },
          { image: bent, value: "with", card: pair.card },
        ].filter((x) => x.image),
        label: "bend",
        name: `bend_compare_${Date.now().toString().slice(-6)}`,
      }, "bend-compare.zip");
    } catch (e) { toast(e.message || "Download failed", "error"); }
  };

  const runSweep = async () => {
    const b = stack[sweep.bend];
    if (!b || !sweep.param) { toast("Pick a bend and a numeric parameter", "error"); return; }
    if (!modelPath) { toast("Pick a model", "error"); return; }
    setSweep((s) => ({ ...s, busy: true, frames: null }));
    setGif(null);
    try {
      const { job: j } = await api.post("/craft/bend/sweep", {
        model_path: modelPath,
        bends: stack,
        bend_index: sweep.bend,
        param: sweep.param,
        from: sweep.from,
        to: sweep.to,
        count: sweep.count,
        image_size: sampleParams.image_size,
        steps: sampleParams.steps,
        eta: sampleParams.eta,
        ema: sampleParams.ema,
        sampler: sampleParams.sampler,
        seed: sampleParams.seed === "" ? null : sampleParams.seed,
      });
      setSweep((s) => ({ ...s, job: j }));
      const done = await pollJob(j.id, (u) => setSweep((s) => ({ ...s, job: u })), 400);
      if (done.status === "error") throw new Error(done.message);
      setSweep((s) => ({ ...s, busy: false, job: done, frames: done.detail?.frames || [] }));
      if (done.status === "done") toast(`Swept ${sweep.param} over ${done.detail?.frames?.length || 0} samples`, "success");
    } catch (e) {
      toast(e.message, "error");
      setSweep((s) => ({ ...s, busy: false }));
    }
  };

  const stopSweep = async () => {
    if (sweep.job) await api.post(`/jobs/${sweep.job.id}/cancel`);
  };

  // A sweep is a set of candidate runs, not one artefact: any frame can be
  // carried into Create, and dropping the weak ones is how a usable GIF gets made.
  const openFrameInCreate = (f) => {
    commitFrame(f.image, null, f.card || null);
    if (f.card) applyCard(f.card);
    setPlayTab("create");
  };

  const dropFrame = (i) => {
    setSweep((s) => ({ ...s, frames: (s.frames || []).filter((_, k) => k !== i) }));
    setGif(null);  // the built GIF no longer matches the strip
  };

  const downloadFrames = async () => {
    const frames = sweep.frames || [];
    if (!frames.length) { toast("Run a sweep first", "error"); return; }
    try {
      await downloadPost("/tools/zip", {
        images: frames.map((f) => ({ image: f.image, value: f.value, card: f.card || null })),
        label: sweep.param,
        name: `bend_${sweep.param}_sweep`,
        card: sweep.job?.detail?.card || null,
      }, `bend-${sweep.param}-sweep.zip`);
    } catch (e) { toast(e.message || "Download failed", "error"); }
  };

  const makeGif = async () => {
    const frames = sweep.frames || [];
    if (frames.length < 2) { toast("Run a sweep with at least two values first", "error"); return; }
    setGifBusy(true);
    try {
      const r = await api.post("/tools/gif", {
        images: frames.map((f) => f.image),
        fps: sweep.fps,
        pingpong: sweep.pingpong,
        name: `bend_${sweep.param}_${Date.now().toString().slice(-6)}`,
        card: sweep.job?.detail?.card || null,
        kind: "sweep",
      });
      setGif(r);
      toast(`GIF ready — ${r.frames} frames, saved to Sweeps`, "success");
    } catch (e) { toast(e.message, "error"); }
    setGifBusy(false);
  };

  const opMap = Object.fromEntries((ops || []).map((o) => [o.name, o]));
  const sweepBend = stack[sweep.bend];
  const sweepParams = (opMap[sweepBend?.op]?.params || []).filter((p) => p.kind !== "select");

  return (
    <div className="col">
      {/* One row for what the screen is and how to look at it. The map's view
          controls live here rather than on the map card, so changing them never
          moves the map. */}
      <div className="bend-toolbar">
        <div className="bend-toolbar-id">
          <h2 className="bend-title">Bend</h2>
          {graph?.model && (
            <span className="pill tnum">{graph.model.mtype} · {graph.model.mults?.join("-")}{graph.model.attn ? ` · ${graph.model.attn}` : ""}</span>
          )}
          <span className="sub">
            Rewrites activations at the layers you pick, mid-generation. The model file is untouched.
          </span>
        </div>
        <div className="bend-toolbar-tools">
          <Seg ariaLabel="How to show the model" tabs={VIEWS} value={view}
               onChange={setView} size="sm" />
          {view === "structure" && (
            <Seg ariaLabel="How much detail the map shows" tabs={GRAINS} value={grain}
                 onChange={setGrain} size="sm" />
          )}
          {view === "structure" && grain === "layers" && graph?.inner?.length > 0 && (
            <button type="button" className={`btn sm ${showInner ? "on" : ""}`.trim()}
              aria-pressed={showInner}
              title="Show the layers inside each block and attention: conv, norm, film, q, k, v"
              onClick={() => setShowInner(!showInner)}>
              Inside blocks
            </button>
          )}
          <span className="bend-toolbar-sep" aria-hidden="true" />
          <Popover
            label="Bend presets"
            triggerClass="btn sm"
            trigger={<><TilesIcon /> Presets</>}
            open={presetsOpen}
            onOpenChange={setPresetsOpen}
            panelClass="bend-presets-pop"
          >
            <BendPresets
              starters={starterPresets}
              saved={savedPresets}
              ops={ops}
              nodes={nodes}
              saveName={saveName}
              setSaveName={setSaveName}
              canSave={stack.length > 0 && !!saveName.trim()}
              onSave={savePreset}
              onLoad={loadPreset}
              onAdd={addPreset}
            />
          </Popover>
          <Popover
            label="Share with other tools"
            triggerLabel="Import or export bends as JSON"
            triggerClass="btn icon"
            trigger={<TransferIcon size={15} />}
          >
            {(close) => (
              <div className="bend-share">
                <div className="section-title">Share with other tools</div>
                <p className="sub">
                  The shared network-bending JSON, as other bending tools write it. Layer paths are
                  per-architecture, so a file from elsewhere usually needs retargeting on the map.
                </p>
                <div className="row gap-2">
                  <button type="button" className="btn sm" disabled={!stack.length}
                    onClick={() => { close(); exportBends(); }}>
                    <DownloadIcon /> Export JSON
                  </button>
                  <button type="button" className="btn sm"
                    onClick={() => { importRef.current?.click(); close(); }}>
                    Import JSON
                  </button>
                </div>
              </div>
            )}
          </Popover>
          {/* Outside the popover, which unmounts on close: the file picker
              outlives it. */}
          <input
            ref={importRef}
            type="file"
            accept="application/json,.json"
            className="hidden-file"
            onChange={(e) => { importBends(e.target.files?.[0]); e.target.value = ""; }}
          />
        </div>
      </div>

      <div className="bend-work">
        <div className="card bend-bench">
          {/* The map and the settings of the bend it is editing, side by side:
              the map sets where a bend acts, the panel everything else. They
              wrap into one column when the card is too narrow for both. */}
          <div className="bend-bench-top">
            <div className="bend-bench-map">
              <div className="row between center wrap gap-2">
                <div className="row center wrap gap-2">
                  <h3 className="mb-0">Model map</h3>
                  {focusedBend ? (
                    <span className="pill accent">
                      Editing #{focusIndex + 1} {opMap[focusedBend.op]?.label || focusedBend.op}
                    </span>
                  ) : (
                    <span className="pill">No bend selected</span>
                  )}
                </div>
              </div>
              <p className="hint mt-1 mb-2">{MAP_HINT[shown][focusedBend ? "focus" : "empty"]}</p>
              {view === "simple" ? (
                <BendPipeline
                  graph={graph}
                  focusTargets={focusTargets}
                  holders={holders}
                  hasFocus={!!focusedBend}
                  activeGroups={(focusedBend?.targets || []).filter(isGroup)}
                  onToggle={onMapToggle}
                  onToggleGroup={toggleGroup}
                  onCreateFromNode={onCreateFromNode}
                />
              ) : (
                <UnetVisualizer
                  graph={graph}
                  focusTargets={focusTargets}
                  holders={holders}
                  hasFocus={!!focusedBend}
                  density={grain}
                  showInner={showInner}
                  onToggle={onMapToggle}
                  onCreateFromNode={onCreateFromNode}
                />
              )}
            </div>
            <BendInspector
              b={focusedBend}
              index={focusIndex}
              opDef={opMap[focusedBend?.op]}
              ops={ops}
              nodes={nodes}
              points={points}
              note={note?.bendId === focusedBendId ? note.groups : null}
              update={updateBend}
              remove={removeBend}
              duplicate={duplicateBend}
            />
          </div>

          {/* The stack lives in the same card as the map: the map is how you set
              the layers of whichever card below is focused. */}
          <div className="bend-work-sep" />
          <BendEditor
            ops={ops}
            nodes={nodes}
            points={points}
            stack={stack}
            setStack={setStack}
            focusedId={focusedBendId}
            setFocusedId={(id) => { setFocusedBendId(id); setNote(null); }}
            addBend={() => addBend()}
            headExtra={(
              <button type="button" className="btn ghost sm" disabled={!stack.length}
                onClick={() => setPresetsOpen(true)}>
                Save as preset…
              </button>
            )}
          />
        </div>

        <div className="card bend-compare-card">
          <div className="row between center gap-2">
            <h3 className="mb-0">Compare</h3>
            <div className="row center gap-2">
              <Seg ariaLabel="How to show the pair" tabs={MODES} value={compareMode}
                   onChange={setCompareMode} size="sm" />
              <button type="button" className="btn icon" aria-label="Enlarge the compare"
                title="Enlarge" disabled={!hasPair} onClick={() => setCompareOpen(true)}>
                <ExpandIcon />
              </button>
            </div>
          </div>
          {hasPair ? (
            <CompareViewer plain={pair.plain} bent={pair.bent} mode={compareMode} />
          ) : (
            <div className="cmp-empty sub">
              {genBusy ? "Sampling…" : "The same seed, without and then with your bends, shows up here."}
            </div>
          )}
          <CompareMeta meta={pair.meta} />
          {genBusy ? (
            <div className="col gap-1">
              <button type="button" className="btn danger w-full"
                onClick={async () => { if (genJob) await api.post(`/jobs/${genJob.id}/cancel`); }}>
                Stop
              </button>
              {genJob && (
                <>
                  <Progress value={genJob.progress || 0} />
                  <span className="sub">{genJob.message || "Generating…"}</span>
                </>
              )}
            </div>
          ) : (
            <button type="button" className="btn primary w-full" onClick={generateCompare}
              disabled={!modelPath || !stack.length}>
              Compare samples
            </button>
          )}
          <p className="hint mb-0">
            Same seed, plain then bent. The plain side is reused until the model, settings or seed change.
          </p>
          <CompareHistory
            history={history}
            shownId={shownTry}
            ops={ops}
            onShow={setShownTry}
            onRestore={restoreTry}
            restorable={restorable}
          />
        </div>
      </div>

      {compareOpen && (
        <Modal
          title="Compare"
          wide
          onClose={() => setCompareOpen(false)}
          footer={(
            <>
              <button type="button" className="btn" onClick={downloadPair} disabled={!hasPair}>
                <DownloadIcon /> Download pair
              </button>
              <button type="button" className="btn primary" disabled={!pair.bent}
                onClick={() => { setCompareOpen(false); openFrameInCreate({ image: pair.bent, card: pair.card }); }}>
                Open bent in Create
              </button>
              <button type="button" className="btn ghost" onClick={() => setCompareOpen(false)}>Close</button>
            </>
          )}
        >
          <div className="row between center wrap gap-2 mb-2">
            <Seg ariaLabel="How to show the pair" tabs={MODES_LARGE} value={bigMode}
                 onChange={setBigMode} size="sm" />
            <CompareMeta meta={pair.meta} />
          </div>
          <CompareViewer plain={pair.plain} bent={pair.bent} mode={bigMode} large />
          <StackChips stack={pair.stack} ops={ops} />
        </Modal>
      )}

      <Disclose
        title="Sweep a bend parameter"
        tip="Runs a full sample for each value, changing only this one number — everything else, including the seed and the layers you targeted, stays fixed."
      >
        <p className="hint mb-2">
          One generation per value, same seed, same targets. Only the number moves, so the strip
          can be saved as a GIF.
        </p>
        <div className="row wrap gap-3">
          <div className="w-130">
            <Select label="Bend" value={String(sweep.bend)}
              onChange={(v) => setSweep((s) => ({ ...s, bend: parseInt(v, 10), param: "" }))}
              options={stack.map((b, i) => ({ value: String(i), label: `#${i + 1} ${opMap[b.op]?.label || b.op}` }))}
              tip="Which bend in the stack to vary." />
          </div>
          <div className="w-130">
            <Select label="Parameter" value={sweep.param}
              onChange={(v) => {
                const def = sweepParams.find((x) => x.name === v);
                setSweep((s) => ({
                  ...s, param: v,
                  from: def?.min ?? s.from,
                  to: def?.max ?? s.to,
                }));
              }}
              options={[{ value: "", label: "—" }, ...sweepParams.map((p) => ({ value: p.name, label: p.label }))]}
              tip="Which number on that bend to sweep. Picking one snaps the range to its limits." />
          </div>
          <div className="w-80"><Num label="From" value={sweep.from} onChange={(v) => setSweep((s) => ({ ...s, from: v }))} tip="Start of the sweep range." /></div>
          <div className="w-80"><Num label="To" value={sweep.to} onChange={(v) => setSweep((s) => ({ ...s, to: v }))} tip="End of the sweep range." /></div>
          <div className="w-70"><Num label="Frames" value={sweep.count} min={2} max={24}
            onChange={(v) => setSweep((s) => ({ ...s, count: Math.max(2, Math.min(24, Math.round(v) || 2)) }))}
            tip="How many full samples to render." /></div>
          {sweep.busy ? (
            <button type="button" className="btn danger self-end mb-2" onClick={stopSweep}>Stop</button>
          ) : (
            <button type="button" className="btn primary self-end mb-2" onClick={runSweep}
              disabled={!stack.length || !sweep.param || !modelPath}>
              Run sweep
            </button>
          )}
        </div>

        {sweep.busy && sweep.job && (
          <>
            <Progress value={sweep.job.progress || 0} />
            <p className="sub mt-1">{sweep.job.message || "Sampling…"}</p>
          </>
        )}

        {sweep.frames?.length > 0 && (
          <>
            <div className="row between center wrap gap-2 mt-2">
              <span className="sub">
                {sweep.frames.length} run{sweep.frames.length === 1 ? "" : "s"} — click one to open it in Create,
                or drop the ones you don&apos;t want before building the GIF.
              </span>
              <button type="button" className="btn sm" onClick={downloadFrames}
                title="Save every run in the strip as a zip of PNGs, each with its recipe embedded.">
                Download all ({sweep.frames.length})
              </button>
            </div>
            <div className="bend-sweep-strip">
              {sweep.frames.map((f, i) => (
                <div key={`${f.value}-${i}`} className="bend-sweep-cell">
                  <div className="bend-sweep-shot">
                    <button
                      type="button"
                      className="bend-sweep-pick"
                      title={`Open in Create · ${sweep.param} ${f.value}`}
                      onClick={() => openFrameInCreate(f)}
                    >
                      <img src={f.image} alt={`${sweep.param} = ${f.value}`} />
                      <span className="sweep-cell-hint">Open in Create</span>
                    </button>
                    <button
                      type="button"
                      className="bend-sweep-drop"
                      aria-label={`Remove ${sweep.param} ${f.value} from the strip`}
                      title="Remove this run"
                      onClick={() => dropFrame(i)}
                    >
                      ✕
                    </button>
                  </div>
                  <span className="sub mono">{sweep.param} {f.value}</span>
                </div>
              ))}
            </div>

            <div className="row center wrap gap-2 mt-2">
              <div className="w-70">
                <Num label="GIF fps" value={sweep.fps} min={1} max={30}
                  onChange={(v) => setSweep((s) => ({ ...s, fps: Math.max(1, Math.min(30, Math.round(v) || 1)) }))}
                  tip="Playback speed of the animation." />
              </div>
              <label className="row center gap-2 self-end mb-2">
                <input type="checkbox" checked={sweep.pingpong}
                  onChange={(e) => setSweep((s) => ({ ...s, pingpong: e.target.checked }))} />
                <span className="sub">Ping-pong</span>
              </label>
              <button type="button" className="btn primary self-end mb-2" onClick={makeGif} disabled={gifBusy}>
                {gifBusy ? "Building…" : "Make GIF"}
              </button>
              {gif && (
                <a className="btn self-end mb-2" href={gif.gif} download={`${sweep.param}-sweep.gif`}>
                  Download GIF
                </a>
              )}
            </div>
            <p className="hint mb-0">
              Ping-pong plays the sweep forwards then back, so the loop has no jump.
            </p>

            {gif && (
              <div className="preview-box preview-max mt-2">
                <img src={gif.gif} alt="bend sweep animation" />
              </div>
            )}
          </>
        )}
      </Disclose>
    </div>
  );
}
