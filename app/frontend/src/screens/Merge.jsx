import React, { useEffect, useMemo, useState } from "react";
import { api, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay, usePlayState } from "./playContext.jsx";
import { Modal, Popover, Seg, TipLabel } from "../components/ui.jsx";
import { BookmarkIcon, ExpandIcon } from "../components/icons.jsx";
import { selectOptions, tagStars } from "./ModelList.jsx";
import { buildSamplePayload, parseSeed } from "../sampleSettings.jsx";
import { presetThumb } from "../presetThumb.js";
import {
  MERGE_OPEN_EVENT, describeMix as describe, mixOf, pct, recipeOf, recipeSummary, round4,
  takeMergeOpen,
} from "../mergeRecipes.js";

const METHODS = [
  { id: "linear", label: "Linear", tip: "A straight weighted average of A's and B's weights" },
  { id: "slerp", label: "Slerp", tip: "Blends along a sphere so each layer keeps its size" },
  { id: "blockwise", label: "Block-wise", tip: "A different mix per UNet stage" },
];
const METHOD_TIPS = {
  linear: "Each step is a straight weighted average. The ends are A and B themselves, so only the steps between are new merges.",
  slerp: "Each step blends along a sphere, so layers keep their size. Slerp and linear share their ends, so comparing them only adds the steps between.",
  blockwise: "Rows set how much of the structure (encoder) comes from B; columns do the same for texture (decoder). The diagonal mixes both evenly. The mid stage, the bottleneck between them, is on neither axis: every cell uses the one share set in “Mid stage, every cell”.",
};
const STAGES = [
  { id: "encoder", label: "Encoder", role: "structure" },
  { id: "mid", label: "Mid", role: "bottleneck" },
  { id: "decoder", label: "Decoder", role: "texture" },
];
// Where each ladder starts, per method; zooming pushes onto the method's own stack.
const START_VIEWS = {
  linear: [{ kind: "1d", lo: 0, hi: 1 }],
  slerp: [{ kind: "1d", lo: 0, hi: 1 }],
  blockwise: [{ kind: "grid", enc: [0, 1], dec: [0, 1] }],
};
const EMPTY_CACHE = { key: "", seed: null, refs: {}, cells: {} };
const LIVE = ["queued", "running"];

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const linspace = (lo, hi, n) =>
  Array.from({ length: n }, (_, i) => round4(lo + (hi - lo) * (i / Math.max(n - 1, 1))));
const recipeKey = (r) => (r.method === "blockwise"
  ? `blockwise:${STAGES.map((s) => r.block_weights[s.id]).join("/")}`
  : `${r.method}:${r.alpha}`);

/** "a" or "b" when a recipe is exactly that model (the server reuses its sample). */
function endOf(r) {
  const vals = r.method === "blockwise" ? Object.values(r.block_weights) : [r.alpha];
  if (vals.every((v) => v === 0)) return "a";
  if (vals.every((v) => v === 1)) return "b";
  return null;
}

function gridTag(r) {
  const { encoder: e, decoder: d } = r.block_weights;
  if (e === 0 && d === 1) return "structure A · texture B";
  if (e === 1 && d === 0) return "structure B · texture A";
  if (e === 0 && d === 0) return "mostly A";
  if (e === 1 && d === 1) return "mostly B";
  if (e === 0.5 && d === 0.5) return "even mix";
  return "";
}

/** What one view of the ladder asks the server for, and how it lays out. */
function ladderFor(method, view, { steps, compare, grid, mid }) {
  if (view.kind === "1d") {
    const values = linspace(view.lo, view.hi, steps);
    const methods = method === "slerp" && compare ? ["slerp", "linear"] : [method];
    const alpha = { param: "alpha", values };
    return {
      method,
      fixed: {},
      axes: methods.length > 1 ? [{ param: "method", values: methods }, alpha] : [alpha],
      heads: values.map((v) => `${pct(v)} B`),
      label: 72,
      rows: methods.map((m) => ({
        label: m === "slerp" ? "Slerp" : "Linear",
        dim: method === "slerp" && m === "linear",
        cells: values.map((v) => recipeOf(m, v)),
      })),
    };
  }
  if (view.kind === "grid") {
    const encs = linspace(view.enc[0], view.enc[1], grid);
    const decs = linspace(view.dec[0], view.dec[1], grid);
    return {
      method: "blockwise",
      fixed: { block_weights: { mid } },
      axes: [{ param: "encoder", values: encs }, { param: "decoder", values: decs }],
      heads: decs.map((d) => `Decoder ${pct(d)}`),
      headStage: "decoder",
      label: 128,
      rows: encs.map((e) => ({
        label: `Encoder ${pct(e)}`,
        stage: "encoder",
        cells: decs.map((d) => recipeOf("blockwise", 0, { encoder: e, mid, decoder: d })),
      })),
    };
  }
  const mids = linspace(0, 1, 5);
  return {
    method: "blockwise",
    fixed: { block_weights: { encoder: view.enc, decoder: view.dec } },
    axes: [{ param: "mid", values: mids }],
    heads: mids.map((m) => `Mid ${pct(m)}`),
    headStage: "mid",
    label: 72,
    rows: [{
      label: "Mid",
      stage: "mid",
      cells: mids.map((m) => recipeOf("blockwise", 0, { encoder: view.enc, mid: m, decoder: view.dec })),
    }],
  };
}

function crumbLabel(view) {
  if (view.kind === "1d") return `${pct(view.lo)}–${pct(view.hi)} B`;
  if (view.kind === "mid") return "Mid stage sweep";
  return view.enc[0] === 0 && view.enc[1] === 1 && view.dec[0] === 0 && view.dec[1] === 1
    ? "Whole grid"
    : `Encoder ${pct(view.enc[0])}–${pct(view.enc[1])}, decoder ${pct(view.dec[0])}–${pct(view.dec[1])}`;
}

function autoName(a, b, r) {
  const tag = r.method === "blockwise"
    ? `b${STAGES.map((s) => Math.round(r.block_weights[s.id] * 100)).join("-")}`
    : `${r.method === "slerp" ? "slerp" : ""}${Math.round(r.alpha * 100)}`;
  return `${a?.name || "a"}-x-${b?.name || "b"}-${tag}`.replace(/[^A-Za-z0-9_-]+/g, "-");
}

// The mid stage is the one stage the grid does not vary, so the control says
// what it holds still and why.
const MID_TIP = "The UNet has three stages: the encoder (structure), the mid stage "
  + "(the bottleneck between them) and the decoder (texture). The grid varies the "
  + "encoder down the rows and the decoder across the columns, so the mid stage "
  + "needs one value for the whole grid: this is how much of it comes from B in "
  + "every cell. To vary it, pick a cell and choose “Sweep the mid stage”.";

const SwapIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M7 4L3 8l4 4" /><path d="M3 8h14" /><path d="M17 20l4-4-4-4" /><path d="M21 16H7" />
  </svg>
);

export default function Merge() {
  const {
    toast, modelPath, setModelPath, stars, setAppMode, setPlayTab,
    models: allModels, refreshModels,
  } = useApp();
  const { sampleParams } = usePlay();

  // Everything lives in the Play provider: a ladder is a long run, and the
  // recipe you are tuning should still be there when you come back to the tab.
  const [b, setB] = usePlayState("merge.b", "");
  const [method, setMethod] = usePlayState("merge.method", "linear");
  const [steps, setSteps] = usePlayState("merge.steps", 5);
  const [compare, setCompare] = usePlayState("merge.compare", true);
  const [grid, setGrid] = usePlayState("merge.grid", 3);
  const [mid, setMid] = usePlayState("merge.mid", 0.5);
  const [views, setViews] = usePlayState("merge.views", START_VIEWS);
  const [cacheState, setCache] = usePlayState("merge.cache", EMPTY_CACHE);
  const [job, setJob] = usePlayState("merge.job", null);
  const [pick, setPick] = usePlayState("merge.pick", null);
  const [fine, setFine] = usePlayState("merge.fine", null);
  const [name, setName] = usePlayState("merge.name", null);
  const [keepRecipe, setKeepRecipe] = usePlayState("merge.keepRecipe", false);
  const [which, setWhich] = usePlayState("merge.which", "both");
  const [saving, setSaving] = usePlayState("merge.writing", false);
  const [result, setResult] = usePlayState("merge.result", null);
  const [compat, setCompat] = useState(null);
  // The recipe shown enlarged next to A and B, or null.
  const [big, setBig] = useState(null);

  const models = useMemo(
    () => tagStars(allModels || [], stars).filter((x) => x.role !== "checkpoint"),
    [allModels, stars],
  );
  const byPath = useMemo(() => Object.fromEntries(models.map((m) => [m.path, m])), [models]);
  const modelA = byPath[modelPath];
  const modelB = byPath[b];
  const optsA = useMemo(() => selectOptions(models), [models]);
  const optsB = useMemo(() => optsA.filter((o) => o.value !== modelPath), [optsA, modelPath]);

  useEffect(() => {
    if (!models.length) return;
    // Prefer a partner A can actually merge with: same engine and UNet shape.
    const a = byPath[modelPath];
    const others = models.filter((x) => x.path !== modelPath);
    const fits = (x) => a && x.backend === a.backend && x.mtype === a.mtype
      && String(x.mults) === String(a.mults) && (x.attn || null) === (a.attn || null);
    const alt = (others.find(fits) || others[0])?.path || "";
    setB((prev) => (prev && prev !== modelPath && byPath[prev] ? prev : alt));
  }, [models, modelPath, byPath, setB]);

  useEffect(() => {
    setCompat(null);
    if (!modelPath || !b) return;
    let live = true;
    api.post("/craft/merge/check", { model_a: modelPath, model_b: b })
      .then((c) => live && setCompat(c)).catch(() => {});
    return () => { live = false; };
  }, [modelPath, b]);

  // The samples a ladder holds are only good for these two models and these
  // sampling settings. The seed is kept apart: a blank Create seed is resolved
  // by the first ladder and then reused, so zooms and refines line up with it.
  const payload = useMemo(() => {
    const { seed, model_path: _m, batch_size: _b, ...rest } = buildSamplePayload(sampleParams, { postproc: {} });
    return { seed, rest };
  }, [sampleParams]);
  const settingsKey = useMemo(
    () => JSON.stringify({ a: modelPath, b, ...payload.rest }), [modelPath, b, payload.rest],
  );
  const userSeed = parseSeed(sampleParams.seed);
  const cacheValid = cacheState.key === settingsKey
    && (userSeed == null || cacheState.seed === userSeed);
  const cache = cacheValid ? cacheState : EMPTY_CACHE;

  const viewStack = views[method] || START_VIEWS[method];
  const view = viewStack[viewStack.length - 1];
  const ladder = useMemo(
    () => ladderFor(method, view, { steps, compare, grid, mid }),
    [method, view, steps, compare, grid, mid],
  );

  const running = LIVE.includes(job?.status);
  const incompatible = compat && !compat.compatible;
  const ready = !!(modelPath && b) && !incompatible;

  const shotOf = (r) => {
    const end = endOf(r);
    return end ? cache.refs[end] : cache.cells[recipeKey(r)];
  };
  // What a render of this view would still have to sample.
  const missing = useMemo(() => {
    const keys = new Set();
    ladder.rows.forEach((row) => row.cells.forEach((r) => {
      if (!endOf(r) && !cache.cells[recipeKey(r)]) keys.add(recipeKey(r));
    }));
    return keys.size;
  }, [ladder, cache]);
  const refsMissing = !(cache.refs.a && cache.refs.b);
  const newCount = missing + (refsMissing ? 2 : 0);

  const run = async ({ fresh = false, only = null } = {}) => {
    if (!ready) return;
    const usable = cacheValid && !fresh;
    const spec = only
      ? { method: only.method, fixed: { alpha: only.alpha, block_weights: only.block_weights }, axes: [] }
      : ladder;
    const known = usable
      ? (only ? [] : ladder.rows.flatMap((row) => row.cells)).filter((r) => cache.cells[recipeKey(r)])
      : [];
    const seed = fresh ? null : (userSeed ?? (usable ? cache.seed : null));
    try {
      const { job: j } = await api.post("/craft/merge/ladder", {
        model_a: modelPath, model_b: b,
        method: spec.method, fixed: spec.fixed, axes: spec.axes, known,
        refs: !(usable && cache.refs.a && cache.refs.b),
        sample: { ...payload.rest, seed },
      });
      const reqKey = settingsKey;
      const reqSeed = j.detail.seed;
      setCache((prev) => (usable && prev.key === reqKey && prev.seed === reqSeed
        ? prev : { key: reqKey, seed: reqSeed, refs: {}, cells: {} }));
      setJob(j);
      // Polls send each picture once (pollJob's delta), so the cache is the
      // only place a finished sample is kept.
      const absorb = (snap) => {
        setJob(snap);
        setCache((prev) => {
          if (prev.key !== reqKey || prev.seed !== reqSeed) return prev;
          const refs = { ...prev.refs };
          Object.entries(snap.detail?.refs || {}).forEach(([k, v]) => {
            if (v?.image) refs[k] = { image: v.image, card: v.card };
          });
          const cells = { ...prev.cells };
          (snap.detail?.cells || []).forEach((c) => {
            if (c.image) cells[recipeKey(c.recipe)] = { image: c.image, card: c.card };
          });
          return { ...prev, refs, cells };
        });
      };
      const done = await pollJob(j.id, absorb, 600, { delta: true });
      if (done.status === "error") toast(done.message || "The ladder failed", "error");
      else if (done.status === "done" && only) { setPick(only); setFine(null); }
    } catch (e) {
      toast(e.message, "error");
    }
  };

  const stop = async () => { if (job) await api.post(`/jobs/${job.id}/cancel`).catch(() => {}); };

  const setView = (next) => setViews((prev) => ({ ...prev, [method]: next }));
  const pushView = (v) => setView([...viewStack, v]);

  const chosen = fine || pick;
  const chosenShot = chosen ? shotOf(chosen) : null;
  const sampledSlot = sampleParams.ema === false ? "model" : "ema";
  const showWhich = modelA?.ema === "distinct" && modelB?.ema === "distinct";
  const slotKept = showWhich && which !== "both" && which !== sampledSlot;
  const outName = name ?? (chosen ? autoName(modelA, modelB, chosen) : "merge");

  // Saved merge recipes: the mix and both models, like a saved bend stack.
  const [recipes, setRecipes] = useState([]);
  const [recipesOpen, setRecipesOpen] = useState(false);
  const loadRecipes = () => api.get("/library/recipes").then(setRecipes).catch(() => {});
  useEffect(() => { loadRecipes(); }, []);

  const saveRecipe = async () => {
    if (!chosen) return;
    try {
      // The picked sample is of exactly this mix on these two models (the cache
      // is only valid for them), so it can stand for the recipe.
      const thumbnail = chosenShot ? await presetThumb(chosenShot.image) : null;
      await api.post("/craft/merge/recipes", {
        name: outName, model_a: modelPath, model_b: b, ...chosen, thumbnail,
      });
      toast(`Saved the recipe “${outName}”. Use it in Create with “Merge with”.`, "success");
      loadRecipes();
    } catch (e) { toast(e.message, "error"); }
  };

  /** Show a saved recipe: its models, method and mix, picked on the ladder. */
  const loadRecipe = (r) => {
    const mix = mixOf(r);
    const got = [];
    if (r.model_a && byPath[r.model_a] && r.model_a !== modelPath) { setModelPath(r.model_a); got.push(`A ${byPath[r.model_a].name}`); }
    if (r.model_b && byPath[r.model_b] && r.model_b !== b) { setB(r.model_b); got.push(`B ${byPath[r.model_b].name}`); }
    setMethod(mix.method);
    setViews((prev) => ({ ...prev, [mix.method]: START_VIEWS[mix.method] }));
    if (mix.method === "blockwise" && [0, 0.25, 0.5, 0.75, 1].includes(mix.block_weights.mid)) setMid(mix.block_weights.mid);
    setPick(mix);
    setFine(null);
    setName(r.name);
    setRecipesOpen(false);
    const lost = r.model_b && !byPath[r.model_b] ? " Its model B is no longer in the library." : "";
    toast(`Loaded “${r.name}”${got.length ? `: ${got.join(", ")}` : ""}.${lost}`, lost ? "warn" : "success");
  };

  // Library ▸ Merges ▸ Open in Merge leaves the recipe's name for this tab.
  useEffect(() => {
    if (!models.length) return undefined;
    const open = async () => {
      const name = takeMergeOpen();
      if (!name) return;
      const list = await api.get("/library/recipes").catch(() => []);
      setRecipes(list);
      const r = list.find((x) => x.name === name);
      if (r) loadRecipe(r);
    };
    open();
    window.addEventListener(MERGE_OPEN_EVENT, open);
    return () => window.removeEventListener(MERGE_OPEN_EVENT, open);
  }, [models.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const save = async () => {
    if (!chosen) return;
    setSaving(true);
    try {
      const res = await api.post("/craft/merge", {
        model_a: modelPath, model_b: b, out_name: outName,
        method: chosen.method, alpha: chosen.alpha ?? 0.5,
        block_weights: chosen.block_weights || {},
        which: showWhich ? which : "both", save_recipe: keepRecipe,
        thumbnail: chosenShot?.image || null, card: chosenShot?.card || null,
        recipe_thumbnail: keepRecipe && chosenShot ? await presetThumb(chosenShot.image) : null,
      });
      if (keepRecipe) loadRecipes();
      setResult(res);
      if (res.thumbnail_skipped) toast(`Saved ${res.name}. No thumbnail: ${res.thumbnail_skipped}`, "warn");
      else toast(`Saved ${res.name}${res.thumbnail ? " with the picked sample as its thumbnail" : ""}`, "success");
      await refreshModels({ force: true });
    } catch (e) {
      toast(e.message, "error");
    }
    setSaving(false);
  };

  // The job's current cell, so its live frame can stand in for the picture.
  const liveKey = (() => {
    const cur = job?.detail?.current;
    if (!running || cur == null) return null;
    if (cur === "a" || cur === "b") return cur;
    const c = job.detail.cells?.[cur];
    return c ? recipeKey(c.recipe) : null;
  })();
  const queuedKeys = useMemo(() => new Set(running
    ? (job?.detail?.cells || []).filter((c) => c.order != null && c.rev == null).map((c) => recipeKey(c.recipe))
    : []), [running, job]);

  const cellProps = (r, rowLabel) => {
    const end = endOf(r);
    const key = end || recipeKey(r);
    const shot = shotOf(r);
    const picked = !!chosen && recipeKey(chosen) === recipeKey(r);
    let tag = end === "a" ? "model A" : end === "b" ? "model B" : "";
    if (!tag && r.method === "blockwise" && ladder.headStage === "decoder") tag = gridTag(r);
    if (picked) tag = tag ? `picked · ${tag}` : "picked";
    return {
      key,
      shot,
      live: liveKey === key ? job?.detail?.frame : null,
      queued: !shot && (queuedKeys.has(key) || (running && end && !cache.refs[end])),
      picked,
      src: !!end,
      tag,
      label: `${rowLabel}, ${r.method === "blockwise" ? describe(r) : `${pct(r.alpha)} B`}`,
      onPick: () => { setPick(r); setFine(null); },
      onEnlarge: shot ? () => setBig(r) : null,
    };
  };

  // What the enlarged view steps through: every rendered picture in this
  // view, in reading order, once each. A previewed mix that is not on the
  // ladder (a fine-tuned pick) stands alone.
  const bigList = useMemo(() => {
    const seen = new Set();
    const out = [];
    ladder.rows.forEach((row) => row.cells.forEach((r) => {
      const k = recipeKey(r);
      if (seen.has(k) || !shotOf(r)) return;
      seen.add(k);
      out.push(r);
    }));
    if (big && !seen.has(recipeKey(big)) && shotOf(big)) return [big];
    return out;
  }, [ladder, cache, big]); // eslint-disable-line react-hooks/exhaustive-deps
  const bigAt = big ? bigList.findIndex((r) => recipeKey(r) === recipeKey(big)) : -1;

  // Zoom and refine offers, around the picked cell when it is in this view.
  const zooms = (() => {
    if (!pick) return [];
    if (view.kind === "1d") {
      const row = ladder.rows.find((rw) => rw.cells.some((r) => recipeKey(r) === recipeKey(pick)));
      if (!row) return [];
      const vals = row.cells.map((r) => r.alpha);
      const i = vals.indexOf(pick.alpha);
      const out = [];
      if (i > 0) out.push({ label: `Between ${pct(vals[i - 1])} and ${pct(vals[i])}`, view: { kind: "1d", lo: vals[i - 1], hi: vals[i] } });
      if (i < vals.length - 1) out.push({ label: `Between ${pct(vals[i])} and ${pct(vals[i + 1])}`, view: { kind: "1d", lo: vals[i], hi: vals[i + 1] } });
      return out;
    }
    if (pick.method !== "blockwise") return [];
    const { encoder: e, decoder: d } = pick.block_weights;
    const out = [];
    if (view.kind === "grid") {
      const half = (lohi) => (lohi[1] - lohi[0]) / Math.max(grid - 1, 1) / 2;
      const he = half(view.enc);
      const hd = half(view.dec);
      if (he >= 0.01 && hd >= 0.01) {
        out.push({
          label: "A finer grid around the picked cell",
          view: { kind: "grid", enc: [round4(clamp01(e - he)), round4(clamp01(e + he))], dec: [round4(clamp01(d - hd)), round4(clamp01(d + hd))] },
        });
      }
    }
    if (view.kind !== "mid") out.push({ label: "Sweep the mid stage for the picked cell", view: { kind: "mid", enc: e, dec: d } });
    return out;
  })();
  // How many merges a zoom would add, so the offer says what it costs.
  const newIn = (v) => {
    const keys = new Set();
    ladderFor(method, v, { steps, compare, grid, mid }).rows.forEach((row) => row.cells.forEach((r) => {
      if (!endOf(r) && !cache.cells[recipeKey(r)]) keys.add(recipeKey(r));
    }));
    return keys.size;
  };

  const renderLabel = running
    ? `Rendering ${Math.round((job?.progress || 0) * (job?.detail?.planned || 0))} of ${job?.detail?.planned || 0}…`
    : newCount > 0 ? "Render ladder" : userSeed == null ? "New seed" : "Up to date";
  const renderTip = !running && newCount === 0 && userSeed == null
    ? "Everything here is rendered. Render the ladder again on a new random seed."
    : undefined;

  const methodTabs = METHODS.map((m) => ({ id: m.id, label: m.label, tip: m.tip }));
  const isGrid = method === "blockwise";

  return (
    <div className="col">
      <div className="merge-head">
        <h3 className="merge-title">Merge</h3>
        <div className="merge-pick">
          <span className="merge-badge" aria-hidden="true">A</span>
          <select aria-label="Model A" value={modelPath || ""} onChange={(e) => setModelPath(e.target.value)}>
            {!modelPath && <option value="">— pick model A —</option>}
            {optsA.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </div>
        <button type="button" className="btn icon" aria-label="Swap A and B" title="Swap A and B"
          disabled={!modelPath || !b}
          onClick={() => { const a = modelPath; setModelPath(b); setB(a); }}>
          <SwapIcon />
        </button>
        <div className="merge-pick">
          <span className="merge-badge" aria-hidden="true">B</span>
          <select aria-label="Model B" value={b} onChange={(e) => setB(e.target.value)}>
            {!optsB.length && <option value="">— no other library models —</option>}
            {optsB.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </div>
        {compat && (compat.compatible
          ? <span className="pill good">Compatible</span>
          : <span className="pill bad">Incompatible</span>)}
      </div>
      {incompatible && (
        <ul className="hint merge-reasons">{compat.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
      )}

      <section className="card merge-ladder" aria-label="Blend ladder">
        <div className="merge-bar">
          <h3 className="mb-0">Blend ladder</h3>
          <Popover
            label="Saved merge recipes"
            triggerClass="btn sm"
            trigger={<><BookmarkIcon /> Saved{recipes.length ? ` (${recipes.length})` : ""}</>}
            open={recipesOpen}
            onOpenChange={setRecipesOpen}
            panelClass="bend-presets-pop merge-recipes-pop"
          >
            <div className="bend-presets">
              <p className="sub preset-note mb-0">
                A recipe keeps the mix and both models. Load one to see it on the ladder; in Create,
                &ldquo;Merge with&rdquo; blends its B into whatever model is selected.
              </p>
              {recipes.length === 0 ? (
                <p className="sub mb-0">Nothing saved yet. Pick a mix, then &ldquo;Save recipe&rdquo; below the ladder.</p>
              ) : (
                <ul className="preset-rows">
                  {recipes.map((r) => (
                    <li key={r.name} title={`${r.name}\n${recipeSummary(r, allModels)}`}>
                      {r.thumbnail
                        ? <img className="preset-row-thumb" src={r.thumbnail} alt="" loading="lazy" />
                        : <span className="preset-row-thumb" aria-hidden="true" />}
                      <span className="preset-row-name">{r.name}</span>
                      <span className="sub preset-row-sum">{recipeSummary(r, allModels)}</span>
                      <button type="button" className="btn xs" onClick={() => loadRecipe(r)}>Load</button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </Popover>
          <Seg tabs={methodTabs} value={method} onChange={setMethod} ariaLabel="Merge method" size="sm" />
          {!isGrid && (
            <label className="merge-inline">Steps
              <select value={steps} onChange={(e) => setSteps(Number(e.target.value))}>
                {[5, 7, 9].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
          )}
          {method === "slerp" && (
            <label className="merge-inline">
              <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
              Compare with linear
            </label>
          )}
          {isGrid && (
            <>
              <label className="merge-inline">Grid
                <select value={grid} onChange={(e) => setGrid(Number(e.target.value))}>
                  <option value={3}>3 × 3</option>
                  <option value={5}>5 × 5</option>
                </select>
              </label>
              <label className="merge-inline">
                <span className="merge-dot stage-mid" aria-hidden="true" />
                <TipLabel tip={MID_TIP}>Mid stage, every cell</TipLabel>
                <select value={mid} disabled={view.kind === "mid"} aria-label="Share of model B in the mid stage, for every cell"
                  onChange={(e) => setMid(Number(e.target.value))}>
                  {[0, 0.25, 0.5, 0.75, 1].map((v) => <option key={v} value={v}>{pct(v)} B</option>)}
                </select>
              </label>
            </>
          )}
          <span className="spacer" />
          <span className="sub">
            Seed {userSeed ?? cache.seed ?? "random"} · {payload.rest.steps} steps ·{" "}
            {newCount ? `${missing} new merge${missing === 1 ? "" : "s"}${refsMissing ? " + A and B" : ""}` : "all rendered"}
          </span>
          {running && <button type="button" className="btn" onClick={stop}>Stop</button>}
          <button
            type="button"
            className={`btn ${newCount > 0 && !running ? "primary" : ""}`}
            title={renderTip}
            disabled={!ready || running || (newCount === 0 && userSeed != null)}
            onClick={() => run({ fresh: newCount === 0 })}
          >
            {renderLabel}
          </button>
        </div>
        <p className="hint mb-0">{METHOD_TIPS[method]}</p>
        <p className="hint mb-0">
          No bends here: merging blends the two models&rsquo; weights, while bends rewrite activations as an
          image forms. The ladder renders A, B and the mixes without bends, and a saved merge carries none.
          Add bends to the new model afterwards, in Create or Bend.
        </p>

        {viewStack.length > 1 && (
          <nav className="merge-crumbs" aria-label="Ladder zoom">
            {viewStack.map((v, i) => (i === viewStack.length - 1
              ? <span key={i} aria-current="true">{crumbLabel(v)}</span>
              : (
                <React.Fragment key={i}>
                  <button type="button" onClick={() => setView(viewStack.slice(0, i + 1))}>
                    {crumbLabel(v)}
                  </button>
                  <span aria-hidden="true">›</span>
                </React.Fragment>
              )))}
          </nav>
        )}

        <div className={isGrid ? "merge-grid-wrap" : ""}>
          <div className="merge-rows" style={{ "--n": ladder.heads.length, "--label": `${ladder.label}px` }}>
            <div className="merge-row merge-heads" aria-hidden="true">
              <span />
              {ladder.heads.map((h) => (
                <span key={h}>
                  {ladder.headStage && <span className={`merge-dot stage-${ladder.headStage}`} />}
                  {h}
                </span>
              ))}
            </div>
            {ladder.rows.map((row, y) => (
              <div className="merge-row" key={row.label}>
                <span className={`merge-rowlabel ${row.dim ? "dim" : ""}`}>
                  {row.stage && <span className={`merge-dot stage-${row.stage}`} aria-hidden="true" />}
                  {row.label}
                </span>
                {row.cells.map((r) => {
                  if (y > 0 && !isGrid && endOf(r)) {
                    return <div className="merge-same" key={recipeKey(r)}>Same as the row above: {endOf(r).toUpperCase()}</div>;
                  }
                  const c = cellProps(r, row.label);
                  return <LadderCell key={c.key} {...c} />;
                })}
              </div>
            ))}
          </div>
          {isGrid && (
            <aside className="merge-refs">
              <div className="section-title">For reference</div>
              <div className="merge-refs-pair">
                {["a", "b"].map((k) => (
                  <figure key={k}>
                    <div className="merge-shot-sq">
                      {cache.refs[k]
                        ? <img src={cache.refs[k].image} alt={`Sample from model ${k.toUpperCase()}`} />
                        : liveKey === k && job?.detail?.frame
                          ? <img src={job.detail.frame} alt="" className="live" />
                          : <span className="sub">{k.toUpperCase()}</span>}
                    </div>
                    <figcaption className="sub"><b>{k.toUpperCase()}</b> on its own</figcaption>
                  </figure>
                ))}
              </div>
              <p className="sub mb-0">
                {view.kind === "mid"
                  ? "Encoder and decoder stay at the picked cell; only the mid stage moves."
                  : mid === 0 || mid === 1
                    ? `The mid stage is ${pct(mid)} B in every cell, so the matching corner is ${mid === 0 ? "A" : "B"} itself.`
                    : `The mid stage is ${pct(mid)} B in every cell, so even the corners are mixes, not A or B themselves.`}
              </p>
            </aside>
          )}
        </div>

        {zooms.length > 0 ? (
          <div className="merge-next" role="group" aria-label={isGrid ? "Refine around the pick" : "Zoom in around the pick"}>
            <div className="merge-next-head">
              <b>{isGrid ? "Refine around your pick" : "Zoom in around your pick"}</b>
              <span className="sub">A closer look at the mixes next to the one you picked.</span>
            </div>
            <div className="merge-next-opts">
              {zooms.map((z) => {
                const n = newIn(z.view);
                return (
                  <button key={z.label} type="button" className="btn merge-next-btn" onClick={() => pushView(z.view)}>
                    <span className="merge-next-label">{z.label}</span>
                    <span className="sub">{n ? `${n} new merge${n === 1 ? "" : "s"}` : "already rendered"}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <p className="merge-next-empty mb-0">
            Click a {isGrid ? "cell" : "step"} to pick it{view.kind === "mid" ? "." : `, then ${isGrid ? "refine around it" : "zoom in around it"}.`}
            {" "}Double-click a picture, or use its <ExpandIcon size={12} aria-hidden="true" /> button, to see it large between A and B.
          </p>
        )}
      </section>

      <div className="merge-bottom">
        <section className="card merge-picked" aria-label="Picked mix">
          <div className="row center gap-2">
            <h3 className="mb-0">Picked mix</h3>
            <span className="sub">{chosen ? describe(chosen) : "Nothing picked yet"}</span>
          </div>
          {chosen && (
            <div className="merge-picked-body">
              {chosenShot ? (
                <button type="button" className="merge-shot-sq small merge-enlarge-thumb"
                  aria-label="Enlarge the picked sample between A and B" title="Enlarge between A and B"
                  onClick={() => setBig(chosen)}>
                  <img src={chosenShot.image} alt="The picked sample" />
                </button>
              ) : (
                <div className="merge-shot-sq small"><span className="sub">not previewed</span></div>
              )}
              <div className="merge-fine">
                {chosen.method === "blockwise"
                  ? STAGES.map((s) => (
                    <FineSlider
                      key={s.id}
                      label={<><span className={`merge-dot stage-${s.id}`} aria-hidden="true" />{s.label}</>}
                      aria={`${s.label}: share of model B`}
                      value={chosen.block_weights[s.id]}
                      onChange={(v) => setFine(recipeOf("blockwise", 0, { ...chosen.block_weights, [s.id]: v }))}
                    />
                  ))
                  : (
                    <FineSlider
                      label="A ↔ B"
                      aria="Share of model B"
                      value={chosen.alpha}
                      onChange={(v) => setFine(recipeOf(chosen.method, v))}
                    />
                  )}
                <div className="row center gap-2 wrap">
                  {!chosenShot && (
                    <button type="button" className="btn" disabled={!ready || running} onClick={() => run({ only: chosen })}>
                      Preview this mix
                    </button>
                  )}
                  {fine && (
                    <button type="button" className="btn ghost" onClick={() => setFine(null)}>Back to the picked step</button>
                  )}
                </div>
                <p className="sub mb-0">
                  Fine-tune in 1% steps. A mix that is not on the ladder needs a preview before it can be the thumbnail.
                </p>
              </div>
            </div>
          )}
        </section>

        <section className="card merge-save" aria-label="Save">
          <h3>Save the picked mix</h3>
          <div className="merge-name">
            <input type="text" aria-label="Name for the new model or recipe" value={outName}
              onChange={(e) => setName(e.target.value)} />
            <span className="sub">.pt</span>
          </div>
          {showWhich && (
            <label className="merge-inline mt-2">Weights to merge
              <select value={which} onChange={(e) => setWhich(e.target.value)}>
                <option value="both">EMA + raw</option>
                <option value="ema">EMA only</option>
                <option value="model">Raw only</option>
              </select>
            </label>
          )}
          <div className="row center gap-2 wrap mt-2">
            <label className="merge-inline">
              <TipLabel tip="Also keep this mix as a merge recipe under the same name, as “Save recipe” does.">
                <input type="checkbox" checked={keepRecipe} onChange={(e) => setKeepRecipe(e.target.checked)} />
                Keep recipe
              </TipLabel>
            </label>
            <span className="spacer" />
            <button type="button" className="btn"
              title="Keep the mix and both models as a recipe, without writing a model file. Create can use it with “Merge with”."
              disabled={!chosen || !ready || saving}
              onClick={saveRecipe}>
              Save recipe
            </button>
            <button type="button"
              className={`btn ${chosenShot && newCount === 0 && !saving ? "primary" : ""}`}
              disabled={!chosen || !ready || saving}
              onClick={save}>
              {saving ? "Saving…" : "Save model"}
            </button>
          </div>
          <p className="sub mb-0 mt-1">
            {!chosen ? "Pick a step on the ladder to save it."
              : slotKept ? `No thumbnail: the samples show ${sampledSlot === "ema" ? "EMA" : "raw"} weights, which this choice keeps from model A.`
                : chosenShot ? "The picked sample becomes the new model's thumbnail."
                  : "Preview this mix first to give the new model a thumbnail."}
          </p>
          {result && (
            <div className="row center gap-2 wrap mt-2">
              <span className="sub">Saved <b>{result.name}</b></span>
              <button type="button" className="btn sm"
                onClick={() => {
                  setModelPath(result.path);
                  setAppMode("play");
                  setPlayTab("create");
                  toast(`Using ${result.name} in Create`, "success");
                }}>
                Use in Create
              </button>
            </div>
          )}
        </section>
      </div>

      {big && bigAt >= 0 && (
        <MergeLightbox
          list={bigList}
          at={bigAt}
          onAt={(i) => setBig(bigList[i])}
          shotOf={shotOf}
          refs={cache.refs}
          nameA={modelA?.name || "Model A"}
          nameB={modelB?.name || "Model B"}
          pickedKey={chosen ? recipeKey(chosen) : null}
          onPick={(r) => { setPick(r); setFine(null); }}
          onClose={() => setBig(null)}
        />
      )}
    </div>
  );
}

/** One merge, large, with A on its left and B on its right. */
function MergeLightbox({ list, at, onAt, shotOf, refs, nameA, nameB, pickedKey, onPick, onClose }) {
  const r = list[at];
  const shot = shotOf(r);
  const end = endOf(r);
  const picked = pickedKey === recipeKey(r);
  const step = (d) => { const i = at + d; if (i >= 0 && i < list.length) onAt(i); };

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
      if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const side = (k, name) => (
    <figure className="merge-big-side">
      <div className="merge-big-shot">
        {refs[k] ? <img src={refs[k].image} alt={`Sample from model ${k.toUpperCase()}`} /> : <span className="sub">not rendered</span>}
      </div>
      <figcaption><span className="merge-badge" aria-hidden="true">{k.toUpperCase()}</span>{name}</figcaption>
    </figure>
  );

  return (
    <Modal
      title={end ? `Model ${end.toUpperCase()} itself` : describe(r)}
      wide
      onClose={onClose}
      footer={(
        <>
          {list.length > 1 && (
            <>
              <button type="button" className="btn" disabled={at === 0} onClick={() => step(-1)} aria-label="Previous merge">‹ Previous</button>
              <span className="sub merge-big-count">{at + 1} of {list.length}</span>
              <button type="button" className="btn" disabled={at === list.length - 1} onClick={() => step(1)} aria-label="Next merge">Next ›</button>
              <span className="spacer" />
            </>
          )}
          <button type="button" className="btn primary" disabled={picked} onClick={() => onPick(r)}>
            {picked ? "Picked" : "Pick this mix"}
          </button>
          <button type="button" className="btn ghost" onClick={onClose}>Close</button>
        </>
      )}
    >
      <div className="merge-big">
        {side("a", nameA)}
        <figure className="merge-big-mid">
          <div className={`merge-big-shot${picked ? " on" : ""}`}>
            {shot && <img src={shot.image} alt={end ? `Sample from model ${end.toUpperCase()}` : `Merge: ${describe(r)}`} />}
          </div>
          <figcaption>{end ? `Model ${end.toUpperCase()}, as rendered for the ladder` : describe(r)}</figcaption>
        </figure>
        {side("b", nameB)}
      </div>
    </Modal>
  );
}

function LadderCell({ shot, live, queued, picked, src, tag, label, onPick, onEnlarge }) {
  const img = shot?.image || live;
  return (
    <div className="merge-cell-wrap">
      <button
        type="button"
        className={`merge-cell${picked ? " on" : ""}${src ? " src" : ""}`}
        aria-pressed={picked}
        aria-label={label}
        disabled={!shot}
        onClick={onPick}
        onDoubleClick={onEnlarge || undefined}
      >
        <span className={`shot${!shot && live ? " live" : ""}`}>
          {img ? <img src={img} alt="" /> : <span className="wait">{queued ? "queued" : "–"}</span>}
        </span>
        <span className="tag">{tag || " "}</span>
      </button>
      {onEnlarge && (
        <button type="button" className="merge-cell-enlarge" onClick={onEnlarge}
          aria-label={`Enlarge ${label} between A and B`} title="Enlarge between A and B">
          <ExpandIcon size={13} />
        </button>
      )}
    </div>
  );
}

function FineSlider({ label, aria, value, onChange }) {
  const p = Math.round(value * 100);
  return (
    <div className="merge-fine-row">
      <span className="merge-fine-label">{label}</span>
      <span className="num">{100 - p}% A</span>
      <input type="range" min={0} max={100} step={1} value={p} aria-label={aria}
        onChange={(e) => onChange(Number(e.target.value) / 100)} />
      <span className="num">{p}% B</span>
    </div>
  );
}
