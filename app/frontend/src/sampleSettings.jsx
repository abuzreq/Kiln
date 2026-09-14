import React from "react";
import { ShuffleIcon } from "./components/icons.jsx";
import { Slider, Select, Text, Num, Disclose } from "./components/ui.jsx";

const STORAGE_KEY = "kiln.sampleParams";
// Bumped when a *default* changes in a way stored settings should follow. Only
// for defaults nobody would have chosen deliberately, since this overwrites what
// is on disk: v2 re-points the old UniPC default at DPM-Solver++, which can
// harmonize a region fill where UniPC cannot.
const PARAMS_VERSION = 2;

export const DEFAULT_SAMPLE_PARAMS = {
  image_size: 512,
  steps: 50,
  eta: 0.5,
  skip: 0,
  seed: "",
  text: "",
  // Not a switch and not a strength -- a prompt turns guidance on by existing,
  // and Guidance strength is the amount. This only balances a text prompt
  // against an image prompt, so it has no panel control.
  text_weight: 1,
  guidance_step: 0.02,
  guidance_power: 1,
  spherical: false,
  image_prompt_weight: 0,
  cuts: 0.5,
  noise_level: 1,
  attenuation: 1,
  ema: true,
  // DPM-Solver++, not UniPC: same speed class, but it is the only fast solver
  // that can resample, so region fills can harmonize (see RESAMPLE_SOLVERS).
  sampler: "dpmpp",
};

/** Solvers that can do RePaint resampling.
 *
 *  Mirrors the `resample` flag on SAMPLERS in app/core/engine/sampler.py, which
 *  is the source of truth — the backend refuses the others outright. The
 *  multistep solvers carry outputs from previous timesteps; jumping back up the
 *  schedule invalidates them, and only DPM++ has a state reset wired.
 */
export const RESAMPLE_SOLVERS = ["ddim", "dpmpp"];

export function solverCanResample(sampler) {
  return RESAMPLE_SOLVERS.includes(sampler || "ddim");
}

/** Range and label for each sweepable sampler parameter.
 *
 *  Shared by the Sample-settings sliders and Sweep's axis defaults so the two
 *  can never disagree about what a sensible span for a parameter is.
 */
export const PARAM_RANGES = {
  steps: { label: "Steps", min: 5, max: 200, step: 1, int: true },
  eta: { label: "Eta", min: 0, max: 1, step: 0.05 },
  // No panel control any more (see DEFAULT_SAMPLE_PARAMS); kept for the label
  // and for recipes written while it was a slider.
  text_weight: { label: "Text weight", min: 0, max: 100, step: 1 },
  guidance_step: { label: "Guidance strength", min: 0, max: 0.3, step: 0.005 },
  guidance_power: { label: "Guidance ramp", min: 0, max: 4, step: 0.25 },
  cuts: { label: "Detail ↔ Structure", min: 0, max: 1, step: 0.05 },
  // 1.0 means "leave the schedule alone". Below that the sampler injects
  // extra noise at every step, so the image gets grainier as the number
  // goes *down* — measured detail (total variation) rises from 0.039 at
  // 1.0 to 0.201 at 0.25. "Noise level" read as though 0 meant no noise.
  noise_level: { label: "Extra noise", min: 0, max: 1, step: 0.05 },
  // Scales the seed noise the run starts from, before anything else happens.
  // Unlike Extra noise this is a one-off at step 0, so it survives every mode:
  // pure generation, img2img, and the noise a painted region is filled with.
  attenuation: { label: "Noise attenuation", min: 0, max: 1, step: 0.05 },
  // A seed has no meaningful range — any integer is as valid as any other — so
  // this is just a comfortable default span of eight consecutive seeds.
  seed: { label: "Seed", min: 0, max: 7, step: 1, int: true },
};

// Display names for sweepable settings that are not sampler parameters and so
// have no PARAM_RANGES entry. Kept here rather than in Sweep.jsx because the
// Library list and the settings panel's "sweep is setting X" callout need the
// same names -- when they did not have them, both printed the raw wire key.
const EXTRA_PARAM_LABELS = {
  sampler: "Sampler",
  image_size: "Image size",
  text: "Prompt",
  spherical: "Spherical distance",
};

/** The one display name for any sweepable/sampling key. Never returns a raw key
 *  unless the key is genuinely unknown. */
export function paramLabel(key) {
  return PARAM_RANGES[key]?.label || EXTRA_PARAM_LABELS[key] || key;
}

// The knobs that only shape guidance once it is actually running. The prompt
// and its weight are what turn guidance on, so they stay visible either way —
// these are the ones with nothing to act on until it is.
export const GUIDANCE_PARAMS = ["guidance_step", "guidance_power", "cuts", "spherical"];

/** Is CLIP guidance switched on? A prompt is the switch. */
export function guidanceActive(params = {}) {
  return !!(params.text || "").trim();
}

/** Why a parameter cannot currently affect the output, or null if it can.
 *
 *  ``model`` is the descriptor for the model actually selected, because several
 *  of these knobs are only meaningful for some models — the sampler has always
 *  fallen back silently rather than saying so.
 */
export function inertReason(name, params = {}, model = null, overriddenBy = null) {
  const by = overriddenBy?.[name];
  if (by) return `${by} sets this, so the value here is not used.`;
  if (name === "eta" && (params.sampler || "ddim") !== "ddim") {
    return "Eta only affects DDIM. The fast solvers take a fixed path, so this does nothing with the current sampler.";
  }
  if (name === "ema" && model) {
    if (model.ema === "none") {
      return model.backend === "diffusers"
        ? "This model publishes a single set of weights, so there is nothing to switch between. "
          + "(Some Hugging Face models ship an averaged copy as a separate repo instead — the "
          + "google/ddpm-ema-* ones.)"
        : "This model has only one set of weights, so there is nothing to switch between.";
    }
    if (model.ema === "same") {
      return "This checkpoint's averaged weights are identical to its raw ones, so this changes "
        + "nothing. Training only starts averaging at step 2000 — snapshots saved before that "
        + "carry a straight copy.";
    }
  }
  if (GUIDANCE_PARAMS.includes(name) && !guidanceActive(params)) {
    return "Guidance is off until there is a prompt.";
  }
  return null;
}

/** Note about a size the selected model will not handle well, or null. */
export function imageSizeNote(size, model) {
  if (!model || !size) return null;
  const step = model.size_multiple || 16;
  if (size % step !== 0) {
    const snapped = Math.max(step, Math.floor(size / step) * step);
    return `This model works in multiples of ${step}, so ${size} is sampled at ${snapped}.`;
  }
  const native = model.sample_size;
  if (native && (size >= native * 2 || size <= native / 2)) {
    return `Trained at ${native}px. Sampling this far from that usually degrades the result.`;
  }
  return null;
}

/** Controls to hide, rather than explain, when they cannot do anything.
 *
 *  The distinction is whether the user can get the control back:
 *
 *  - Inert because of *another setting* (Eta needs DDIM; Extra noise is set by
 *    Create's Change slider) -- hide it. It is not a property of anything the
 *    user needs to understand, and a nearby hint says how to bring it back.
 *  - Inert because of *the model they loaded* (no EMA weights, or averaged
 *    weights identical to the raw ones) -- show it and say why. There is
 *    nothing to switch to, and the reason is worth knowing about the model.
 *
 *  The guidance sub-knobs follow the first rule: they appear once a prompt and a
 *  weight are set, because until then there is nothing for them to shape.
 */
export const HIDE_WHEN_INERT = new Set(["eta", "noise_level", ...GUIDANCE_PARAMS]);

export function paramRange(name) {
  return PARAM_RANGES[name] || null;
}

export function loadSampleParams() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SAMPLE_PARAMS, _v: PARAMS_VERSION };
    const stored = JSON.parse(raw);
    const out = { ...DEFAULT_SAMPLE_PARAMS, ...stored };
    if ((stored._v || 1) < 2 && stored.sampler === "unipc") out.sampler = "dpmpp";
    out._v = PARAMS_VERSION;
    return out;
  } catch {
    return { ...DEFAULT_SAMPLE_PARAMS, _v: PARAMS_VERSION };
  }
}

export function saveSampleParams(params) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
  } catch { /* ignore quota */ }
}

/** Merge a patch into the stored sampler params.
 *
 *  Play owns the live params, so anywhere outside it (the library drawer, the
 *  Start hub) writes straight to storage instead — that way a restore survives
 *  the switch into Play rather than being overwritten when it mounts.
 */
export function mergeStoredSampleParams(patch) {
  saveSampleParams({ ...loadSampleParams(), ...patch });
}

/** Longest side region fill will sample at before scaling back (see MAX_FILL_SIDE). */
export const MAX_FILL_SIDE = 1024;

/** Mirrors align_size / fill_size in app/core/engine/inpaint.py.
 *  Every dimension must divide by 2**len(mults) or the UNet's skip connections
 *  do not line up. */
export function fillSizeFor(width, height, mults) {
  if (!width || !height) return null;
  const step = 2 ** ((mults || [1, 2, 2, 2]).length);
  let [w, h] = [width, height];
  const longest = Math.max(w, h);
  if (longest > MAX_FILL_SIDE) {
    const k = MAX_FILL_SIDE / longest;
    w = Math.round(w * k);
    h = Math.round(h * k);
  }
  const snap = (v) => Math.max(step, Math.floor(v / step) * step);
  return { w: snap(w), h: snap(h) };
}

/** Map a 0–1 "change amount" onto the skip + noise pair img2img actually uses. */
export function changeToParams(change, steps, hasInit) {
  if (!hasInit) return { skip: 0, noise_level: 1 };
  const c = Math.min(1, Math.max(0, change));
  return {
    skip: Math.round((1 - c) * Math.max(steps - 4, 0) * 0.9),
    noise_level: 0.25 + c * 0.75,
  };
}

/** How many steps an img2img / region-fill run will actually take.
 *
 *  `Steps` sets the resolution of the noise schedule; `Change` decides how far
 *  along it you start. Keeping some of the init image means skipping the early,
 *  high-noise steps, so the run is shorter than the Steps box suggests — worth
 *  showing rather than leaving the user to wonder where the steps went.
 */
export function effectiveSteps(change, steps, hasInit) {
  const n = Math.max(0, Math.round(steps) || 0);
  if (!hasInit) return n;
  const { skip } = changeToParams(change, n, true);
  return Math.max(1, n - Math.min(skip, n));
}

/** How many of the schedule's steps a given Change amount skips over.
 *
 *  The counterpart to effectiveSteps, and the number worth showing: skipping is
 *  the actual mechanism behind "keep more of the init image", so saying how many
 *  steps are being skipped explains the slider better than any adjective.
 */
export function skippedSteps(change, steps, hasInit) {
  const n = Math.max(0, Math.round(steps) || 0);
  return n - effectiveSteps(change, n, hasInit);
}

/** Sampler settings carried by an image's embedded card, ready for setSampleParams. */
export function paramsFromCard(card) {
  const p = card?.params;
  if (!p) return null;
  const out = {};
  for (const k of Object.keys(DEFAULT_SAMPLE_PARAMS)) {
    if (p[k] !== undefined && p[k] !== null) out[k] = p[k];
  }
  // Cards written before the sampler was selectable were all rendered with
  // DDIM — pin it, or restoring them would silently produce a different image.
  if (out.sampler === undefined) out.sampler = "ddim";
  return Object.keys(out).length ? out : null;
}

/** Short human label for a card, e.g. "flowers · seed 418823". */
export function cardLabel(card) {
  if (!card) return "";
  const bits = [];
  if (card.model) bits.push(card.model);
  if (card.params?.seed != null) bits.push(`seed ${card.params.seed}`);
  if (card.bend_preset) bits.push(card.bend_preset);
  return bits.join(" · ");
}

/** A fresh seed, in the same range the sampler draws from when one is blank.
 *
 *  Written into the field rather than left empty on purpose: a blank seed is
 *  redrawn every run, so the image you liked cannot be got back. A pinned seed
 *  can. (Mirrors resolve_seed in app/core/engine/sampler.py.)
 */
export function randomSeed() {
  return Math.floor(Math.random() * 2 ** 31);
}

export function parseSeed(seed) {
  if (seed === "" || seed == null) return null;
  const n = parseInt(seed, 10);
  return Number.isNaN(n) ? null : n;
}

/** Build a /perform/sample or sweep base payload from shared params. */
export function buildSamplePayload(params, opts = {}) {
  const {
    model_path,
    bends = null,
    bend_preset = "",
    init_image = null,
    postproc = {},
    overrides = {},
    batch_size,
  } = opts;
  const merged = { ...params, ...overrides };
  const body = {
    model_path,
    image_size: merged.image_size,
    steps: merged.steps,
    eta: merged.eta,
    skip: merged.skip,
    seed: parseSeed(merged.seed),
    batch_size: Math.max(1, Math.min(4, Math.round(merged.batch_size || opts.batch_size || 1))),
    text: merged.text,
    text_weight: merged.text_weight,
    guidance_step: merged.guidance_step,
    guidance_power: merged.guidance_power,
    spherical: merged.spherical,
    image_prompt_weight: merged.image_prompt_weight,
    cuts: merged.cuts,
    noise_level: merged.noise_level,
    attenuation: merged.attenuation,
    ema: merged.ema,
    sampler: merged.sampler,
    postproc,
  };
  if (bends) body.bends = bends;
  if (bend_preset) body.bend_preset = bend_preset;
  if (init_image) body.init_image = init_image;
  return body;
}

/** Build a /perform/inpaint payload from shared params. */
export function buildInpaintPayload(params, opts = {}) {
  const {
    model_path,
    init_image,
    mask,
    bends = null,
    bend_preset = "",
    feather = 8,
    overrides = {},
    batch_size,
  } = opts;
  const merged = { ...params, ...overrides };
  const body = {
    model_path,
    init_image,
    mask,
    image_size: merged.image_size,
    steps: merged.steps,
    eta: merged.eta,
    skip: merged.skip,
    seed: parseSeed(merged.seed),
    batch_size: Math.max(1, Math.min(4, Math.round(batch_size ?? merged.batch_size ?? 1))),
    text: merged.text,
    text_weight: merged.text_weight,
    guidance_step: merged.guidance_step,
    guidance_power: merged.guidance_power,
    spherical: merged.spherical,
    image_prompt_weight: merged.image_prompt_weight,
    cuts: merged.cuts,
    noise_level: merged.noise_level,
    attenuation: merged.attenuation,
    ema: merged.ema,
    sampler: merged.sampler,
    resample: merged.resample || 1,
    jump_length: merged.jump_length || 0,
    feather,
  };
  if (bends) body.bends = bends;
  if (bend_preset) body.bend_preset = bend_preset;
  return body;
}

/** Sweep base dict (no model_path). Swept axes override matching keys per cell. */
export function buildSweepBase(params, postproc = {}) {
  return {
    image_size: params.image_size,
    steps: params.steps,
    eta: params.eta,
    skip: params.skip,
    seed: parseSeed(params.seed),
    text: params.text,
    text_weight: params.text_weight,
    guidance_step: params.guidance_step,
    guidance_power: params.guidance_power,
    spherical: params.spherical,
    image_prompt_weight: params.image_prompt_weight,
    cuts: params.cuts,
    noise_level: params.noise_level,
    attenuation: params.attenuation,
    ema: params.ema,
    sampler: params.sampler,
    postproc,
  };
}

// Short display names. The select wants a descriptor after the name; chips and
// running prose want the bare name. Both come from here so a sampler can never
// show as its raw wire id ("dpmpp") in one place and "DPM-Solver++" in another.
export const SAMPLER_LABELS = {
  unipc: "UniPC",
  dpmpp: "DPM-Solver++",
  deis: "DEIS",
  ddim: "DDIM",
};

export const samplerLabel = (id) => SAMPLER_LABELS[id] || id;

const SAMPLER_OPTIONS = [
  { value: "dpmpp", label: `${SAMPLER_LABELS.dpmpp} — fast (recommended)` },
  { value: "unipc", label: `${SAMPLER_LABELS.unipc} — fast, no region harmonize` },
  { value: "deis", label: `${SAMPLER_LABELS.deis} — fast` },
  { value: "ddim", label: `${SAMPLER_LABELS.ddim} — classic, slowest` },
];

/** Settings a paused run can be resumed with.
 *
 *  Mirrors what the sampler re-reads mid-run: the solver knobs it can adjust in
 *  place, plus every guidance setting (app/core/engine/sampler.py rebuilds the
 *  prompt embedding from scratch on any of those, which is what lets someone
 *  pause, retype the prompt, and resume onto it). One list, because the pause
 *  snapshot, the resume diff and the panel's editable set all have to agree.
 */
export const LIVE_PARAM_KEYS = [
  "steps", "seed", "eta", "noise_level",
  "text", "guidance_step", "guidance_power", "cuts", "spherical",
];

/** The live-editable controls actually on screen, by label.
 *
 * A control in HIDE_WHEN_INERT disappears when it cannot do anything, so any
 * notice that names one is only truthful if it checks first. Both the panel's
 * own "Paused — ..." line and Create's pause toast go through here so they
 * cannot drift apart again. The guidance knobs collapse into one phrase: naming
 * all six would bury the two that most runs actually touch.
 */
export function liveEditLabels(params, model, overriddenBy) {
  const labels = ["steps", "seed", "eta", "noise_level"]
    .filter((k) => !(HIDE_WHEN_INERT.has(k) && !!inertReason(k, params, model, overriddenBy)))
    .map((k) => PARAM_RANGES[k]?.label || k);
  labels.push(guidanceActive(params) ? "the guidance settings" : "the prompt");
  return labels;
}

/** "a, b and c" — an empty list reads as "nothing". */
export function joinLabels(labels, join) {
  if (!labels.length) return "nothing";
  if (labels.length === 1) return labels[0];
  return labels.slice(0, -1).join(", ") + ` ${join} ` + labels[labels.length - 1];
}

export function SampleSettingsPanel({
  params,
  setParam,
  title = "Sample settings",
  hint = "Applies to Create, Bend, Merge, and Sweep.",
  disabled = false,
  editable = null,
  running = false,
  stepsMin,
  sweptBy = null,
  model = null,
  overriddenBy = null,
}) {
  // A sweep decides these per cell, so the value shown here does not apply.
  const swept = (key) => !!sweptBy?.includes(key);
  const can = (key) => {
    if (disabled) return false;
    if (swept(key)) return false;
    if (editable == null) return true;
    return editable.includes(key);
  };
  const inert = (key) => inertReason(key, params, model, overriddenBy);
  const hidden = (key) => HIDE_WHEN_INERT.has(key) && !!inert(key);
  const liveList = (join) => joinLabels(liveEditLabels(params, model, overriddenBy), join);
  const sizeNote = imageSizeNote(params.image_size, model);
  const sweptNote = (key, tip) => (swept(key)
    ? "The sweep grid sets this per cell — change it on the axis, not here."
    : tip);
  const frozenTip = "Only applies to the next generation; it cannot change while one is sampling.";
  const tipFor = (key, tip) => {
    const dead = inert(key);
    if (dead) return `${dead}\n\n${tip}`;
    if (swept(key)) return sweptNote(key, tip);
    return can(key) ? tip : frozenTip;
  };

  return (
    <div className="card">
      <h3>{title}</h3>
      {hint && <p className="hint mb-0">{hint}</p>}
      {disabled && <p className="sub mb-2">Frozen while generating — Pause to edit {liveList("or")}.</p>}
      {sweptBy?.length > 0 && (
        <p className="callout mb-2">
          The sweep grid is setting{" "}
          <b>{sweptBy.map(paramLabel).join(" and ")}</b>{" "}
          per cell. Everything else here still applies to every cell.
        </p>
      )}
      {editable && !disabled && (
        <p className="sub mb-2">
          {running
            ? <>Sampling — {liveList("and")} can still change; Pause, then Resume to apply them. The rest waits for the next run.</>
            : <>Paused — {liveList("and")} apply on Resume.</>}
        </p>
      )}
      <div className="row gap-3 wrap sample-steps-seed">
        <div className="grow">
          <Slider
            label={swept("steps") ? "Steps — set by the sweep" : "Steps"}
            value={params.steps}
            min={stepsMin != null ? Math.max(PARAM_RANGES.steps.min, stepsMin) : PARAM_RANGES.steps.min}
            max={PARAM_RANGES.steps.max}
            step={PARAM_RANGES.steps.step}
            onChange={(v) => setParam("steps", v)}
            disabled={!can("steps")}
            tip={tipFor("steps", "How many denoising steps. More is slower and usually cleaner.")}
          />
        </div>
        <div className="sample-seed">
          <div className="row gap-1 seed-row">
            <div className="grow">
              <Num
                label={swept("seed") ? "Seed — set by the sweep" : "Seed"}
                value={params.seed}
                onChange={(v) => setParam("seed", v)}
                disabled={!can("seed")}
                tip={tipFor("seed",
                  "Same seed + settings = same image. Leave empty to draw a fresh one every run; "
                  + "the shuffle button pins a new one you can keep and come back to.")}
              />
            </div>
            <button
              type="button"
              className="btn ghost seed-random"
              onClick={() => setParam("seed", randomSeed())}
              disabled={!can("seed")}
              title="Pin a new random seed"
              aria-label="Pin a new random seed"
            >
              <ShuffleIcon />
            </button>
          </div>
        </div>
      </div>
      <Disclose title="Advanced" tip="Less-used sampling knobs.">
        <Select
          label="Sampler"
          value={params.sampler || "dpmpp"}
          onChange={(v) => setParam("sampler", v)}
          options={SAMPLER_OPTIONS}
          disabled={!can("sampler")}
          tip={tipFor("sampler",
            "Which ODE solver walks the noise back to an image. The fast solvers reach "
            + "comparable detail in roughly a third of the steps, but give a different look — "
            + "the choice is saved into each PNG so results stay reproducible.")}
        />
        {params.sampler && params.sampler !== "ddim" && (
          <p className="hint mb-2">
            Fast solver — try 15–25 steps. Eta and live step changes are DDIM-only.
          </p>
        )}
        {!hidden("eta") && (
          <Slider label={PARAM_RANGES.eta.label}
            value={params.eta}
            min={PARAM_RANGES.eta.min} max={PARAM_RANGES.eta.max} step={PARAM_RANGES.eta.step}
            onChange={(v) => setParam("eta", v)}
            disabled={!can("eta")}
            tip={tipFor("eta", "How much extra noise is mixed in at each step. DDIM only.")} />
        )}
        <Num label="Image size" value={params.image_size} onChange={(v) => setParam("image_size", v)}
          step={model?.size_multiple || 64}
          disabled={!can("image_size")}
          tip={tipFor("image_size",
            "Output resolution for new generations (square). Region fill ignores this — "
            + "it works at your canvas's own size and aspect."
            + (model?.size_multiple ? `

This model works in multiples of ${model.size_multiple}.` : "")
            + (model?.sample_size ? ` It was trained at ${model.sample_size}px.` : ""))} />
        {sizeNote && <p className="hint mb-2">{sizeNote}</p>}
        {!hidden("noise_level") && (
          <Slider
            label={PARAM_RANGES.noise_level.label}
            value={params.noise_level}
            min={PARAM_RANGES.noise_level.min} max={PARAM_RANGES.noise_level.max}
            step={PARAM_RANGES.noise_level.step}
            fmt={(v) => (v >= 1 ? "none" : `+${Math.round((1 - v) * 100)}% grain`)}
            onChange={(v) => setParam("noise_level", v)}
            disabled={!can("noise_level")}
            tip={tipFor("noise_level",
              "Extra noise injected at every denoise step. 1.0 leaves the schedule alone; "
              + "lowering it makes the result grainier and less settled, not cleaner. "
              + "Useful for roughening an output on purpose.")} />
        )}
        <Slider
          label={PARAM_RANGES.attenuation.label}
          value={params.attenuation ?? 1}
          min={PARAM_RANGES.attenuation.min} max={PARAM_RANGES.attenuation.max}
          step={PARAM_RANGES.attenuation.step}
          fmt={(v) => (v >= 1 ? "none" : `×${v.toFixed(2)} noise`)}
          onChange={(v) => setParam("attenuation", v)}
          disabled={!can("attenuation")}
          tip={tipFor("attenuation",
            "Scales the noise the run starts from — once, at step 0. Below 1 the sampler begins "
            + "with less variance than the model expects, which comes out calmer and flatter, and "
            + "at 0 there is no randomness left at all.\n\n"
            + "It applies to whatever the run starts with: pure noise, the noise mixed into an "
            + "init image, or the noise a painted region is filled with. Unlike Extra noise, which "
            + "keeps injecting at every step, this is a single change to the starting point.")} />

        <p className="sub mt-2 mb-1">CLIP guidance</p>
        <Text
          label="Prompt"
          value={params.text || ""}
          placeholder="e.g. a rust-red coastline, aerial"
          onChange={(v) => setParam("text", v)}
          disabled={!can("text")}
          tip={tipFor("text",
            "Steers the image toward a description. CLIP scores crops of the picture as it forms, "
            + "and every step is nudged toward a better score. Typing a prompt switches guidance on; "
            + "clearing it switches guidance off.\n\n"
            + "The model has no text input of its own, so this is a pull in a direction it can "
            + "already go — not an instruction.")} />
        {!hidden("guidance_step") && (
          <Slider
            label={PARAM_RANGES.guidance_step.label}
            value={params.guidance_step ?? 0.02}
            min={PARAM_RANGES.guidance_step.min} max={PARAM_RANGES.guidance_step.max}
            step={PARAM_RANGES.guidance_step.step}
            onChange={(v) => setParam("guidance_step", v)}
            disabled={!can("guidance_step")}
            tip={tipFor("guidance_step",
              "How hard each step is pulled toward the prompt — the main dial. Too high and the "
              + "picture turns into CLIP's own texture instead of the subject.")} />
        )}
        {!hidden("guidance_power") && (
          <Slider
            label={PARAM_RANGES.guidance_power.label}
            value={params.guidance_power ?? 1}
            min={PARAM_RANGES.guidance_power.min} max={PARAM_RANGES.guidance_power.max}
            step={PARAM_RANGES.guidance_power.step}
            fmt={(v) => (v === 0 ? "even" : `×${v}`)}
            onChange={(v) => setParam("guidance_power", v)}
            disabled={!can("guidance_power")}
            tip={tipFor("guidance_power",
              "When the pull happens. 0 pulls evenly the whole way through; higher holds off until "
              + "there is an image to work on rather than noise.")} />
        )}
        {!hidden("cuts") && (
          <Slider
            label={PARAM_RANGES.cuts.label}
            value={params.cuts ?? 0.5}
            min={PARAM_RANGES.cuts.min} max={PARAM_RANGES.cuts.max}
            step={PARAM_RANGES.cuts.step}
            fmt={(v) => (v < 0.35 ? "detail" : v > 0.65 ? "structure" : "balanced")}
            onChange={(v) => setParam("cuts", v)}
            disabled={!can("cuts")}
            tip={tipFor("cuts",
              "CLIP only ever sees small crops, never the whole frame. Toward detail it takes many "
              + "little ones, so the prompt reaches texture; toward structure a few big ones, so it "
              + "shapes the composition.")} />
        )}
        {!hidden("spherical") && (
          <label className="row center gap-2" title={tipFor("spherical",
            "A distance measure that keeps pulling as the image nears the prompt instead of easing "
            + "off. Worth trying when a strong prompt seems to stall.")}>
            <input type="checkbox" checked={!!params.spherical}
              disabled={!can("spherical")}
              onChange={(e) => setParam("spherical", e.target.checked)} />
            <span className="sub">Spherical distance</span>
          </label>
        )}

        <label className="row center gap-2" title={tipFor("ema",
          "Averaged weights are usually smoother than the raw ones. Kiln falls back to the raw "
          + "weights when a model has no averaged copy.")}>
          <input type="checkbox" checked={params.ema}
            disabled={!can("ema") || !!inert("ema")}
            onChange={(e) => setParam("ema", e.target.checked)} />
          <span className="sub">
            {!inert("ema") ? "Use EMA weights (recommended)"
              : model?.ema === "same"
                // The weights exist, they are just the same weights -- saying
                // "not available" here would be wrong.
                ? "Use EMA weights — identical for this model"
                : "Use EMA weights — not available for this model"}
          </span>
        </label>
        {inert("ema") && <p className="hint mb-0">{inert("ema")}</p>}
      </Disclose>
    </div>
  );
}
