import React, { useMemo, useState } from "react";
import { api, downloadPost, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay, usePlayState } from "./playContext.jsx";
import { Select, Num, Progress, Modal } from "../components/ui.jsx";
import { ExpandIcon } from "../components/icons.jsx";
import { buildSweepBase, PARAM_RANGES, paramRange, inertReason, samplerLabel, paramLabel } from "../sampleSettings.jsx";
// Starting points, for anyone who has not run a sweep before: what one is for,
// shown rather than described. Clicking one fills in the axes below; nothing
// runs until Run sweep, so it can be changed first. A JSON file because
// scripts/make_example_media.py renders the picture each one shows from it.
import EXAMPLES from "../sweepExamples.json";
import MEDIA from "../exampleMedia.json";

// Sweepable axes.
//
// Eta and Extra noise were dropped: Eta only does anything on DDIM, and Extra
// noise is set by Create's Change slider, so both spent most of their time
// producing grids of identical cells. Sampler and Image size replace them —
// both change every cell, always.
const AXIS_KEYS = ["seed", "steps", "sampler", "image_size"];

// How an axis lays out its values. Not every parameter is a line between two
// numbers: resolution is more useful doubled than added to, and a sampler is a
// set of names with no order at all.
const AXIS_KINDS = {
  seed: "linear",
  steps: "linear",
  image_size: "geometric",
  sampler: "choice",
};



const SAMPLER_CHOICES = ["dpmpp", "unipc", "deis", "ddim"];

export function axisKind(param) {
  return AXIS_KINDS[param] || "linear";
}

const axisLabel = paramLabel;

function axisOptions(sampleParams) {
  return AXIS_KEYS.map((k) => {
    const why = inertReason(k, sampleParams);
    const r = PARAM_RANGES[k];
    return {
      value: k,
      label: why ? `${axisLabel(k)} (no effect)` : axisLabel(k),
      title: why || (r ? `${r.min} – ${r.max}` : axisLabel(k)),
    };
  });
}

function linspace(from, to, count, asInt) {
  const n = Math.max(2, Math.min(12, Number(count) || 2));
  const a = Number(from);
  const b = Number(to);
  const out = [];
  for (let i = 0; i < n; i++) {
    const v = a + (b - a) * (i / Math.max(n - 1, 1));
    out.push(asInt ? Math.round(v) : Math.round(v * 1000) / 1000);
  }
  return out;
}

/** Values for one axis, whatever kind it is. */
export function axisValues(param, { from, to, count, mult, picks }) {
  const kind = axisKind(param);
  if (kind === "choice") {
    const chosen = (picks || []).filter((p) => SAMPLER_CHOICES.includes(p));
    return chosen.length ? chosen : [SAMPLER_CHOICES[0]];
  }
  if (kind === "geometric") {
    // 256 with x2 and N=3 gives 256, 512, 1024 — resolution is far more useful
    // stepped by a factor than by an offset.
    const n = Math.max(2, Math.min(12, Number(count) || 2));
    const base = Math.max(1, Number(from) || 1);
    const m = Number(mult) || 2;
    return Array.from({ length: n }, (_, i) => Math.round(base * m ** i));
  }
  // Steps and seeds are both whole numbers; a grid labelled "steps=27.5" was
  // never doing half a step.
  return linspace(from, to, count, true);
}

function fmt(param, v) {
  if (typeof v === "string") return v;
  if (Number.isInteger(v)) return String(v);
  return String(v);
}

function ChoicePicker({ picks, setPicks }) {
  const toggle = (name) => {
    const has = picks.includes(name);
    if (has && picks.length <= 1) return;      // an axis needs at least one cell
    setPicks(has ? picks.filter((p) => p !== name)
                 : SAMPLER_CHOICES.filter((s) => picks.includes(s) || s === name));
  };
  return (
    <>
      <div className="row wrap gap-2 mb-2">
        {SAMPLER_CHOICES.map((name) => (
          <button
            key={name}
            type="button"
            className={`pill chip ${picks.includes(name) ? "on" : ""}`}
            onClick={() => toggle(name)}
          >
            {samplerLabel(name)}
          </button>
        ))}
      </div>
      <p className="hint mb-0">
        Each chosen sampler gets a column. Everything else stays fixed, so the grid
        shows only what the solver changes.
      </p>
    </>
  );
}

function AxisEditor({
  title, tip, options, param, setParam,
  from, setFrom, to, setTo, count, setCount,
  mult, setMult, picks, setPicks, sizeNote,
}) {
  const kind = axisKind(param);
  const values = useMemo(
    () => axisValues(param, { from, to, count, mult, picks }),
    [param, from, to, count, mult, picks],
  );
  const range = paramRange(param);
  // Seeds are unbounded; everything else is clamped to the slider's own span.
  const bounds = range && param !== "seed"
    ? { min: range.min, max: range.max, step: range.step }
    : { step: range?.step };

  return (
    <div className="card">
      <h3 className="mb-0" title={tip}>{title}</h3>
      <Select
        label="Parameter"
        value={param}
        onChange={setParam}
        options={options}
        tip="Which setting to vary along this axis."
      />

      {kind === "choice" && <ChoicePicker picks={picks} setPicks={setPicks} />}

      {kind === "geometric" && (
        <div className="row gap-2">
          <div className="grow">
            <Num label="Start" value={from} onChange={setFrom} min={32} max={2048} step={32}
              tip="Smallest size in the grid. Sizes are square." />
          </div>
          <div className="w-64">
            <Num label="×" value={mult} onChange={setMult} min={1.25} max={4} step={0.25}
              tip="Each cell is this many times the previous one. 2 doubles: 256, 512, 1024." />
          </div>
          <div className="w-64">
            <Num label="N" value={count} onChange={setCount} min={2} max={6}
              tip="How many sizes. Large sizes are slow and can run a small card out of memory." />
          </div>
        </div>
      )}

      {kind === "linear" && (
        <div className="row gap-2">
          <div className="grow">
            <Num label="From" value={from} onChange={setFrom} {...bounds}
              tip={range && param !== "seed"
                ? `Start of the range (${range.min} – ${range.max}).`
                : "Start of the range."} />
          </div>
          <div className="grow">
            <Num label="To" value={to} onChange={setTo} {...bounds}
              tip={range && param !== "seed"
                ? `End of the range (${range.min} – ${range.max}).`
                : "End of the range."} />
          </div>
          <div className="w-64">
            <Num label="N" value={count} onChange={setCount} min={2} max={12}
              tip="How many full samples on this axis." />
          </div>
        </div>
      )}

      <div className="sweep-values" aria-label={`Values for ${param}`}>
        {values.map((v, i) => (
          <span key={`${param}-${i}`} className="pill sm">{axisLabel(param)}={fmt(param, v)}</span>
        ))}
      </div>
      {kind === "geometric" && sizeNote && <p className="hint mb-0">{sizeNote}</p>}
    </div>
  );
}

export default function SweepPanel() {
  const { toast, modelPath, setPlayTab, models } = useApp();
  const { sampleParams, commitFrame, applyCard } = usePlay();
  // Axis config lives in Play state so it survives leaving the tab, and so the
  // shared Sample-settings panel can see which parameters it no longer controls.
  const [param, setParam] = usePlayState("sweep.param", "seed");
  const [from, setFrom] = usePlayState("sweep.from", PARAM_RANGES.seed.min);
  const [to, setTo] = usePlayState("sweep.to", PARAM_RANGES.seed.max);
  const [count, setCount] = usePlayState("sweep.count", 4);
  const [twoD, setTwoD] = usePlayState("sweep.twoD", false);
  const [param2, setParam2] = usePlayState("sweep.param2", "sampler");
  const [from2, setFrom2] = usePlayState("sweep.from2", 256);
  const [to2, setTo2] = usePlayState("sweep.to2", 1024);
  const [count2, setCount2] = usePlayState("sweep.count2", 3);
  // Extra state the non-linear axis kinds need. Kept alongside from/to rather
  // than replacing it so switching axis kind and back does not lose the numbers.
  const [mult, setMult] = usePlayState("sweep.mult", 2);
  const [mult2, setMult2] = usePlayState("sweep.mult2", 2);
  const [picks, setPicks] = usePlayState("sweep.picks", SAMPLER_CHOICES);
  const [picks2, setPicks2] = usePlayState("sweep.picks2", SAMPLER_CHOICES);

  /** Choosing an axis snaps its span to something sensible for that parameter.
   *  Otherwise From/To keep the previous parameter's numbers, which are usually
   *  nonsense for the new one (0-1 on a control that runs to 200). */
  const chooseParam = (setName, setLo, setHi) => (next) => {
    setName(next);
    if (axisKind(next) === "geometric") { setLo(256); setHi(1024); return; }
    if (axisKind(next) === "choice") return;
    const r = paramRange(next);
    if (r) { setLo(r.min); setHi(r.max); }
  };

  const [viewing, setViewing] = useState(null);
  const loadExample = (ex) => {
    const fill = (a, set) => {
      set.param(a.param);
      if (a.from != null) set.from(a.from);
      if (a.to != null) set.to(a.to);
      if (a.count != null) set.count(a.count);
      if (a.mult != null) set.mult(a.mult);
      if (a.picks) set.picks(a.picks);
    };
    fill(ex.x, { param: setParam, from: setFrom, to: setTo, count: setCount, mult: setMult, picks: setPicks });
    setTwoD(!!ex.y);
    if (ex.y) {
      fill(ex.y, { param: setParam2, from: setFrom2, to: setTo2, count: setCount2, mult: setMult2, picks: setPicks2 });
    }
  };
  // The example the axes still match, if any, so its button shows as loaded.
  const matches = (a, cur) => a.param === cur.param
    && ["from", "to", "count", "mult"].every((k) => a[k] == null || a[k] === cur[k])
    && (!a.picks || a.picks.join() === cur.picks.join());
  const loadedExample = EXAMPLES.find((ex) => (
    matches(ex.x, { param, from, to, count, mult, picks })
    && !!ex.y === twoD
    && (!ex.y || matches(ex.y, {
      param: param2, from: from2, to: to2, count: count2, mult: mult2, picks: picks2,
    }))
  ))?.id;

  // survives leaving the Sweep tab mid-run
  const [job, setJob] = usePlayState("sweep.job", null);

  // What the grid decides, and what it inherits — derived rather than written
  // out, so the two lists can never contradict each other.
  const options = useMemo(() => axisOptions(sampleParams), [sampleParams]);

  // A model can only work in multiples of its own downsampling factor; the
  // sampler snaps anything else down silently, so say which values will move.
  const activeModel = useMemo(
    () => (models || []).find((m) => m.path === modelPath) || null,
    [models, modelPath],
  );
  const sizeNoteFor = (values) => {
    const step = activeModel?.size_multiple;
    if (!step) return null;
    const off = values.filter((v) => typeof v === "number" && v % step !== 0);
    const big = values.filter((v) => typeof v === "number" && v > 768);
    const bits = [];
    if (off.length) {
      bits.push(`This model works in multiples of ${step}, so `
        + off.map((v) => `${v} runs at ${Math.max(step, Math.floor(v / step) * step)}`).join(", ")
        + ".");
    }
    if (big.length) bits.push("Sizes above 768px are slow and can exhaust a small card.");
    return bits.join(" ") || null;
  };

  const sweptKeys = useMemo(
    () => [param, twoD && param2 !== param ? param2 : null].filter(Boolean),
    [param, twoD, param2],
  );
  const sweptLabels = sweptKeys.map((k) => axisLabel(k));
  // An axis that cannot change the output makes every cell in that direction
  // identical — expensive and confusing. Catch it before the grid runs.
  const inertAxes = sweptKeys
    .map((k) => ({ key: k, label: axisLabel(k), why: inertReason(k, sampleParams) }))
    .filter((a) => a.why);
  const inheritedLabels = useMemo(() => {
    const out = [];
    if (!sweptKeys.includes("steps")) out.push(`Steps (${sampleParams.steps})`);
    if (!sweptKeys.includes("seed")) {
      out.push(sampleParams.seed === "" ? "a random seed" : `Seed (${sampleParams.seed})`);
    }
    if (!sweptKeys.includes("sampler")) out.push(samplerLabel(sampleParams.sampler || "dpmpp"));
    if (!sweptKeys.includes("image_size")) out.push(`size ${sampleParams.image_size}px`);
    return out;
  }, [sweptKeys, sampleParams]);

  const xs = useMemo(
    () => axisValues(param, { from, to, count, mult, picks }),
    [param, from, to, count, mult, picks],
  );
  const ys = useMemo(
    () => (twoD ? axisValues(param2, { from: from2, to: to2, count: count2, mult: mult2, picks: picks2 }) : [null]),
    [twoD, param2, from2, to2, count2, mult2, picks2],
  );
  const planned = useMemo(() => {
    const labels = [];
    for (const y of ys) {
      for (const x of xs) {
        labels.push(
          `${axisLabel(param)}=${fmt(param, x)}`
            + (y != null ? ` · ${axisLabel(param2)}=${fmt(param2, y)}` : ""),
        );
      }
    }
    return labels;
  }, [xs, ys, param, param2]);

  const running = job?.status === "running";
  const cols = job?.detail?.cols || xs.length;
  const rows = job?.detail?.rows || ys.length;
  const cells = job?.detail?.cells || [];

  const run = async () => {
    if (!modelPath) { toast("Pick a model", "error"); return; }
    if (twoD && param2 === param) { toast("Pick two different parameters for a 2D sweep", "error"); return; }
    try {
      // Values are computed here rather than re-derived from from/to/count on
      // the server: two of the three axis kinds are not a line between numbers,
      // and the route already accepts an explicit list.
      const body = {
        model_path: modelPath,
        param, from, to, count, values: xs,
        ...buildSweepBase(sampleParams),
      };
      if (twoD) {
        body.param2 = param2;
        body.from2 = from2;
        body.to2 = to2;
        body.count2 = count2;
        body.values2 = ys;
      }
      const { job: j } = await api.post("/tools/sweep", body);
      setJob(j);
      const done = await pollJob(j.id, setJob, 500);
      if (done.status !== "done") toast(done.message || "Sweep failed", "error");
      else toast(done.message || "Sweep ready", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  };

  const stop = async () => { if (job) await api.post(`/jobs/${job.id}/cancel`); };

  return (
    <div className="col">
      <div className="card">
        <h3>Sweep</h3>
        <p className="hint mb-2">
          Every cell is a full sample run using your Sample settings and the model
          picked above — except the parameters below, which this grid varies.
        </p>
        <p className="sub mb-0">
          <b>{sweptLabels.join(" and ")}</b>{" "}
          {sweptLabels.length > 1 ? "come" : "comes"} from the grid.
          {" "}Everything else — {inheritedLabels.join(", ")} — follows Sample settings.
        </p>
        <div className="section-title mt-3">Start from an example</div>
        <div className="sweep-examples">
          {EXAMPLES.map((ex) => (
            <div key={ex.id} className={`sweep-example ${loadedExample === ex.id ? "on" : ""}`.trim()}>
              <button
                type="button"
                className="sweep-example-pick"
                aria-pressed={loadedExample === ex.id}
                title={`${ex.tip}\n\nClick to fill in the axes below.`}
                onClick={() => loadExample(ex)}
              >
                {MEDIA.sweeps?.[ex.id] && (
                  <img src={MEDIA.sweeps[ex.id]} alt="" loading="lazy" />
                )}
                <span className="sweep-example-label">{ex.label}</span>
                <span className="sub">{ex.desc}</span>
              </button>
              {MEDIA.sweeps?.[ex.id] && (
                <button type="button" className="sweep-example-view"
                  aria-label={`See the ${ex.label} example larger`} title="See this run larger"
                  onClick={() => setViewing(ex)}>
                  <ExpandIcon size={14} />
                </button>
              )}
            </div>
          ))}
        </div>
        <p className="hint mb-0 mt-2">
          Each picture is a real run on the sample model {MEDIA.model}, seed {MEDIA.seed}. Yours will
          use your model and Sample settings.
        </p>
      </div>

      {viewing && (
        <Modal
          title={viewing.label}
          wide
          onClose={() => setViewing(null)}
          footer={(
            <>
              <button type="button" className="btn" onClick={() => setViewing(null)}>Close</button>
              <button type="button" className="btn primary"
                onClick={() => { loadExample(viewing); setViewing(null); }}>
                Use these axes
              </button>
            </>
          )}
        >
          <p className="hint mt-0">{viewing.tip}</p>
          <img className="sweep-example-full" src={MEDIA.sweeps[viewing.id]}
            alt={`${viewing.label}: ${viewing.desc}`} />
          <p className="sub mb-0">
            {viewing.desc}, on the sample model {MEDIA.model}: seed {MEDIA.seed},{" "}
            {MEDIA.steps} steps, {samplerLabel(MEDIA.sampler)}, {MEDIA.image_size}px, except
            where the grid varies one of them.
          </p>
        </Modal>
      )}

      {inertAxes.length > 0 && (
        <div className="card">
          <span className="pill bad">No effect</span>
          {inertAxes.map((a) => (
            <p className="hint mb-0 mt-2" key={a.key}>
              <b>{a.label}</b> would produce identical cells: {a.why}
            </p>
          ))}
        </div>
      )}

      <AxisEditor
        title="Axis X"
        tip="Horizontal axis of the contact sheet."
        options={options}
        param={param} setParam={chooseParam(setParam, setFrom, setTo)}
        from={from} setFrom={setFrom}
        to={to} setTo={setTo}
        count={count} setCount={setCount}
        mult={mult} setMult={setMult}
        picks={picks} setPicks={setPicks}
        sizeNote={sizeNoteFor(xs)}
      />

      <div className="card">
        <label className="row center gap-2">
          <input type="checkbox" checked={twoD} onChange={(e) => setTwoD(e.target.checked)} />
          <span>2D sweep (second axis)</span>
        </label>
      </div>

      {twoD && (
        <AxisEditor
          title="Axis Y"
          tip="Vertical axis of the contact sheet."
          options={options}
          param={param2} setParam={chooseParam(setParam2, setFrom2, setTo2)}
          from={from2} setFrom={setFrom2}
          to={to2} setTo={setTo2}
          count={count2} setCount={setCount2}
          mult={mult2} setMult={setMult2}
          picks={picks2} setPicks={setPicks2}
          sizeNote={sizeNoteFor(ys)}
        />
      )}

      <div className="card">
        <div className="section-title">Will sample ({planned.length})</div>
        <div className="sweep-values">
          {planned.map((lab, i) => (
            <span key={i} className="pill sm">{lab}</span>
          ))}
        </div>
      </div>

      {running ? (
        <button type="button" className="btn danger w-full" onClick={stop}>Stop</button>
      ) : (
        <button
          type="button"
          className="btn primary w-full"
          onClick={run}
          disabled={!modelPath || inertAxes.length > 0}
          title={inertAxes.length > 0
            ? `${inertAxes.map((a) => a.label).join(" and ")} cannot change the output right now`
            : undefined}
        >
          {inertAxes.length > 0 ? "Pick an axis that changes something" : "Run sweep"}
        </button>
      )}

      {(running || cells.length > 0 || job?.detail?.sheet) && (
        <div className="card">
          <div className="row between center mb-2">
            <h3 className="mb-0">Results</h3>
            {job?.detail?.sheet && (
              <button
                type="button"
                className="btn sm"
                onClick={() => downloadPost(
                  "/perform/export",
                  { image: job.detail.sheet, card: job.detail.card, filename: "kiln-sweep-grid" },
                  "kiln-sweep-grid.png",
                ).catch((e) => toast(e.message || "Download failed", "error"))}
                title="Save the full sweep grid as one PNG, with the swept axes recorded inside it."
              >
                Download sweep grid
              </button>
            )}
          </div>
          {job?.detail?.sheet && (
            <p className="hint">
              <b>Sweep grid</b> = one PNG of every cell laid out together. Browse cells below, or download the grid to share or archive.
            </p>
          )}
          {running && (
            <>
              <Progress value={job.progress || 0} />
              <p className="sub">{job.message || "Sweeping…"}</p>
            </>
          )}
          {cells.length > 0 && (
            <div
              className="sweep-grid"
              style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
            >
              {Array.from({ length: rows * cols }, (_, i) => {
                const cell = cells[i];
                const label = (job?.detail?.planned || planned)[i] || "";
                return (
                  <div key={i} className="sweep-cell">
                    {cell?.image ? (
                      <button
                        type="button"
                        className="sweep-cell-pick"
                        title={`Open in Create · ${label}`}
                        onClick={() => {
                          // a cell is a complete run — carry it into Create,
                          // settings and all, instead of retyping the values
                          commitFrame(cell.image, null, cell.card || null);
                          if (cell.card) applyCard(cell.card);
                          setPlayTab("create");
                        }}
                      >
                        <img src={cell.image} alt={label} />
                        <span className="sweep-cell-hint">Open in Create</span>
                      </button>
                    ) : (
                      <div className="sweep-placeholder">{running ? "…" : "—"}</div>
                    )}
                    <span className="sub">{label}</span>
                  </div>
                );
              })}
            </div>
          )}
          {!cells.length && job?.detail?.sheet && (
            <img className="sweep-sheet" src={job.detail.sheet} alt="Sweep contact sheet" />
          )}
        </div>
      )}
    </div>
  );
}
