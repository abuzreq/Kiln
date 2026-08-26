import React, { useMemo } from "react";
import { api, downloadPost, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay, usePlayState } from "./playContext.jsx";
import { Select, Num, Progress } from "../components/ui.jsx";
import { buildSweepBase, PARAM_RANGES, paramRange, inertReason } from "../sampleSettings.jsx";

// Sweepable axes, labelled and bounded by the same table the Sample-settings
// sliders use, so an axis always starts on that parameter's full span.
// text_weight and cuts are omitted while CLIP guidance is unimplemented —
// sweeping them would render identical cells (see GUIDANCE_PARAMS).
const AXIS_KEYS = ["seed", "steps", "eta", "noise_level"];

function axisOptions(sampleParams) {
  return AXIS_KEYS.map((k) => {
    const why = inertReason(k, sampleParams);
    const r = PARAM_RANGES[k];
    return {
      value: k,
      label: why ? `${r.label} (no effect)` : r.label,
      title: why || `${r.min} – ${r.max}`,
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

function fmt(param, v) {
  if (param === "seed" || Number.isInteger(v)) return String(Math.round(v));
  return String(v);
}

function AxisEditor({ title, param, setParam, from, setFrom, to, setTo, count, setCount, tip, options }) {
  const values = useMemo(
    () => linspace(from, to, count, param === "seed"),
    [from, to, count, param],
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
        <div className="w-64"><Num label="N" value={count} onChange={setCount} min={2} max={12} tip="How many full samples on this axis." /></div>
      </div>
      <div className="sweep-values" aria-label={`Values for ${param}`}>
        {values.map((v, i) => (
          <span key={`${param}-${i}`} className="pill sm">{param}={fmt(param, v)}</span>
        ))}
      </div>
    </div>
  );
}

export default function SweepPanel() {
  const { toast, modelPath, setPlayTab } = useApp();
  const { sampleParams, commitFrame, applyCard } = usePlay();
  // Axis config lives in Play state so it survives leaving the tab, and so the
  // shared Sample-settings panel can see which parameters it no longer controls.
  const [param, setParam] = usePlayState("sweep.param", "seed");
  const [from, setFrom] = usePlayState("sweep.from", PARAM_RANGES.seed.min);
  const [to, setTo] = usePlayState("sweep.to", PARAM_RANGES.seed.max);
  const [count, setCount] = usePlayState("sweep.count", 4);
  const [twoD, setTwoD] = usePlayState("sweep.twoD", false);
  const [param2, setParam2] = usePlayState("sweep.param2", "eta");
  const [from2, setFrom2] = usePlayState("sweep.from2", PARAM_RANGES.eta.min);
  const [to2, setTo2] = usePlayState("sweep.to2", PARAM_RANGES.eta.max);
  const [count2, setCount2] = usePlayState("sweep.count2", 3);

  /** Choosing an axis snaps its span to that parameter's own range — otherwise
   *  From/To keep the previous parameter's numbers, which are usually nonsense
   *  for the new one (0-1 on a control that runs to 100). */
  const chooseParam = (setName, setLo, setHi) => (next) => {
    setName(next);
    const r = paramRange(next);
    if (r) { setLo(r.min); setHi(r.max); }
  };
  // survives leaving the Sweep tab mid-run
  const [job, setJob] = usePlayState("sweep.job", null);

  // What the grid decides, and what it inherits — derived rather than written
  // out, so the two lists can never contradict each other.
  const options = useMemo(() => axisOptions(sampleParams), [sampleParams]);

  const sweptKeys = useMemo(
    () => [param, twoD && param2 !== param ? param2 : null].filter(Boolean),
    [param, twoD, param2],
  );
  const sweptLabels = sweptKeys.map((k) => PARAM_RANGES[k].label);
  // An axis that cannot change the output makes every cell in that direction
  // identical — expensive and confusing. Catch it before the grid runs.
  const inertAxes = sweptKeys
    .map((k) => ({ key: k, label: PARAM_RANGES[k].label, why: inertReason(k, sampleParams) }))
    .filter((a) => a.why);
  const inheritedLabels = useMemo(() => {
    const out = [];
    if (!sweptKeys.includes("steps")) out.push(`Steps (${sampleParams.steps})`);
    if (!sweptKeys.includes("seed")) {
      out.push(sampleParams.seed === "" ? "a random seed" : `Seed (${sampleParams.seed})`);
    }
    if (!sweptKeys.includes("eta")) out.push(`Eta (${sampleParams.eta})`);
    out.push(`size ${sampleParams.image_size}px`);
    return out;
  }, [sweptKeys, sampleParams]);

  const xs = useMemo(() => linspace(from, to, count, param === "seed"), [from, to, count, param]);
  const ys = useMemo(
    () => (twoD ? linspace(from2, to2, count2, param2 === "seed") : [null]),
    [twoD, from2, to2, count2, param2],
  );
  const planned = useMemo(() => {
    const labels = [];
    for (const y of ys) {
      for (const x of xs) {
        labels.push(
          `${param}=${fmt(param, x)}` + (y != null ? ` · ${param2}=${fmt(param2, y)}` : ""),
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
      const body = {
        model_path: modelPath,
        param, from, to, count,
        ...buildSweepBase(sampleParams),
      };
      if (twoD) {
        body.param2 = param2;
        body.from2 = from2;
        body.to2 = to2;
        body.count2 = count2;
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
      </div>

      {inertAxes.length > 0 && (
        <div className="card">
          <span className="pill bad">no effect</span>
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
