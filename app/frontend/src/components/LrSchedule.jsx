import React, { useMemo, useState } from "react";
import { Modal, Num, Seg, Select, Tooltip } from "./ui.jsx";

/** Rate as "5e-4", matching lr_plan.fmt_lr so a label reads the same wherever
 *  it came from. Neither toExponential nor %g gives this consistently. */
export function fmtRate(x) {
  const v = Number(x);
  if (!Number.isFinite(v) || v <= 0) return "—";
  let e = Math.floor(Math.log10(v));
  let m = v / 10 ** e;
  if (m >= 9.995) { m /= 10; e += 1; }
  return `${String(m.toFixed(1)).replace(/\.0$/, "")}e${e}`;
}

export function fmtStep(n) {
  const v = Number(n) || 0;
  if (v >= 1000 && v % 100 === 0) return v % 1000 === 0 ? `${v / 1000}k` : `${v / 1000}k`;
  return String(v);
}

/** The three schedules worth one click. The rest live behind "Edit…": drops-3
 *  and cyclical are real choices, just not first-run ones. */
const QUICK = ["constant", "drops-2", "cosine-floor"];

const CONST_ONLY = (plan) =>
  (plan?.segments || []).every((s) => (s.kind || "const") === "const");

/**
 * Picking a learning-rate schedule before a run.
 *
 * The summary under the tabs comes from the server, compiled against this run's
 * own base rate and step target, so the step numbers shown are the ones the
 * trainer will actually use.
 */
export function LrScheduleField({ presets, value, onChange, onEdit, disabled }) {
  const rows = presets || {};
  const tabs = QUICK.filter((id) => rows[id]).map((id) => ({
    id,
    label: rows[id].label,
    tip: `${rows[id].blurb}\n\n${rows[id].summary}`,
  }));
  const current = rows[value];
  return (
    <>
      <div className="section-title mt-2">Learning rate over the run</div>
      {tabs.length > 0 && (
        <Seg ariaLabel="Learning rate schedule" tabs={tabs} value={value} onChange={onChange} />
      )}
      <p className="hint mt-1 row between center gap-2">
        <span>{current?.summary || (value ? value : "constant")}</span>
        <button type="button" className="btn ghost sm" onClick={onEdit} disabled={disabled}>
          Edit…
        </button>
      </p>
    </>
  );
}

/**
 * The full editor: every shipped schedule, plus hand-placed breakpoints.
 *
 * Breakpoints are the shape the engine author actually works in ("drop it to
 * 5e-5 around here"), so they are directly editable. The curves are not
 * expressible as breakpoints, so choosing one replaces the list rather than
 * pretending to convert it.
 */
export function LrPlanModal({ presets, plan, baseLr, onClose, onApply, title = "Learning rate schedule" }) {
  const rows = presets || {};
  const ids = Object.keys(rows).sort((a, b) => (rows[a].order ?? 0) - (rows[b].order ?? 0));
  const [picked, setPicked] = useState(() => (plan?.preset && rows[plan.preset] ? plan.preset : "custom"));
  const [points, setPoints] = useState(() => {
    const segs = (plan?.segments || []).filter((s) => (s.kind || "const") === "const");
    return segs.length
      ? segs.map((s) => ({ from: s.from ?? 0, lr: s.lr }))
      : [{ from: 0, lr: Number(baseLr) || 4e-4 }];
  });

  const shown = picked === "custom" ? null : rows[picked];
  const curvy = picked === "custom" && plan && !CONST_ONLY(plan);

  const setPoint = (i, key, v) =>
    setPoints((ps) => ps.map((p, j) => (j === i ? { ...p, [key]: v } : p)));
  const addPoint = () =>
    setPoints((ps) => {
      const last = ps[ps.length - 1] || { from: 0, lr: Number(baseLr) || 4e-4 };
      return [...ps, { from: Number(last.from) + 1000, lr: Number(last.lr) / 5 }];
    });
  const dropPoint = (i) => setPoints((ps) => (ps.length > 1 ? ps.filter((_, j) => j !== i) : ps));

  const problem = useMemo(() => {
    if (picked !== "custom") return null;
    const ps = points.map((p) => ({ from: Number(p.from), lr: Number(p.lr) }));
    if (ps.some((p) => !Number.isFinite(p.lr) || p.lr <= 0)) return "Every rate must be a positive number.";
    if (ps[0].from !== 0) return "The first breakpoint has to start at step 0.";
    for (let i = 1; i < ps.length; i += 1) {
      if (!(ps[i].from > ps[i - 1].from)) return "Each breakpoint must come after the one before it.";
    }
    return null;
  }, [picked, points]);

  const apply = () => {
    if (picked !== "custom") { onApply({ preset: picked }); return; }
    onApply({
      segments: points.map((p) => ({
        from: Math.max(Math.round(Number(p.from)), 0), kind: "const", lr: Number(p.lr),
      })),
    });
  };

  return (
    <Modal
      title={title}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn ghost" onClick={onClose}>Cancel</button>
          <button type="button" className="btn primary" onClick={apply} disabled={!!problem}>
            Use this schedule
          </button>
        </>
      }
    >
      <Select
        label="Schedule"
        value={picked}
        onChange={setPicked}
        options={[...ids, "custom"]}
        tip={"How the learning rate moves over the run. Lowering it once the model has the general idea is the usual way to sharpen detail without the model drifting away from what it learned."}
      />
      {shown && (
        <>
          <p className="hint mt-1 mb-0">{shown.blurb}</p>
          <p className="sub mt-1"><b>{shown.summary}</b></p>
        </>
      )}
      {picked === "custom" && (
        <>
          <p className="hint mt-2 mb-1">
            A rate, and the step it takes over from. The first one starts the run.
          </p>
          {curvy && (
            <p className="warn-banner mb-2">
              This run is on a curve, not breakpoints. Saving here replaces it with the
              steps below.
            </p>
          )}
          <div className="lr-points">
            {points.map((p, i) => (
              <div className="row gap-2 center" key={i}>
                <div className="grow">
                  <Num label={i === 0 ? "From step (start)" : "From step"} value={p.from}
                       onChange={(v) => setPoint(i, "from", v)} min={0} step={1000}
                       disabled={i === 0} />
                </div>
                <div className="grow">
                  <Num label="Rate" value={p.lr} onChange={(v) => setPoint(i, "lr", v)}
                       step={0.00001} />
                </div>
                <button type="button" className="btn ghost sm" onClick={() => dropPoint(i)}
                        disabled={points.length < 2} aria-label={`Remove breakpoint ${i + 1}`}>
                  Remove
                </button>
              </div>
            ))}
          </div>
          <button type="button" className="btn ghost sm mt-2" onClick={addPoint}>Add a drop</button>
          {problem && <p className="warn-banner mt-2 mb-0">{problem}</p>}
        </>
      )}
    </Modal>
  );
}

/**
 * Changing the rate while a run is going.
 *
 * Three divisors rather than a number field because the author's method is a
 * judgement made at a moment ("it has the general idea now, go slower"), and
 * because 5e-4 -> 1e-4 -> 5e-5 is exactly divide by five then by two.
 */
export function LiveRateCard({ lr, summary, onDrop, onEdit, busy }) {
  const rate = Number(lr);
  const known = Number.isFinite(rate) && rate > 0;
  return (
    <div className="continue-run-box mt-2">
      <div className="row between center gap-2">
        <span className="sub">Learning rate <b>{known ? fmtRate(rate) : "—"}</b></span>
        {summary && <span className="sub">{summary}</span>}
      </div>
      <div className="row gap-2 mt-2 wrap">
        {[2, 5, 10].map((f) => (
          <button type="button" key={f} className="btn ghost sm" disabled={!known || busy}
                  onClick={() => onDrop(rate / f)}>
            Drop to {known ? fmtRate(rate / f) : "—"}
          </button>
        ))}
        <button type="button" className="btn ghost sm" onClick={onEdit} disabled={busy}>
          Edit schedule…
        </button>
      </div>
      <Tooltip text={"Adam reads the rate fresh every step, so there is nothing to restart and nothing to corrupt.\n\nStopping and relaunching at a lower rate also works, but the checkpoint stores no optimizer state, so that path throws away the momentum Adam has built up and the first steps afterwards land harder than the rate you asked for."}>
        <p className="hint mb-0 has-tip">
          Applies on the next step, and keeps the optimizer&rsquo;s momentum. The loss
          curve reacts within tens of steps; snapshot previews follow the averaged
          weights, so give it two or three snapshots before judging.
        </p>
      </Tooltip>
    </div>
  );
}
