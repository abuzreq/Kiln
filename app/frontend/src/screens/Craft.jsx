import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, downloadPost, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay, usePlayState } from "./playContext.jsx";
import { Select, Num, Text, Disclose, Progress, Modal } from "../components/ui.jsx";
import UnetVisualizer from "../components/UnetVisualizer.jsx";
import BendEditor from "../components/BendEditor.jsx";
import BendPresetList from "../components/BendPresetList.jsx";
import { buildSamplePayload } from "../sampleSettings.jsx";
import {
  expandGroup, isGroup, resolveMany, resolveTargets, toggleNodeTargets,
} from "../bendTargets.js";

const newBendId = () => Math.random().toString(36).slice(2);

const BEND_SWEEP_EMPTY = {
  bend: 0, param: "", from: 0, to: 1, count: 5,
  frames: null, busy: false, job: null, fps: 8, pingpong: true,
};

/** The two samples side by side. Same markup inline and in the modal, so the
 *  enlarged view cannot drift from the one in the column. */
function ComparePair({ plain, bent, large = false, onEnlarge }) {
  const box = (src, label, alt) => (
    <div className="grow">
      <div className="section-title">{label}</div>
      <div className={`preview-box preview-square ${large ? "" : "preview-max"}`}>
        {src ? (
          onEnlarge
            ? <img src={src} alt={alt} onClick={onEnlarge} className="clickable" />
            : <img src={src} alt={alt} />
        ) : <span className="sub">—</span>}
      </div>
    </div>
  );
  return (
    <div className={`row gap-2 mt-2 ${large ? "compare-large" : ""}`}>
      {box(plain, "Without bends", "without bends")}
      {box(bent, "With bends", "with bends")}
    </div>
  );
}

export function BendWorkspace({ stack, setStack }) {
  const { toast, modelPath, ops: sharedOps, setPlayTab } = useApp();
  const { sampleParams, commitFrame, applyCard } = usePlay();
  const [graph, setGraph] = useState(null);
  const [groups, setGroups] = useState([]);
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
  const [compareOpen, setCompareOpen] = useState(false);
  const [gifBusy, setGifBusy] = useState(false);

  useEffect(() => {
    // groups only ever come from this endpoint; the op list is shared via context
    api.get("/craft/ops").then((d) => setGroups(d.groups || [])).catch(() => {});
    api.get("/craft/bends").then(setPresets).catch(() => {});
  }, []);

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
    api.post("/craft/introspect", { model_path: modelPath })
      .then(setGraph)
      .catch((e) => toast(e.message, "error"));
  }, [modelPath]);

  // Keep focus pointing at a bend that still exists.
  useEffect(() => {
    if (focusedBendId && stack.some((b) => b.id === focusedBendId)) return;
    setFocusedBendId(stack[0]?.id || null);
  }, [stack, focusedBendId]);

  const nodes = graph?.nodes || [];
  const focusedBend = stack.find((b) => b.id === focusedBendId) || null;
  const focusIndex = stack.findIndex((b) => b.id === focusedBendId);

  const focusTargets = useMemo(
    () => resolveTargets(focusedBend?.targets, nodes),
    [focusedBend, nodes],
  );
  // Everything the rest of the stack hits, so editing one bend never hides the others.
  const otherTargets = useMemo(
    () => resolveMany(stack.filter((b) => b.active && b.id !== focusedBendId), nodes),
    [stack, focusedBendId, nodes],
  );

  const updateBend = (id, patch) => setStack(stack.map((b) => (b.id === id ? { ...b, ...patch } : b)));

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
    const { targets, expanded } = toggleNodeTargets(focusedBend, ids, nodes, opts);
    updateBend(focusedBend.id, { targets });
    setNote(expanded.length ? { bendId: focusedBend.id, groups: expanded } : null);
  };

  const onCreateFromNode = (id) => {
    if (addBend([id])) toast("Started a bend on that layer", "success");
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
      api.get("/craft/bends").then(setPresets);
    } catch (e) { toast(e.message, "error"); }
  };

  const loadPreset = (name) => {
    const p = presets.find((x) => x.name === name);
    if (p) {
      const loaded = (p.bends || []).map((b) => ({ ...b, id: newBendId() }));
      setStack(loaded);
      setFocusedBendId(loaded[0]?.id || null);
      setNote(null);
      toast(`Loaded “${name}”`, "success");
    }
  };

  // Interchange with other network-bending tools. Groups are resolved to real
  // layer names first: "encoder" means nothing outside Kiln.
  const exportBends = async () => {
    if (!stack.length) { toast("Nothing to export", "error"); return; }
    const resolved = stack.filter((b) => b.active).map((b) => ({
      ...b, targets: [...resolveTargets(b.targets, nodes)],
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
      if (report?.schedule_flattened) {
        toast(`Exported ${report.bends_written} layer bends — the format has one schedule for the whole file, so the per-bend windows were merged`, "warn");
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
        doc, layers: nodes.map((n) => n.id),
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
      setGenBent(bent.detail?.frame || null);
      toast(reusedPlain
        ? "Compared — reused the unbent image, only the bent side was sampled"
        : "Compared with vs without bends", "success");
    } catch (e) { toast(e.message, "error"); }
    setGenBusy(false);
    setGenJob(null);
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
      <div className="card">
        <div className="row between center wrap gap-2">
          <h3 className="mb-0">Bend</h3>
          {graph?.model && (
            <span className="pill">{graph.model.mtype} · {graph.model.mults?.join("-")}</span>
          )}
        </div>
        <p className="hint mb-0 mt-1">
          Rewrites activations mid-generation at the layers you pick. Changes the output, not the
          model file. Save a stack to reuse it in Create.
        </p>
      </div>

      <div className="bend-layout">
        <div className="bend-side">
          <div className="card bend-save-card">
            <h3>Save this setup</h3>
            <p className="hint mb-2">Name it to reuse in Create.</p>
            <div className="row center wrap gap-2">
              <div className="grow">
                <Text label="" value={saveName} onChange={setSaveName} placeholder="e.g. melt-decoder" />
              </div>
              <button type="button" className="btn primary" onClick={savePreset} disabled={!stack.length}>Save bend</button>
            </div>
            <p className="hint mb-0 mt-2">
              Each starter changes one thing. Load one, Compare, then edit it. Hover for details.
            </p>
            <BendPresetList
              groups={[
                { label: "Starters", presets: starterPresets },
                { label: "Saved", presets: savedPresets },
              ]}
              onLoad={loadPreset}
              ops={ops}
            />
            <div className="section-title mt-2">Share with other tools</div>
            <p className="hint mb-2">
              The shared network-bending JSON. Layer paths are per-architecture, so an imported
              file usually needs retargeting.
            </p>
            <div className="row center wrap gap-2">
              <button type="button" className="btn sm" onClick={exportBends} disabled={!stack.length}>
                Export JSON
              </button>
              <button type="button" className="btn sm" onClick={() => importRef.current?.click()}>
                Import JSON
              </button>
              <input
                ref={importRef}
                type="file"
                accept="application/json,.json"
                className="hidden-file"
                onChange={(e) => { importBends(e.target.files?.[0]); e.target.value = ""; }}
              />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="row between center wrap gap-2">
            <h3 className="mb-0">Model map</h3>
            {focusedBend ? (
              <span className="pill accent">
                Editing #{focusIndex + 1} {opMap[focusedBend.op]?.label || focusedBend.op}
              </span>
            ) : (
              <span className="pill">No bend selected</span>
            )}
          </div>
          <p className="hint mt-1">
            {focusedBend
              ? "Click to target, shift+click a range, drag a span (alt+drag removes). Solid rings: this bend. Dashed: the rest."
              : "Click a layer to start a bend on it."}
          </p>
          <div className="row wrap gap-2 bend-map-chips">
            {groups.map((g) => (
              <button type="button" key={g}
                className={`pill chip ${focusedBend?.targets.includes(g) ? "on" : ""}`}
                onClick={() => toggleGroup(g)}>{g}</button>
            ))}
          </div>
          <UnetVisualizer
            graph={graph}
            focusTargets={focusTargets}
            otherTargets={otherTargets}
            hasFocus={!!focusedBend}
            onToggle={onMapToggle}
            onCreateFromNode={onCreateFromNode}
          />
        </div>
        <div className="bend-compare-col">
          <div className="card bend-compare-card">
            <div className="row between center wrap gap-2">
              <h3 className="mb-0">Compare</h3>
              {(genPlain || genBent) && (
                <button type="button" className="btn ghost sm" onClick={() => setCompareOpen(true)}>
                  Enlarge
                </button>
              )}
            </div>
            <p className="hint mb-2">
              Same seed, plain then bent. The plain side is reused until the model, settings or seed change.
            </p>
            <div className="row center wrap gap-2">
              {genBusy ? (
                <button type="button" className="btn danger" onClick={async () => { if (genJob) await api.post(`/jobs/${genJob.id}/cancel`); }}>Stop</button>
              ) : (
                <button type="button" className="btn primary" onClick={generateCompare} disabled={!modelPath || !stack.length}>
                  Compare samples
                </button>
              )}
              {genBusy && genJob && <span className="sub">{genJob.message || "Generating…"}</span>}
            </div>
            {(genPlain || genBent) && (
              <ComparePair plain={genPlain} bent={genBent} onEnlarge={() => setCompareOpen(true)} />
            )}
          </div>
        </div>
      </div>

      <div className="card bend-stack-card">
        <BendEditor
          ops={ops}
          nodes={nodes}
          stack={stack}
          setStack={setStack}
          focusedId={focusedBendId}
          setFocusedId={(id) => { setFocusedBendId(id); setNote(null); }}
          addBend={() => addBend()}
          updateBend={updateBend}
          note={note}
        />
      </div>

      {compareOpen && (
        <Modal
          title="Compare: without vs with bends"
          wide
          onClose={() => setCompareOpen(false)}
          footer={<button type="button" className="btn ghost" onClick={() => setCompareOpen(false)}>Close</button>}
        >
          <ComparePair plain={genPlain} bent={genBent} large />
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
