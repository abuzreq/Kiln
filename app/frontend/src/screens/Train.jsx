import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, mediaUrl, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { Text, Num, Select, Progress, Empty, Modal, Disclose, ConfirmModal, DeleteBtn, Loading, Seg, Tooltip } from "../components/ui.jsx";
import { modelSubtitle } from "../components/modelMeta.jsx";
import LossChart from "../components/LossChart.jsx";
import CheckpointGallery from "../components/CheckpointGallery.jsx";

export function statusPillClass(status) {
  if (status === "training") return "good";
  if (status === "stopped") return "warn";
  return "";
}

function nextRunName(runs) {
  const names = new Set((runs || []).map((r) => r.name));
  let i = 1;
  while (names.has(`run${i}`)) i += 1;
  return `run${i}`;
}

function resumeLabel(path) {
  if (!path) return "Scratch (random init)";
  const base = path.split(/[/\\]/).pop() || path;
  return base.replace(/\.pt$/i, "");
}

function runLabel(r) {
  if (r.status === "training") return "Training";
  if (r.status === "done") return "Done";
  if (r.status === "cancelled") return "Stopped";
  if (r.status === "error") return "Error";
  return r.checkpoints ? "Stopped" : "—";
}

const EMPTY_FORM = {
  backend: "xurdif", mode: "scratch", preset: "standard-128",
  lora_r: 8, precision: "no", gradient_checkpointing: false,
  dataset: "", run_name: "run1", image_size: 512, batch_size: 8,
  diffusion_steps: 1000, train_steps: 280000, accum: 10, lr: 0.0004,
  loss_type: "l1", ssimw: 0, l1w: 1, pred: "x0",
  mtype: "tinyunet_with_attention3", mults: "1,2,2,2", save_every: 1000,
  nsamples: 1, sample_seed: 42, fit: "resize", amp: false, resume: "", nostrict: false,
};

const MODE_TABS = [
  { id: "new", label: "Train new model", tip: "Configure and start a training run" },
  { id: "run", label: "View previous runs", tip: "Inspect a finished or stopped run" },
];


/**
 * Engine-specific settings for a Diffusers run.
 *
 * Which controls appear is driven by the backend's reported capabilities, not
 * by its name -- an install without peft reports lora:false and simply does not
 * offer the mode, rather than offering a button that fails.
 */
function DiffusersOptions({ form, set, engine, seeded }) {
  const caps = engine?.capabilities || {};
  const presets = engine?.presets || {};
  const modes = [];
  if (seeded && caps.finetune) modes.push("finetune");
  if (seeded && caps.lora) modes.push("lora");

  return (
    <>
      {seeded ? (
        modes.length ? (
          <Select label="How to train" value={modes.includes(form.mode) ? form.mode : modes[0]}
            onChange={(v) => set("mode", v)} options={modes}
            tip="finetune updates every weight. lora trains a small adapter instead — far fewer parameters, and the adapter saves separately." />
        ) : (
          <p className="hint mb-0">This install cannot fine-tune Diffusers models. Install accelerate (and peft for LoRA).</p>
        )
      ) : (
        <Select label="Model size" value={form.preset} onChange={(v) => set("preset", v)}
          options={Object.keys(presets).length ? Object.keys(presets) : [form.preset]}
          tip={presets[form.preset]?.label || "Architecture preset for a new model."} />
      )}

      {seeded && form.mode === "lora" && (
        <Num label="LoRA rank" value={form.lora_r} onChange={(v) => set("lora_r", Math.round(v))} step={4}
          tip="Adapter capacity. Higher fits more, costs more. 4–16 is usual." />
      )}

      <div className="row gap-2">
        <div className="grow">
          <Select label="Precision" value={form.precision} onChange={(v) => set("precision", v)}
            options={["no", "fp16", "bf16"]}
            tip="Mixed precision. fp16 is faster and lighter on most NVIDIA cards; bf16 needs a newer one." />
        </div>
        <div className="grow">
          <Select label="Fit" value={form.fit} onChange={(v) => set("fit", v)} options={["resize", "crop"]} />
        </div>
      </div>

      <label className="row center gap-2">
        <input type="checkbox" checked={!!form.gradient_checkpointing}
          onChange={(e) => set("gradient_checkpointing", e.target.checked)} />
        <span>Gradient checkpointing <span className="hint">(less VRAM, slower steps)</span></span>
      </label>
    </>
  );
}

export default function Train() {
  const {
    device, toast, trainFromPath, setTrainFromPath,
    trainDataset, setTrainDataset, setModelPath, setAppMode, setPlayTab, models,
  } = useApp();
  const [info, setInfo] = useState(null);
  const [presets, setPresets] = useState({});
  const [archs, setArchs] = useState([]);
  const [engines, setEngines] = useState([]);
  const [job, setJob] = useState(null);
  const [runView, setRunView] = useState(null);
  const [inspectRun, setInspectRun] = useState(null);
  const [mode, setMode] = useState("new"); // "new" | "run"
  const [est, setEst] = useState(null);
  const [showLogs, setShowLogs] = useState(false);
  const [saveTarget, setSaveTarget] = useState(null);
  const [saveName, setSaveName] = useState("");
  const [fromMode, setFromMode] = useState("scratch");
  const [pendingRun, setPendingRun] = useState(null);
  const [nameAvailable, setNameAvailable] = useState(true);
  const [continueSteps, setContinueSteps] = useState(280000);
  const logRef = useRef(null);

  const [form, setForm] = useState({ ...EMPTY_FORM });
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const train = info?.train || { status: "not_started", label: "Not started" };
  const running = job && job.status === "running";
  const canStart = !running;
  const runs = info?.runs || [];
  const orderedRuns = useMemo(() => runs.slice().reverse(), [runs]);
  const hasRuns = orderedRuns.length > 0;
  const trimmedRunName = (form.run_name || "").trim();
  const libraryModels = useMemo(
    () => (models || []).filter((m) => m.role !== "checkpoint"),
    [models],
  );
  const baseModel = libraryModels.find((m) => m.path === form.resume);

  const loadRun = async (runName) => {
    if (!runName) return;
    try {
      const view = await api.get(`/runs/${runName}`);
      setRunView(view);
      setInspectRun(runName);
      setMode("run");
      if (view.log?.length) setShowLogs(true);
    } catch { /* run may not exist yet */ }
  };

  const startNew = (runsList) => {
    setMode("new");
    setInspectRun(null);
    setRunView(null);
    setShowLogs(false);
    setFromMode("scratch");
    setForm((f) => ({
      ...EMPTY_FORM,
      dataset: f.dataset || info?.datasets?.[0]?.name || "",
      run_name: nextRunName(runsList || info?.runs || runs),
      image_size: f.image_size,
      batch_size: f.batch_size,
      mtype: f.mtype,
    }));
  };

  const applyLibraryStart = async (path) => {
    if (!path) {
      setFromMode("scratch");
      setForm((f) => ({ ...f, resume: "", nostrict: false }));
      return;
    }
    setFromMode("library");
    setMode("new");
    try {
      const d = await api.get(`/train/continue/info?path=${encodeURIComponent(path)}`);
      setForm((f) => ({
        ...f,
        resume: path,
        image_size: d.config?.image_size ?? f.image_size,
        batch_size: d.config?.batch_size ?? f.batch_size,
        train_steps: d.suggested_steps ?? f.train_steps,
        save_every: d.config?.save_every ?? f.save_every,
        mtype: d.mtype || f.mtype,
        mults: Array.isArray(d.mults) ? d.mults.join(",") : f.mults,
        pred: d.pred || f.pred,
        lr: d.config?.lr ?? f.lr,
        accum: d.config?.accum ?? f.accum,
      }));
    } catch (e) {
      toast(e.message, "error");
      setForm((f) => ({ ...f, resume: path }));
    }
  };

  useEffect(() => {
    if (mode !== "new" || !trimmedRunName) {
      setNameAvailable(true);
      return undefined;
    }
    const t = setTimeout(async () => {
      try {
        const r = await api.get(`/runs/available?name=${encodeURIComponent(trimmedRunName)}`);
        setNameAvailable(!!r.available);
      } catch {
        setNameAvailable(false);
      }
    }, 300);
    return () => clearTimeout(t);
  }, [mode, trimmedRunName]);

  useEffect(() => {
    let cancelled = false;
    setJob(null);
    setRunView(null);
    setInspectRun(null);
    setShowLogs(false);
    setInfo(null);
    (async () => {
      try {
        const p = await api.get("/studio");
        if (cancelled) return;
        setInfo(p);
        if (p.datasets?.length) set("dataset", p.datasets[0].name);
        set("run_name", nextRunName(p.runs));
        const t = p.train || {};
        if (t.job_id) {
          const j = await api.get(`/jobs/${t.job_id}`);
          if (!cancelled && j && j.status === "running") {
            setJob(j);
            if (t.run) {
              setInspectRun(t.run);
              setMode("run");
              loadRun(t.run);
            }
            pollJob(j.id, setJob, 800).then((done) => {
              if (cancelled) return;
              setJob(done);
              toast(done.message, done.status === "done" ? "success" : "error");
              api.get("/studio").then(setInfo);
              if (t.run) loadRun(t.run);
            }).catch(() => {});
            return;
          }
        }
        if (t.run) await loadRun(t.run);
        else if (p.runs?.length) await loadRun(p.runs[p.runs.length - 1].name);
        else setMode("new");
      } catch (e) {
        if (!cancelled) toast(e.message, "error");
      }
    })();
    api.get("/train/presets").then(setPresets);
    api.get("/architectures").then((d) => setArchs(d.architectures));
    api.get("/train/backends").then((d) => setEngines(d.backends || [])).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!trainFromPath) return;
    applyLibraryStart(trainFromPath);
    setTrainFromPath("");
  }, [trainFromPath]);

  // arriving from "Train on this dataset" in Workshop ▸ Data
  useEffect(() => {
    if (!trainDataset) return;
    setMode("new");
    set("dataset", trainDataset);
    setTrainDataset("");
  }, [trainDataset]);

  useEffect(() => {
    const t = setTimeout(() => {
      api.get(`/train/estimate?image_size=${form.image_size || 0}&batch_size=${form.batch_size || 1}`)
        .then(setEst).catch(() => {});
    }, 250);
    return () => clearTimeout(t);
  }, [form.image_size, form.batch_size]);

  /** Which preset (if any) the current form still matches. */
  const activePreset = useMemo(() => {
    const hit = Object.entries(presets).find(([, p]) => (
      p.image_size === form.image_size
      && p.batch_size === form.batch_size
      && p.lr === form.lr
      && p.train_steps === form.train_steps
      && (p.mults || []).join(",") === form.mults
    ));
    return hit ? hit[0] : null;
  }, [presets, form.image_size, form.batch_size, form.lr, form.train_steps, form.mults]);

  const applyPreset = (id) => {
    const p = presets[id];
    if (!p) return;
    setForm((f) => ({
      ...f,
      image_size: p.image_size, batch_size: p.batch_size, lr: p.lr,
      train_steps: p.train_steps, save_every: p.save_every,
      mults: (p.mults || [1, 2, 2, 2]).join(","),
    }));
    toast(`Applied ${p.label}`, "success");
  };

  const start = async () => {
    if (!nameAvailable) {
      toast(`Run name “${trimmedRunName}” is already taken`, "error");
      return;
    }
    try {
      const seeded = fromMode === "library" ? (form.resume || "") : "";
      const payload = {
        ...form,
        mults: form.mults.split(",").map((x) => parseInt(x.trim(), 10)),
        resume: seeded,
        nostrict: !!form.nostrict,
        // Each engine names continuing from a checkpoint differently: xurdif
        // resumes the same run, Diffusers fine-tunes (or trains an adapter).
        mode: form.backend === "xurdif"
          ? (seeded ? "continue" : "scratch")
          : (seeded ? form.mode : "scratch"),
        base_model: seeded || undefined,
      };
      const { job: j, warning } = await api.post("/train", payload);
      setJob(j);
      setInspectRun(form.run_name);
      setMode("run");
      setInfo((prev) => ({ ...prev, train: { status: "training", label: "Training", job_id: j.id, run: form.run_name } }));
      if (warning) toast(warning, "warn");
      const done = await pollJob(j.id, setJob, 800);
      toast(done.message, done.status === "done" ? "success" : "error");
      const p = await api.get("/studio");
      setInfo(p);
      loadRun(form.run_name);
    } catch (e) { toast(e.message, "error"); }
  };

  const stop = async () => {
    if (job) { await api.post(`/jobs/${job.id}/cancel`); toast("Stopping training…"); }
  };

  const continueTraining = async () => {
    if (!inspectRun || running) return;
    try {
      const { job: j, warning } = await api.post(`/runs/${encodeURIComponent(inspectRun)}/continue`, {
        train_steps: continueSteps,
      });
      setJob(j);
      setMode("run");
      setInfo((prev) => ({ ...prev, train: { status: "training", label: "Training", job_id: j.id, run: inspectRun } }));
      if (warning) toast(warning, "warn");
      const done = await pollJob(j.id, setJob, 800);
      toast(done.message, done.status === "done" ? "success" : "error");
      const p = await api.get("/studio");
      setInfo(p);
      loadRun(inspectRun);
    } catch (e) { toast(e.message, "error"); }
  };

  const deleteRun = async () => {
    const name = pendingRun;
    setPendingRun(null);
    if (!name) return;
    try {
      await api.del(`/runs/${name}`);
      toast(`Deleted run ${name}`, "success");
      const p = await api.get("/studio");
      setInfo(p);
      if (inspectRun === name) {
        if (p.runs?.length) loadRun(p.runs[p.runs.length - 1].name);
        else startNew(p.runs);
      } else {
        setForm((f) => ({ ...f, run_name: nextRunName(p.runs) }));
      }
    } catch (e) { toast(e.message, "error"); }
  };

  const copyCmd = async (text) => {
    try { await navigator.clipboard.writeText(text); toast("Command copied", "success"); }
    catch { toast("Copy failed — select and copy manually", "error"); }
  };

  const openSave = (ckpt) => {
    const def = `${inspectRun || form.run_name}-step${ckpt.step ?? ckpt.milestone}`
      .replace(/[^a-zA-Z0-9_-]+/g, "-");
    setSaveName(def);
    setSaveTarget(ckpt);
  };

  const confirmSave = async (thenUse = false) => {
    try {
      const r = await api.post(
        `/runs/${inspectRun || form.run_name}/checkpoints/save`,
        { filename: saveTarget.filename, save_as: saveName.trim() }
      );
      setSaveTarget(null);
      if (thenUse && r.model?.path) {
        setModelPath(r.model.path);
        setAppMode("play");
        setPlayTab("create");
        toast(`Saved “${r.model.name}” — now selected in Create`, "success");
      } else {
        toast(`Saved “${r.model.name}” to the library`, "success");
      }
    } catch (e) { toast(e.message, "error"); }
  };

  const runMeta = runView?.meta || {};
  const multsLabel = Array.isArray(runMeta.mults) ? runMeta.mults.join(",") : (runMeta.mults || "—");
  const sample = job?.detail?.sample || runView?.sample;
  const checkpoints = (running && job?.detail?.checkpoints?.length ? job.detail.checkpoints : (runView?.checkpoints || [])) || [];
  const logLines = (running && job?.detail?.log?.length ? job.detail.log : (runView?.log || [])) || [];
  const losses = (running && job?.detail?.losses?.length ? job.detail.losses : (runView?.losses || [])) || [];
  const cmd = job?.detail?.cmd;
  const showRun = mode === "run" && (inspectRun || running);
  const latestCkptStep = checkpoints.length
    ? Math.max(...checkpoints.map((c) => c.step ?? 0))
    : (runView?.meta?.step || 0);
  const canContinue = showRun && !running && checkpoints.length > 0
    && runView?.status !== "training";

  useEffect(() => {
    if (!canContinue || continueSteps > latestCkptStep) return;
    const target = runView?.meta?.train_steps || form.train_steps || 280000;
    setContinueSteps(Math.max(latestCkptStep + Math.round(target / 4), latestCkptStep + 1000));
  }, [inspectRun, latestCkptStep, canContinue]);

  useEffect(() => {
    if (showLogs && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines.length, showLogs]);

  if (!info) return <Loading>Loading training…</Loading>;

  return (
    <div className="col">
      <div className="card prepare-mode-card">
        <div className="row between center wrap gap-2 mb-2">
          <h3 className="mb-0">Training</h3>
          <div className="row center gap-2 wrap">
            <Seg
              ariaLabel="Training mode"
              tabs={MODE_TABS.map((t) => (t.id === "run" && !hasRuns ? { ...t, tip: "Start a training run first" } : t))}
              value={mode}
              onChange={(m) => {
                if (m === "run" && !hasRuns) {
                  toast("Start a training run first", "error");
                  return;
                }
                if (m === "new") startNew();
                else setMode("run");
              }}
            />
            <span className={`pill ${statusPillClass(running ? "training" : train.status)}`}>
              {running ? "Training" : train.label}
            </span>
          </div>
        </div>
        {mode === "new" ? (
          <p className="hint mb-0">Configure a dataset and hyperparameters, then start training.</p>
        ) : (
          <p className="hint mb-2">Pick a run to inspect samples, loss, and snapshots.</p>
        )}
        {mode === "run" && (
          <div className="run-list mt-2" role="list" aria-label="Training runs">
            {orderedRuns.length ? orderedRuns.map((r) => (
              <div
                key={r.name}
                className={`run-chip ${inspectRun === r.name ? "on" : ""}`}
                role="listitem"
              >
                <button
                  type="button"
                  className="run-chip-main"
                  onClick={() => loadRun(r.name)}
                >
                  <b>{r.name}</b>
                  <span className="sub">{runLabel(r)} · {r.checkpoints} snapshots</span>
                </button>
                <DeleteBtn label={`Delete run ${r.name}`} onClick={(e) => { e.stopPropagation(); setPendingRun(r.name); }} />
              </div>
            )) : (
              <Empty>No runs yet.</Empty>
            )}
          </div>
        )}
      </div>

      {mode === "new" && (
        <div className="work-split">
          <div className="col">
            {!device?.cuda && canStart && (
              <div className="card">
                <span className="pill warn">No GPU</span>
                <p className="hint mt-2 mb-0">Training requires an NVIDIA GPU.</p>
              </div>
            )}

            <div className="card">
              <h3>New run</h3>
              <p className="hint">
                Configure the run, then start. The run folder is created when training begins.
              </p>
              <Select label="Dataset" value={form.dataset} onChange={(v) => set("dataset", v)} disabled={running}
                options={(info?.datasets || []).map((d) => ({ value: d.name, label: `${d.name} (${d.count})` }))}
                tip="A prepared image folder from Data. The same dataset can feed many runs." />
              <Text label="Run name" value={form.run_name} onChange={(v) => set("run_name", v)} disabled={running}
                tip="Folder name for this run’s snapshots. Must be unique." />
              <p className={`hint ${!nameAvailable ? "warn-text" : ""}`}>
                {trimmedRunName && !nameAvailable && "Name already taken"}
                {trimmedRunName && nameAvailable && "Name available"}
              </p>
              <Select
                label="Start from"
                value={fromMode}
                onChange={(v) => { if (v === "scratch") applyLibraryStart(""); else setFromMode(v); }}
                disabled={running}
                options={[
                  { value: "scratch", label: "Scratch (new weights)" },
                  { value: "library", label: "Library model (continue / fine-tune)" },
                ]}
                tip="Scratch builds a new model. Library loads a named model you saved or downloaded."
              />
              {fromMode === "library" && (
                <>
                  <Select
                    label="Base model"
                    value={form.resume}
                    onChange={(v) => applyLibraryStart(v)}
                    disabled={running}
                    options={[
                      { value: "", label: "— pick a library model —" },
                      ...libraryModels.map((m) => ({
                        value: m.path,
                        label: m.original_name && m.original_name !== m.name
                          ? `${m.name} (trained as ${m.original_name})`
                          : m.name,
                      })),
                    ]}
                    tip="Named models in the library — not raw training snapshots. Save a snapshot to the library first."
                  />
                  {baseModel && (
                    <p className="sub mt-0">{modelSubtitle(baseModel)}</p>
                  )}
                </>
              )}
              <div className="row gap-2">
                <div className="grow"><Num label="Image size" value={form.image_size} onChange={(v) => set("image_size", v)} step={32} disabled={running} tip="Training resolution. Higher uses much more VRAM." /></div>
                <div className="grow"><Num label="Batch" value={form.batch_size} onChange={(v) => set("batch_size", v)} min={1} disabled={running} tip="Images per step. Lower if you run out of VRAM." /></div>
              </div>
              <div className="row gap-2">
                <div className="grow"><Num label={fromMode === "library" ? "Train until step" : "Iterations"} value={form.train_steps} onChange={(v) => set("train_steps", v)} step={1000} disabled={running} tip={fromMode === "library" ? "Total optimizer steps. Should be higher than the step count already in the library model." : "How many optimizer steps to run."} /></div>
                <div className="grow"><Num label="Save every" value={form.save_every} onChange={(v) => set("save_every", v)} disabled={running} tip="Write a snapshot every N steps." /></div>
              </div>
              <div className="row gap-2">
                <div className="grow"><Num label="Learning rate" value={form.lr} onChange={(v) => set("lr", v)} step={0.0001} disabled={running} tip="How big each weight update is." /></div>
                <div className="grow"><Num label="Grad accum" value={form.accum} onChange={(v) => set("accum", v)} disabled={running} tip="Accumulate this many micro-batches before an optimizer step." /></div>
              </div>
              {est && est.total_mib > 0 && canStart && (
                <div className={`vram-hint ${est.risky ? "risky" : ""}`}>
                  <span>Est. peak VRAM <b>~{est.estimate_mib} MiB</b> / {est.total_mib} MiB{est.free_mib ? ` · ${est.free_mib} free` : ""}</span>
                  {est.risky && (
                    <span className="warn-text">May exceed VRAM — try batch ≤ {est.recommended_batch} at {form.image_size}px.</span>
                  )}
                </div>
              )}
            </div>

            {canStart && (
              <Disclose
                title="Snapshot thumbnails"
                tip="How the preview image saved beside each snapshot is rendered. A fixed seed makes the thumbnails across a run comparable — the only thing that changes is how far training got."
              >
                <p className="hint mb-2">
                  Rendered every <b>{form.save_every}</b> steps, beside each snapshot. Keep the seed fixed
                  and the run reads as one image evolving; set it to <b>-1</b> for a fresh random sample each time.
                </p>
                <div className="row gap-2">
                  <div className="grow">
                    <Num label="Images per snapshot" value={form.nsamples} min={1} max={16}
                      onChange={(v) => set("nsamples", Math.max(1, Math.min(16, Math.round(v) || 1)))}
                      disabled={running}
                      tip="How many samples go into the preview. 1 gives a single image; more are laid out as a grid." />
                  </div>
                  <div className="grow">
                    <Num label="Seed" value={form.sample_seed} min={-1}
                      onChange={(v) => set("sample_seed", Math.round(v))}
                      disabled={running}
                      tip="Seed for the snapshot preview. -1 draws a new random sample each time." />
                  </div>
                </div>
              </Disclose>
            )}

            {canStart && (
              <Disclose title="Advanced"
                tip="How the network is built and how it learns. The defaults are the ones Kiln's presets are measured against — change these only when you want a different kind of model, not to fix a slow or poor run.">
                {engines.length > 1 && (
                  <Select label="Engine" value={form.backend} onChange={(v) => set("backend", v)}
                    options={engines.map((e) => e.name)}
                    tip="xurdif trains the compact models Kiln started with. diffusers trains Hugging Face UNet2DModel models and can fine-tune ones you import." />
                )}

                {form.backend === "xurdif" ? (
                  <>
                    <Select label="Architecture" value={form.mtype} onChange={(v) => set("mtype", v)} options={archs.length ? archs : [form.mtype]}
                      tip="Which network shape to build. The variants with attention see more of the image at once, which helps with overall composition but costs speed and memory. Fixed for the life of a model: you cannot change it later and resume." />
                    <Text label="Channel multipliers" value={form.mults} onChange={(v) => set("mults", v)}
                      tip={"Width of the network at each resolution, coarsest last. \"1 2 2 2\" is the default.\n\nBigger numbers mean more capacity and a larger, slower model; the count of numbers sets how many times the image is halved, so it also decides the smallest resolution the model works at. Two models can only be merged if these match."} />
                    <div className="row gap-2">
                      <div className="grow"><Select label="Prediction" value={form.pred} onChange={(v) => set("pred", v)} options={["x0", "eps"]}
                        tip={"What the network is asked to output at each step.\n\nx0 predicts the finished image directly and tends to settle faster on small datasets. eps predicts the noise to remove, the classic formulation. Fixed for the life of a model."} /></div>
                      <div className="grow"><Select label="Loss" value={form.loss_type} onChange={(v) => set("loss_type", v)} options={["l1", "l2"]}
                        tip={"How error is measured while training.\n\nl1 is the absolute difference — more forgiving of outliers, and it keeps edges crisp. l2 squares the error, punishing big mistakes harder, which tends to look smoother and blurrier."} /></div>
                    </div>
                    <div className="row gap-2">
                      <div className="grow"><Num label="SSIM weight" value={form.ssimw} onChange={(v) => set("ssimw", v)} step={0.5}
                        tip="How much structural similarity is mixed into the loss on top of Loss above. 0 turns it off. Raising it pushes the model toward matching local structure and texture rather than just pixel values; too high and training can stall." /></div>
                      <div className="grow"><Select label="Fit" value={form.fit} onChange={(v) => set("fit", v)} options={["resize", "crop"]}
                        tip={"How training images that are not square are made to fit.\n\nresize squashes the whole image to the training size, keeping everything but distorting proportions. crop takes a center square, keeping proportions but discarding the edges."} /></div>
                    </div>
                  </>
                ) : (
                  <DiffusersOptions form={form} set={set} engine={engines.find((e) => e.name === form.backend)} seeded={fromMode === "library"} />
                )}
                {form.backend === "xurdif" && (
                  <Tooltip text="Runs much of the math at half precision. Roughly doubles training speed and halves memory use, at a small risk of numerical instability. Leave it on unless a run produces NaN losses.">
                    <label className="row center gap-2 has-tip">
                      <input type="checkbox" checked={form.amp} onChange={(e) => set("amp", e.target.checked)} />
                      <span className="sub">Mixed precision (AMP)</span>
                    </label>
                  </Tooltip>
                )}
              </Disclose>
            )}

            {running ? (
              <button type="button" className="btn danger w-full" onClick={stop}>Stop training</button>
            ) : (
              <button type="button" className="btn primary w-full" onClick={start}
                disabled={!form.dataset || (fromMode === "library" && !form.resume) || !nameAvailable || !trimmedRunName}>
                Start training
              </button>
            )}
          </div>
          <div className="col">
            <div className="card">
              <h3>Presets</h3>
              <p className="hint">
                A preset fills in image size, batch, learning rate and schedule for a
                given GPU budget. Start there, then adjust.
              </p>
              {form.backend !== "xurdif" ? (
                <Empty>
                  These presets are measured for the xurdif engine. For {form.backend},
                  pick a model size under Advanced and set image size and batch to suit
                  your GPU.
                </Empty>
              ) : Object.entries(presets).length === 0 ? (
                <Empty>No presets available.</Empty>
              ) : (
                <div className="col gap-2">
                  {Object.entries(presets).map(([id, pr]) => (
                    <button
                      key={id}
                      type="button"
                      className={`asset-row ${activePreset === id ? "on" : ""}`}
                      onClick={() => applyPreset(id)}
                      disabled={running}
                      aria-pressed={activePreset === id}
                    >
                      <div className="meta">
                        <b>{pr.label}</b>
                        <span className="sub">
                          {pr.image_size}px · batch {pr.batch_size} · lr {pr.lr}
                          {pr.train_steps ? ` · ${pr.train_steps.toLocaleString()} steps` : ""}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
              <p className="callout mt-2 mb-0">
                Once started, this view switches to the live run — samples, loss curve,
                and snapshots you can save into the library.
              </p>
            </div>
          </div>
        </div>
      )}

      {showRun && (
        <div className="work-split">
          <div className="col">
            <div className="card">
              <h3>{running ? "Live output" : `Run · ${inspectRun}`}</h3>
              {job || runView ? (
                <>
                  <div className="row between center mb-2 gap-3">
                    <span className={`pill ${statusPillClass(running ? "training" : train.status)} ${job?.status === "error" ? "bad" : ""} ${job?.status === "done" ? "good" : ""}`}>
                      {running ? "Training" : (job?.status === "done" ? "Completed" : job?.status === "cancelled" ? "Stopped" : job?.status === "error" ? "Error" : runLabel({ status: runView?.status, checkpoints: checkpoints.length }))}
                    </span>
                    <div className="row center gap-3 grow">
                      <span className="sub grow">{job?.message || train.message}</span>
                      {running && <button type="button" className="btn danger sm" onClick={stop}>Stop</button>}
                    </div>
                  </div>
                  {running && <Progress value={job.progress} />}
                  {job?.detail?.warning && <div className="warn-banner">⚠ {job.detail.warning}</div>}
                  {canContinue && (
                    <div className="continue-run-box mt-2">
                      <p className="hint mb-2">Resume this run from its latest checkpoint ({latestCkptStep} steps).</p>
                      <Num
                        label="Train until step"
                        value={continueSteps}
                        onChange={setContinueSteps}
                        min={latestCkptStep + 1}
                        step={1000}
                        tip="Total optimizer steps — must exceed the current checkpoint step."
                      />
                      <button type="button" className="btn primary w-full mt-2" onClick={continueTraining}
                        disabled={continueSteps <= latestCkptStep}>
                        Continue training
                      </button>
                    </div>
                  )}
                  {cmd && running && (
                    <div className="cmd-box">
                      <div className="row between center mb-2">
                        <span className="sub">
                          Subprocess{job?.detail?.pid ? ` · pid ${job.detail.pid}` : ""}
                          {job?.detail?.cuda === true ? " · CUDA ✓" : job?.detail?.cuda === false ? " · CPU ⚠ (no CUDA)" : ""}
                        </span>
                        <button type="button" className="btn ghost sm" onClick={() => copyCmd(cmd)}>Copy</button>
                      </div>
                      <code>{cmd}</code>
                    </div>
                  )}
                  <div className="preview-box preview-scroll preview-h-lg mt-2">
                    {sample ? <img src={mediaUrl(sample)} alt="sample" /> : <span className="sub">Sample grid appears at the first snapshot.</span>}
                  </div>
                </>
              ) : <Empty>Loading run…</Empty>}
            </div>

            <div className="card">
              <h3>Loss</h3>
              <LossChart points={losses} />
            </div>

            {Object.keys(runMeta).length > 0 && (
              <Disclose title="Training settings" tip="Parameters recorded in run.json for this run.">
                <div className="run-params">
                  <div className="kv"><span>Dataset</span><b>{runMeta.dataset || "—"}</b></div>
                  <div className="kv"><span>Image size</span><b>{runMeta.image_size ?? "—"} px</b></div>
                  <div className="kv"><span>Batch size</span><b>{runMeta.batch_size ?? "—"}</b></div>
                  <div className="kv"><span>Train steps</span><b>{runMeta.train_steps?.toLocaleString?.() ?? runMeta.train_steps ?? "—"}</b></div>
                  <div className="kv"><span>Save every</span><b>{runMeta.save_every ?? "—"} steps</b></div>
                  <div className="kv"><span>Architecture</span><b>{runMeta.mtype || "—"}</b></div>
                  <div className="kv"><span>Channel multipliers</span><b>{multsLabel}</b></div>
                  <div className="kv"><span>Prediction</span><b>{runMeta.pred || "—"}</b></div>
                  <div className="kv"><span>Learning rate</span><b>{runMeta.lr ?? "—"}</b></div>
                  <div className="kv"><span>Accumulation</span><b>{runMeta.accum ?? "—"}</b></div>
                  <div className="kv"><span>Diffusion steps</span><b>{runMeta.diffusion_steps ?? "—"}</b></div>
                  <div className="kv"><span>Loss</span><b>{runMeta.loss_type || "—"} (L1 {runMeta.l1w ?? 1}, SSIM {runMeta.ssimw ?? 0})</b></div>
                  <div className="kv"><span>Fit mode</span><b>{runMeta.fit || "—"}</b></div>
                  <div className="kv"><span>Samples / snapshot</span><b>{runMeta.nsamples ?? "—"}</b></div>
                  <div className="kv"><span>Snapshot seed</span><b>{runMeta.sample_seed == null ? "—" : (runMeta.sample_seed < 0 ? "random" : runMeta.sample_seed)}</b></div>
                  <div className="kv"><span>AMP</span><b>{runMeta.amp ? "On" : "Off"}</b></div>
                  <div className="kv"><span>Starting model</span><b>{resumeLabel(runMeta.resume)}</b></div>
                  {runMeta.continued_from && (
                    <div className="kv"><span>Continued from</span><b>{runMeta.continued_from}</b></div>
                  )}
                  {(runMeta.continues?.length > 0) && (
                    <div className="kv"><span>Continue history</span><b>{runMeta.continues.length} resume{runMeta.continues.length === 1 ? "" : "s"}</b></div>
                  )}
                </div>
              </Disclose>
            )}
          </div>

          <div className="col">
            <div className="card">
              <p className="callout mb-2">
                These are training checkpoints, not library models. <b>Save one to the library</b> to
                sample, paint, or continue training from it later.
              </p>
              <CheckpointGallery checkpoints={checkpoints} onSave={openSave} />
            </div>

            <div className="card">
              <div className="row between center">
                <h3 className="mb-0">Logs</h3>
                <button type="button" className="btn ghost sm" onClick={() => setShowLogs((s) => !s)}>
                  {showLogs ? "Hide" : "Show"}
                </button>
              </div>
              {showLogs && (
                <pre className="logbox" ref={logRef}>
                  {logLines.length ? logLines.join("\n") : "No trainer output yet."}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}

      {saveTarget && (
        <Modal
          title="Save snapshot to library"
          onClose={() => setSaveTarget(null)}
          footer={<>
            <button type="button" className="btn ghost" onClick={() => setSaveTarget(null)}>Cancel</button>
            <button type="button" className="btn" onClick={() => confirmSave(false)} disabled={!saveName.trim()}>Save</button>
            <button type="button" className="btn primary" onClick={() => confirmSave(true)} disabled={!saveName.trim()}>
              Save &amp; use in Create
            </button>
          </>}
        >
          <p className="sub mt-0">
            Copies <b>{saveTarget.filename}</b> into the library as a named model you can sample, paint, or continue training from.
            The original training name is kept if you rename it later.
          </p>
          <Text label="Model name" value={saveName} onChange={setSaveName} />
        </Modal>
      )}

      {pendingRun && (
        <ConfirmModal
          title="Delete training run"
          body={`Delete training run “${pendingRun}”? This removes its checkpoints, samples, and logs.`}
          confirmLabel="Delete run"
          danger
          onCancel={() => setPendingRun(null)}
          onConfirm={deleteRun}
        />
      )}
    </div>
  );
}
