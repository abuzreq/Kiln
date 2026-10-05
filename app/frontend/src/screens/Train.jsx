import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, mediaUrl, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { Text, Num, Select, Progress, Empty, Modal, Disclose, ConfirmModal, DeleteBtn, Loading, Seg, Tooltip } from "../components/ui.jsx";
import { modelSubtitle } from "../components/modelMeta.jsx";
import {
  KINDS as ATTN_KINDS, depthOf, kindAt, layoutIdFor, trimToDepth, withKind,
} from "../attnLayout.js";
import ArchSketch, { levelInfo } from "../components/ArchSketch.jsx";
import LossChart from "../components/LossChart.jsx";
import { LrScheduleField, LiveRateCard, fmtRate } from "../components/LrSchedule.jsx";
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

function edgeLossDefault(backend) {
  // xurdif's native training loss. Diffusers' stock target is MSE; the same
  // edge-weighted L1 is available there as an opt-in, not the starting point.
  return backend === "xurdif";
}

// The schedule a from-scratch run starts on. Lowering the rate once the model
// has the general idea is the engine author's own practice and the reason this
// exists, so it is the default rather than something to go and find. A fine-tune
// overrides it to "constant": its rate is already chosen by the distance control,
// and decaying away from that is not what the control means.
const SCRATCH_LR_SCHEDULE = "drops-2";

const EMPTY_FORM = {
  backend: "xurdif", mode: "scratch", preset: "standard-128",
  lora_r: 8, precision: "no", gradient_checkpointing: false,
  dataset: "", run_name: "run1", image_size: 512, batch_size: 8,
  diffusion_steps: 1000, train_steps: 280000, accum: 10, lr: 0.0004,
  loss_type: "l1", ssimw: 0, l1w: 1, pred: "x0",
  // One control, both engines. start() translates it: diffusers spells it
  // objective:"xurdif"|"mse", xurdif takes edge_loss straight through.
  edge_loss: edgeLossDefault("xurdif"),
  // The attention layout only means something to the configurable architecture.
  // It is the vendor's spec string; the pickers under "custom" are a view of it.
  mtype: "tinyunet_conf_attention", attn: "-1:linear,mid:full", mults: "1,2,2,4", save_every: 1000,
  nsamples: 1, sample_seed: 42, fit: "resize", amp: true, resume: "", nostrict: false,
  // A named schedule, or "custom" with lr_plan holding the segments. Either way
  // the server compiles it to absolute steps before the run starts.
  lr_schedule: SCRATCH_LR_SCHEDULE, lr_plan: null,
  // Warm the rate up over the run's first steps; the server sizes it (2% of the
  // run, at most 1000 steps). Off sends lr_warmup: 0.
  warmup: true,
};

/** The run-size fields an xurdif preset sets. */
function presetFields(p) {
  return {
    image_size: p.image_size, batch_size: p.batch_size, lr: p.lr,
    train_steps: p.train_steps, save_every: p.save_every,
    mults: (p.mults || [1, 2, 2, 2]).join(","),
  };
}

// How far a fine-tune should travel from the model it starts on. Learning rate
// is the whole mechanism, so this is the only dial that decides it. It
// deliberately does not touch the step target, and a fine-tune holds its rate
// for the whole run rather than decaying from it.
const FT_DISTANCE = [
  { id: "close", label: "Stay close", lr: 5e-5,
    tip: "Small steps. Keeps the base model's look and picks up your images slowly." },
  { id: "some", label: "Move somewhat", lr: 1.5e-4,
    tip: "The middle setting. Shifts noticeably toward your images while keeping the base model's habits." },
  { id: "far", label: "Go far", lr: 4e-4,
    tip: "Big steps. Learns your images fastest and is the most likely to lose what the base model knew." },
];
// Fine-tuning starts here rather than at whatever lr the base model was trained
// with -- inheriting 4e-4 from a scratch run is the worst default for a tune.
const FT_DEFAULT_LR = 1.5e-4;

/** What a recorded run actually optimised.
 *
 *  Not `loss_type`: that field is written to run.json but the vendored trainer
 *  parses `--losstype` and never uses it. What runs is the edge-weighted L1
 *  (plus SSIM when weighted), or plain MSE on the Diffusers engine.
 */
function lossLabel(meta) {
  if (!meta) return "—";
  if (meta.backend === "diffusers") {
    return meta.objective === "xurdif" ? "Edge-aware L1 (xurdif objective)" : "MSE";
  }
  const edges = meta.edge_loss === false ? "plain L1" : "edge-weighted L1";
  return `${edges} (SSIM ${meta.ssimw ?? 0})`;
}

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
function DiffusersOptions({ form, set, engine, seeded, onMps }) {
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
        <Select label="Model size" value={form.preset}
          onChange={(v) => {
            // Each network is built for one size; train it at that size unless
            // the image size is changed afterwards.
            set("preset", v);
            if (presets[v]?.sample_size) set("image_size", presets[v].sample_size);
          }}
          options={Object.keys(presets).length ? Object.keys(presets) : [form.preset]}
          tip={presets[form.preset]?.label || "Architecture preset for a new model."} />
      )}

      {seeded && form.mode === "lora" && (
        <Num label="LoRA rank" value={form.lora_r} onChange={(v) => set("lora_r", Math.round(v))} step={4}
          tip="Adapter capacity. Higher fits more, costs more. 4–16 is usual." />
      )}

      <div className="row gap-2">
        {/* Apple's GPU trains in fp32 for now; the server would override it anyway. */}
        {!onMps && (
          <div className="grow">
            <Select label="Precision" value={form.precision} onChange={(v) => set("precision", v)}
              options={["no", "fp16", "bf16"]}
              tip="Mixed precision. fp16 is faster and lighter on most NVIDIA cards; bf16 needs a newer one." />
          </div>
        )}
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
    device, toast, trainFromPath, setTrainFromPath, trainEngines, trainableHere,
    trainDataset, setTrainDataset, setModelPath, setAppMode, setPlayTab, models,
  } = useApp();
  const [info, setInfo] = useState(null);
  const [presets, setPresets] = useState({});
  const [archs, setArchs] = useState([]);
  // Named attention layouts and the default spec come from the server, so the
  // list lives in one place (app/core/backends/xurdif/attn.py).
  const [archInfo, setArchInfo] = useState({ layouts: [], default_attn: "", conf_mtype: "tinyunet_conf_attention", base_dim: 64 });
  // "Custom" stays open once chosen even while the pickers happen to spell a
  // named layout, so the panel does not snap shut mid-edit.
  const [customLayout, setCustomLayout] = useState(false);
  const engines = trainEngines.backends;
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
  // Whether the form has already been seeded (by the recommended preset, or by
  // a model handed over from Library). Guards against the two racing.
  const presetSeeded = useRef(false);
  const arrivedWithDataset = useRef("");
  const [continueSteps, setContinueSteps] = useState(280000);
  const [lrPresets, setLrPresets] = useState({});
  const [lrBusy, setLrBusy] = useState(false);
  // What the server last said the schedule is. run.json carries a summary too,
  // but it goes stale the moment the rate is changed live.
  const [liveSummary, setLiveSummary] = useState(null);
  const [continuePlan, setContinuePlan] = useState(null);
  // The step a library checkpoint's xurdif run counts on from, so the schedule
  // preview sizes the warm-up over the steps the run will actually take.
  const [resumeStep, setResumeStep] = useState(0);
  const logRef = useRef(null);

  const [form, setForm] = useState({ ...EMPTY_FORM });
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  // Attention layout, derived from the spec string in the form.
  const isConf = form.backend === "xurdif" && form.mtype === archInfo.conf_mtype;
  const layouts = archInfo.layouts || [];
  const layoutId = layoutIdFor(form.attn, layouts);
  const layoutLabel = (spec) => {
    const hit = layouts.find((l) => l.spec === spec);
    return hit ? `${hit.label} (${spec})` : (spec || "—");
  };

  const train = info?.train || { status: "not_started", label: "Not started" };
  const running = job && job.status === "running";
  const canStart = !running;
  // The engine the form names, and why this machine cannot train it (a Mac
  // has no GPU for the xurdif trainer), or null when it can.
  const engineInfo = engines.find((e) => e.name === form.backend);
  const engineBlocked = engineInfo?.trainable_here === false
    ? (engineInfo.unavailable_reason || `${form.backend} cannot train on this machine.`)
    : null;
  const onMps = device?.device === "mps";

  // A form moved to another engine for a new model: that engine's loss and its
  // own starting numbers. xurdif's come from the run-size preset that suits this
  // GPU; a Diffusers model's from the engine, at the size its network is built for.
  const withEngine = (f, name, presetMap = presets) => {
    const engine = engines.find((e) => e.name === name);
    const next = { ...f, backend: name, edge_loss: edgeLossDefault(name), ...(engine?.defaults || {}) };
    if (name === "xurdif") {
      const rec = Object.keys(presetMap || {}).find((k) => presetMap[k].recommended);
      return rec ? { ...next, ...presetFields(presetMap[rec]) } : next;
    }
    const size = engine?.presets?.[f.preset]?.sample_size;
    return size ? { ...next, image_size: size } : next;
  };
  const runs = info?.runs || [];
  const orderedRuns = useMemo(() => runs.slice().reverse(), [runs]);
  const hasRuns = orderedRuns.length > 0;
  const trimmedRunName = (form.run_name || "").trim();
  // Only models some engine here can train: on a Mac that leaves Diffusers ones.
  const libraryModels = useMemo(
    () => (models || []).filter((m) => m.role !== "checkpoint" && trainableHere(m.backend)),
    [models, trainableHere],
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
    setResumeStep(0);
    // A new run lands on its engine's own starting numbers -- for xurdif, the
    // preset that suits this GPU -- rather than on the size and batch the
    // previous run happened to leave behind.
    setForm((f) => withEngine({
      ...EMPTY_FORM,
      dataset: f.dataset || info?.datasets?.[0]?.name || "",
      run_name: nextRunName(runsList || info?.runs || runs),
      mtype: f.mtype,
    }, trainEngines.default));
  };

  const applyLibraryStart = async (path) => {
    if (!path) {
      // Back to a new model: its engine's scratch numbers and schedule, not the
      // fine-tune rate the base model left behind.
      setFromMode("scratch");
      setResumeStep(0);
      setForm((f) => ({
        ...withEngine(f, f.backend),
        resume: "", nostrict: false, lr_schedule: SCRATCH_LR_SCHEDULE, lr_plan: null,
      }));
      return;
    }
    setFromMode("library");
    setMode("new");
    presetSeeded.current = true;
    // The model decides the engine. A Diffusers model fine-tunes (or takes an
    // adapter) on the Diffusers trainer, and /train/continue/info reads only
    // xurdif checkpoints -- so it is never asked about one.
    const picked = (models || []).find((m) => m.path === path);
    if (picked?.backend === "diffusers") {
      const caps = engines.find((e) => e.name === "diffusers")?.capabilities || {};
      setResumeStep(0);
      setForm((f) => ({
        ...withEngine(f, "diffusers"),
        mode: caps.finetune ? "finetune" : caps.lora ? "lora" : "finetune",
        resume: path,
        image_size: picked.sample_size ?? f.image_size,
        lr: FT_DEFAULT_LR,
        lr_schedule: "constant", lr_plan: null,   // see SCRATCH_LR_SCHEDULE
      }));
      return;
    }
    if (picked && form.backend !== "xurdif") {
      const defaults = engines.find((e) => e.name === "xurdif")?.defaults || {};
      setForm((f) => ({ ...f, ...defaults, backend: "xurdif", edge_loss: edgeLossDefault("xurdif") }));
    }
    try {
      const d = await api.get(`/train/continue/info?path=${encodeURIComponent(path)}`);
      setResumeStep(Number(d.step) || 0);
      setForm((f) => ({
        ...f,
        resume: path,
        image_size: d.config?.image_size ?? f.image_size,
        batch_size: d.config?.batch_size ?? f.batch_size,
        train_steps: d.suggested_steps ?? f.train_steps,
        // save_every and lr are deliberately NOT inherited. The first is a disk
        // preference, not an architecture fact, and older runs carry 100. The
        // second is the base model's *training* rate -- often 4e-4, the worst
        // place to start a fine-tune from. Everything below is architecture and
        // must match, or the checkpoint will not load.
        lr: FT_DEFAULT_LR,
        lr_schedule: "constant", lr_plan: null,   // see SCRATCH_LR_SCHEDULE
        mtype: d.mtype || f.mtype,
        mults: Array.isArray(d.mults) ? d.mults.join(",") : f.mults,
        attn: d.attn ?? f.attn,
        pred: d.pred || f.pred,
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
        const arrived = arrivedWithDataset.current;
        arrivedWithDataset.current = "";
        if (!arrived && p.datasets?.length) set("dataset", p.datasets[0].name);
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
        // Opening Train means starting a run far more often than revisiting an
        // old one, so it always lands on New run; previous runs are one click
        // away. A run that is actually training is the exception, above.
        if (arrived) set("dataset", arrived);
        setMode("new");
      } catch (e) {
        if (!cancelled) toast(e.message, "error");
      }
    })();
    api.get("/train/presets").then((p) => {
      setPresets(p);
      const rec = Object.keys(p).find((k) => p[k].recommended);
      // The ref, not a check on form state: the deep link from Library resolves
      // asynchronously too, and whichever lands second must not win.
      if (rec && !presetSeeded.current) {
        presetSeeded.current = true;
        applyPresetFrom(p, rec, { announce: false });
      }
    });
    api.get("/architectures").then((d) => {
      setArchs(d.architectures);
      setArchInfo({
        layouts: d.layouts || [], default_attn: d.default_attn || "",
        conf_mtype: d.conf_mtype || "tinyunet_conf_attention", base_dim: d.base_dim || 64,
      });
      if (d.default_attn) setForm((f) => ({ ...f, attn: f.attn || d.default_attn }));
    });
    return () => { cancelled = true; };
  }, []);

  // EMPTY_FORM names xurdif; on a machine that cannot train it, start on the
  // engine the server offers instead, once the list has arrived.
  useEffect(() => {
    if (trainableHere(form.backend)) return;
    setForm((f) => withEngine(f, trainEngines.default));
  }, [trainEngines]);

  useEffect(() => {
    if (!trainFromPath) return;
    applyLibraryStart(trainFromPath);
    setTrainFromPath("");
  }, [trainFromPath]);

  // arriving from "Train on this dataset" in Prepare ▸ Data. The /studio load
  // below is still in flight on a fresh mount and would otherwise replace this
  // dataset with the first one in the list, so it checks the ref.
  useEffect(() => {
    if (!trainDataset) return;
    arrivedWithDataset.current = trainDataset;
    setMode("new");
    set("dataset", trainDataset);
    setTrainDataset("");
  }, [trainDataset]);

  useEffect(() => {
    const t = setTimeout(() => {
      const q = new URLSearchParams({
        image_size: form.image_size || 0,
        batch_size: form.batch_size || 1,
        // Width changes what a run costs -- mostly through optimizer state rather
        // than activations -- so the readout has to be told about it.
        mults: form.mults || "",
        mtype: form.mtype || "",
        attn: form.attn || "",
        // Mixed precision roughly halves the activation memory, which is the
        // whole batch term -- without it the readout is ~2x out.
        amp: form.amp ? "1" : "0",
      });
      api.get(`/train/estimate?${q}`).then(setEst).catch(() => {});
    }, 250);
    return () => clearTimeout(t);
  }, [form.image_size, form.batch_size, form.mults, form.mtype, form.attn, form.amp]);

  // The schedules, compiled by the server against this run's own base rate and
  // step target, so the step numbers shown are the ones the trainer will use.
  // Debounced like the estimate above, and for the same reason.
  useEffect(() => {
    const t = setTimeout(() => {
      const q = new URLSearchParams({
        lr: form.lr || 0.0004,
        train_steps: form.train_steps || 280000,
        save_every: form.save_every || 1000,
        // The warm-up is sized over the steps the run will take, which for an
        // xurdif library start begin at the checkpoint's own step.
        start_step: fromMode === "library" && form.backend === "xurdif" ? resumeStep : 0,
      });
      if (!form.warmup) q.set("warmup", "0");
      api.get(`/train/lr_presets?${q}`).then(setLrPresets).catch(() => {});
    }, 250);
    return () => clearTimeout(t);
  }, [form.lr, form.train_steps, form.save_every, form.warmup, form.backend, fromMode, resumeStep]);

  const datasetInfo = useMemo(
    () => (info?.datasets || []).find((d) => d.name === form.dataset) || null,
    [info, form.dataset]);

  /** Which preset (if any) the current form still matches. */
  const activePreset = useMemo(() => {
    const hit = Object.entries(presets).find(([, p]) => (
      p.image_size === form.image_size
      && p.batch_size === form.batch_size
      && p.lr === form.lr
      && p.train_steps === form.train_steps
      && p.save_every === form.save_every
      && (p.mults || []).join(",") === String(form.mults ?? "")
    ));
    return hit ? hit[0] : null;
  }, [presets, form.image_size, form.batch_size, form.lr, form.train_steps, form.save_every, form.mults]);

  // Split in two so the mount handler can seed from the response it just got,
  // before `presets` state exists on that tick.
  const applyPresetFrom = (map, id, { announce = true } = {}) => {
    const p = map?.[id];
    if (!p) return;
    // These are xurdif's run sizes. The mount-time seeding can land after the
    // form has moved to Diffusers (a Mac), and must not overwrite its numbers.
    setForm((f) => (f.backend === "xurdif" ? { ...f, ...presetFields(p) } : f));
    if (announce) toast(`Applied ${p.label}`, "success");
  };

  const applyPreset = (id, opts) => applyPresetFrom(presets, id, opts);

  // Flask sorts JSON keys, so trust the server's rank rather than key order.
  const presetRows = useMemo(
    () => Object.entries(presets).sort((a, b) => (a[1].order ?? 0) - (b[1].order ?? 0)),
    [presets],
  );

  // Derived, not stored: the lr is the single source of truth, so typing one by
  // hand under Advanced simply deselects all three tabs.
  const ftDistance = useMemo(
    () => FT_DISTANCE.find((d) => d.lr === form.lr)?.id || "",
    [form.lr],
  );

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
        // One checkbox, two spellings. Diffusers picks a loss by name; the
        // xurdif trainer takes a flag. Sending both is harmless -- each backend
        // reads only the keys it knows.
        objective: form.edge_loss ? "xurdif" : "mse",
        // Left out, the server sizes the warm-up; 0 turns it off.
        lr_warmup: form.warmup ? undefined : 0,
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

  /** Write a run's schedule. One path for a run that is training and one that is
   *  stopped: the server writes a file either way and says when it takes hold. */
  const putLrPlan = async (body) => {
    if (!inspectRun) return;
    setLrBusy(true);
    try {
      const r = await api.put(`/runs/${encodeURIComponent(inspectRun)}/lr_plan`, body);
      setLiveSummary(r.summary);
      setContinuePlan(r.plan);
      toast(r.at_step != null
        ? `Learning rate ${fmtRate(r.lr_now)} from step ${r.at_step} — applies ${r.applies}`
        : `Schedule set to ${r.summary} — applies on the ${r.applies}`, "success");
    } catch (e) { toast(e.message, "error"); }
    finally { setLrBusy(false); }
  };

  const continueTraining = async () => {
    if (!inspectRun || running) return;
    try {
      const { job: j, warning } = await api.post(`/runs/${encodeURIComponent(inspectRun)}/continue`, {
        train_steps: continueSteps,
        // Continuing at a lower rate is the habit this replaces. null means
        // "whatever the run already has", which is read back from its plan file.
        lr_plan: continuePlan,
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
  // The past (recorded events) while a run goes, the whole picture from disk
  // afterwards; see lr_plan.marks_for_chart.
  const lrMarks = (running && job?.detail?.lr_marks?.length
    ? job.detail.lr_marks : (runView?.lr_marks || [])) || [];
  const liveRate = running ? job?.detail?.lr : null;
  const formSchedule = form.lr_plan ? null : lrPresets[form.lr_schedule];
  const runSchedule = liveSummary || runMeta.lr_schedule_summary || null;
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
                // Previous runs open on the newest one rather than on nothing,
                // or on whichever run was last looked at in this session.
                else loadRun(orderedRuns[0]?.name);
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
        <div className="work-split form-first">
          <div className="col">
            {engineBlocked && canStart ? (
              <div className="card">
                <span className="pill warn">Not on this machine</span>
                <p className="hint mt-2 mb-0">{engineBlocked}</p>
              </div>
            ) : !device?.cuda && !device?.mps && canStart && (
              <div className="card">
                <span className="pill warn">No GPU</span>
                <p className="hint mt-2 mb-0">{device?.hint || "Training requires an NVIDIA GPU."}</p>
              </div>
            )}

            <div className="card">
              <h3>New run</h3>
              <p className="hint">
                Configure the run, then start. The run folder is created when training begins.
              </p>
              {/* The three things every run needs, on one line. Each keeps its own
                  note underneath, so the row stays readable when one of them has
                  something to say. */}
              <div className="field-row">
                <div>
                  <Select label="Dataset" value={form.dataset} onChange={(v) => set("dataset", v)} disabled={running}
                    options={(info?.datasets || []).map((d) => ({ value: d.name, label: `${d.name} (${d.count})` }))}
                    tip="A prepared image folder from Data. The same dataset can feed many runs." />
                  {datasetInfo && (
                    <p className="hint mb-0">
                      {datasetInfo.total > datasetInfo.count
                        ? `${datasetInfo.total.toLocaleString()} images per pass: `
                          + `${datasetInfo.count.toLocaleString()} × ${Math.round(datasetInfo.total / datasetInfo.count)} `
                          + "versions from its recipe."
                        : `${datasetInfo.count.toLocaleString()} images per pass.`}
                    </p>
                  )}
                </div>
                <div>
                  <Text label="Run name" value={form.run_name} onChange={(v) => set("run_name", v)} disabled={running}
                    tip="Folder name for this run’s snapshots. Must be unique." />
                  <p className={`hint mb-0 ${!nameAvailable ? "warn-text" : ""}`}>
                    {trimmedRunName && !nameAvailable && "Name already taken"}
                    {trimmedRunName && nameAvailable && "Name available"}
                  </p>
                </div>
                <div>
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
                </div>
              </div>
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
              {fromMode === "library" ? (
                <>
                  <div className="section-title mt-2">How far from the base model</div>
                  <Seg tabs={FT_DISTANCE} value={ftDistance} onChange={(id) => set("lr", FT_DISTANCE.find((d) => d.id === id).lr)}
                    ariaLabel="How far to move from the base model" />
                  <p className="hint mt-1">
                    Learning rate <b>{form.lr}</b>{ftDistance ? "" : " (custom)"}. Fine-tuning starts
                    lower than the rate the base model was trained at.
                  </p>
                </>
              ) : form.backend !== "xurdif" ? (
                <p className="hint mt-2">
                  Pick a model size under Advanced — Diffusers presets choose the network
                  shape rather than a GPU budget.
                </p>
              ) : presetRows.length > 0 && (
                <>
                  <div className="section-title mt-2">Preset</div>
                  <div className="col gap-2">
                    {presetRows.map(([id, pr]) => (
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
                          {pr.blurb && <span className="sub">{pr.blurb}</span>}
                          <span className="sub">
                            {pr.image_size}px · batch {pr.batch_size} · lr {pr.lr}
                            {pr.train_steps ? ` · ${pr.train_steps.toLocaleString()} steps` : ""}
                          </span>
                          {pr.fits === false && (
                            <span className="sub warn-text">Larger than your GPU</span>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                  {!activePreset && (
                    <p className="hint mt-1 mb-0">Custom — edited under Advanced.</p>
                  )}
                </>
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
                <div className="adv-grid">
                <div>
                <div className="section-title">Run size</div>
                <div className="row gap-2">
                  <div className="grow"><Num label="Image size" value={form.image_size} onChange={(v) => set("image_size", v)} step={32} disabled={running} tip="Training resolution. Higher uses much more VRAM." /></div>
                  <div className="grow"><Num label="Batch" value={form.batch_size} onChange={(v) => set("batch_size", v)} min={1} disabled={running} tip="Images per step. Lower if you run out of VRAM." /></div>
                </div>
                <div className="row gap-2">
                  <div className="grow"><Num label={fromMode === "library" ? "Train until step" : "Iterations"} value={form.train_steps} onChange={(v) => set("train_steps", v)} step={1000} disabled={running} tip={fromMode === "library" ? "Total optimizer steps. Should be higher than the step count already in the library model." : "How many optimizer steps to run."} /></div>
                  <div className="grow"><Num label="Save every" value={form.save_every} onChange={(v) => set("save_every", v)} disabled={running} tip="Write a snapshot every N steps." /></div>
                </div>
                <div className="row gap-2">
                  <div className="grow"><Num label="Learning rate" value={form.lr} onChange={(v) => set("lr", v)} step={0.0001} disabled={running} tip="How big each weight update is, at the start of the run. Fine-tuning sets this from the distance control instead." /></div>
                  <div className="grow"><Num label="Grad accum" value={form.accum} onChange={(v) => set("accum", v)} disabled={running} tip="Accumulate this many micro-batches before an optimizer step." /></div>
                </div>
                <LrScheduleField
                  presets={lrPresets}
                  value={form.lr_plan ? "custom" : form.lr_schedule}
                  plan={form.lr_plan || lrPresets[form.lr_schedule]?.plan}
                  baseLr={form.lr}
                  onChange={(id) => setForm((f) => ({ ...f, lr_schedule: id, lr_plan: null }))}
                  onApply={(spec) => setForm((f) => (spec.preset
                    ? { ...f, lr_schedule: spec.preset, lr_plan: null }
                    : { ...f, lr_schedule: "custom", lr_plan: spec }))}
                  disabled={running}
                />
                <Tooltip text={"Starts at a tiny rate and climbs to the full one over the run's first steps. The optimizer's first updates are its least informed, and at full rate they can knock a trained model off course — so this matters most when starting from a library model.\n\n2% of the run, at most 1,000 steps. On a 280k-step xurdif run it is over by step 1,000, before the trainer starts averaging its weights at step 2,000. Continuing a run never adds one."}>
                  <label className="row center gap-2 has-tip">
                    <input type="checkbox" checked={!!form.warmup} disabled={running}
                      onChange={(e) => set("warmup", e.target.checked)} />
                    <span className="sub">Warm up the rate first <span className="hint">(2% of the run, at most 1,000 steps)</span></span>
                  </label>
                </Tooltip>
                </div>
                <div>
                <div className="section-title">Architecture & loss</div>
                {engines.length > 1 && (
                  <Select label="Engine" value={form.backend} onChange={(v) => {
                    // A new model takes the engine's own starting numbers; from a
                    // library model, the model already decided them.
                    setForm((f) => (fromMode === "library"
                      ? { ...f, backend: v, edge_loss: edgeLossDefault(v) }
                      : withEngine(f, v)));
                  }}
                    options={engines.map((e) => ({
                      value: e.name,
                      label: e.trainable_here === false ? `${e.name} (not on this machine)` : e.name,
                    }))}
                    tip="xurdif trains the compact models Kiln started with. diffusers trains Hugging Face UNet2DModel models and can fine-tune ones you import." />
                )}

                {form.backend === "xurdif" ? (
                  <>
                    <Select label="Architecture" value={form.mtype} onChange={(v) => set("mtype", v)} options={archs.length ? archs : [form.mtype]}
                      tip={"Which network shape to build. Fixed for the life of a model: you cannot change it later and resume.\n\ntinyunet_conf_attention is the current default and lets you choose where attention goes. tinyunet_with_attention3 is the earlier shape, with one attention layer at the bottleneck; models made before this option are that kind."} />
                    {/* Multipliers first: they decide how many levels exist, so the
                        sketch and the per-level pickers below grow as the text is typed. */}
                    <Text label="Channel multipliers" value={form.mults}
                      onChange={(v) => setForm((f) => ({ ...f, mults: v, attn: trimToDepth(f.attn, depthOf(v)) }))}
                      tip={"Width of the network at each resolution, shallowest first, coarsest last. Separate the numbers with commas.\n\n1,2,2,2 — four levels, one doubling. The smallest and fastest, and a probe rather than a result\n1,2,2,4 — Standard 512: wider at the deepest level, so detail has somewhere to live\n1,2,2,4,4 — Detailed 512: a level deeper, and needs image sizes that divide by 32\n\nBigger numbers mean more capacity and a slower model. The count of numbers sets how many times the image is halved. Two models can only be merged if these match."} />
                    <ArchSketch mults={form.mults} imageSize={form.image_size}
                      attn={isConf ? form.attn : null} baseDim={archInfo.base_dim} />
                    {isConf && (
                      <>
                        <Select label="Attention layout" value={customLayout ? "custom" : layoutId}
                          onChange={(v) => {
                            if (v === "custom") { setCustomLayout(true); return; }
                            const hit = layouts.find((l) => l.id === v);
                            if (hit) setForm((f) => ({ ...f, attn: hit.spec }));
                            setCustomLayout(false);
                          }}
                          options={[
                            ...layouts.map((l) => ({ value: l.id, label: l.label, title: l.tip })),
                            { value: "custom", label: "Custom…", title: "Choose the attention kind at every level yourself." },
                          ]}
                          tip={"Attention lets each spot in the image look at every other spot, which helps overall composition. Here you choose where in the network it sits and what kind it is.\n\nThe bottleneck is the smallest, deepest level. Level -1 is one step above it, -2 the step above that. Full attention above the bottleneck is expensive (it grows with the square of the pixels); linear and window attention are cheap. Fixed for the life of a model."} />
                        {(customLayout || layoutId === "custom") && (
                          <div className="attn-levels">
                            {/* Shallowest first, bottleneck last: the same order as the
                                multipliers string and the sketch's encoder, left to right. */}
                            {levelInfo(form.mults, form.image_size, archInfo.base_dim).map((lv) => (
                              <div className="attn-level" key={lv.loc}>
                                <Tooltip text={lv.loc === "mid"
                                  ? "The deepest level, where the image is smallest. Full attention is cheap here and is what every earlier model had."
                                  : `${-parseInt(lv.loc, 10)} step${lv.loc === "-1" ? "" : "s"} above the bottleneck. full sees the whole image at this size (costly); linear approximates that cheaply; window looks only within 8x8 patches; none skips attention at this level.`}>
                                  <span className="lbl">
                                    <b>{lv.label}</b>
                                    {lv.res != null ? ` · ${Number.isInteger(lv.res) ? lv.res : lv.res.toFixed(1)}px` : ""}
                                    {` · ${lv.ch} ch`}
                                  </span>
                                </Tooltip>
                                <Select ariaLabel={`Attention at ${lv.label}`}
                                  value={kindAt(form.attn, lv.loc)}
                                  onChange={(v) => setForm((f) => ({ ...f, attn: withKind(f.attn, lv.loc, v) }))}
                                  options={ATTN_KINDS} />
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                    <div className="row gap-2">
                      <div className="grow"><Select label="Prediction" value={form.pred} onChange={(v) => set("pred", v)} options={["x0", "eps"]}
                        tip={"What the network is asked to output at each step.\n\nx0 predicts the finished image directly and tends to settle faster on small datasets. eps predicts the noise to remove, the classic formulation. Fixed for the life of a model."} /></div>
                      <div className="grow"><Select label="Fit" value={form.fit} onChange={(v) => set("fit", v)} options={["resize", "crop"]}
                        tip={"How training images that are not square are made to fit.\n\nresize squashes the whole image to the training size, keeping everything but distorting proportions. crop takes a square from a random position each time, keeping proportions but not the whole image at once. A dataset's own framing (Data screen) happens first."} /></div>
                    </div>
                    <Num label="SSIM weight" value={form.ssimw} onChange={(v) => set("ssimw", v)} step={0.5}
                      tip="How much structural similarity is mixed into the loss on top of the edge-aware L1. 0 turns it off. Raising it pushes the model toward matching local structure and texture rather than just pixel values; too high and training can stall." />
                  </>
                ) : (
                  <DiffusersOptions form={form} set={set} engine={engineInfo} seeded={fromMode === "library"} onMps={onMps} />
                )}
                <Tooltip text={form.backend === "xurdif"
                  ? "Weights the training loss toward the edges found in your images, so lines and texture stay sharp instead of averaging out. Costs a little speed per step.\n\nOn by default for xurdif — this is how Kiln's own models were trained. Turning it off leaves a plain L1 loss."
                  : "Uses xurdif's edge-weighted L1 instead of Diffusers' usual MSE, so lines and texture stay sharp instead of averaging out. Costs a little speed per step.\n\nOff by default on this engine."}>
                  <label className="row center gap-2 has-tip">
                    <input type="checkbox" checked={form.edge_loss}
                      onChange={(e) => set("edge_loss", e.target.checked)} />
                    <span className="sub">Edge-aware loss <span className="hint">(crisper edges, slightly slower)</span></span>
                  </label>
                </Tooltip>
                {form.backend === "xurdif" && (
                  <Tooltip text="Runs much of the math at half precision. Roughly doubles training speed and halves memory use, at a small risk of numerical instability. Leave it on unless a run produces NaN losses.">
                    <label className="row center gap-2 has-tip">
                      <input type="checkbox" checked={form.amp} onChange={(e) => set("amp", e.target.checked)} />
                      <span className="sub">Mixed precision (AMP)</span>
                    </label>
                  </Tooltip>
                )}
                </div>
                </div>
              </Disclose>
            )}

            {running ? (
              <button type="button" className="btn danger w-full" onClick={stop}>Stop training</button>
            ) : (
              <button type="button" className="btn primary w-full" onClick={start}
                disabled={!!engineBlocked || !form.dataset || (fromMode === "library" && !form.resume) || !nameAvailable || !trimmedRunName}>
                Start training
              </button>
            )}
          </div>
          <div className="col">
            <div className="card">
              <h3>This run</h3>
              <p className="hint">
                What the preset works out to. Change any of it under Advanced.
              </p>
              <div className="run-params">
                <div className="kv"><span>Image size</span><b>{form.image_size}px</b></div>
                <div className="kv"><span>Batch</span><b>{form.batch_size}</b></div>
                <div className="kv"><span>{fromMode === "library" ? "Train until step" : "Iterations"}</span><b>{Number(form.train_steps).toLocaleString()}</b></div>
                <div className="kv"><span>Save every</span><b>{form.save_every}</b></div>
                <div className="kv"><span>Learning rate</span><b>{form.lr}</b></div>
                <div className="kv"><span>Schedule</span>
                  <b>{formSchedule?.summary
                    || (form.lr_plan ? form.lr_plan.segments.map((s) => fmtRate(s.lr)).join(" → ") : "constant")}</b>
                </div>
                <div className="kv"><span>Grad accum</span><b>{form.accum}</b></div>
                {form.backend === "xurdif" && (
                  <div className="kv"><span>Channel multipliers</span><b>{form.mults}</b></div>
                )}
                {isConf && (
                  <div className="kv"><span>Attention</span><b>{layoutLabel(form.attn)}</b></div>
                )}
                <div className="kv"><span>Edge-aware loss</span><b>{form.edge_loss ? "on" : "off"}</b></div>
              </div>
              {form.backend !== "xurdif" ? (
                <p className="hint mt-2">
                  VRAM estimates are measured for the xurdif engine, so none is shown
                  for {form.backend}.
                </p>
              ) : est && est.total_mib > 0 && canStart && (
                <div className={`vram-hint mt-2 ${est.risky ? "risky" : ""}`}>
                  <span>Est. peak VRAM <b>~{est.estimate_mib} MiB</b> / {est.total_mib} MiB{est.free_mib ? ` · ${est.free_mib} free` : ""}</span>
                  {est.risky && (
                    <span className="warn-text">May exceed VRAM — try batch ≤ {est.recommended_batch} at {form.image_size}px.</span>
                  )}
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
        <div className="work-split run-split">
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
                  {running && (
                    <LiveRateCard
                      lr={liveRate}
                      step={job?.detail?.step}
                      summary={runSchedule}
                      plan={continuePlan || job?.detail?.lr_plan || runView?.lr_plan}
                      presets={lrPresets}
                      busy={lrBusy}
                      onDrop={(rate) => putLrPlan({ from_now: true, lr: rate })}
                      onApply={putLrPlan}
                    />
                  )}
                  {canContinue && (
                    <div className="continue-run-box mt-2">
                      <p className="hint mb-2">Resume this run from its latest checkpoint ({latestCkptStep} steps).</p>
                      <LiveRateCard
                        bare
                        summary={runSchedule || "as recorded"}
                        plan={continuePlan || runView?.lr_plan}
                        presets={lrPresets}
                        busy={lrBusy}
                        canDrop={false}
                        onApply={putLrPlan}
                      />
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
                  <div className="preview-box preview-scroll preview-natural preview-h-lg mt-2">
                    {sample ? <img src={mediaUrl(sample)} alt="sample" /> : <span className="sub">Sample grid appears at the first snapshot.</span>}
                  </div>
                  {/* Below the sample, not above it: the command is reference you
                      reach for once, the sample is what you came to watch. */}
                  {cmd && running && (
                    <div className="cmd-box mt-2">
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
                </>
              ) : <Empty>Loading run…</Empty>}
            </div>

            <div className="card">
              <h3>Loss</h3>
              <LossChart points={losses} marks={lrMarks} />
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
                  {runMeta.attn && <div className="kv"><span>Attention</span><b>{layoutLabel(runMeta.attn)}</b></div>}
                  <div className="kv"><span>Prediction</span><b>{runMeta.pred || "—"}</b></div>
                  <div className="kv"><span>Learning rate</span><b>{runMeta.lr ?? "—"}</b></div>
                  <div className="kv"><span>Schedule</span><b>{runSchedule || "constant"}</b></div>
                  {lrMarks.length > 1 && (
                    <div className="kv"><span>Rate changes</span>
                      <b>{lrMarks.map((m) => m.label).join(" → ")}</b></div>
                  )}
                  <div className="kv"><span>Accumulation</span><b>{runMeta.accum ?? "—"}</b></div>
                  <div className="kv"><span>Diffusion steps</span><b>{runMeta.diffusion_steps ?? "—"}</b></div>
                  <div className="kv"><span>Loss</span><b>{lossLabel(runMeta)}</b></div>
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
