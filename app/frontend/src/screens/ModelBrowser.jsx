import React, { useEffect, useMemo, useState } from "react";
import { api, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { Text, Select, Progress, Empty, ConfirmModal, Loading } from "../components/ui.jsx";
import { RenameModal } from "../components/modelMeta.jsx";
import { ModelCards, tagStars } from "./ModelList.jsx";

function GetModels({ onDone }) {
  const { toast } = useApp();
  const [catalog, setCatalog] = useState([]);
  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [job, setJob] = useState(null);
  useEffect(() => { api.get("/library/catalog").then(setCatalog); }, []);

  const download = async () => {
    if (!url || !name) return;
    try {
      const { job: j } = await api.post("/library/download", { url, name });
      setJob(j);
      const done = await pollJob(j.id, setJob, 500);
      toast(done.message, done.status === "done" ? "success" : "error");
      setJob(null);
      if (done.status === "done") onDone?.();
    } catch (e) { toast(e.message, "error"); }
  };

  return (
    <div className="col">
      <Text label="Name" value={name} onChange={setName} placeholder="my-download"
        tip="Filename for the downloaded .pt in the library." />
      <Text label="URL" value={url} onChange={setUrl} placeholder="https://…/model.pt"
        tip="Direct link to a .pt checkpoint." />
      {job && <div><Progress value={job.progress} /><div className="sub">{job.message}</div></div>}
      <button type="button" className="btn primary" onClick={download} disabled={!url || !name || !!job}>Download into library</button>
      {catalog.map((c) => (
        <div key={c.name} className="kv">
          <span>{c.label}</span>
          <a className="btn sm" href={c.url} target="_blank" rel="noreferrer">Open</a>
        </div>
      ))}
    </div>
  );
}

/**
 * Import an unconditional Diffusers model from Hugging Face.
 *
 * Validation is a separate, explicit step before any download: the backend only
 * reads a repo's metadata to decide, so an unsupported model costs one small
 * request instead of gigabytes. Refusals come back as a readable reason rather
 * than an error, and are shown next to the input.
 */
function GetFromHuggingFace({ onDone }) {
  const { toast } = useApp();
  const [available, setAvailable] = useState(true);
  const [suggested, setSuggested] = useState([]);
  const [ref, setRef] = useState("");
  const [name, setName] = useState("");
  const [checking, setChecking] = useState(false);
  const [verdict, setVerdict] = useState(null);
  const [job, setJob] = useState(null);

  useEffect(() => {
    api.get("/library/hf/suggested")
      .then((d) => { setSuggested(d.models || []); setAvailable(d.available !== false); })
      .catch(() => setAvailable(false));
  }, []);

  const check = async (value) => {
    const target = value ?? ref;
    if (!target) return;
    setChecking(true);
    setVerdict(null);
    try {
      const v = await api.post("/library/hf/validate", { ref: target });
      setVerdict(v);
      if (v.ok && !name) setName(v.model?.name || "");
    } catch (e) {
      setVerdict({ ok: false, reason: e.message });
    } finally { setChecking(false); }
  };

  const pick = (repo) => { setRef(repo); setName(""); check(repo); };

  const doImport = async () => {
    if (!verdict?.ok) return;
    try {
      const { job: j } = await api.post("/library/hf/import", { ref, name });
      setJob(j);
      const done = await pollJob(j.id, setJob, 500);
      toast(done.message, done.status === "done" ? "success" : "error");
      setJob(null);
      if (done.status === "done") { setRef(""); setName(""); setVerdict(null); onDone?.(); }
    } catch (e) { toast(e.message, "error"); }
  };

  if (!available) {
    return <p className="hint mb-0">The Diffusers backend is not available in this install.</p>;
  }

  return (
    <div className="col">
      <Text label="Model" value={ref} onChange={(v) => { setRef(v); setVerdict(null); }}
        placeholder="google/ddpm-celebahq-256"
        tip="A Hugging Face repo id, a huggingface.co URL, or a local model folder." />
      <button type="button" className="btn" onClick={() => check()}
        disabled={!ref || checking || !!job}>
        {checking ? "Checking…" : "Check compatibility"}
      </button>

      {verdict && !verdict.ok && (
        <p className="hint mb-0" style={{ color: "var(--bad, #d66)" }}>{verdict.reason}</p>
      )}

      {verdict?.ok && (
        <>
          <div className="kv"><span>Type</span><b>{verdict.model.mtype}</b></div>
          <div className="kv"><span>Pipeline</span><b>{verdict.pipeline || "—"}</b></div>
          <div className="kv"><span>Sizes</span><b>multiples of {verdict.model.size_multiple}</b></div>
          {(verdict.requires || []).map((r) => (
            <p key={r} className="hint mb-0">· {r}</p>
          ))}
          <Text label="Save as" value={name} onChange={setName} placeholder={verdict.model.name}
            tip="Folder name for the imported model in your library." />
          <button type="button" className="btn primary" onClick={doImport} disabled={!!job}>
            Import into library
          </button>
        </>
      )}

      {job && <div><Progress value={job.progress} /><div className="sub">{job.message}</div></div>}

      {!!suggested.length && (
        <>
          <p className="hint mb-0">Known-compatible starting points:</p>
          {suggested.map((sug) => (
            <div key={sug.repo} className="kv">
              <span>{sug.label}{sug.cached ? " · cached" : ""}</span>
              <button type="button" className="btn sm" onClick={() => pick(sug.repo)}>Check</button>
            </div>
          ))}
        </>
      )}
    </div>
  );
}


/**
 * Re-home a xurdif checkpoint into Diffusers format.
 *
 * Additive and lossless: the network is identical on both sides, so the weights
 * transfer verbatim and the converted model samples bit-identically. The
 * original .pt stays where it is.
 */
function RehomeModel({ onDone }) {
  const { toast } = useApp();
  const [available, setAvailable] = useState(true);
  const [models, setModels] = useState([]);
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  const load = () => api.get("/library/model/convertible")
    .then((d) => { setModels(d.models || []); setAvailable(d.available !== false); })
    .catch(() => setAvailable(false));
  useEffect(() => { load(); }, []);

  const chosen = models.find((m) => m.path === path);

  const convert = async () => {
    if (!path) return;
    setBusy(true);
    try {
      const res = await api.post("/library/model/convert", { path, name: name || undefined });
      toast(`Re-homed ${res.source_name} → ${res.name}`, "success");
      setPath(""); setName("");
      load(); onDone?.();
    } catch (e) { toast(e.message, "error"); }
    finally { setBusy(false); }
  };

  if (!available) {
    return <p className="hint mb-0">The Diffusers backend is not available in this install.</p>;
  }
  if (!models.length) {
    return <Empty>No xurdif models to re-home yet.</Empty>;
  }

  return (
    <div className="col">
      <Select label="Model" value={path} onChange={(v) => { setPath(v); setName(""); }}
        options={[{ value: "", label: "Pick a model…" },
                  ...models.map((m) => ({
                    value: m.path,
                    label: m.step != null ? `${m.name} · step ${m.step}` : m.name,
                    title: m.source,
                  }))]}
        tip="Any tinyunet checkpoint in your library." />
      {chosen && (
        <>
          <div className="kv"><span>Step</span><b>{chosen.step ?? "—"}</b></div>
          <div className="kv"><span>Multipliers</span><b>{(chosen.mults || []).join(",")}</b></div>
          <div className="kv"><span>Diffusion steps</span><b>{chosen.num_train_timesteps}</b></div>
          {!chosen.timesteps_known && (
            <p className="hint mb-0">
              This checkpoint does not record its schedule length, so {chosen.default_timesteps} is
              assumed — the same value Kiln samples it with today. If you trained it with a
              different <code>--steps</code>, convert from the run folder instead so its
              run.json can be read.
            </p>
          )}
          <Text label="Save as" value={name} onChange={setName}
            placeholder={`${chosen.name}-diffusers`}
            tip="Folder name for the re-homed model." />
        </>
      )}
      <button type="button" className="btn primary" onClick={convert} disabled={!path || busy}>
        {busy ? "Re-homing…" : "Re-home to Diffusers format"}
      </button>
      <p className="hint mb-0">
        Adds a copy in Diffusers format — safetensors, loadable by the wider ecosystem, and
        eligible for LoRA fine-tuning. Your original .pt is untouched and keeps working.
      </p>
    </div>
  );
}


export default function ModelBrowser() {
  const {
    toast, setModelPath, stars, toggleStar, setPrepareTab, setTrainFromPath,
    setAppMode, setPlayTab, models, refreshModels,
  } = useApp();
  const [renameModel, setRenameModel] = useState(null);
  const [pendingDel, setPendingDel] = useState(null);

  const load = () => refreshModels({ force: true });

  const tagged = useMemo(() => tagStars(models || [], stars), [models, stars]);
  const library = useMemo(() => tagged.filter((m) => m.role !== "checkpoint"), [tagged]);

  const del = async () => {
    const m = pendingDel;
    setPendingDel(null);
    if (!m) return;
    try {
      await api.del("/library/model", { path: m.path });
      toast(`Deleted ${m.name}`, "success");
      load();
    } catch (e) { toast(e.message, "error"); }
  };

  const trainFrom = (m) => {
    setTrainFromPath(m.path);
    setPrepareTab("train");
    toast(`Train will load ${m.name}`, "success");
  };

  const useInPlay = (m) => {
    setModelPath(m.path);
    setAppMode("play");
    setPlayTab("create");
    toast(`Using ${m.name} in Create`, "success");
  };

  const cardProps = {
    onUse: useInPlay,
    onStar: toggleStar,
    onTrain: trainFrom,
    onRename: setRenameModel,
    onDelete: setPendingDel,
  };

  if (!models) return <Loading>Loading models…</Loading>;

  return (
    <div className="work-split">
      <div className="col">
        <div className="card">
          <h3>Models</h3>
          <p className="hint mb-0">
            Named library models you keep. ★ pins favorites. Training snapshots live under Train ▸ each run —
            save one here when you want to sample or merge it.
          </p>
        </div>
        <div className="card">
          <h3>Get a model</h3>
          <p className="hint">Download a xurdif .pt into the library from a direct URL.</p>
          <GetModels onDone={load} />
        </div>
        <div className="card">
          <h3>Re-home a model</h3>
          <p className="hint">
            Convert a xurdif checkpoint into Diffusers format. The network is the same, so
            the converted model produces identical images.
          </p>
          <RehomeModel onDone={load} />
        </div>
        <div className="card">
          <h3>From Hugging Face</h3>
          <p className="hint">
            Unconditional, pixel-space diffusion models (DDPM and friends). These sample,
            paint, bend and merge like your own models, and can be fine-tuned on your
            datasets. Text-to-image models such as Stable Diffusion are not supported.
          </p>
          <GetFromHuggingFace onDone={load} />
        </div>
      </div>

      <div className="col">
        <div className="card">
          <h3>Library</h3>
          {library.length
            ? <ModelCards models={library} {...cardProps} />
            : <Empty>None yet. Save a snapshot from Train, merge two models in Play, or download one.</Empty>}
        </div>
      </div>

      {renameModel && (
        <RenameModal model={renameModel} onClose={() => setRenameModel(null)} onRenamed={() => load()} />
      )}
      {pendingDel && (
        <ConfirmModal
          title="Delete model"
          body={`Delete ${pendingDel.name}? This removes the .pt from the library.`}
          confirmLabel="Delete"
          danger
          onCancel={() => setPendingDel(null)}
          onConfirm={del}
        />
      )}
    </div>
  );
}
