import React, { useEffect, useMemo, useState } from "react";
import { api, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import { Text, Progress, Empty, ConfirmModal, Loading } from "../components/ui.jsx";
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
          <p className="hint">Download a .pt into the library from a direct URL.</p>
          <GetModels onDone={load} />
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
