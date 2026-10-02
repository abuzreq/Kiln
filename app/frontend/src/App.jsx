import React, { useEffect, useState } from "react";
import { AppProvider, useApp } from "./state.jsx";
import { api } from "./api.js";
import Start from "./screens/Start.jsx";
import Studio from "./screens/Studio.jsx";
import Play from "./screens/Play.jsx";
import LibraryDrawer from "./screens/Library.jsx";
import WorkspaceModal from "./components/WorkspaceModal.jsx";
import DiscoveriesDrawer from "./components/DiscoveriesDrawer.jsx";
import { Seg, SubNav, Tooltip } from "./components/ui.jsx";
import { MODE_TABS, PREPARE_TABS, PLAY_TABS, PREPARE_TAB_IDS, PLAY_TAB_IDS, anyBusy } from "./navTabs.js";

function DeviceBadge() {
  const { device, refreshDevice, toast } = useApp();
  if (!device) return null;
  const gpu = device.gpus?.[0];
  const label = device.cuda
    ? `${gpu?.name?.replace("NVIDIA GeForce ", "") || "CUDA"}${gpu?.used_pct != null ? ` · ${gpu.used_pct}%` : ""}`
    : "CPU only";
  // Running on CPU is a huge slowdown, and the cause is usually a CPU-only
  // torch wheel — say so rather than leaving the user to guess.
  const cls = device.cuda ? "pill good" : device.reason === "no_gpu" ? "pill" : "pill bad";
  const tip = device.cuda
    ? `${gpu?.name || "CUDA device"}${gpu?.total_mem_mb ? ` · ${gpu.free_mem_mb} of ${gpu.total_mem_mb} MiB free` : ""}`
    : device.hint || "Running on CPU.";
  const freeGpu = async () => {
    await api.post("/gpu/free");
    refreshDevice();
    toast("GPU cache cleared", "success");
  };
  return (
    <div className="row center gap-2">
      {device.cuda && (
        <button type="button" className="btn ghost sm" onClick={freeGpu}>Free GPU</button>
      )}
      <Tooltip text={tip}>
        <span className={cls}>{!device.cuda && device.reason !== "no_gpu" ? "⚠ " : ""}{label}</span>
      </Tooltip>
    </div>
  );
}

function TrainingBadge() {
  const { setAppMode, setPrepareTab, setTabBusy } = useApp();
  const [train, setTrain] = useState(null);

  useEffect(() => {
    const poll = () => api.get("/studio/train").then((p) => setTrain(p.train)).catch(() => {});
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  // this badge polls regardless of which tab is open, so it is the right
  // place to own the Train tab's running indicator
  const training = train?.status === "training";
  useEffect(() => { setTabBusy("train", training); }, [training, setTabBusy]);

  if (!training) return null;

  return (
    <button
      type="button"
      className="pill good training-pill"
      onClick={() => { setAppMode("prepare"); setPrepareTab("train"); }}
      title={train.message || "Training in progress"}
    >
      Training{train.run ? ` · ${train.run}` : ""}
    </button>
  );
}

function Shell() {
  const {
    appMode, setAppMode, prepareTab, setPrepareTab, playTab, setPlayTab, busyTabs,
  } = useApp();
  const [assets, setAssets] = useState(false);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const mode = ["start", "prepare", "play"].includes(appMode) ? appMode : "start";
  const onStart = mode === "start";

  // Roll each pane's tab flags up to the mode switch, so a job running in Play
  // is still visible while you are over in Prepare. On Start the switch is
  // hidden, and the hub puts the same flags on its own two buttons instead.
  const modeBusy = {
    prepare: anyBusy(busyTabs, PREPARE_TAB_IDS),
    play: anyBusy(busyTabs, PLAY_TAB_IDS),
  };

  // Panes stay mounted once visited: unmounting Play threw away the canvas,
  // the mask, the init image and the whole Results history, so a quick look at
  // a loss curve used to cost you your session. Unvisited panes are not
  // mounted at all, so startup still only pays for what you actually open.
  const [visited, setVisited] = useState(() => ({ [mode]: true }));
  useEffect(() => {
    setVisited((v) => (v[mode] ? v : { ...v, [mode]: true }));
  }, [mode]);

  return (
    <div className="app">
      <div className="topbar">
        <Tooltip text={onStart ? "" : "Back to your hub"}>
          <button
            type="button"
            className={`brand ${onStart ? "home" : ""}`.trim()}
            onClick={() => setAppMode("start")}
            aria-label="Kiln — back to your hub"
            aria-current={onStart ? "page" : undefined}
          >
            <span className="logo" />
            <h1>Kiln</h1>
          </button>
        </Tooltip>
        {!onStart && (
          <Seg
            ariaLabel="App mode"
            tabs={MODE_TABS}
            value={mode}
            onChange={setAppMode}
            busy={modeBusy}
          />
        )}
        <div className="spacer" />
        <TrainingBadge />
        <DeviceBadge />
        <Tooltip text="Where Kiln keeps your datasets, runs, models and captures on disk.">
          <button type="button" className="btn sm" onClick={() => setWorkspaceOpen(true)}>
            Workspace
          </button>
        </Tooltip>
        <button type="button" className="btn sm" onClick={() => setAssets(true)}>
          Library
        </button>
      </div>

      {!onStart && (
        <div className="subnav-bar">
          {mode === "prepare" ? (
            <SubNav
              ariaLabel="Prepare section"
              tabs={PREPARE_TABS}
              value={PREPARE_TAB_IDS.includes(prepareTab) ? prepareTab : "data"}
              onChange={setPrepareTab}
              busy={busyTabs}
            />
          ) : (
            <SubNav
              ariaLabel="Create tool"
              tabs={PLAY_TABS}
              value={playTab}
              onChange={setPlayTab}
              busy={busyTabs}
            />
          )}
        </div>
      )}

      <div className="content">
        {visited.start && (
          <div className={onStart ? "" : "pane-off"}><Start /></div>
        )}
        {visited.prepare && (
          <div className={mode === "prepare" ? "" : "pane-off"}><Studio /></div>
        )}
        {visited.play && (
          <div className={mode === "play" ? "pane-play" : "pane-off"}><Play /></div>
        )}
      </div>

      {/* What the explorer found, along the bottom of Create -- where the
          images it turns up can actually be opened, bent or sampled from.
          Prepare and the hub have nothing to do with it. */}
      {mode === "play" && <DiscoveriesDrawer />}

      {assets && <LibraryDrawer onClose={() => setAssets(false)} />}
      {workspaceOpen && <WorkspaceModal onClose={() => setWorkspaceOpen(false)} />}
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}
