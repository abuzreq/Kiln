import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";
import { api } from "./api.js";

const AppCtx = createContext(null);

// Every session opens on the Start hub, whatever you were doing last: it is the
// overview of your saved work, and dropping straight back into a working screen
// is exactly what skips it. The mode is therefore not persisted at all — the sub
// tabs still are, so choosing a path lands you where you left off inside it.
const loadAppMode = () => "start";

const loadPrepareTab = () => localStorage.getItem("kiln.prepareTab") || localStorage.getItem("kiln.studioTab") || "data";

const loadPlayTab = () => {
  const v = localStorage.getItem("kiln.playTab") || "create";
  if (v === "sample" || v === "paint") return "create";
  return v;
};

function loadBendStack() {
  try {
    const raw = localStorage.getItem("kiln.bendStack");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function AppProvider({ children }) {
  const [modelPath, setModelPathState] = useState(() => localStorage.getItem("kiln.model") || "");
  const [device, setDevice] = useState(null);
  const [workspace, setWorkspace] = useState(null);
  const [stars, setStars] = useState([]);
  const [prepareTab, setPrepareTabState] = useState(loadPrepareTab);
  const [playTab, setPlayTabState] = useState(loadPlayTab);
  const [trainFromPath, setTrainFromPath] = useState("");
  const [trainDataset, setTrainDataset] = useState("");
  const [appMode, setAppModeState] = useState(loadAppMode);
  const [bendStack, setBendStackState] = useState(loadBendStack);
  const [toasts, setToasts] = useState([]);
  const [models, setModels] = useState(null);
  const [ops, setOps] = useState(null);
  // Which tabs currently have work running, so the nav can show it.
  const [busyTabs, setBusyTabs] = useState({});
  const [modelsBusy, setModelsBusy] = useState(false);
  const modelsInflight = useRef(null);

  const setModelPath = useCallback((path) => {
    setModelPathState(path || "");
    if (path) localStorage.setItem("kiln.model", path);
    else localStorage.removeItem("kiln.model");
  }, []);

  const setAppMode = useCallback((mode) => {
    setAppModeState(mode);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const toast = useCallback((message, kind = "info") => {
    const id = Math.random().toString(36).slice(2);
    setToasts((t) => [...t, { id, message, kind }].slice(-4));
    if (kind !== "error") {
      setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4200);
    }
  }, []);

  /** Scanning models reads every .pt on disk, so callers share one request.
   *  Concurrent callers await the same promise rather than starting their own. */
  const refreshModels = useCallback(async ({ force = false } = {}) => {
    if (!force && modelsInflight.current) return modelsInflight.current;
    const run = (async () => {
      setModelsBusy(true);
      try {
        const list = await api.get("/models");
        setModels(list || []);
        return list || [];
      } catch {
        setModels((prev) => prev || []);
        return [];
      } finally {
        setModelsBusy(false);
        modelsInflight.current = null;
      }
    })();
    modelsInflight.current = run;
    return run;
  }, []);

  const setTabBusy = useCallback((tab, busy) => {
    setBusyTabs((s) => {
      if (!tab) return s;
      if (Boolean(s[tab]) === Boolean(busy)) return s;
      const next = { ...s };
      if (busy) next[tab] = true;
      else delete next[tab];
      return next;
    });
  }, []);

  const refreshDevice = useCallback(async () => {
    try {
      setDevice(await api.get("/device"));
    } catch {
      /* ignore */
    }
  }, []);

  const refreshStars = useCallback(async () => {
    try {
      const r = await api.get("/library/stars");
      setStars(r.paths || []);
    } catch { /* ignore */ }
  }, []);

  const toggleStar = useCallback(async (path) => {
    if (!path) return;
    try {
      const r = await api.post("/library/stars", { path });
      setStars(r.paths || []);
    } catch (e) {
      toast(e.message, "error");
    }
  }, [toast]);

  const setPrepareTab = useCallback((tab) => {
    setPrepareTabState(tab);
    if (tab) localStorage.setItem("kiln.prepareTab", tab);
  }, []);

  const setPlayTab = useCallback((tab) => {
    const next = tab === "sample" || tab === "paint" ? "create" : tab;
    setPlayTabState(next);
    if (next) localStorage.setItem("kiln.playTab", next);
  }, []);

  const setBendStack = useCallback((stackOrFn) => {
    setBendStackState((prev) => {
      const next = typeof stackOrFn === "function" ? stackOrFn(prev) : stackOrFn;
      try {
        localStorage.setItem("kiln.bendStack", JSON.stringify(next));
      } catch { /* ignore quota */ }
      return next;
    });
  }, []);

  const openPrepare = useCallback((opts = {}) => {
    if (opts.tab) setPrepareTab(opts.tab);
    if (opts.trainFrom) setTrainFromPath(opts.trainFrom);
    setAppMode("prepare");
  }, [setPrepareTab, setAppMode]);

  const openPlay = useCallback((opts = {}) => {
    if (opts.tab) setPlayTab(opts.tab);
    if (opts.model) setModelPath(opts.model);
    setAppMode("play");
  }, [setPlayTab, setModelPath, setAppMode]);

  useEffect(() => {
    refreshDevice();
    refreshStars();
    refreshModels();
    // The bend op catalog is small, static per session, and needed by both
    // Create (for preset summaries) and Bend (for the editor).
    api.get("/craft/ops").then((d) => setOps(d.ops || [])).catch(() => setOps([]));
    api.get("/workspace").then(setWorkspace).catch(() => {});
    const id = setInterval(refreshDevice, 6000);
    return () => clearInterval(id);
  }, [refreshDevice, refreshStars, refreshModels]);

  const value = {
    modelPath, setModelPath, device, refreshDevice,
    workspace, toast, dismissToast, stars, toggleStar, refreshStars,
    models, modelsBusy, refreshModels,
    ops,
    busyTabs, setTabBusy,
    prepareTab, setPrepareTab, playTab, setPlayTab,
    trainFromPath, setTrainFromPath,
    trainDataset, setTrainDataset,
    appMode, setAppMode, openPrepare, openPlay,
    bendStack, setBendStack,
  };
  return (
    <AppCtx.Provider value={value}>
      {children}
      <div className="toast-wrap" aria-live="polite" aria-relevant="additions">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind === "error" ? "bad" : t.kind === "success" ? "good" : t.kind === "warn" ? "warn" : ""}`} role="status">
            <span className="toast-msg">{t.message}</span>
            <button type="button" className="toast-x" aria-label="Dismiss" onClick={() => dismissToast(t.id)}>✕</button>
          </div>
        ))}
      </div>
    </AppCtx.Provider>
  );
}

export const useApp = () => useContext(AppCtx);
