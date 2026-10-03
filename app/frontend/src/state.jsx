import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";
import { api, streamModels } from "./api.js";
import { relocateModelPath } from "./modelPaths.js";
import { bendCount, normalizeStack } from "./bendStack.js";

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

// The stored stack, checked: another build or tool may have written this key in
// a shape the Bend tab cannot draw. When anything had to be converted or
// dropped, the fixed stack replaces it -- so the note shows once, not every
// session -- and the original text is kept under kiln.bendStack.backup.
function loadBendStack() {
  let raw = null;
  let result;
  try {
    raw = localStorage.getItem("kiln.bendStack");
    result = normalizeStack(raw ? JSON.parse(raw) : []);
  } catch {
    result = { bends: [], dropped: 0, converted: 0, unreadable: true };
  }
  if (raw && (result.dropped || result.converted || result.unreadable)) {
    try {
      localStorage.setItem("kiln.bendStack.backup", raw);
      localStorage.setItem("kiln.bendStack", JSON.stringify(result.bends));
    } catch { /* ignore quota */ }
  }
  return result;
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
  // What loading the stored stack had to fix, reported once the toasts exist.
  const bendLoadNote = useRef(null);
  const [bendStack, setBendStackState] = useState(() => {
    const result = loadBendStack();
    bendLoadNote.current = result;
    return result.bends;
  });
  // A before/after handed to the Bend tab from outside it -- the Discoveries
  // drawer, which already has both pictures and should not make you re-render
  // them. Deliberately not persisted: it describes one click, and a stale one
  // restored next session would caption itself with the wrong model.
  const [bendCompare, setBendCompare] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [models, setModels] = useState(null);
  const [ops, setOps] = useState(null);
  // Which tabs currently have work running, so the nav can show it.
  const [busyTabs, setBusyTabs] = useState({});
  const [modelsBusy, setModelsBusy] = useState(false);
  // The models read so far while a scan streams in, null otherwise. Kept apart
  // from `models` on purpose: pickers treat a model missing from `models` as
  // gone and move off it, so only the finished list may go there. The Start
  // hub, which only displays, shows this one so cards appear as they are read.
  const [modelsPartial, setModelsPartial] = useState(null);
  const modelsInflight = useRef(null);

  // Read by setModelPath without making it change identity on every scan.
  const modelsRef = useRef(null);
  modelsRef.current = models;

  const setModelPath = useCallback((raw) => {
    // A recipe may name a model where it was before it moved into the workspace.
    const path = relocateModelPath(raw, modelsRef.current);
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
        let list;
        try {
          list = await streamModels(setModelsPartial);
        } catch {
          // A server without the streamed listing, or a stream that broke:
          // fall back to the one-shot request.
          list = await api.get("/models");
        }
        // The remembered pick may name a model where it was before it moved
        // into the workspace; follow it there in the same render as the list,
        // so nothing sees the stale path as a missing model and moves off it.
        setModelPathState((prev) => {
          const moved = relocateModelPath(prev, list || []);
          if (moved !== prev) localStorage.setItem("kiln.model", moved);
          return moved;
        });
        setModels(list || []);
        return list || [];
      } catch {
        setModels((prev) => prev || []);
        return [];
      } finally {
        setModelsPartial(null);
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

  // Every writer goes through here, so the stack is always Kiln's shape. A plain
  // value is checked before it is set, which is where an outside list (a
  // discovery, a preset) arrives and where a drop can be reported. An updater
  // derives from a stack that was already checked, so it is fixed up quietly.
  const setBendStack = useCallback((stackOrFn) => {
    let value = stackOrFn;
    if (typeof stackOrFn !== "function") {
      const { bends, dropped } = normalizeStack(stackOrFn);
      if (dropped) toast(`Left out ${bendCount(dropped, "entry", "entries")} this build cannot read as a bend`, "warn");
      value = bends;
    }
    setBendStackState((prev) => {
      const next = typeof value === "function" ? normalizeStack(value(prev)).bends : value;
      try {
        localStorage.setItem("kiln.bendStack", JSON.stringify(next));
      } catch { /* ignore quota */ }
      return next;
    });
  }, [toast]);

  useEffect(() => {
    const note = bendLoadNote.current;
    bendLoadNote.current = null;
    if (!note) return;
    if (note.unreadable) {
      toast("The saved bend stack could not be read, so the Bend tab starts empty. The original is kept in localStorage as kiln.bendStack.backup", "warn");
    } else if (note.dropped) {
      toast(`Left out ${bendCount(note.dropped, "saved entry", "saved entries")} the Bend tab cannot read${note.bends.length ? `; kept ${bendCount(note.bends.length)}` : ""}. The original is kept in localStorage as kiln.bendStack.backup`, "warn");
    } else if (note.converted) {
      toast(`Converted ${bendCount(note.converted, "saved bend")} from Bends JSON`, "info");
    }
  }, [toast]);

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
    models, modelsPartial, modelsBusy, refreshModels,
    ops,
    busyTabs, setTabBusy,
    prepareTab, setPrepareTab, playTab, setPlayTab,
    trainFromPath, setTrainFromPath,
    trainDataset, setTrainDataset,
    appMode, setAppMode, openPrepare, openPlay,
    bendStack, setBendStack,
    bendCompare, setBendCompare,
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
