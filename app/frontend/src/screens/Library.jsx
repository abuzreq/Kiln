import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, mediaUrl, thumbUrl } from "../api.js";
import { useApp } from "../state.jsx";
import { ConfirmModal, DeleteBtn, Empty, Loading } from "../components/ui.jsx";
import { RemoveModelModal, RenameModal } from "../components/modelMeta.jsx";
import { ModelRows } from "./ModelList.jsx";
import { cardLabel, mergeStoredSampleParams, paramsFromCard, paramLabel } from "../sampleSettings.jsx";
import { cardModelPath, restoreSelection, restoredMessage } from "../createSelection.js";
import { recipeSummary, requestMergeOpen } from "../mergeRecipes.js";

function Models({ onPick }) {
  const {
    toast, modelPath, setModelPath, openPrepare,
    models: allModels, modelsBusy, refreshModels,
  } = useApp();
  const [renameModel, setRenameModel] = useState(null);
  const [pendingDel, setPendingDel] = useState(null);

  const models = useMemo(
    () => (allModels || []).filter((x) => x.role !== "checkpoint"),
    [allModels],
  );
  const load = () => refreshModels({ force: true });

  const trainFrom = (m) => {
    openPrepare({ tab: "train", trainFrom: m.path });
    toast(`Train will load ${m.name}`, "success");
    onPick?.();
  };

  return (
    <>
      <ModelRows
        models={models}
        loading={allModels === null || modelsBusy}
        onPick={(m) => { setModelPath(m.path); toast(`Using ${m.name}`, "success"); onPick?.(); }}
        onRename={setRenameModel}
        onTrain={trainFrom}
        onDelete={setPendingDel}
      />
      {renameModel && (
        <RenameModal model={renameModel} onClose={() => setRenameModel(null)} onRenamed={() => load()} />
      )}
      {pendingDel && (
        <RemoveModelModal model={pendingDel} onClose={() => setPendingDel(null)} onDone={load} />
      )}
    </>
  );
}

function NamedList({ kind, empty, subtitle, onClose }) {
  const { toast, models, setAppMode, setPlayTab } = useApp();
  const [items, setItems] = useState(null);
  const [pending, setPending] = useState(null);
  const load = () => api.get(`/library/${kind}`).then(setItems).catch(() => setItems([]));
  useEffect(() => { load(); }, [kind]);

  if (!items) return <Loading />;
  if (!items.length) return <Empty>{empty}</Empty>;
  return (
    <div className="col gap-2">
      {subtitle && <p className="hint mb-0">{subtitle}</p>}
      {items.map((it) => (
        <div key={it.name} className="asset-row static">
          {it.thumbnail
            ? <span className="thumb-sm"><img src={it.thumbnail} alt="" loading="lazy" /></span>
            : kind === "recipes" && <span className="thumb-sm" aria-hidden="true" />}
          <div className="meta">
            <b>{it.name}</b>
            <span className="sub">
              {kind === "bends"
                ? ((it.bends || []).map((b) => b.op).join(", ") || "empty")
                : recipeSummary(it, models)}
            </span>
          </div>
          {kind === "recipes" && (
            <button type="button" className="btn xs"
              onClick={() => {
                requestMergeOpen(it.name);
                setAppMode("play");
                setPlayTab("merge");
                onClose?.();
              }}>
              Open in Merge
            </button>
          )}
          <DeleteBtn label={`Delete ${it.name}`} onClick={() => setPending(it)} />
        </div>
      ))}
      {pending && (
        <ConfirmModal
          title={`Delete ${kind === "bends" ? "bend" : "merge"}`}
          body={`Delete “${pending.name}”?`}
          confirmLabel="Delete"
          danger
          onCancel={() => setPending(null)}
          onConfirm={async () => {
            const name = pending.name;
            setPending(null);
            try {
              await api.del(kind === "bends" ? `/craft/bends/${name}` : `/library/recipes/${name}`);
              toast(`Deleted ${name}`, "success");
              load();
            } catch (e) { toast(e.message, "error"); }
          }}
        />
      )}
    </div>
  );
}

function Saved({ onClose }) {
  const { toast, setModelPath, setAppMode, setPlayTab } = useApp();
  const [items, setItems] = useState(null);
  const [pending, setPending] = useState(null);

  const load = () => api.get("/captures")
    .then((d) => setItems(d.captures || []))
    .catch(() => setItems([]));
  useEffect(() => { load(); }, []);

  const restore = async (card) => {
    const params = paramsFromCard(card);
    if (!params) { toast("This image has no recorded settings", "warn"); return; }
    mergeStoredSampleParams(params);
    if (cardModelPath(card)) setModelPath(cardModelPath(card));
    const missing = await restoreSelection(card);
    setAppMode("play");
    setPlayTab("create");
    toast(restoredMessage(cardLabel(card), missing), missing.length ? "warn" : "success");
    onClose?.();
  };

  if (!items) return <Loading />;
  if (!items.length) return <Empty>No saved images yet. Capture outputs from Create.</Empty>;

  return (
    <div className="col gap-2">
      <p className="hint mb-0">
        Captured images carry the settings that made them. Restore puts those settings back in Create.
      </p>
      {items.map((it) => {
        const label = cardLabel(it.card);
        return (
          <div key={it.path} className="asset-row static">
            <a className="thumb-sm" href={mediaUrl(it.path)} target="_blank" rel="noreferrer">
              <img src={thumbUrl(it.path)} alt="" loading="lazy" />
            </a>
            <div className="meta">
              <b title={it.name}>{it.name}</b>
              <span className="sub">{label || "no recorded settings"}</span>
            </div>
            <div className="asset-actions">
              <div className="asset-actions-primary">
                <button
                  type="button"
                  className="btn sm primary"
                  onClick={() => restore(it.card)}
                  disabled={!it.card?.params}
                >
                  Restore
                </button>
              </div>
              <div className="asset-actions-secondary">
                <DeleteBtn label={`Delete ${it.name}`} onClick={() => setPending(it)} />
              </div>
            </div>
          </div>
        );
      })}
      {pending && (
        <ConfirmModal
          title="Delete capture"
          body={`Delete “${pending.name}”? This removes the PNG from your captures folder.`}
          confirmLabel="Delete"
          danger
          onCancel={() => setPending(null)}
          onConfirm={async () => {
            const target = pending;
            setPending(null);
            try {
              await api.del("/captures", { path: target.path });
              toast(`Deleted ${target.name}`, "success");
              load();
            } catch (e) { toast(e.message, "error"); }
          }}
        />
      )}
    </div>
  );
}

/** Sweep grids and animations — parameter studies, kept apart from captures. */
function Sweeps({ onClose }) {
  const { toast, setModelPath, setAppMode, setPlayTab } = useApp();
  const [items, setItems] = useState(null);
  const [pending, setPending] = useState(null);

  const load = () => api.get("/sweeps")
    .then((d) => setItems(d.sweeps || []))
    .catch(() => setItems([]));
  useEffect(() => { load(); }, []);

  const restore = async (it) => {
    const params = paramsFromCard(it.card);
    if (!params) { toast("This sweep has no recorded settings", "warn"); return; }
    mergeStoredSampleParams(params);
    if (cardModelPath(it.card)) setModelPath(cardModelPath(it.card));
    const missing = await restoreSelection(it.card);
    setAppMode("play");
    setPlayTab("create");
    toast(restoredMessage(cardLabel(it.card), missing), missing.length ? "warn" : "success");
    onClose?.();
  };

  const axisLabel = (card) => {
    const axes = card?.axes;
    if (!Array.isArray(axes) || !axes.length) return null;
    return axes
      .map((a) => `${paramLabel(a.param)} ×${(a.values || []).length}`)
      .join(" · ");
  };

  if (!items) return <Loading />;
  if (!items.length) {
    return <Empty>No sweeps yet. Run one in Create ▸ Sweep, or sweep a bend parameter in Create ▸ Bend.</Empty>;
  }

  return (
    <div className="col gap-2">
      <p className="hint mb-0">
        Parameter studies: grids and animations. Restore puts a sweep&apos;s settings back in Create.
      </p>
      {items.map((it) => (
        <div key={it.path} className="asset-row static">
          <a className="thumb-sm" href={mediaUrl(it.path)} target="_blank" rel="noreferrer">
            {it.video ? (
              <video src={mediaUrl(it.path)} muted loop playsInline preload="metadata"
                onMouseEnter={(e) => e.currentTarget.play().catch(() => {})}
                onMouseLeave={(e) => e.currentTarget.pause()} />
            ) : (
              <img src={it.animated ? mediaUrl(it.path) : thumbUrl(it.path)} alt="" loading="lazy" />
            )}
          </a>
          <div className="meta">
            <b title={it.name}>{it.name}</b>
            <span className="sub">
              {it.animated && <span className="pill sm">GIF</span>}
              {it.video && <span className="pill sm">{it.format === "mp4" ? "MP4" : "WebM"}</span>}
              {axisLabel(it.card) || cardLabel(it.card) || "no recorded settings"}
            </span>
          </div>
          <div className="asset-actions">
            <div className="asset-actions-primary">
              <button type="button" className="btn sm primary"
                onClick={() => restore(it)} disabled={!it.card?.params}>
                Restore
              </button>
            </div>
            <div className="asset-actions-secondary">
              <DeleteBtn label={`Delete ${it.name}`} onClick={() => setPending(it)} />
            </div>
          </div>
        </div>
      ))}
      {pending && (
        <ConfirmModal
          title="Delete sweep"
          body={`Delete “${pending.name}”? This cannot be undone.`}
          danger
          onCancel={() => setPending(null)}
          onConfirm={async () => {
            const it = pending;
            setPending(null);
            try {
              await api.del("/sweeps", { path: it.path });
              toast(`Deleted ${it.name}`, "success");
              load();
            } catch (e) { toast(e.message, "error"); }
          }}
        />
      )}
    </div>
  );
}

const TABS = [
  { id: "models", label: "Models" },
  { id: "bends", label: "Bends" },
  { id: "merges", label: "Merges" },
  // Id is the wire value and keys the persisted tab state, so it stays "saved"
  // while showing as Captures — the name used by the delete modal, the Start
  // screen and the workspace folder.
  { id: "saved", label: "Captures" },
  { id: "sweeps", label: "Sweeps" },
];

export default function LibraryDrawer({ onClose }) {
  const [tab, setTab] = useState("models");
  const drawerRef = useRef(null);
  const prev = useRef(typeof document !== "undefined" ? document.activeElement : null);

  useEffect(() => {
    drawerRef.current?.querySelector("button")?.focus();
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      prev.current?.focus?.();
    };
  }, [onClose]);

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" ref={drawerRef} role="dialog" aria-modal="true" aria-labelledby="library-title">
        <div className="drawer-head">
          <h3 id="library-title" className="grow mb-0">Library</h3>
          <button type="button" className="btn ghost sm" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="tabs" role="tablist" aria-label="Library sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={tab === t.id}
              className={`tab ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="drawer-body">
          {tab === "models" && <Models onPick={onClose} />}
          {tab === "bends" && <NamedList kind="bends" empty="No saved bends. Create some in Create ▸ Bend, then use them in Create ▸ Canvas." />}
          {tab === "merges" && (
            <NamedList kind="recipes" onClose={onClose}
              subtitle="Merge recipes keep a mix and its two models. Create ▸ Canvas uses them with “Merge with”."
              empty="No saved merge recipes. Pick a mix in Create ▸ Merge, then Save recipe." />
          )}
          {tab === "saved" && <Saved onClose={onClose} />}
          {tab === "sweeps" && <Sweeps onClose={onClose} />}
        </div>
      </aside>
    </>
  );
}
