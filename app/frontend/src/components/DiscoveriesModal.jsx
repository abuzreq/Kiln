import React, { useEffect, useMemo, useState } from "react";
import { api, mediaUrl, thumbUrl } from "../api.js";
import { useApp } from "../state.jsx";
import { Modal, Seg, Select, Tooltip } from "./ui.jsx";
import { bendPresetSummary, bendPresetSynopsis } from "../bendSynopsis.js";

// A bigger room for the discoveries than the strip: filter them, lay them out
// by similarity, and walk from one to its neighbours and its lineage. The
// embeddings the explorer already keeps are what make the map and the
// "similar" view possible; nothing is recomputed here.

const VIEW_TABS = [
  { id: "grid", label: "Grid", tip: "Every discovery that passes the filters" },
  { id: "map", label: "Map", tip: "Laid out by similarity: near each other on the map means near each other to the metric. One metric at a time" },
  { id: "similar", label: "Similar", tip: "The nearest neighbours of the selected discovery, closest first" },
];
const SORT_OPTS = [
  { value: "newest", label: "Newest first" },
  { value: "novel", label: "Most novel first" },
  { value: "oldest", label: "Oldest first" },
];
const METRIC_LABEL = { clip: "CLIP", dinov2: "DINOv2" };
const parentOf = (e) => (e?.source || "").startsWith("mutate:") ? e.source.slice(7) : null;

export default function DiscoveriesModal({ onClose, actions, ops }) {
  const { modelPath, toast } = useApp();
  const [all, setAll] = useState([]);
  const [view, setView] = useState("grid");
  const [sort, setSort] = useState("newest");
  const [model, setModel] = useState("");
  const [metric, setMetric] = useState("");
  const [starredOnly, setStarredOnly] = useState(false);
  const [opFilter, setOpFilter] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [mapPts, setMapPts] = useState(null);
  const [similar, setSimilar] = useState(null);

  const reload = () => api.get("/craft/discoveries?limit=500").then((d) => setAll(d.entries || [])).catch(() => {});
  useEffect(() => { reload(); }, []);

  const modelsPresent = useMemo(() => {
    const seen = new Map();
    all.forEach((e) => { if (!seen.has(e.model_path)) seen.set(e.model_path, e.model_name); });
    return [...seen.entries()].map(([value, label]) => ({ value, label }));
  }, [all]);
  const opsPresent = useMemo(() => {
    const s = new Set();
    all.forEach((e) => (e.bends || []).forEach((b) => s.add(b.op)));
    return [...s].sort();
  }, [all]);
  const opLabel = (name) => (ops || []).find((o) => o.name === name)?.label || name;

  const passes = (e) => (
    (!model || e.model_path === model)
    && (!metric || e.metric === metric)
    && (!starredOnly || e.starred)
    && (!opFilter.length || opFilter.every((op) => (e.bends || []).some((b) => b.op === op)))
  );
  const filtered = useMemo(() => {
    const list = all.filter(passes);
    if (sort === "novel") list.sort((a, b) => b.novelty - a.novelty);
    else if (sort === "oldest") list.sort((a, b) => a.created_at - b.created_at);
    else list.sort((a, b) => b.created_at - a.created_at);
    return list;
  }, [all, model, metric, starredOnly, opFilter, sort]);

  const selected = all.find((e) => e.id === selectedId) || null;
  const parent = selected ? all.find((e) => e.id === parentOf(selected)) : null;
  const children = selected ? all.filter((e) => parentOf(e) === selected.id) : [];

  // The map is one metric's space. With no metric filter, it follows the
  // selected discovery's metric, else CLIP.
  const mapMetric = metric || selected?.metric || "clip";
  useEffect(() => {
    if (view !== "map") return undefined;
    let alive = true;
    const q = new URLSearchParams({ metric: mapMetric });
    if (model) q.set("model_path", model);
    setMapPts(null);
    api.get(`/craft/discoveries/map?${q}`).then((d) => { if (alive) setMapPts(d.entries || []); }).catch((e) => toast(e.message, "error"));
    return () => { alive = false; };
  }, [view, mapMetric, model, all.length]);

  useEffect(() => {
    if (view !== "similar" || !selected) { setSimilar(null); return undefined; }
    let alive = true;
    api.get(`/craft/discoveries/${selected.id}/similar?limit=30&model_path=${encodeURIComponent(selected.model_path || "")}`)
      .then((d) => { if (alive) setSimilar(d); })
      .catch((e) => toast(e.message, "error"));
    return () => { alive = false; };
  }, [view, selectedId, all.length]);

  const star = async (e) => { await actions.star(e); setAll((prev) => prev.map((x) => (x.id === e.id ? { ...x, starred: true } : x))); };
  const remove = async (e) => {
    await actions.remove(e);
    setAll((prev) => prev.filter((x) => x.id !== e.id));
    if (selectedId === e.id) setSelectedId(null);
  };
  const openInBend = (e) => { actions.openInBend(e); onClose(); };
  const findSimilar = (e) => { setSelectedId(e.id); setView("similar"); };

  const toggleOp = (op) => setOpFilter((f) => (f.includes(op) ? f.filter((x) => x !== op) : [...f, op]));

  const tile = (e, extra) => (
    <button
      key={e.id}
      type="button"
      className={`dm-tile ${e.id === selectedId ? "sel" : ""} ${e.model_path === modelPath ? "" : "other"}`.trim()}
      onClick={() => setSelectedId(e.id)}
      onDoubleClick={() => openInBend(e)}
      title={`${bendPresetSummary({ bends: e.bends }, ops)} · novelty ${e.novelty} by ${METRIC_LABEL[e.metric] || "CLIP"} · ${e.model_name}`}
    >
      <img src={thumbUrl(e.image)} alt="" loading="lazy" decoding="async" />
      <span className="dm-tile-foot">
        {e.starred ? "★ " : ""}{extra ?? e.model_name}
      </span>
    </button>
  );

  const mapView = () => {
    if (mapPts === null) return <div className="dm-empty">Laying out…</div>;
    const visible = mapPts.filter(passes);
    if (!visible.length) return <div className="dm-empty">Nothing to map under {METRIC_LABEL[mapMetric]} with these filters.</div>;
    return (
      <div className="dm-map-wrap">
        <div className="dm-map">
          {visible.map((e) => (
            <button
              key={e.id}
              type="button"
              className={`dm-dot ${e.id === selectedId ? "sel" : ""} ${e.model_path === modelPath ? "" : "other"}`.trim()}
              style={{ left: `${e.x * 100}%`, top: `${e.y * 100}%` }}
              onClick={() => setSelectedId(e.id)}
              onDoubleClick={() => openInBend(e)}
              title={`${bendPresetSummary({ bends: e.bends }, ops)} · ${e.model_name}`}
            >
              <img src={thumbUrl(e.image)} alt="" loading="lazy" decoding="async" />
            </button>
          ))}
        </div>
        <p className="hint dm-map-hint">
          {visible.length} discoveries by {METRIC_LABEL[mapMetric]}, the two main directions of their embeddings.
          Clusters are the model's recurring looks; the gaps between them are where nothing has been found yet.
        </p>
      </div>
    );
  };

  const similarView = () => {
    if (!selected) return <div className="dm-empty">Select a discovery first, then this view ranks its neighbours.</div>;
    if (!similar) return <div className="dm-empty">Ranking…</div>;
    const list = similar.entries.filter(passes);
    return (
      <div className="dm-grid">
        {tile(selected, "this one")}
        {list.map((e) => tile(e, `${e.distance.toFixed(2)} away`))}
      </div>
    );
  };

  return (
    <Modal title="Discoveries" onClose={onClose} wide>
      <div className="dm-body">
        <aside className="dm-filters">
          <Select label="Model" value={model} onChange={setModel} options={[{ value: "", label: "All models" }, ...modelsPresent]} />
          <Select label="Metric" value={metric} onChange={setMetric} options={[{ value: "", label: "Both" }, { value: "clip", label: "CLIP" }, { value: "dinov2", label: "DINOv2" }]} />
          <Select label="Order" value={sort} onChange={setSort} options={SORT_OPTS} />
          <label className="row center gap-2 dm-check">
            <input type="checkbox" checked={starredOnly} onChange={(e) => setStarredOnly(e.target.checked)} />
            Starred only
          </label>
          {opsPresent.length > 0 && (
            <div className="dm-ops">
              <div className="hint">Uses these bends</div>
              <div className="row wrap gap-2">
                {opsPresent.map((op) => (
                  <button
                    key={op}
                    type="button"
                    className={`pill chip ${opFilter.includes(op) ? "on" : ""}`}
                    onClick={() => toggleOp(op)}
                    aria-pressed={opFilter.includes(op)}
                  >
                    {opLabel(op)}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="hint dm-count">{filtered.length} of {all.length}</div>
        </aside>

        <section className="dm-main">
          <Seg ariaLabel="View" tabs={VIEW_TABS} value={view} onChange={setView} size="sm" />
          <div className="dm-view">
            {view === "grid" && (
              filtered.length
                ? <div className="dm-grid">{filtered.map((e) => tile(e))}</div>
                : <div className="dm-empty">{all.length ? "Nothing passes these filters." : "Nothing found yet. Start exploring from the bar below."}</div>
            )}
            {view === "map" && mapView()}
            {view === "similar" && similarView()}
          </div>
        </section>

        <aside className="dm-detail">
          {selected ? (
            <>
              <img className="dm-big" src={mediaUrl(selected.image)} alt="" />
              <div className="dm-detail-body">
                <div className="dm-detail-model">{selected.model_name}{selected.model_missing ? " · model gone" : ""}</div>
                <pre className="dm-synopsis">{bendPresetSynopsis({ bends: selected.bends }, ops)}</pre>
                <div className="hint">
                  novelty {selected.novelty} by {METRIC_LABEL[selected.metric] || "CLIP"} · {selected.source === "random" ? "random stack" : selected.source === "inject" ? "kept by chance" : "mutation"}
                </div>
                {parentOf(selected) && (
                  <div className="hint">
                    parent:{" "}
                    {parent
                      ? <button type="button" className="dm-link" onClick={() => setSelectedId(parent.id)}>{bendPresetSummary({ bends: parent.bends }, ops)}</button>
                      : <span>no longer in the archive</span>}
                  </div>
                )}
                {children.length > 0 && (
                  <div className="hint">
                    {children.length} mutation{children.length === 1 ? "" : "s"} of this one:{" "}
                    {children.map((c) => (
                      <button key={c.id} type="button" className="dm-link" onClick={() => setSelectedId(c.id)}>{c.novelty}</button>
                    ))}
                  </div>
                )}
                <div className="row wrap gap-2 dm-actions">
                  <Tooltip text="Load these bends in the Bend tab, on the model that made them">
                    <button type="button" className="btn primary sm" onClick={() => openInBend(selected)}>Open in Bend</button>
                  </Tooltip>
                  <Tooltip text="Rank the archive by distance from this one">
                    <button type="button" className="btn sm" onClick={() => findSimilar(selected)}>Similar</button>
                  </Tooltip>
                  <Tooltip text={selected.starred ? "Saved as a preset" : "Save as a bend preset; starred entries are never replaced"}>
                    <button type="button" className="btn sm" onClick={() => star(selected)} disabled={selected.starred}>★ Star</button>
                  </Tooltip>
                  <Tooltip text="Drop this discovery">
                    <button type="button" className="btn ghost sm" onClick={() => remove(selected)}>Delete</button>
                  </Tooltip>
                </div>
              </div>
            </>
          ) : (
            <div className="dm-empty">Click a discovery to see its bends and its neighbours. Double-click opens it in Bend.</div>
          )}
        </aside>
      </div>
    </Modal>
  );
}
