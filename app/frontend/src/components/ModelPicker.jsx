import React, { useEffect, useMemo, useState } from "react";
import { thumbUrl } from "../api.js";
import { useApp } from "../state.jsx";
import { Seg, Select, Empty, Loading } from "./ui.jsx";
import { pickDefaultModel, tagStars } from "../screens/ModelList.jsx";
import { modelSubtitle } from "./modelMeta.jsx";
import { GALLERY_SORT_OPTS, ThumbSizeSeg } from "./ThumbGalleryToolbar.jsx";
import { THUMB_LEVELS } from "./LazySourceGallery.jsx";

const VIEW_TABS = [
  { id: "gallery", label: "Gallery", tip: "Thumbnail grid" },
  { id: "list", label: "List", tip: "Compact rows" },
];

// The panel grows with the thumbnails so a bigger size shows two rows rather
// than cropping the first one. Derived from the level, not a second table.
const thumbMin = (level) => THUMB_LEVELS.find((t) => t.id === level)?.min || 108;
const scrollFor = (level) => thumbMin(level) * 2 + 44;

function sortLibrary(list, sort) {
  const pinned = list.filter((m) => m.starred);
  const rest = list.filter((m) => !m.starred);
  const cmp = sort === "date"
    ? (a, b) => (b.mtime || 0) - (a.mtime || 0)
    : (a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
  return [...pinned.sort(cmp), ...rest.sort(cmp)];
}

function GalleryThumb({ m, selected, onPick, onStar }) {
  return (
    <div className="mp-cell">
      <div className="mp-thumb-wrap">
        <button
          type="button"
          className={`mp-thumb ${selected ? "on" : ""}`}
          onClick={() => onPick(m.path)}
          title={m.name}
          aria-label={m.name}
        >
          {m.thumbnail
            ? <img src={thumbUrl(m.thumbnail)} alt="" loading="lazy" decoding="async" />
            : <span>{m.name.slice(0, 2)}</span>}
        </button>
        <button
          type="button"
          className={`star ${m.starred ? "on" : ""}`}
          aria-label={m.starred ? `Unpin ${m.name}` : `Pin ${m.name}`}
          onClick={(e) => { e.stopPropagation(); onStar(m.path); }}
        >
          {m.starred ? "★" : "☆"}
        </button>
      </div>
      <span className="mp-name">{m.name}</span>
    </div>
  );
}

function ListRow({ m, selected, onPick, onStar }) {
  const sub = modelSubtitle(m) || `${m.mtype} · ${m.size_mb} MB`;
  return (
    <div className={`mp-row ${selected ? "on" : ""}`}>
      <button type="button" className="mp-row-main" onClick={() => onPick(m.path)}>
        <b className="mp-row-name" title={m.name}>{m.name}</b>
        <span className="mp-row-sub sub" title={sub}>{sub}</span>
      </button>
      <button
        type="button"
        className={`star ${m.starred ? "on" : ""}`}
        aria-label={m.starred ? `Unpin ${m.name}` : `Pin ${m.name}`}
        onClick={() => onStar(m.path)}
      >
        {m.starred ? "★" : "☆"}
      </button>
    </div>
  );
}

export default function ModelPicker() {
  const {
    modelPath, setModelPath, stars, toggleStar, models: finished, modelsPartial, modelsBusy,
    refreshModels,
  } = useApp();
  // While the first scan streams in, show what has arrived. Only the finished
  // list may decide the picked model is gone (the effect below): a model not
  // read yet is not missing.
  const allModels = finished ?? (modelsPartial?.length ? modelsPartial : null);
  const [view, setView] = useState(() => localStorage.getItem("kiln.modelPickerView") || "list");
  const [sort, setSort] = useState(() => localStorage.getItem("kiln.modelPickerSort") || "name");
  const [thumbLevel, setThumbLevel] = useState(() => localStorage.getItem("kiln.modelPickerSize") || "md");

  const models = useMemo(
    () => (allModels === null ? null : allModels.filter((x) => x.role !== "checkpoint")),
    [allModels],
  );
  const load = () => refreshModels({ force: true });

  useEffect(() => {
    if (finished === null || !models?.length) return;
    if (!models.some((x) => x.path === modelPath)) {
      setModelPath(pickDefaultModel(models, modelPath));
    }
  }, [finished, models, modelPath, setModelPath]);

  useEffect(() => {
    localStorage.setItem("kiln.modelPickerView", view);
  }, [view]);

  useEffect(() => {
    localStorage.setItem("kiln.modelPickerSort", sort);
  }, [sort]);

  useEffect(() => {
    localStorage.setItem("kiln.modelPickerSize", thumbLevel);
  }, [thumbLevel]);

  const ordered = useMemo(() => {
    const tagged = tagStars(models || [], stars);
    return sortLibrary(tagged, sort);
  }, [models, stars, sort]);

  const selected = ordered.find((m) => m.path === modelPath);

  if (models === null) return <Loading>Loading models…</Loading>;

  return (
    <div className="model-picker-panel">
      <div className="model-picker-header">
        <div className="model-picker-title-row">
          <span className="section-title">Model</span>
          {selected
            ? <span className="sub">Selected: <b>{selected.name}</b></span>
            : <span className="sub">Pick a library model</span>}
        </div>
        <div className="model-picker-toolbar">
          <div className="model-picker-toolbar-group">
            <span className="toolbar-label">View</span>
            <Seg ariaLabel="Model picker view" tabs={VIEW_TABS} value={view} onChange={setView} size="sm" />
          </div>
          {view === "gallery" && (
            <div className="model-picker-toolbar-group">
              <span className="toolbar-label">Size</span>
              <ThumbSizeSeg value={thumbLevel} onChange={setThumbLevel} ariaLabel="Model thumbnail size" />
            </div>
          )}
          <div className="model-picker-toolbar-group model-picker-toolbar-actions">
            <span className="toolbar-label">Sort</span>
            <Select label="" value={sort} onChange={setSort} options={GALLERY_SORT_OPTS} ariaLabel="Sort order" />
            <button type="button" className="btn ghost sm" onClick={load} disabled={modelsBusy}>
              {modelsBusy ? <><span className="spinner sm" aria-hidden /> Refreshing…</> : "Refresh"}
            </button>
          </div>
        </div>
      </div>
      {!ordered.length ? (
        <Empty>No library models yet. Save a training snapshot, merge models, or download one in Models.</Empty>
      ) : view === "gallery" ? (
        <div className="model-picker-scroll" style={{ "--mp-scroll": `${scrollFor(thumbLevel)}px` }}>
          <div className="model-picker-grid" style={{ "--mp-thumb": `${thumbMin(thumbLevel)}px` }}>
            {ordered.map((m) => (
              <GalleryThumb
                key={m.path}
                m={m}
                selected={modelPath === m.path}
                onPick={setModelPath}
                onStar={toggleStar}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="model-picker-scroll model-picker-list">
          {ordered.map((m) => (
            <ListRow
              key={m.path}
              m={m}
              selected={modelPath === m.path}
              onPick={setModelPath}
              onStar={toggleStar}
            />
          ))}
        </div>
      )}
    </div>
  );
}
