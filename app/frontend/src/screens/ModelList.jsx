import React, { useMemo, useState } from "react";
import { thumbUrl } from "../api.js";
import { useApp } from "../state.jsx";
import { modelSubtitle } from "../components/modelMeta.jsx";
import { DeleteBtn, Empty, Loading, Modal } from "../components/ui.jsx";
import { ThumbGalleryToolbar } from "../components/ThumbGalleryToolbar.jsx";

export function pickDefaultModel(models, current) {
  if (current && models.some((m) => m.path === current)) return current;
  const starred = models.find((m) => m.starred && m.role !== "checkpoint");
  if (starred) return starred.path;
  const main = models.find((m) => m.role !== "checkpoint");
  if (main) return main.path;
  return models[0]?.path || "";
}

export function splitModels(models) {
  const library = models.filter((m) => m.role !== "checkpoint");
  const pinned = library.filter((m) => m.starred);
  const rest = library.filter((m) => !m.starred);
  const groups = {};
  models.filter((m) => m.role === "checkpoint").forEach((m) => {
    const g = m.group || "training run";
    (groups[g] = groups[g] || []).push(m);
  });
  Object.values(groups).forEach((list) => list.sort((a, b) => (a.step || 0) - (b.step || 0)));
  return { library, pinned, rest, groups };
}

export function pathsMatch(a, b) {
  if (!a || !b) return false;
  if (a === b) return true;
  return a.replace(/\\/g, "/").toLowerCase() === b.replace(/\\/g, "/").toLowerCase();
}

export function isStarred(path, stars) {
  return (stars || []).some((s) => pathsMatch(s, path));
}

export function tagStars(models, stars) {
  return (models || []).map((m) => ({ ...m, starred: isStarred(m.path, stars) }));
}

export function selectOptions(models) {
  const { library } = splitModels(models);
  return library.map((m) => ({ value: m.path, label: m.starred ? `★ ${m.name}` : m.name }));
}

// Model cards are larger than image thumbs, but the control is the same one.
const CARD_LEVELS = [
  { id: "sm", label: "S", min: 160 },
  { id: "md", label: "M", min: 200 },
  { id: "lg", label: "L", min: 260 },
];

function sortModels(list, sort) {
  const cmp = sort === "date"
    ? (a, b) => (b.mtime || 0) - (a.mtime || 0)
    : (a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
  return [...list].sort(cmp);
}

function StarBtn({ m, onStar }) {
  return (
    <button
      type="button"
      className={`star ${m.starred ? "on" : ""}`}
      aria-label={m.starred ? "Unpin from favorites" : "Pin to the top of the library"}
      onClick={() => onStar(m.path)}
    >
      {m.starred ? "★" : "☆"}
    </button>
  );
}

/** Compact row list used in the Library drawer. */
export function ModelRows({ models, loading, onPick, onRename, onTrain, onDelete }) {
  const { modelPath, stars, toggleStar } = useApp();
  const tagged = useMemo(() => tagStars(models, stars), [models, stars]);
  const { pinned, rest } = useMemo(() => splitModels(tagged), [tagged]);
  const ordered = [...pinned, ...rest];

  if (loading) return <Loading>Loading models…</Loading>;
  if (!ordered.length) {
    return (
      <Empty>
        No library models yet. Save a training snapshot in Prepare ▸ Train, merge in Create, or download one in Prepare ▸ Models.
      </Empty>
    );
  }
  return (
    <div className="col gap-2">
      <p className="hint mb-0">Named models you keep. Pin favorites with ★. Use in Create opens the canvas with that model selected.</p>
      {ordered.map((m) => (
        <div
          key={m.path}
          className={`asset-row ${modelPath === m.path ? "on" : ""}`}
          role="button"
          tabIndex={0}
          onClick={() => onPick?.(m)}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick?.(m); } }}
        >
          <div className="thumb-sm">
            {m.thumbnail ? <img src={thumbUrl(m.thumbnail)} alt="" /> : null}
          </div>
          <div className="meta">
            <b title={m.name}>{m.name}</b>
            <span className="sub">{modelSubtitle(m) || `${m.mtype} · ${m.size_mb} MB`}</span>
          </div>
          <div className="asset-actions" onClick={(e) => e.stopPropagation()}>
            <div className="asset-actions-primary">
              <button type="button" className="btn sm primary" onClick={() => onPick?.(m)}>Use in Create</button>
              <StarBtn m={m} onStar={toggleStar} />
            </div>
            <div className="asset-actions-secondary">
              <button type="button" className="btn ghost sm" onClick={() => onTrain?.(m)}>Train</button>
              {m.renamable && (
                <button type="button" className="btn ghost sm" onClick={() => onRename?.(m)}>Rename</button>
              )}
              <DeleteBtn onClick={() => onDelete?.(m)} label={`Delete ${m.name}`} />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/** Card grid used in Prepare ▸ Models. */
export function ModelCards({
  models, onUse, onStar, onTrain, onRename, onDelete,
  sort: sortProp, onSortChange, thumbLevel: sizeProp, onSizeChange,
}) {
  const [sortLocal, setSortLocal] = useState("name");
  const [sizeLocal, setSizeLocal] = useState("md");
  const [lightbox, setLightbox] = useState(null);
  const sort = sortProp ?? sortLocal;
  const setSort = onSortChange ?? setSortLocal;
  const thumbLevel = sizeProp ?? sizeLocal;
  const setThumbLevel = onSizeChange ?? setSizeLocal;
  const cardMin = (CARD_LEVELS.find((l) => l.id === thumbLevel) || CARD_LEVELS[1]).min;

  const ordered = useMemo(() => {
    const pinned = models.filter((m) => m.starred);
    const rest = models.filter((m) => !m.starred);
    return [...sortModels(pinned, sort), ...sortModels(rest, sort)];
  }, [models, sort]);

  return (
    <>
      <ThumbGalleryToolbar
        title="Model"
        count={ordered.length}
        sort={sort}
        onSortChange={setSort}
        thumbLevel={thumbLevel}
        onThumbLevelChange={setThumbLevel}
        levels={CARD_LEVELS}
        heading={false}
      />
      <div className="model-scroll" style={{ "--card-min": `${cardMin}px` }}>
        <div className="mb-grid">
          {ordered.map((m) => (
            <div key={m.path} className="mb-card">
              <button
                type="button"
                className="mb-shot"
                onClick={() => setLightbox(m)}
                aria-label={`Enlarge ${m.name}`}
              >
                {m.thumbnail ? <img src={thumbUrl(m.thumbnail)} alt="" /> : <span>{(m.name || "?").slice(0, 2)}</span>}
              </button>
              <b title={m.name}>{m.name}</b>
              <span className="sub">{modelSubtitle(m)}</span>
              <div className="mb-actions">
                <button type="button" className="btn sm primary w-full" onClick={() => onUse?.(m)}>Use in Create</button>
                <div className="mb-actions-row">
                  <button type="button" className="btn sm" onClick={() => onTrain?.(m)}>Train</button>
                  {m.renamable && (
                    <button type="button" className="btn ghost sm" onClick={() => onRename?.(m)}>Rename</button>
                  )}
                  {onStar && <StarBtn m={m} onStar={() => onStar(m.path)} />}
                  <DeleteBtn onClick={() => onDelete?.(m)} label={`Delete ${m.name}`} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      {lightbox && (
        <Modal wide title={lightbox.name} onClose={() => setLightbox(null)}>
          <div className="model-lightbox">
            {lightbox.thumbnail
              ? <img src={thumbUrl(lightbox.thumbnail)} alt={lightbox.name} />
              : <div className="model-lightbox-empty sub">No sample image</div>}
            <div className="model-lightbox-meta">
              <div className="kv"><span>Type</span><b>{lightbox.mtype || "—"}</b></div>
              {lightbox.step != null && <div className="kv"><span>Step</span><b>{lightbox.step}</b></div>}
              <div className="kv"><span>Size</span><b>{lightbox.size_mb} MB</b></div>
            </div>
            <button type="button" className="btn primary" onClick={() => { onUse?.(lightbox); setLightbox(null); }}>
              Use in Create
            </button>
          </div>
        </Modal>
      )}
    </>
  );
}
