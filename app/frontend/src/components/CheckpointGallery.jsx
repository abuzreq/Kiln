import React, { useMemo, useState } from "react";
import { mediaUrl } from "../api.js";
import { Empty, Modal } from "./ui.jsx";
import { THUMB_LEVELS } from "./LazySourceGallery.jsx";
import { ThumbGalleryToolbar } from "./ThumbGalleryToolbar.jsx";

const CKPT_SORT_OPTS = [
  { value: "step", label: "Step" },
  { value: "name", label: "Name" },
  { value: "date", label: "Date" },
];

function sortCheckpoints(items, sort) {
  const list = items.slice();
  if (sort === "step") {
    list.sort((a, b) => (b.step ?? b.milestone ?? 0) - (a.step ?? a.milestone ?? 0));
  } else if (sort === "date") {
    list.sort((a, b) => (b.mtime ?? 0) - (a.mtime ?? 0));
  } else {
    list.sort((a, b) => (a.filename || "").localeCompare(b.filename || ""));
  }
  return list;
}

export default function CheckpointGallery({ checkpoints, onSave }) {
  const [sort, setSort] = useState("step");
  const [thumbLevel, setThumbLevel] = useState("md");
  const [lightbox, setLightbox] = useState(null);
  const thumbMin = THUMB_LEVELS.find((t) => t.id === thumbLevel)?.min || 108;
  const ordered = useMemo(() => sortCheckpoints(checkpoints, sort), [checkpoints, sort]);

  if (!checkpoints.length) {
    return <Empty>Snapshots saved during training appear here.</Empty>;
  }

  return (
    <>
      <ThumbGalleryToolbar
        title="Snapshots"
        count={checkpoints.length}
        sort={sort}
        onSortChange={setSort}
        thumbLevel={thumbLevel}
        onThumbLevelChange={setThumbLevel}
        sortOptions={CKPT_SORT_OPTS}
      />
      <div className="model-scroll">
        <div className="mb-grid" style={{ "--card-min": `${thumbMin}px` }}>
          {ordered.map((c) => (
            <div key={c.filename} className="mb-card ckpt-card">
              <button
                type="button"
                className="mb-shot"
                onClick={() => setLightbox(c)}
                aria-label={`Enlarge step ${c.step ?? c.milestone}`}
              >
                {c.sample
                  ? <img src={mediaUrl(c.sample)} alt="" />
                  : <span className="sub">No sample</span>}
              </button>
              <b>{c.step != null ? `Step ${c.step}` : `Milestone ${c.milestone}`}</b>
              <span className="sub">{c.size_mb} MB · {c.filename}</span>
              <button type="button" className="btn xs primary w-full ckpt-save-btn" onClick={() => onSave?.(c)}>
                Save to library
              </button>
            </div>
          ))}
        </div>
      </div>
      {lightbox && (
        <Modal
          wide
          title={lightbox.step != null ? `Step ${lightbox.step}` : `Milestone ${lightbox.milestone}`}
          onClose={() => setLightbox(null)}
        >
          <div className="model-lightbox">
            {lightbox.sample
              ? <img src={mediaUrl(lightbox.sample)} alt="" />
              : <div className="model-lightbox-empty sub">No sample image</div>}
            <div className="model-lightbox-meta">
              <div className="kv"><span>File</span><b>{lightbox.filename}</b></div>
              {lightbox.step != null && <div className="kv"><span>Step</span><b>{lightbox.step}</b></div>}
              <div className="kv"><span>Size</span><b>{lightbox.size_mb} MB</b></div>
            </div>
            <button type="button" className="btn sm primary" onClick={() => { onSave?.(lightbox); setLightbox(null); }}>
              Save to library
            </button>
          </div>
        </Modal>
      )}
    </>
  );
}
