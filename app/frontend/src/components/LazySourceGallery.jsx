import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, mediaUrl, thumbUrl } from "../api.js";
import { DeleteBtn, Modal } from "./ui.jsx";

export const THUMB_LEVELS = [
  { id: "sm", label: "S", min: 72 },
  { id: "md", label: "M", min: 108 },
  { id: "lg", label: "L", min: 152 },
];

const GAP = 8;
const PAGE = 48;
const OVERSCAN = 2;

export default function LazySourceGallery({
  sourceKey,
  listPath,
  totalCount,
  thumbLevel = "md",
  sort = "name",
  deletable = false,
  onRemove,
  removeLabel = "Remove file",
}) {
  const thumbMin = THUMB_LEVELS.find((t) => t.id === thumbLevel)?.min || 108;
  const wrapRef = useRef(null);
  const loadingRef = useRef(false);
  const [paths, setPaths] = useState([]);
  const [loaded, setLoaded] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewH, setViewH] = useState(360);
  const [cols, setCols] = useState(4);
  const [lightbox, setLightbox] = useState(null);

  const query = useCallback((from) => {
    const q = new URLSearchParams({ offset: String(from), limit: String(PAGE), sort });
    if (listPath.includes("?")) return `${listPath}&${q}`;
    return `${listPath}?${q}`;
  }, [listPath, sort]);

  const loadMore = useCallback(async (from) => {
    if (loadingRef.current || from >= totalCount) return;
    loadingRef.current = true;
    try {
      const data = await api.get(query(from));
      const batch = data.files || [];
      setPaths((prev) => (from === 0 ? batch : [...prev, ...batch]));
      setLoaded(from + batch.length);
    } finally {
      loadingRef.current = false;
    }
  }, [query, totalCount]);

  useEffect(() => {
    setPaths([]);
    setLoaded(0);
    setScrollTop(0);
    if (wrapRef.current) wrapRef.current.scrollTop = 0;
  }, [sourceKey]);

  useEffect(() => {
    if (totalCount > 0 && loaded === 0) loadMore(0);
  }, [sourceKey, totalCount, loaded, loadMore]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const measure = () => {
      const w = Math.max(el.clientWidth - GAP, thumbMin);
      setCols(Math.max(1, Math.floor((w + GAP) / (thumbMin + GAP))));
      setViewH(el.clientHeight);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [thumbMin, sourceKey]);

  const rowH = thumbMin + GAP + (deletable ? 4 : 0);
  const totalRows = Math.max(1, Math.ceil(totalCount / cols));
  const totalH = totalRows * rowH;
  const startRow = Math.max(0, Math.floor(scrollTop / rowH) - OVERSCAN);
  const endRow = Math.min(totalRows, Math.ceil((scrollTop + viewH) / rowH) + OVERSCAN);
  const needCount = Math.min(totalCount, endRow * cols);

  useEffect(() => {
    if (needCount > loaded && totalCount > 0) loadMore(loaded);
  }, [needCount, loaded, totalCount, loadMore]);

  const onScroll = (e) => setScrollTop(e.target.scrollTop);

  if (!totalCount) {
    return <div className="empty">No images.</div>;
  }

  const cells = [];
  for (let row = startRow; row < endRow; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const idx = row * cols + col;
      if (idx >= totalCount) continue;
      const path = paths[idx];
      cells.push(
        <div
          key={path || idx}
          className={`thumb virtual-cell ${path ? "clickable" : ""} ${deletable ? "deletable" : ""}`}
          style={{
            left: col * (thumbMin + GAP),
            top: row * rowH,
            width: thumbMin,
            height: thumbMin,
          }}
          onClick={path ? () => setLightbox(path) : undefined}
          onKeyDown={path ? (e) => { if (e.key === "Enter") setLightbox(path); } : undefined}
          role={path ? "button" : undefined}
          tabIndex={path ? 0 : undefined}
        >
          {path ? <img src={thumbUrl(path)} alt="" loading="lazy" decoding="async" /> : null}
          {deletable && path && onRemove && (
            <DeleteBtn label={removeLabel} onClick={(e) => { e.stopPropagation(); onRemove(path); }} />
          )}
        </div>,
      );
    }
  }

  return (
    <>
      <div ref={wrapRef} className="model-scroll source-scroll" onScroll={onScroll}>
        <div className="source-inner" style={{ height: totalH }}>
          {cells}
        </div>
      </div>
      {lightbox && (
        <Modal wide title={lightbox.split(/[/\\]/).pop()} onClose={() => setLightbox(null)}>
          <div className="model-lightbox">
            <img src={mediaUrl(lightbox)} alt="" />
          </div>
        </Modal>
      )}
    </>
  );
}
