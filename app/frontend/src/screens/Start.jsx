import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, mediaUrl, thumbUrl } from "../api.js";
import { useApp } from "../state.jsx";
import { Empty, Loading } from "../components/ui.jsx";
import OpenFolder from "../components/OpenFolder.jsx";
import { splitModels, tagStars } from "./ModelList.jsx";
import { modelSubtitle } from "../components/modelMeta.jsx";
import { cardLabel, mergeStoredSampleParams, paramsFromCard } from "../sampleSettings.jsx";
import { PLAY_TAB_IDS, PREPARE_TAB_IDS, anyBusy } from "../navTabs.js";

const CYCLE_MS = 2000;     // one card re-rolls every two seconds
const POLL_MS = 4000;
const CAPTURE_LIMIT = 60;  // a wrapping grid, so this can hold a lot more than a strip

/** Seed number out of a cached preview path (".../seed-44.png"). */
function seedOf(path) {
  const m = /seed-(\d+)\.png$/i.exec(path || "");
  return m ? parseInt(m[1], 10) : null;
}

/** Does the viewer want motion? Live, because the OS setting can change mid-session. */
function useMotionAllowed() {
  const [allowed, setAllowed] = useState(() => {
    if (typeof window === "undefined" || !window.matchMedia) return true;
    return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  });
  useEffect(() => {
    if (!window.matchMedia) return undefined;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setAllowed(!mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return allowed;
}

/** Is the tab in the foreground? A hidden tab should not be animating. */
function usePageVisible() {
  const [visible, setVisible] = useState(() => !document.hidden);
  useEffect(() => {
    const on = () => setVisible(!document.hidden);
    document.addEventListener("visibilitychange", on);
    return () => document.removeEventListener("visibilitychange", on);
  }, []);
  return visible;
}

export default function Start() {
  const {
    models, modelsPartial, stars, appMode, busyTabs, toast,
    setModelPath, openPrepare, openPlay,
  } = useApp();

  const [previews, setPreviews] = useState({});
  const [failed, setFailed] = useState({});
  const [captures, setCaptures] = useState(null);
  const [frameIdx, setFrameIdx] = useState({});
  const [featured, setFeatured] = useState(null);
  const cursor = useRef(0);

  const motionAllowed = useMotionAllowed();
  const pageVisible = usePageVisible();

  // The first scan streams in: show each model as it is read rather than an
  // empty spinner until the slowest checkpoint opens. A later refresh keeps the
  // finished list on screen instead of shrinking it back to the first few.
  const listed = models ?? (modelsPartial?.length ? modelsPartial : null);
  const stillReading = !models && !!modelsPartial;
  const cards = useMemo(() => {
    if (!listed) return null;
    const { pinned, rest } = splitModels(tagStars(listed, stars));
    return [...pinned, ...rest];
  }, [listed, stars]);

  // --- fill the preview cache -----------------------------------------
  // One polled call does double duty: it reports what is cached and keeps the
  // background queue topped up (enqueue de-dupes, so re-asking is free).
  useEffect(() => {
    if (!cards?.length) return undefined;
    const paths = cards.map((m) => m.path);
    let stopped = false;
    let timer = null;

    const poll = async () => {
      try {
        const d = await api.post("/library/previews", { paths, generate: true });
        if (stopped) return;
        setPreviews(d.previews || {});
        setFailed(d.failed || {});
        // A model that could not be sampled counts as settled: the server will
        // not retry it until the file changes, so waiting on it is pointless.
        const done = paths.every((p) => d.failed?.[p] || (d.previews?.[p] || []).length >= d.count);
        // Stop when everything is cached, and also when nothing is queued or in
        // flight — that means generation has stopped making progress, so polling
        // on would just spin forever.
        if (!done && (d.queued > 0 || d.pending > 0)) {
          timer = setTimeout(poll, POLL_MS);
        }
      } catch {
        /* keep whatever we already have */
      }
    };
    poll();
    return () => { stopped = true; if (timer) clearTimeout(timer); };
  }, [cards]);

  useEffect(() => {
    api.get("/captures")
      .then((d) => {
        const all = d.captures || [];
        setCaptures(all.slice(0, CAPTURE_LIMIT));
        // Picked once per app start, from everything saved rather than just the
        // recent slice — so opening Kiln resurfaces old work you had forgotten.
        setFeatured(all.length ? all[Math.floor(Math.random() * all.length)] : null);
      })
      .catch(() => { setCaptures([]); setFeatured(null); });
  }, []);

  // --- the stagger ------------------------------------------------------
  // A single timer with a round-robin cursor: each tick advances one card, so
  // the wave rolls down the grid a second apart without a timer per card.
  const animating = motionAllowed && pageVisible && appMode === "start" && !!cards?.length;
  useEffect(() => {
    if (!animating) return undefined;
    const id = setInterval(() => {
      const list = cards;
      if (!list.length) return;
      const card = list[cursor.current % list.length];
      cursor.current += 1;
      const n = (previews[card.path] || []).length;
      // Still consumed the tick even with nothing to show, so the rhythm holds.
      if (n < 2) return;
      setFrameIdx((m) => ({ ...m, [card.path]: ((m[card.path] || 0) + 1) % n }));
    }, CYCLE_MS);
    return () => clearInterval(id);
  }, [animating, cards, previews]);

  const openModel = useCallback((m, seed) => {
    if (seed != null) mergeStoredSampleParams({ seed });
    openPlay({ tab: "create", model: m.path });
    toast(seed == null ? `${m.name} ready in Create` : `${m.name} · seed ${seed}`, "success");
  }, [openPlay, toast]);

  const restoreCapture = useCallback((it) => {
    const params = paramsFromCard(it.card);
    if (!params) { toast("This image has no recorded settings", "warn"); return; }
    mergeStoredSampleParams(params);
    if (it.card.model_path) setModelPath(it.card.model_path);
    openPlay({ tab: "create" });
    toast(`Settings restored — ${cardLabel(it.card) || "sampler updated"}`, "success");
  }, [openPlay, setModelPath, toast]);

  const prepareBusy = anyBusy(busyTabs, PREPARE_TAB_IDS);
  const playBusy = anyBusy(busyTabs, PLAY_TAB_IDS);

  return (
    <div className="start-hub">
      <div className="start-hero">
        <div className="start-hero-text">
          <h2>Your kiln</h2>
          <p className="hint mb-0">
            Everything you have made lives here. Pick up a model, or head somewhere to make more.
          </p>
          <OpenFolder folder="root" label="Open workspace folder" className="mt-2" />
        </div>
        <div className="start-paths">
          <button type="button" className="start-path" onClick={() => openPrepare({ tab: "data" })}>
            <span className="start-path-name">Prepare</span>
            <span className="start-path-sub">Build datasets, train models</span>
            {prepareBusy && <span className="tab-runbar" aria-hidden="true" />}
          </button>
          <button type="button" className="start-path primary" onClick={() => openPlay({ tab: "create" })}>
            <span className="start-path-name">Create</span>
            <span className="start-path-sub">Generate, bend, merge, sweep</span>
            {playBusy && <span className="tab-runbar" aria-hidden="true" />}
          </button>
        </div>
      </div>

      {featured && (
        <button
          type="button"
          className="start-featured"
          onClick={() => restoreCapture(featured)}
          title={`Restore the settings behind ${featured.name}`}
        >
          <img src={thumbUrl(featured.path)} alt="" />
          <span className="start-featured-body">
            <span className="section-title">From your captures</span>
            <b>{featured.name}</b>
            <span className="sub">{cardLabel(featured.card) || "no recorded settings"}</span>
            <span className="start-featured-cta">Pick up where this left off →</span>
          </span>
        </button>
      )}

      <div className="start-section">
        <div className="row between center wrap gap-2">
          <h3 className="mb-0">Models{cards?.length ? ` (${cards.length})` : ""}</h3>
          {stillReading ? (
            <span className="sub"><span className="spinner sm" aria-hidden /> Reading more models…</span>
          ) : cards?.length > 0 && (
            <span className="sub">Click a card to open that seed</span>
          )}
        </div>
        {cards === null ? (
          <Loading>Loading models…</Loading>
        ) : !cards.length ? (
          <Empty>
            No models yet. Train one in Prepare, or download a starting point from the Library.
          </Empty>
        ) : (
          <div className="start-grid">
            {cards.map((m) => (
              <ModelCard
                key={m.path}
                m={m}
                frames={previews[m.path] || []}
                failure={failed[m.path]}
                index={frameIdx[m.path] || 0}
                onOpen={openModel}
              />
            ))}
          </div>
        )}
      </div>

      <div className="start-section">
        <h3 className="mb-0">Captures{captures?.length ? ` (${captures.length})` : ""}</h3>
        {captures === null ? (
          <Loading>Loading captures…</Loading>
        ) : !captures.length ? (
          <Empty>No saved images yet. Capture an output in Create and it shows up here.</Empty>
        ) : (
          <div className="start-strip">
            {captures.map((it) => (
              <button
                key={it.path}
                type="button"
                className="start-capture"
                onClick={() => restoreCapture(it)}
                title={`${it.name}${it.card ? ` · ${cardLabel(it.card) || ""}` : ""}`}
              >
                <img src={thumbUrl(it.path)} alt="" loading="lazy" decoding="async" />
                <span className="start-capture-hint">Restore</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ModelCard({ m, frames, index, onOpen, failure }) {
  // Every frame is stacked and only opacity changes, so a swap cross-fades
  // instead of flashing through an unloaded image.
  const has = frames.length > 0;
  const active = has ? Math.min(index, frames.length - 1) : 0;
  const seed = has ? seedOf(frames[active]) : null;

  return (
    <div className="start-card">
      <button
        type="button"
        className="start-card-shot"
        onClick={() => onOpen(m, seed)}
        aria-label={`Open ${m.name} in Create${seed != null ? `, seed ${seed}` : ""}`}
      >
        {has ? (
          frames.map((f, i) => (
            <img
              key={f}
              src={mediaUrl(f)}
              className={i === active ? "on" : ""}
              alt=""
              decoding="async"
            />
          ))
        ) : m.thumbnail ? (
          <img src={thumbUrl(m.thumbnail)} className="on" alt="" loading="lazy" />
        ) : (
          <span className="start-card-mark">{m.name.slice(0, 2)}</span>
        )}
        {m.starred && <span className="start-card-star" aria-hidden="true">★</span>}
        <span className="start-card-hint">
          {seed != null ? `Open · seed ${seed}` : "Open in Create"}
        </span>
        {failure && !has && (
          <span className="start-card-filling" title={failure}>Can't preview this model</span>
        )}
        {has && frames.length < 6 && !failure && (
          <span className="start-card-filling" aria-hidden="true">
            <span className="spinner sm" /> {frames.length}/6
          </span>
        )}
      </button>
      <b title={m.name}>{m.name}</b>
      <span className="sub">{modelSubtitle(m) || `${m.mtype} · ${m.size_mb} MB`}</span>
    </div>
  );
}
