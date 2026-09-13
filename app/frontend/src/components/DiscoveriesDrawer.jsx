import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, thumbUrl } from "../api.js";
import { useApp } from "../state.jsx";
import { Seg, Tooltip } from "./ui.jsx";
import { bendPresetSummary } from "../bendSynopsis.js";

// A drawer along the bottom of the app for what the novelty explorer found.
// The strip keeps scrolling while the explorer runs, and new entries join it
// at the trailing end, so it reads as a growing place rather than a gallery.
// See docs/exploration-design.md, "The drawer".

const SPEED = 30;          // px per second
const HOLD_END = 2500;     // ms to rest at the end before starting over
const HOLD_START = 800;    // ms to rest after jumping back
const POLL = 3000;

const loadOpen = () => {
  try { return localStorage.getItem("kiln.discoveries") === "open"; } catch { return false; }
};
const newId = () => `b-${Math.random().toString(36).slice(2, 10)}`;

const SCOPE_TABS = [
  { id: "all", label: "All models", tip: "Every model's discoveries" },
  { id: "current", label: "This model", tip: "Only the model picked in Play" },
];
const SORT_TABS = [
  { id: "newest", label: "Newest", tip: "In the order they were found" },
  { id: "novel", label: "Most novel", tip: "Farthest from everything else kept" },
];

// The strip runs oldest → newest left to right, so scrolling leftwards is
// always moving toward what just arrived.
const orient = (entries, sort) => (sort === "newest" ? [...entries].reverse() : entries);

export default function DiscoveriesDrawer() {
  const { modelPath, models, ops, setBendStack, openPlay, toast } = useApp();
  const [open, setOpen] = useState(loadOpen);
  const [status, setStatus] = useState(null);
  const [entries, setEntries] = useState([]);
  const [scope, setScope] = useState("all");
  const [sort, setSort] = useState("newest");
  // null = follow the explorer (move while it runs); true / false = the user said so
  const [manual, setManual] = useState(null);
  const [fresh, setFresh] = useState(() => new Set());
  const sinceRef = useRef(0);
  const viewRef = useRef(null);
  const trackRef = useRef(null);
  const offRef = useRef(0);
  const hoverRef = useRef(false);
  const movingRef = useRef(false);

  const running = !!status?.running;
  const here = running && status?.model_path === modelPath;
  const currentName = models?.find((m) => m.path === modelPath)?.name || (modelPath ? modelPath.split(/[\\/]/).pop() : "");
  const moving = manual == null ? running : manual;
  movingRef.current = moving;

  useEffect(() => {
    try { localStorage.setItem("kiln.discoveries", open ? "open" : "closed"); } catch { /* ignore */ }
  }, [open]);

  const query = useCallback((extra) => {
    const q = new URLSearchParams(extra);
    if (scope === "current" && modelPath) q.set("model_path", modelPath);
    return `/craft/discoveries?${q.toString()}`;
  }, [scope, modelPath]);

  // Full fetch when the filters change; from then on, only what is new.
  useEffect(() => {
    let alive = true;
    api.get(query({ sort, limit: 200 })).then((d) => {
      if (!alive) return;
      setEntries(orient(d.entries || [], sort));
      sinceRef.current = d.now || 0;
    }).catch(() => {});
    return () => { alive = false; };
  }, [query, sort]);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await api.get("/craft/explore/status");
        if (alive) setStatus(s);
        const d = await api.get(query({ since: sinceRef.current, sort: "newest", limit: 50 }));
        if (!alive) return;
        sinceRef.current = d.now || sinceRef.current;
        const incoming = (d.entries || []).reverse();
        if (incoming.length) {
          setEntries((prev) => {
            const seen = new Set(prev.map((e) => e.id));
            return [...prev, ...incoming.filter((e) => !seen.has(e.id))];
          });
          setFresh((f) => new Set([...f, ...incoming.map((e) => e.id)]));
          setTimeout(() => setFresh((f) => {
            const n = new Set(f);
            incoming.forEach((e) => n.delete(e.id));
            return n;
          }), 4000);
        }
      } catch { /* backend away; try again next tick */ }
    };
    tick();
    const id = setInterval(tick, POLL);
    return () => { alive = false; clearInterval(id); };
  }, [query]);

  // The motion: a timer nudging a translateX offset. Not a CSS keyframe, so an
  // appended card never restarts the animation; a timer rather than
  // requestAnimationFrame because embedded webviews can leave rAF silent for
  // a visible page, and a strip that only moves in some windows is worse
  // than one that never does.
  useEffect(() => {
    if (!open) return undefined;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    let last = performance.now();
    let holdUntil = 0;
    let atEnd = false;
    const apply = () => {
      if (trackRef.current) trackRef.current.style.transform = `translateX(${-offRef.current}px)`;
    };
    const step = () => {
      const t = performance.now();
      const dt = Math.min(0.1, (t - last) / 1000);
      last = t;
      const view = viewRef.current;
      const track = trackRef.current;
      if (view && track) {
        const max = Math.max(0, track.scrollWidth - view.clientWidth);
        const go = !reduced && movingRef.current && !hoverRef.current && max > 0;
        if (go && t >= holdUntil) {
          if (atEnd) {
            offRef.current = 0;
            atEnd = false;
            holdUntil = t + HOLD_START;
          } else {
            offRef.current = Math.min(max, offRef.current + SPEED * dt);
            if (offRef.current >= max - 0.5) {
              atEnd = true;
              holdUntil = t + HOLD_END;
            }
          }
        } else if (offRef.current > max) {
          offRef.current = max;
        }
        apply();
      }
    };
    const timer = setInterval(step, 33);

    const onWheel = (e) => {
      const view = viewRef.current;
      const track = trackRef.current;
      if (!view || !track) return;
      e.preventDefault();
      const max = Math.max(0, track.scrollWidth - view.clientWidth);
      offRef.current = Math.max(0, Math.min(max, offRef.current + (e.deltaX || e.deltaY)));
      atEnd = false;
      apply();
    };
    const view = viewRef.current;
    view?.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      clearInterval(timer);
      view?.removeEventListener("wheel", onWheel);
    };
  }, [open]);

  const toggleExplore = async (e) => {
    e.stopPropagation();
    try {
      if (here) {
        setStatus(await api.post("/craft/explore", { run: false }));
      } else {
        if (!modelPath) { toast("Pick a model in Play first", "error"); return; }
        setStatus(await api.post("/craft/explore", { model_path: modelPath, run: true }));
        setOpen(true);
      }
    } catch (err) { toast(err.message, "error"); }
  };

  const openInBend = (entry) => {
    setBendStack((entry.bends || []).map((b) => ({ ...b, id: newId() })));
    if (entry.model_missing) {
      toast("That model is gone; the bends are loaded on the current one", "warn");
      openPlay({ tab: "bend" });
    } else if (entry.model_path && entry.model_path !== modelPath) {
      openPlay({ tab: "bend", model: entry.model_path });
      toast(`Loaded on ${entry.model_name}`, "success");
    } else {
      openPlay({ tab: "bend" });
    }
  };

  const star = async (entry) => {
    const name = `disc-${entry.id}`;
    try {
      await api.post("/craft/bends", {
        name, bends: entry.bends,
        notes: `Found by the explorer on ${entry.model_name} (novelty ${entry.novelty}).`,
        model_hint: entry.model_path,
      });
      await api.post(`/craft/discoveries/${entry.id}/star`, { model_path: entry.model_path });
      setEntries((prev) => prev.map((e) => (e.id === entry.id ? { ...e, starred: true } : e)));
      toast(`Saved “${name}” — it will show up in Create and Bend`, "success");
    } catch (err) { toast(err.message, "error"); }
  };

  const remove = async (entry) => {
    try {
      await api.del(`/craft/discoveries/${entry.id}?model_path=${encodeURIComponent(entry.model_path || "")}`);
      setEntries((prev) => prev.filter((e) => e.id !== entry.id));
    } catch (err) { toast(err.message, "error"); }
  };

  const statusLine = () => {
    if (!status) return "";
    if (status.error && !running) return status.error;
    if (!running) return entries.length ? "not exploring" : "";
    const parts = [`exploring ${status.model_name || "…"}`, `tried ${status.tried}`, `threshold ${status.threshold}`];
    if (status.yielding_to) parts.push("yielding to your run");
    else if (status.error) parts.push(status.error);
    return parts.join(" · ");
  };

  const exploreLabel = here ? "Stop" : running ? "Explore here" : "Start exploring";
  const exploreTip = here
    ? "Stop looking for new bends on this model"
    : running
      ? `Exploring ${status?.model_name}; start here to move it to ${currentName || "this model"}`
      : `Try random bends on ${currentName || "the picked model"} in the background and keep the ones that look new`;

  return (
    <div className={`disc-drawer ${open ? "open" : ""}`.trim()}>
      <div
        className="disc-bar"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen((o) => !o); } }}
      >
        <span className="disc-chev" aria-hidden="true">{open ? "▾" : "▴"}</span>
        <strong>Discoveries</strong>
        <span className="disc-count">{entries.length}</span>
        <span className="disc-status">{statusLine()}</span>
        <span className="spacer" />
        {open && (
          <span className="row center gap-2" onClick={(e) => e.stopPropagation()}>
            <Seg ariaLabel="Which discoveries" tabs={SCOPE_TABS} value={scope} onChange={setScope} size="sm" />
            <Seg ariaLabel="Order" tabs={SORT_TABS} value={sort} onChange={setSort} size="sm" />
            <Tooltip text={moving ? "Hold the strip still" : "Let the strip roll"}>
              <button type="button" className="btn ghost sm" onClick={() => setManual(!moving)} aria-label={moving ? "Pause" : "Play"}>
                {moving ? "⏸" : "▶"}
              </button>
            </Tooltip>
          </span>
        )}
        <Tooltip text={exploreTip}>
          <button
            type="button"
            className={`btn sm ${here ? "" : "primary"}`.trim()}
            onClick={toggleExplore}
            disabled={!modelPath && !running}
          >
            {exploreLabel}
          </button>
        </Tooltip>
      </div>
      {open && (
        <div
          className="disc-strip"
          ref={viewRef}
          onMouseEnter={() => { hoverRef.current = true; }}
          onMouseLeave={() => { hoverRef.current = false; }}
        >
          {entries.length === 0 ? (
            <div className="disc-empty">
              {running
                ? "Looking… the first discoveries take a minute."
                : modelPath
                  ? "Nothing found yet. Start exploring and the strip fills in while you work."
                  : "Pick a model in Play, then start exploring."}
            </div>
          ) : (
            <div className="disc-track" ref={trackRef}>
              {entries.map((e) => (
                <DiscoveryCard
                  key={e.id}
                  entry={e}
                  ops={ops}
                  current={e.model_path === modelPath}
                  fresh={fresh.has(e.id)}
                  onOpen={() => openInBend(e)}
                  onStar={() => star(e)}
                  onRemove={() => remove(e)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DiscoveryCard({ entry, ops, current, fresh, onOpen, onStar, onRemove }) {
  const summary = bendPresetSummary({ bends: entry.bends }, ops || []);
  const modelCls = `disc-model ${entry.model_missing ? "gone" : current ? "" : "other"}`.trim();
  return (
    <div className={`disc-card ${fresh ? "fresh" : ""}`.trim()} tabIndex={0}>
      <Tooltip text={`${summary}\nnovelty ${entry.novelty} · ${entry.model_name}${entry.model_missing ? " (model gone)" : ""}`}>
        <button type="button" className="disc-img" onClick={onOpen} aria-label={`Open in Bend: ${summary}`}>
          <img src={thumbUrl(entry.image)} alt="" loading="lazy" decoding="async" />
        </button>
      </Tooltip>
      <div className="disc-meta">
        <span className={modelCls} title={entry.model_missing ? "This model is no longer in the library" : entry.model_name}>
          {entry.starred ? "★ " : ""}{entry.model_name}
        </span>
      </div>
      <div className="disc-actions">
        <Tooltip text="Load these bends in the Bend tab, on the model that made them">
          <button type="button" onClick={onOpen}>Bend</button>
        </Tooltip>
        <Tooltip text={entry.starred ? "Saved as a preset" : "Save as a bend preset"}>
          <button type="button" onClick={onStar} disabled={entry.starred}>★</button>
        </Tooltip>
        <Tooltip text="Drop this discovery">
          <button type="button" onClick={onRemove} aria-label="Delete">✕</button>
        </Tooltip>
      </div>
    </div>
  );
}
