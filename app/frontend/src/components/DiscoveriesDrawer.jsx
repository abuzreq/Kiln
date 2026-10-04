import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, mediaUrl, thumbUrl } from "../api.js";
import { useApp } from "../state.jsx";
import { ConfirmModal, Seg, Tooltip } from "./ui.jsx";
import { bendPresetSummary } from "../bendSynopsis.js";
import { bendCount, newBendId, normalizeStack } from "../bendStack.js";
import { presetThumb } from "../presetThumb.js";
import DiscoveriesModal from "./DiscoveriesModal.jsx";

// A drawer along the bottom of the app for what the novelty explorer found.
// The strip keeps scrolling while the explorer runs, and new entries join it
// at the trailing end, so it reads as a growing place rather than a gallery.

const SPEED = 30;          // px per second
const HOLD_END = 2500;     // ms to rest at the end before starting over
const HOLD_START = 800;    // ms to rest after jumping back
const POLL = 3000;

const loadOpen = () => {
  try { return localStorage.getItem("kiln.discoveries") === "open"; } catch { return false; }
};
const loadMetric = () => {
  try { return localStorage.getItem("kiln.discoveries.metric") || "clip"; } catch { return "clip"; }
};
// What "looks new" is measured with. The two disagree in an interesting way:
// CLIP groups by what a caption would say, DINOv2 by structure and texture.
const METRIC_TABS = [
  { id: "clip", label: "CLIP", tip: "Judge newness by CLIP image features: a find is new when the picture reads as something different" },
  { id: "dinov2", label: "DINOv2", tip: "Judge newness by DINOv2 features: a find is new when its structure and texture differ, whatever it shows. Loads a 350 MB model the first time" },
];
const METRIC_LABEL = Object.fromEntries(METRIC_TABS.map((t) => [t.id, t.label]));

// What the drawer is, in one breath: the explorer is a novelty search
// (app/core/craft/explore.py), and people took it for a gallery of good results.
const ABOUT = "Kiln tries random bends on a model and keeps one only when its pictures look "
  + "unlike every bend it has kept so far. It is a search for the new and strange, not for "
  + "good pictures: most finds are odd, and a few are worth a closer look.";

const SCOPE_TABS = [
  { id: "all", label: "All models", tip: "Every model's discoveries" },
  { id: "current", label: "This model", tip: "Only the model picked in Create" },
];
const SORT_TABS = [
  { id: "newest", label: "Newest", tip: "In the order they were found" },
  { id: "novel", label: "Most novel", tip: "Farthest from everything else kept" },
];

// The strip runs oldest → newest left to right, so scrolling leftwards is
// always moving toward what just arrived.
const orient = (entries, sort) => (sort === "newest" ? [...entries].reverse() : entries);

export default function DiscoveriesDrawer() {
  const { modelPath, models, ops, setBendStack, setBendCompare, openPlay, toast } = useApp();
  const [open, setOpen] = useState(loadOpen);
  const [status, setStatus] = useState(null);
  const [entries, setEntries] = useState([]);
  const [scope, setScope] = useState("all");
  const [sort, setSort] = useState("newest");
  const [metric, setMetric] = useState(loadMetric);
  // null = follow the explorer (move while it runs); true / false = the user said so
  const [manual, setManual] = useState(null);
  const [fresh, setFresh] = useState(() => new Set());
  const [explore, setExplore] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const sinceRef = useRef(0);
  const viewRef = useRef(null);
  const trackRef = useRef(null);
  const offRef = useRef(0);
  const hoverRef = useRef(false);
  const movingRef = useRef(false);

  const running = !!status?.running;
  const sameModel = running && status?.model_path === modelPath;
  const here = sameModel && (status?.metric || "clip") === metric;
  const currentName = models?.find((m) => m.path === modelPath)?.name || (modelPath ? modelPath.split(/[\\/]/).pop() : "");
  const moving = manual == null ? running : manual;
  movingRef.current = moving;

  useEffect(() => {
    try { localStorage.setItem("kiln.discoveries", open ? "open" : "closed"); } catch { /* ignore */ }
  }, [open]);
  useEffect(() => {
    try { localStorage.setItem("kiln.discoveries.metric", metric); } catch { /* ignore */ }
  }, [metric]);

  const refetch = useCallback(() => {
    const q = new URLSearchParams({ sort, limit: 200 });
    if (scope === "current" && modelPath) q.set("model_path", modelPath);
    return api.get(`/craft/discoveries?${q}`).then((d) => {
      setEntries(orient(d.entries || [], sort));
      sinceRef.current = d.now || sinceRef.current;
    }).catch(() => {});
  }, [scope, sort, modelPath]);

  const clearAll = async () => {
    setConfirmClear(false);
    try {
      const r = await api.del("/craft/discoveries");
      setEntries([]);
      offRef.current = 0;
      toast(`Cleared ${r.removed} discover${r.removed === 1 ? "y" : "ies"}. Saved presets are still in the Library.`, "success");
    } catch (err) { toast(err.message, "error"); }
  };

  const query = useCallback((extra) => {
    const q = new URLSearchParams(extra);
    if (scope === "current" && modelPath) q.set("model_path", modelPath);
    return `/craft/discoveries?${q.toString()}`;
  }, [scope, modelPath]);

  // Full fetch when the filters change; from then on, only what is new. If the
  // full fetch fails (the page loaded while Kiln was restarting), the poll
  // retries it rather than asking "what is new since nothing" forever.
  const loadedRef = useRef(false);
  const loadAll = useCallback(() => api.get(query({ sort, limit: 200 })).then((d) => {
    setEntries(orient(d.entries || [], sort));
    sinceRef.current = d.now || 0;
    loadedRef.current = true;
  }), [query, sort]);
  useEffect(() => {
    let alive = true;
    loadedRef.current = false;
    loadAll().catch(() => { if (alive) loadedRef.current = false; });
    return () => { alive = false; };
  }, [loadAll]);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await api.get("/craft/explore/status");
        if (alive) setStatus(s);
        if (!loadedRef.current) { await loadAll(); return; }
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
        if (!modelPath) { toast("Pick a model in Create first", "error"); return; }
        setStatus(await api.post("/craft/explore", { model_path: modelPath, metric, run: true }));
        setOpen(true);
      }
    } catch (err) { toast(err.message, "error"); }
  };

  const openInBend = (entry) => {
    // An archive written by another build may hold bends in another shape.
    // Opening one must not swap the current stack for nothing.
    const { bends, dropped } = normalizeStack(entry.bends);
    if (!bends.length) {
      toast(dropped ? "This discovery's bends are in a shape this build cannot read" : "This discovery has no bends", "error");
      return;
    }
    if (dropped) toast(`Left out ${bendCount(dropped, "entry", "entries")} of this discovery this build cannot read as a bend`, "warn");
    setBendStack(bends.map((b) => ({ ...b, id: newBendId() })));
    // Both pictures already exist: the discovery itself, and the unbent render
    // the explorer measured its novelty against. Showing them beats re-running
    // a compare that would only reproduce them -- and on the same seed, so the
    // pair differs by the bends and nothing else.
    setBendCompare(entry.image ? {
      plain: entry.baseline ? mediaUrl(entry.baseline) : null,
      bent: mediaUrl(entry.image),
      model: entry.model_path || null,
    } : null);
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
        thumbnail: await presetThumb(thumbUrl(entry.image)),
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

  // The bar is all most people ever see of this, so it says what the explorer
  // is doing and why, not only how far it has got.
  const statusLine = () => {
    if (!status) return "";
    if (status.error && !running) return status.error;
    if (!running) return "novel bends, found by trying random ones · paused";
    const parts = [
      `searching ${status.model_name || "…"} for bends that look new`,
      `${status.tried} tried, ${status.accepted} kept`,
    ];
    if (status.yielding_to) parts.push("waiting for your run");
    else if (status.error) parts.push(status.error);
    return parts.join(" · ");
  };

  const exploreLabel = here ? "Stop" : sameModel ? `Switch to ${METRIC_LABEL[metric]}` : running ? "Search here" : "Find novel bends";
  const exploreTip = here
    ? "Stop searching for new bends on this model"
    : sameModel
      ? `Restart the search on this model, judging novelty by ${METRIC_LABEL[metric]}; each measure keeps its own finds`
      : running
        ? `Searching ${status?.model_name}; move the search to ${currentName || "this model"}`
        : `${ABOUT} Runs on ${currentName || "the picked model"} in the background, and steps aside while you generate.`;

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
        <strong title={ABOUT}>Discoveries</strong>
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
            <Tooltip text="Filter, map by similarity, and walk between neighbours">
              <button type="button" className="btn sm" onClick={() => setExplore(true)}>Explore…</button>
            </Tooltip>
            <Tooltip text="Delete every discovery, starred ones included. Presets you saved stay in the Library">
              <button type="button" className="btn ghost sm" onClick={() => setConfirmClear(true)} disabled={!entries.length} aria-label="Clear all discoveries">🗑</button>
            </Tooltip>
          </span>
        )}
        <span className="row center gap-1" onClick={(e) => e.stopPropagation()}>
          <span className="disc-metric-label" title="Which vision model decides whether a picture looks new">judged by</span>
          <Seg ariaLabel="What judges a bend new" tabs={METRIC_TABS} value={metric} onChange={setMetric} size="sm" />
        </span>
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
              <p className="mb-1">{ABOUT}</p>
              <p className="mb-0">
                {running
                  ? "Searching… the first finds take a minute."
                  : modelPath
                    ? "Nothing found yet. Press Find novel bends and this strip fills in while you work; click a find to open its bends in Bend."
                    : "Pick a model in Create, then press Find novel bends."}
              </p>
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
      {explore && (
        <DiscoveriesModal
          ops={ops || []}
          actions={{ openInBend, star, remove }}
          onClose={() => { setExplore(false); refetch(); }}
        />
      )}
      {confirmClear && (
        <ConfirmModal
          title="Clear all discoveries?"
          body={`This deletes all ${entries.length} discoveries across every model and metric, starred ones included. Presets you saved from them stay in the Library. The explorer keeps running if it is on.`}
          confirmLabel="Delete everything"
          danger
          onCancel={() => setConfirmClear(false)}
          onConfirm={clearAll}
        />
      )}
    </div>
  );
}

function DiscoveryCard({ entry, ops, current, fresh, onOpen, onStar, onRemove }) {
  const summary = bendPresetSummary({ bends: entry.bends }, ops || []);
  const modelCls = `disc-model ${entry.model_missing ? "gone" : current ? "" : "other"}`.trim();
  return (
    <div className={`disc-card ${fresh ? "fresh" : ""}`.trim()} tabIndex={0}>
      <Tooltip text={`${summary}\nnovelty ${entry.novelty} by ${METRIC_LABEL[entry.metric] || "CLIP"} · ${entry.model_name}${entry.model_missing ? " (model gone)" : ""}`}>
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
