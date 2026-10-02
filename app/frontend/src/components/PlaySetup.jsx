import React, { useCallback, useEffect, useRef, useState } from "react";
import { thumbUrl } from "../api.js";
import { PinIcon, ShuffleIcon } from "./icons.jsx";
import {
  PARAM_RANGES, joinLabels, liveEditLabels, paramLabel, randomSeed, samplerLabel,
} from "../sampleSettings.jsx";

// The model and sample settings, out of the way until wanted.
//
// They used to sit in a row above every Play tab -- 224px in a 1024x768
// window, before any tab content started -- though most people set them once
// a session. Now a one-line bar keeps what decides the next run in view (model,
// steps, seed, size, sampler, and anything overriding them), and the row itself
// becomes a drawer that hangs from the bar: floating over the page, or pinned
// into it, which is the old layout.
//
// docs/play-setup-drawer-design.md has the reasoning and the states.

const KEY = "kiln.playSetup";        // "closed" | "pinned"
const PIN_KEY = "kiln.playSetupPin"; // "1" when the last open drawer was pinned

function load(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function save(key, value) {
  try { localStorage.setItem(key, value); } catch { /* the session keeps it */ }
}

/** A number box that only commits whole, in-range values, so a half-typed
 *  "1" on the way to "120" never runs as one step. */
function StepsField({ value, min, max, disabled, onChange }) {
  const [draft, setDraft] = useState(null);
  const commit = (text) => {
    const v = parseInt(text, 10);
    if (!Number.isFinite(v) || v < min || v > max) return;
    onChange(v);
  };
  return (
    <label className="setup-field" title={`Denoising steps, ${min}–${max}`}>
      <input
        type="number"
        inputMode="numeric"
        min={min}
        max={max}
        value={draft ?? value}
        disabled={disabled}
        aria-label="Steps"
        onChange={(e) => { setDraft(e.target.value); commit(e.target.value); }}
        onBlur={() => setDraft(null)}
      />
      <span>steps</span>
    </label>
  );
}

export default function PlaySetup({
  model, params, setParam, sweptBy, overriddenBy, editable, stepsMin, children,
}) {
  const [mode, setMode] = useState(() => {
    const v = load(KEY);
    if (v === "pinned" || v === "closed") return v;
    return model ? "closed" : "floating";
  });
  const wrap = useRef(null);
  const toggleBtn = useRef(null);

  const setAndSave = useCallback((next) => {
    setMode(next);
    // A floating drawer left open is not a preference: the next visit starts closed.
    save(KEY, next === "pinned" ? "pinned" : "closed");
    if (next !== "closed") save(PIN_KEY, next === "pinned" ? "1" : "0");
  }, []);
  const open = useCallback(() => setAndSave(load(PIN_KEY) === "1" ? "pinned" : "floating"), [setAndSave]);
  const close = useCallback((refocus = false) => {
    setAndSave("closed");
    if (refocus) toggleBtn.current?.focus();
  }, [setAndSave]);
  const toggle = useCallback(() => (mode === "closed" ? open() : close()), [mode, open, close]);

  // Ctrl+\ (Cmd+\ on a Mac) shows and hides it from anywhere in Play. Not
  // while writing in a text area, where the keys are the user's.
  useEffect(() => {
    const onKey = (e) => {
      if (!(e.ctrlKey || e.metaKey) || (e.key !== "\\" && e.code !== "Backslash")) return;
      const t = e.target;
      if (t?.tagName === "TEXTAREA" || t?.isContentEditable) return;
      e.preventDefault();
      toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  // A floating drawer closes on Escape or a press anywhere else. A pinned one
  // is part of the page and stays.
  useEffect(() => {
    if (mode !== "floating") return undefined;
    const onKey = (e) => {
      if (e.key !== "Escape" || document.querySelector(".modal-backdrop")) return;
      close(wrap.current?.contains(document.activeElement));
    };
    const onDown = (e) => {
      if (!wrap.current?.contains(e.target)) close();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown);
    };
  }, [mode, close]);

  const swept = (k) => !!sweptBy?.includes(k);
  const can = (k) => !swept(k) && (editable == null || editable.includes(k));
  const range = PARAM_RANGES.steps;
  const size = params.image_size ? `${params.image_size}px` : null;
  const live = editable ? joinLabels(liveEditLabels(params, model, overriddenBy), "and") : "";
  const isOpen = mode !== "closed";

  return (
    <div className={`play-setup ${mode}`} ref={wrap}>
      <div className="setup-bar" role="group" aria-label="Model and sample settings">
        <button
          type="button"
          className="setup-model"
          aria-expanded={isOpen}
          title={model ? `${model.name} — change model` : "Choose a model"}
          onClick={() => (isOpen ? undefined : open())}
        >
          {model ? (
            <>
              <span className="setup-thumb" aria-hidden="true">
                {model.thumbnail ? <img src={thumbUrl(model.thumbnail)} alt="" /> : model.name.slice(0, 2)}
              </span>
              <span className="setup-model-name">{model.name}</span>
              {model.mtype && <span className="setup-model-type">{model.mtype}</span>}
            </>
          ) : (
            <span className="setup-model-none">Choose a model to start</span>
          )}
        </button>

        <span className="setup-sep" aria-hidden="true" />

        {swept("steps") ? (
          <span className="setup-value swept" title="The sweep grid sets this per cell">steps swept</span>
        ) : (
          <StepsField
            value={params.steps}
            min={stepsMin != null ? Math.max(range.min, stepsMin) : range.min}
            max={range.max}
            disabled={!can("steps")}
            onChange={(v) => setParam("steps", v)}
          />
        )}

        {swept("seed") ? (
          <span className="setup-value swept" title="The sweep grid sets this per cell">seed swept</span>
        ) : (
          <span className="setup-field setup-seed">
            <span>seed</span>
            <input
              type="number"
              inputMode="numeric"
              value={params.seed ?? ""}
              placeholder="random"
              disabled={!can("seed")}
              aria-label="Seed — empty draws a new one every run"
              onChange={(e) => setParam("seed", e.target.value === "" ? "" : parseFloat(e.target.value))}
            />
            <button
              type="button"
              className="setup-icon"
              disabled={!can("seed")}
              title="Pin a new random seed"
              aria-label="Pin a new random seed"
              onClick={() => setParam("seed", randomSeed())}
            >
              <ShuffleIcon size={13} />
            </button>
          </span>
        )}

        <button type="button" className="setup-summary" onClick={() => (isOpen ? undefined : open())}
          title="Size and sampler — open the settings to change them">
          {size && <span className="tnum">{size}</span>}
          <span>{samplerLabel(params.sampler || "dpmpp")}</span>
        </button>

        {overriddenBy?.noise_level && (
          <span className="setup-flag warn" title={`${overriddenBy.noise_level} decides noise level here`}>
            noise level: set by Change
          </span>
        )}
        {sweptBy?.length > 0 && (
          <span className="setup-flag info">swept: {sweptBy.map(paramLabel).join(", ").toLowerCase()}</span>
        )}
        {editable && (
          <button type="button" className="setup-flag hot" onClick={() => (isOpen ? undefined : open())}
            title="Only these change what Resume does">
            paused · live: {live}
          </button>
        )}

        <span className="setup-spacer" />

        {isOpen && (
          <button
            type="button"
            className={`setup-icon setup-pin ${mode === "pinned" ? "on" : ""}`.trim()}
            aria-pressed={mode === "pinned"}
            title={mode === "pinned" ? "Unpin: float over the page again" : "Pin: keep it in the page"}
            aria-label={mode === "pinned" ? "Unpin the setup" : "Pin the setup into the page"}
            onClick={() => setAndSave(mode === "pinned" ? "floating" : "pinned")}
          >
            <PinIcon size={14} />
          </button>
        )}
        <button
          ref={toggleBtn}
          type="button"
          className={`btn sm setup-toggle ${isOpen ? "on" : ""}`.trim()}
          aria-expanded={isOpen}
          aria-controls="play-setup-drawer"
          onClick={toggle}
        >
          {isOpen ? "Hide setup" : "Setup"}
          <kbd>Ctrl \</kbd>
        </button>
      </div>

      {isOpen && (
        <div id="play-setup-drawer" className="setup-drawer">
          {children}
        </div>
      )}
    </div>
  );
}
