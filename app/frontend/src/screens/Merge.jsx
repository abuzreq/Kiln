import React, { useEffect, useMemo, useState } from "react";
import { api, pollJob, thumbUrl } from "../api.js";
import { useApp } from "../state.jsx";
import { usePlay, usePlayState } from "./playContext.jsx";
import { Slider, Select, Text } from "../components/ui.jsx";
import { selectOptions, tagStars } from "./ModelList.jsx";
import { buildSamplePayload } from "../sampleSettings.jsx";

const METHODS = [
  { value: "linear", label: "Linear (weighted average)", tip: "Simple mix of the two models' weights. 0 is all A, 1 is all B." },
  { value: "slerp", label: "Slerp (spherical)", tip: "Blends on a sphere so each layer keeps its magnitude — often preserves character better." },
  { value: "blockwise", label: "Block-wise (per stage)", tip: "Different mix for encoder, mid, and decoder. Keep structure from one model and texture from the other." },
];

async function sampleOnce(modelPath, sampleParams) {
  const body = buildSamplePayload(sampleParams, { model_path: modelPath, postproc: {} });
  const { job } = await api.post("/perform/sample", body);
  const done = await pollJob(job.id, () => {}, 300);
  if (done.status === "error") throw new Error(done.message || "Sample failed");
  return { image: done.detail?.frame || null, card: done.detail?.card || null };
}

const MERGE_SHOTS_EMPTY = { a: null, b: null, merge: null, recipeKey: "" };

export default function Merge() {
  const {
    toast, modelPath, setModelPath, stars, setAppMode, setPlayTab,
    models: allModels, refreshModels,
  } = useApp();
  const { sampleParams } = usePlay();
  const [b, setB] = useState("");
  const [compat, setCompat] = useState(null);
  const [method, setMethod] = useState("linear");
  const [alpha, setAlpha] = useState(0.5);
  const [blocks, setBlocks] = useState({ encoder: 0.5, mid: 0.5, decoder: 0.5 });
  const [which, setWhich] = useState("both");
  const [outName, setOutName] = useState("merge1");
  const [saveRecipe, setSaveRecipe] = useState(false);
  const [busy, setBusy] = usePlayState("merge.writing", false);
  // a compare is a long run — keep it across tab switches
  const [result, setResult] = usePlayState("merge.result", null);
  const [compareBusy, setCompareBusy] = usePlayState("merge.busy", false);
  const [compareMsg, setCompareMsg] = usePlayState("merge.msg", "");
  const [shots, setShots] = usePlayState("merge.shots", MERGE_SHOTS_EMPTY);

  const recipeKey = useMemo(
    () => JSON.stringify({ method, alpha, blocks, which, a: modelPath, b }),
    [method, alpha, blocks, which, modelPath, b],
  );
  const recipeDirty = shots.merge && shots.recipeKey && shots.recipeKey !== recipeKey;

  const models = useMemo(
    () => tagStars(allModels || [], stars).filter((x) => x.role !== "checkpoint"),
    [allModels, stars],
  );

  useEffect(() => {
    if (!models.length) return;
    const alt = models.find((x) => x.path !== modelPath)?.path || "";
    setB((prev) => (prev && prev !== modelPath ? prev : alt));
  }, [models, modelPath]);

  useEffect(() => {
    if (modelPath && b) api.post("/craft/merge/check", { model_a: modelPath, model_b: b }).then(setCompat).catch(() => setCompat(null));
    else setCompat(null);
  }, [modelPath, b]);

  const byPath = useMemo(() => Object.fromEntries(models.map((m) => [m.path, m])), [models]);
  const modelA = byPath[modelPath];
  const modelB = byPath[b];
  const selectOpts = useMemo(
    () => selectOptions(models).filter((o) => o.value && o.value !== modelPath),
    [models, modelPath],
  );

  const doMerge = async () => {
    setBusy(true);
    try {
      const res = await api.post("/craft/merge", {
        model_a: modelPath, model_b: b, out_name: outName, method, alpha,
        block_weights: blocks, which, save_recipe: saveRecipe,
        // give the new model the sample this recipe already produced
        thumbnail: !recipeDirty ? shots.merge : null,
        card: !recipeDirty ? shots.mergeCard : null,
      });
      setResult({ ...res, a: modelA, b: modelB });
      toast(
        res.thumbnail
          ? `Saved ${res.name}.pt with its compare sample as the thumbnail`
          : `New model saved as ${res.name}.pt`,
        "success",
      );
      await refreshModels({ force: true });
    } catch (e) { toast(e.message, "error"); }
    setBusy(false);
  };

  const runCompare = async ({ onlyMerged = false } = {}) => {
    if (!modelPath || !b) { toast("Pick two models", "error"); return; }
    if (compat && !compat.compatible) { toast("Models are incompatible", "error"); return; }
    setCompareBusy(true);
    setCompareMsg("Preparing…");
    try {
      let next = { ...shots };
      if (!onlyMerged || !next.a) {
        setCompareMsg("Sampling model A…");
        next.a = (await sampleOnce(modelPath, sampleParams)).image;
      }
      if (!onlyMerged || !next.b) {
        setCompareMsg("Sampling model B…");
        next.b = (await sampleOnce(b, sampleParams)).image;
      }
      setCompareMsg("Merging recipe…");
      const preview = await api.post("/craft/merge/preview", {
        model_a: modelPath, model_b: b, method, alpha,
        block_weights: blocks, which,
      });
      setCompareMsg("Sampling merged model…");
      const merged = await sampleOnce(preview.path, sampleParams);
      next.merge = merged.image;
      next.mergeCard = merged.card || null;
      next.recipeKey = recipeKey;
      setShots(next);
      toast(onlyMerged ? "Merged sample updated" : "A / B / merge compared", "success");
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setCompareBusy(false);
      setCompareMsg("");
    }
  };

  return (
    <div className="col">
      <div className="card">
        <h3>Merge</h3>
        <p className="hint mb-0">
          Combine two compatible models into a new model file. Model A comes from the picker above;
          pick model B below. Compare A, B, and the current recipe with the same seed before you save.
        </p>
      </div>

      <div className="merge-tri">
        <ModelPanelReadonly title="Model A" model={modelA} shot={shots.a} hint="From the model picker above" />
        <ModelPanel title="Model B" model={modelB} value={b} onChange={setB} options={selectOpts} shot={shots.b} />
        <div className="card merge-panel">
          <h3>Merged (recipe)</h3>
          <div className="merge-shot">
            {shots.merge
              ? <img src={shots.merge} alt="merged sample" />
              : (
                <div className="merge-result-mark">
                  <span className="sub">Generate a compare to see this recipe’s sample here.</span>
                </div>
              )}
          </div>
          {recipeDirty && (
            <p className="hint warn-text">Recipe changed — generate again to refresh the merged sample.</p>
          )}
          {result && (
            <p className="sub mt-2">Last saved library model: <b>{result.name}</b></p>
          )}
        </div>
      </div>

      {compat && !compat.compatible && (
        <div className="card">
          <span className="pill bad">incompatible</span>
          <ul className="hint mt-2">{compat.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
        </div>
      )}
      {compat?.compatible && <span className="pill good">compatible</span>}

      <div className="card">
        <div className="row gap-2 wrap mt-2">
          <button
            type="button"
            className="btn primary"
            onClick={() => runCompare({ onlyMerged: false })}
            disabled={compareBusy || !modelPath || !b || (compat && !compat.compatible)}
          >
            {compareBusy ? (compareMsg || "Working…") : "Generate A / B / merge"}
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => runCompare({ onlyMerged: true })}
            disabled={compareBusy || !modelPath || !b || !shots.a || !shots.b || (compat && !compat.compatible)}
            title="Keep A and B samples; only rebuild the merge with the current recipe"
          >
            Refresh merged only
          </button>
        </div>
      </div>

      <div className="work-split">
        <div className="card">
          <h3>Recipe</h3>
          <Select label="Method" value={method} onChange={setMethod} options={METHODS}
            tip="How the two weight tensors are blended." />
          {method === "blockwise" ? (
            <>
              <p className="hint">0 = all A, 1 = all B, per stage of the UNet.</p>
              <Slider label="Encoder" value={blocks.encoder} min={0} max={1} step={0.05} onChange={(v) => setBlocks({ ...blocks, encoder: v })}
                tip="How much of model B to use in the downsampling / structure half." />
              <Slider label="Mid" value={blocks.mid} min={0} max={1} step={0.05} onChange={(v) => setBlocks({ ...blocks, mid: v })}
                tip="How much of model B to use in the bottleneck." />
              <Slider label="Decoder" value={blocks.decoder} min={0} max={1} step={0.05} onChange={(v) => setBlocks({ ...blocks, decoder: v })}
                tip="How much of model B to use in the upsampling / texture half." />
            </>
          ) : (
            <Slider
              label={`A ↔ B  (${Math.round((1 - alpha) * 100)}% / ${Math.round(alpha * 100)}%)`}
              value={alpha} min={0} max={1} step={0.05} onChange={setAlpha}
              tip="0 is entirely model A, 1 is entirely model B."
            />
          )}
          <Select label="Weights to merge" value={which} onChange={setWhich}
            tip="EMA is the smoothed copy used for sampling; raw is the training weights. Usually keep both."
            options={[
              { value: "both", label: "EMA + raw" },
              { value: "ema", label: "EMA only" },
              { value: "model", label: "raw only" },
            ]} />
        </div>
        <div className="card max-w-sm">
          <h3>Save as</h3>
          <Text label="New model name" value={outName} onChange={setOutName}
            tip="Filename for the merged .pt in your model library." />
          <label className="row center gap-2 mt-2">
            <input type="checkbox" checked={saveRecipe} onChange={(e) => setSaveRecipe(e.target.checked)} />
            <span className="sub">Also save as a reusable recipe</span>
          </label>
          <button type="button" className="btn primary w-full mt-2" onClick={doMerge}
            disabled={busy || !modelPath || !b || (compat && !compat.compatible)}>
            {busy ? "Merging…" : "Merge into new model"}
          </button>
          {!shots.merge && (
            <p className="hint mb-0 mt-1">
              Generate a compare first and the merged sample becomes this model&apos;s thumbnail.
            </p>
          )}
          {result && (
            <button type="button" className="btn w-full mt-2"
              onClick={() => {
                setModelPath(result.path);
                setAppMode("play");
                setPlayTab("create");
                toast(`Using ${result.name} in Create`, "success");
              }}>
              Use in Create
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function ModelPanelReadonly({ title, model, shot, hint }) {
  return (
    <div className="card merge-panel">
      <h3>{title}</h3>
      <p className="sub">{hint}</p>
      <div className="merge-shot">
        {shot
          ? <img src={shot} alt={`${title} sample`} />
          : model?.thumbnail
            ? <img src={thumbUrl(model.thumbnail)} alt={model.name} />
            : <span className="sub">{model ? "No sample yet — generate a compare" : "Pick a model above"}</span>}
      </div>
      {model && (
        <>
          <b>{model.name}</b>
          <div className="kv"><span>arch</span><b>{model.mtype}</b></div>
        </>
      )}
    </div>
  );
}

function ModelPanel({ title, model, value, onChange, options, shot }) {
  return (
    <div className="card merge-panel">
      <h3>{title}</h3>
      <Select label="" value={value} onChange={onChange}
        options={options.length ? options : [{ value: "", label: "no models" }]}
        tip="Pick a library model." />
      <div className="merge-shot">
        {shot
          ? <img src={shot} alt={`${title} sample`} />
          : model?.thumbnail
            ? <img src={thumbUrl(model.thumbnail)} alt={model.name} />
            : <span className="sub">{model ? "No sample yet — generate a compare" : "Pick a model"}</span>}
      </div>
      {model && (
        <div className="kv"><span>arch</span><b>{model.mtype}</b></div>
      )}
    </div>
  );
}
