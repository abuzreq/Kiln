import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, pollJob } from "../api.js";
import { useApp } from "../state.jsx";
import {
  Slider, Text, Num, Select, Progress, Empty, ConfirmModal, DeleteBtn, Seg,
} from "../components/ui.jsx";
import LazySourceGallery from "../components/LazySourceGallery.jsx";
import { ThumbGalleryToolbar, GALLERY_SORT_OPTS } from "../components/ThumbGalleryToolbar.jsx";

const RESIZE_MODES = [
  { value: "center_crop", label: "Center crop" },
  { value: "stretch", label: "Stretch" },
  { value: "pad", label: "Pad (keep aspect)" },
];
const AUGS = [
  { id: "hflip", label: "H flip" },
  { id: "vflip", label: "V flip" },
  { id: "rotate", label: "Rotate" },
  { id: "brightness", label: "Brightness" },
  { id: "contrast", label: "Contrast" },
];
const MODE_TABS = [
  { id: "open", label: "Open", tip: "Edit or inspect a dataset" },
  { id: "create", label: "New dataset", tip: "Start a dataset from folders on this computer" },
];
const DEFAULT_AUG_SETTINGS = {
  rotate: { mode: "angles", angle: 90, angles: [90, 180, 270] },
  brightness: { min: 0.8, max: 1.2, levels: 2 },
  contrast: { min: 0.8, max: 1.2, levels: 2 },
};

/** The recipe as the controls edit it, with every setting filled in. */
function editable(recipe) {
  const r = recipe || {};
  const s = r.augment_settings || {};
  return {
    width: r.width ?? 512,
    height: r.height ?? 512,
    resize_mode: r.resize_mode || "center_crop",
    padding_mode: r.padding_mode || "edge",
    augmentations: r.augmentations || [],
    augment_settings: {
      rotate: { ...DEFAULT_AUG_SETTINGS.rotate, ...s.rotate },
      brightness: { ...DEFAULT_AUG_SETTINGS.brightness, ...s.brightness },
      contrast: { ...DEFAULT_AUG_SETTINGS.contrast, ...s.contrast },
    },
    video_fps: r.video_fps ?? 2,
  };
}

const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

export default function Prepare() {
  const { toast, openPrepare, setTrainDataset, setTabBusy } = useApp();
  const [datasets, setDatasets] = useState(null);
  const [dsMode, setDsMode] = useState("open");
  const [selected, setSelected] = useState(null);
  const [ds, setDs] = useState(null);           // summary of the selected dataset
  const [recipe, setRecipe] = useState(editable(null));
  const [newName, setNewName] = useState("");
  const [nameAvailable, setNameAvailable] = useState(true);
  const [linkPath, setLinkPath] = useState("");
  const [previews, setPreviews] = useState([]);
  const [previewCounts, setPreviewCounts] = useState({});
  const [previewBusy, setPreviewBusy] = useState(false);
  const [job, setJob] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(false);
  const [deleteFiles, setDeleteFiles] = useState(false);
  const [thumbLevel, setThumbLevel] = useState("md");
  const [sort, setSort] = useState("name");
  const [showExcluded, setShowExcluded] = useState(false);
  const fileRef = useRef();

  const creating = dsMode === "create";
  const isRecord = ds?.kind === "record";
  const trimmedName = newName.trim();
  const dirty = isRecord && JSON.stringify(editable(ds.recipe)) !== JSON.stringify(recipe);
  // The preview is the set the trainer will build, so its own count -- and what
  // each augmentation contributes to it -- comes back with it.
  const variantsBuilt = Math.max(1, previewCounts.variants || previews.length || 1);
  const trainingTotal = (ds?.count || 0) * variantsBuilt;
  const breakdown = AUGS
    .filter((a) => previewCounts.counts?.[a.id])
    .map((a) => `${a.label} x${previewCounts.counts[a.id]}`)
    .join(" · ") || "";

  const loadList = () => api.get("/datasets")
    .then((list) => { setDatasets(list || []); return list || []; })
    .catch((e) => { toast(e.message, "error"); setDatasets([]); return []; });

  const loadDs = (name) => {
    if (!name) { setDs(null); return Promise.resolve(null); }
    return api.get(`/datasets/${encodeURIComponent(name)}`)
      .then((d) => { setDs(d); return d; })
      .catch((e) => { toast(e.message, "error"); setDs(null); return null; });
  };

  // One place to take a fresh summary from any editing call.
  const applySummary = (d) => { setDs(d); loadList(); };

  useEffect(() => {
    loadList().then((list) => {
      if (!list.length) setDsMode("create");
      else setSelected((cur) => cur || list[0].name);
    });
  }, []);
  useEffect(() => { setTabBusy("data", !!job); }, [job, setTabBusy]);
  useEffect(() => {
    setShowExcluded(false);
    loadDs(selected).then((d) => setRecipe(editable(d?.recipe)));
  }, [selected]);

  useEffect(() => {
    if (!creating || !trimmedName) { setNameAvailable(true); return undefined; }
    const t = setTimeout(async () => {
      try {
        const r = await api.get(`/datasets/available?name=${encodeURIComponent(trimmedName)}`);
        setNameAvailable(!!r.available);
      } catch { setNameAvailable(false); }
    }, 300);
    return () => clearTimeout(t);
  }, [creating, trimmedName]);

  // Live preview of the recipe being edited, on the dataset's first image.
  useEffect(() => {
    if (creating || !isRecord || !ds.count) { setPreviews([]); setPreviewCounts({}); return undefined; }
    let cancelled = false;
    const t = setTimeout(async () => {
      setPreviewBusy(true);
      try {
        const r = await api.post("/datasets/preview", { dataset: ds.name, recipe });
        if (!cancelled) {
          setPreviews(r.previews || []);
          setPreviewCounts({ counts: r.counts || {}, variants: r.variants || 0 });
        }
      } catch (e) {
        if (!cancelled) { setPreviews([]); setPreviewCounts({}); toast(e.message, "error"); }
      } finally {
        if (!cancelled) setPreviewBusy(false);
      }
    }, 200);
    return () => { cancelled = true; clearTimeout(t); };
  }, [creating, isRecord, ds?.name, ds?.count, recipe]);

  const setR = (patch) => setRecipe((r) => ({ ...r, ...patch }));
  const setAug = (key, val) => setRecipe((r) => ({
    ...r,
    augment_settings: { ...r.augment_settings, [key]: { ...r.augment_settings[key], ...val } },
  }));
  const toggleAug = (id) => setRecipe((r) => ({
    ...r,
    augmentations: r.augmentations.includes(id)
      ? r.augmentations.filter((x) => x !== id)
      : [...r.augmentations, id],
  }));

  const call = async (fn, success) => {
    try {
      const d = await fn();
      applySummary(d);
      if (success) toast(typeof success === "function" ? success(d) : success, "success");
      return d;
    } catch (e) { toast(e.message, "error"); return null; }
  };

  const doCreate = async () => {
    try {
      const d = await api.post("/datasets", { name: trimmedName });
      await loadList();
      setNewName("");
      setDsMode("open");
      setSelected(d.name);
      toast(`Created “${d.name}”. Add a folder of images to it.`, "success");
    } catch (e) { toast(e.message, "error"); }
  };

  const doLink = () => call(async () => {
    const d = await api.post(`/datasets/${encodeURIComponent(ds.name)}/link`, { path: linkPath });
    setLinkPath("");
    return d;
  }, (d) => `Added. ${plural(d.count, "image")} in “${d.name}”.`);

  const doUpload = (e) => {
    const files = e.target.files;
    if (!files?.length) return;
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    e.target.value = "";
    call(() => api.upload(`/datasets/${encodeURIComponent(ds.name)}/upload`, fd), "Uploaded");
  };

  const unlinkSource = (path) => call(
    () => api.del(`/datasets/${encodeURIComponent(ds.name)}/source`, { path }),
    "No longer used. The folder itself is untouched.",
  );

  const excludeFile = (path) => call(
    () => api.post(`/datasets/${encodeURIComponent(ds.name)}/exclude`, { path }),
    "Removed from the dataset",
  );
  const restoreFile = (path) => call(
    () => api.post(`/datasets/${encodeURIComponent(ds.name)}/exclude`, { path, restore: true }),
    "Restored",
  );
  const restoreAll = () => call(
    () => api.post(`/datasets/${encodeURIComponent(ds.name)}/exclude`, { all: true, restore: true }),
    "Every removed image is back",
  );

  const saveRecipe = () => call(
    () => api.post(`/datasets/${encodeURIComponent(ds.name)}/recipe`, { recipe }),
    "Saved",
  );

  const extractFrames = async () => {
    try {
      const { job: j } = await api.post(`/datasets/${encodeURIComponent(ds.name)}/frames`, {});
      setJob(j);
      const done = await pollJob(j.id, setJob);
      toast(done.message, done.status === "done" ? "success" : "error");
      applySummary(await loadDs(ds.name));
    } catch (e) { toast(e.message, "error"); }
    setJob(null);
  };

  const confirmDelete = async () => {
    const name = ds.name;
    setPendingDelete(false);
    try {
      const r = await api.del(`/datasets/${encodeURIComponent(name)}`, { delete_files: deleteFiles });
      toast(r.archived ? `Removed “${name}” (archived)` : `Removed “${name}”`, "success");
      const list = await loadList();
      setSelected(list[0]?.name || null);
      if (!list.length) setDsMode("create");
    } catch (e) { toast(e.message, "error"); }
  };

  const hasDatasets = (datasets?.length || 0) > 0;
  const galleryPath = ds
    ? `/datasets/${encodeURIComponent(ds.name)}/${showExcluded ? "excluded" : "files"}`
    : "";
  const galleryCount = showExcluded ? (ds?.excluded_count || 0) : (ds?.count || 0);

  return (
    <div className="col">
      <div className="card prepare-mode-card">
        <div className="row between center wrap gap-2 mb-2">
          <h3 className="mb-0">Dataset</h3>
          <Seg
            ariaLabel="Dataset mode"
            tabs={MODE_TABS.map((t) => (t.id === "open" && !hasDatasets ? { ...t, tip: "Create a dataset first" } : t))}
            value={dsMode}
            onChange={(m) => {
              if (m === "open" && !hasDatasets) { toast("Create a dataset first", "error"); return; }
              setDsMode(m);
            }}
          />
        </div>
        {creating ? (
          <>
            <p className="hint">
              A dataset is Kiln's list of your images: folders and files that stay where they
              are on disk, plus how to frame and augment them for training.
            </p>
            <div className="row gap-2 wrap" style={{ alignItems: "flex-end" }}>
              <div className="grow"><Text label="Name" value={newName} onChange={setNewName} /></div>
              <button type="button" className="btn primary mb-2" onClick={doCreate}
                disabled={!trimmedName || !nameAvailable}>Create</button>
            </div>
            {trimmedName && !nameAvailable && <p className="hint bad mb-0">That name is already taken.</p>}
          </>
        ) : hasDatasets ? (
          <Select
            label="Dataset"
            value={selected || ""}
            onChange={setSelected}
            options={datasets.map((d) => ({ value: d.name, label: `${d.name} (${plural(d.count, "image")})` }))}
          />
        ) : (
          <Empty>No datasets yet.</Empty>
        )}
      </div>

      {!creating && ds && (
        <div className="work-split">
          <div className="col">
            {isRecord ? (
              <>
                <div className="card">
                  <h3>Images</h3>
                  <p className="hint">
                    Kiln reads your images where they are and never changes them. Only files you
                    upload from the browser are copied into the dataset.
                  </p>
                  <Text label="Folder or file" value={linkPath} onChange={setLinkPath}
                    placeholder="e.g. D:\photos\cats"
                    tip="A folder (with its subfolders) or a single image or video. New files you later put in a linked folder join the dataset automatically." />
                  <div className="row gap-2 mb-2">
                    <button type="button" className="btn" onClick={doLink} disabled={!linkPath.trim()}>Add</button>
                    <button type="button" className="btn ghost" onClick={() => fileRef.current.click()}>Upload…</button>
                    <input ref={fileRef} type="file" multiple hidden onChange={doUpload} accept="image/*,video/*" />
                  </div>
                  {[...ds.sources, ...ds.added].length > 0 && (
                    <div className="col gap-1 mb-2">
                      {[...ds.sources, ...ds.added].map((p) => (
                        <div key={p} className="row gap-2 center">
                          <span className="grow sub" style={{ wordBreak: "break-all" }} title={p}>{p}</span>
                          <button type="button" className="btn ghost sm" onClick={() => unlinkSource(p)}
                            title="Stop using this in the dataset. Nothing on disk is deleted.">Remove</button>
                        </div>
                      ))}
                    </div>
                  )}
                  {ds.uploads > 0 && <p className="sub mb-1">{plural(ds.uploads, "uploaded file")} kept in the dataset.</p>}
                  {ds.missing.length > 0 && (
                    <p className="hint bad">
                      Can't find {ds.missing.length === 1 ? "this" : "these"} any more: {ds.missing.join(", ")}
                    </p>
                  )}
                  {ds.videos_pending > 0 && (
                    <div className="callout">
                      {plural(ds.videos_pending, "video")} need frames extracted before training.{" "}
                      <button type="button" className="btn sm" onClick={extractFrames} disabled={!!job}>Extract frames</button>
                    </div>
                  )}
                  {ds.excluded_count > 0 && (
                    <div className="row gap-2 center wrap">
                      <span className="sub grow">{plural(ds.excluded_count, "image")} removed from the dataset (still on disk).</span>
                      <button type="button" className="btn ghost sm" onClick={() => setShowExcluded((v) => !v)}>
                        {showExcluded ? "Show dataset" : "Show removed"}
                      </button>
                      <button type="button" className="btn ghost sm" onClick={restoreAll}>Restore all</button>
                    </div>
                  )}
                  {job && (
                    <div className="mt-2">
                      <Progress value={job.progress} />
                      <div className="sub mt-1">{job.message}</div>
                    </div>
                  )}
                </div>

                <div className="card">
                  <h3>Framing and augmentation</h3>
                  <Select label="Resize" value={recipe.resize_mode} onChange={(v) => setR({ resize_mode: v })} options={RESIZE_MODES} />
                  <div className="row gap-2">
                    <div className="grow"><Num label="Width" value={recipe.width} onChange={(v) => setR({ width: v })} min={32} max={2048} step={32} /></div>
                    <div className="grow"><Num label="Height" value={recipe.height} onChange={(v) => setR({ height: v })} min={32} max={2048} step={32} /></div>
                  </div>
                  {recipe.resize_mode === "pad" && (
                    <Select label="Padding" value={recipe.padding_mode} onChange={(v) => setR({ padding_mode: v })} options={["edge", "reflect", "constant"]} />
                  )}
                  <div className="section-title mt-2">Augmentations</div>
                  <p className="hint">
                    Every augmentation chosen here multiplies the dataset: training sees each
                    image in every combination of them. Nothing is random, and nothing is
                    written to disk — your files are never changed, and the preview below is
                    the real set.
                  </p>
                  <div className="row wrap gap-2 mb-2">
                    {AUGS.map((a) => (
                      <button key={a.id} type="button"
                        className={`pill chip ${recipe.augmentations.includes(a.id) ? "on" : ""}`}
                        aria-pressed={recipe.augmentations.includes(a.id)}
                        onClick={() => toggleAug(a.id)}>
                        {a.label}
                      </button>
                    ))}
                  </div>
                  {recipe.augmentations.includes("rotate") && (
                    <>
                      <Select label="Rotate" value={recipe.augment_settings.rotate.mode} onChange={(v) => setAug("rotate", { mode: v })}
                        options={[{ value: "angles", label: "Quarter turns: 0°, 90°, 180°, 270° (x4)" },
                          { value: "fixed", label: "One angle: as it is, or turned (x2)" }]} />
                      {recipe.augment_settings.rotate.mode === "fixed" && (
                        <Num label="Angle (°)" value={recipe.augment_settings.rotate.angle} onChange={(v) => setAug("rotate", { angle: v })} min={0} max={359} step={1} />
                      )}
                    </>
                  )}
                  {recipe.augmentations.includes("brightness") && (
                    <div className="row gap-2">
                      <div className="grow"><Slider label="Brightness min" value={recipe.augment_settings.brightness.min} min={0.4} max={1.5} step={0.05} onChange={(v) => setAug("brightness", { min: v })} /></div>
                      <div className="grow"><Slider label="Brightness max" value={recipe.augment_settings.brightness.max} min={0.4} max={1.5} step={0.05} onChange={(v) => setAug("brightness", { max: v })} /></div>
                      <Num label="Levels" value={recipe.augment_settings.brightness.levels}
                        onChange={(v) => setAug("brightness", { levels: Math.max(1, Math.min(6, Math.round(Number(v) || 1))) })}
                        min={1} max={6} step={1}
                        tip="How many brightness factors to use between min and max. Each one is another version of every image." />
                    </div>
                  )}
                  {recipe.augmentations.includes("contrast") && (
                    <div className="row gap-2">
                      <div className="grow"><Slider label="Contrast min" value={recipe.augment_settings.contrast.min} min={0.4} max={1.5} step={0.05} onChange={(v) => setAug("contrast", { min: v })} /></div>
                      <div className="grow"><Slider label="Contrast max" value={recipe.augment_settings.contrast.max} min={0.4} max={1.5} step={0.05} onChange={(v) => setAug("contrast", { max: v })} /></div>
                      <Num label="Levels" value={recipe.augment_settings.contrast.levels}
                        onChange={(v) => setAug("contrast", { levels: Math.max(1, Math.min(6, Math.round(Number(v) || 1))) })}
                        min={1} max={6} step={1}
                        tip="How many contrast factors to use between min and max. Each one is another version of every image." />
                    </div>
                  )}
                  <p className="hint">
                    {breakdown && <>{breakdown} = </>}
                    <b>{variantsBuilt}</b> {variantsBuilt === 1 ? "version" : "versions"} of each image.
                    {" "}Training will see <b>{trainingTotal.toLocaleString()}</b> images per pass
                    {" "}({ds.count.toLocaleString()} × {variantsBuilt}).
                  </p>
                  <Slider label="Video fps" value={recipe.video_fps} min={0.5} max={12} step={0.5} onChange={(v) => setR({ video_fps: v })}
                    tip="Frames taken per second from videos in the dataset. Changing it re-extracts them." />
                  <div className="row gap-2 mt-2">
                    <button type="button" className="btn primary grow" onClick={saveRecipe} disabled={!dirty}>
                      {dirty ? "Save" : "Saved"}
                    </button>
                    {dirty && (
                      <button type="button" className="btn ghost" onClick={() => setRecipe(editable(ds.recipe))}>Discard</button>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="card">
                <h3>Details</h3>
                <p className="hint">
                  {ds.linked
                    ? "This dataset is a folder linked into Kiln's workspace. It trains on the images as they are."
                    : "Built by an earlier version of Kiln: a folder of finished images, augmentations included. It trains on them as they are."}
                </p>
                <div className="kv"><span>Images</span><b>{ds.count}</b></div>
              </div>
            )}

            <div className="card">
              <div className="row between center mb-2">
                <h3 className="mb-0">{plural(ds.count, "image")}</h3>
                <DeleteBtn label={`Remove ${ds.name}`} onClick={() => { setDeleteFiles(false); setPendingDelete(true); }} />
              </div>
              {dirty && <p className="hint">Save the recipe first: training uses the saved one.</p>}
              <button
                type="button"
                className="btn primary w-full"
                disabled={!ds.count || dirty || ds.videos_pending > 0}
                onClick={() => { setTrainDataset(ds.name); openPrepare({ tab: "train" }); }}
              >
                Train on this dataset
              </button>
            </div>
          </div>

          <div className="col">
            {isRecord && (
              <div className="card">
                <div className="row between center mb-2">
                  <h3 className="mb-0">Preview</h3>
                  {previewBusy && <span className="sub">Updating…</span>}
                </div>
                <p className="hint mb-2">Every version of the first image this recipe will train on.</p>
                {previews.length > 0 ? (
                  <div className="preview-var-grid">
                    {previews.map((p, i) => (
                      <div key={`${i}-${p.label}`} className="preview-var-cell">
                        <div className="preview-box preview-scroll"><img src={p.image} alt={p.label} /></div>
                        <span className="sub">{p.label}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="preview-box preview-scroll preview-h">
                    <span className="sub">{ds.count ? "Loading preview…" : "Add images to see a preview."}</span>
                  </div>
                )}
              </div>
            )}

            <div className="card">
              <ThumbGalleryToolbar
                title={showExcluded ? "Removed" : "Images"}
                count={galleryCount}
                sort={sort}
                onSortChange={setSort}
                thumbLevel={thumbLevel}
                onThumbLevelChange={setThumbLevel}
                sortOptions={GALLERY_SORT_OPTS}
              />
              {galleryCount > 0 ? (
                <LazySourceGallery
                  sourceKey={`${ds.name}:${showExcluded}:${galleryCount}:${thumbLevel}:${sort}`}
                  listPath={galleryPath}
                  totalCount={galleryCount}
                  thumbLevel={thumbLevel}
                  sort={sort}
                  deletable={isRecord}
                  onRemove={isRecord ? (showExcluded ? restoreFile : excludeFile) : undefined}
                  removeLabel={showExcluded ? "Restore to dataset" : "Remove from dataset"}
                />
              ) : (
                <Empty>{isRecord ? "Add a folder or upload images." : "No images."}</Empty>
              )}
            </div>
          </div>
        </div>
      )}

      {pendingDelete && ds && (
        <ConfirmModal
          title="Remove dataset"
          body={
            isRecord
              ? `Remove “${ds.name}”? Kiln deletes its record of the dataset${ds.uploads ? ` and the ${plural(ds.uploads, "file")} uploaded into it` : ""}. Linked folders and files stay on disk untouched.`
              : ds.linked
                ? `Remove “${ds.name}”? Only the link in Kiln's workspace is removed; the folder it points to is untouched.`
                : `Remove “${ds.name}”?`
          }
          confirmLabel={!isRecord && !ds.linked && deleteFiles ? "Delete files" : "Remove"}
          danger={isRecord || ds.linked || deleteFiles}
          extra={!isRecord && !ds.linked && (
            <label className="row center gap-2 mt-2">
              <input type="checkbox" checked={deleteFiles} onChange={(e) => setDeleteFiles(e.target.checked)} />
              <span className="sub">Also permanently delete the images (otherwise they are archived)</span>
            </label>
          )}
          onCancel={() => setPendingDelete(false)}
          onConfirm={confirmDelete}
        />
      )}
    </div>
  );
}
