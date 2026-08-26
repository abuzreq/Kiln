import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, pollJob, mediaUrl } from "../api.js";
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
  { id: "create", label: "Create new", tip: "Import media, pre-process, and create a dataset" },
  { id: "view", label: "View existing", tip: "Inspect a finished dataset" },
];
const DEFAULT_AUG_SETTINGS = {
  rotate: { mode: "random", angle: 90, all_angles: false, angles: [90, 180, 270] },
  brightness: { min: 0.8, max: 1.2 },
  contrast: { min: 0.8, max: 1.2 },
};

/** Images written per source file (1 base + augmentation extras). */
function imagesPerSource(augs, augSettings) {
  let n = 1;
  if (augs.includes("hflip")) n += 1;
  if (augs.includes("vflip")) n += 1;
  if (augs.includes("rotate")) {
    const rot = { ...DEFAULT_AUG_SETTINGS.rotate, ...augSettings.rotate };
    if (rot.mode === "fixed") n += 1;
    else if (rot.all_angles) n += (rot.angles || [90, 180, 270]).length;
    else n += 1;
  }
  if (augs.includes("brightness")) {
    const b = { ...DEFAULT_AUG_SETTINGS.brightness, ...augSettings.brightness };
    n += b.min === b.max ? 1 : 2;
  }
  if (augs.includes("contrast")) {
    const c = { ...DEFAULT_AUG_SETTINGS.contrast, ...augSettings.contrast };
    n += c.min === c.max ? 1 : 2;
  }
  return n;
}

export default function Prepare() {
  const { toast, openPrepare, setTrainDataset, setTabBusy } = useApp();
  const [info, setInfo] = useState(null);
  const [draftCount, setDraftCount] = useState(0);
  const [dsMode, setDsMode] = useState("create");
  const [viewDs, setViewDs] = useState(null);
  const [importPath, setImportPath] = useState("");
  const [mode, setMode] = useState("center_crop");
  const [padding, setPadding] = useState("edge");
  const [w, setW] = useState(512);
  const [h, setH] = useState(512);
  const [nonSquare, setNonSquare] = useState(false);
  const [augs, setAugs] = useState([]);
  const [augSettings, setAugSettings] = useState({ ...DEFAULT_AUG_SETTINGS });
  const [dsName, setDsName] = useState("dataset");
  const [videoFps, setVideoFps] = useState(2);
  const [previews, setPreviews] = useState([]);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [building, setBuilding] = useState(null);
  const [pendingDs, setPendingDs] = useState(null);
  const [deleteFiles, setDeleteFiles] = useState(false);
  const [thumbLevel, setThumbLevel] = useState("md");
  const [sort, setSort] = useState("name");
  const [nameAvailable, setNameAvailable] = useState(true);
  const fileRef = useRef();

  const creating = dsMode === "create";
  const trimmedName = (dsName || "").trim();
  const selectedMeta = useMemo(
    () => (viewDs ? info?.datasets?.find((d) => d.name === viewDs) : null),
    [viewDs, info],
  );
  const localNameTaken = useMemo(
    () => trimmedName && info?.datasets?.some((d) => d.name === trimmedName),
    [trimmedName, info],
  );

  const reloadDraftCount = () => {
    if (!trimmedName) { setDraftCount(0); return; }
    api.get(`/datasets/draft/files?name=${encodeURIComponent(trimmedName)}&offset=0&limit=1`)
      .then((r) => setDraftCount(r.count || 0))
      .catch(() => setDraftCount(0));
  };

  const reload = () => {
    api.get("/studio").then(setInfo).catch((e) => toast(e.message, "error"));
    reloadDraftCount();
  };

  useEffect(() => { reload(); }, []);
  useEffect(() => { setTabBusy("data", !!building); }, [building, setTabBusy]);
  useEffect(() => { reloadDraftCount(); }, [trimmedName, sort]);

  useEffect(() => {
    if (dsMode !== "view" || !info?.datasets?.length) return;
    if (!viewDs || !info.datasets.some((d) => d.name === viewDs)) {
      setViewDs(info.datasets[0].name);
    }
  }, [dsMode, viewDs, info]);

  useEffect(() => {
    if (!creating || !trimmedName) {
      setNameAvailable(true);
      return undefined;
    }
    const t = setTimeout(async () => {
      try {
        const r = await api.get(`/datasets/available?name=${encodeURIComponent(trimmedName)}`);
        setNameAvailable(!!r.available);
      } catch {
        setNameAvailable(false);
      }
    }, 300);
    return () => clearTimeout(t);
  }, [creating, trimmedName]);

  const previewParams = useMemo(() => ({
    name: trimmedName,
    width: w,
    height: nonSquare ? h : w,
    resize_mode: mode,
    padding_mode: padding,
    augmentations: augs,
    augment_settings: augSettings,
  }), [trimmedName, w, h, nonSquare, mode, padding, augs, augSettings]);

  const perSourceImages = useMemo(() => imagesPerSource(augs, augSettings), [augs, augSettings]);
  const totalOutputImages = draftCount * perSourceImages;

  useEffect(() => {
    if (!creating || !trimmedName || draftCount === 0) {
      setPreviews([]);
      return undefined;
    }
    let cancelled = false;
    const t = setTimeout(async () => {
      setPreviewBusy(true);
      try {
        const r = await api.post("/datasets/preview", previewParams);
        if (!cancelled) setPreviews(r.previews || []);
      } catch (e) {
        if (!cancelled) {
          setPreviews([]);
          toast(e.message, "error");
        }
      } finally {
        if (!cancelled) setPreviewBusy(false);
      }
    }, 200);
    return () => { cancelled = true; clearTimeout(t); };
  }, [creating, trimmedName, draftCount, previewParams, toast]);

  useEffect(() => {
    if (creating) return undefined;
    if (!viewDs) return undefined;
    setPreviews([]);
    api.get(`/datasets/${encodeURIComponent(viewDs)}/samples`)
      .then((r) => {
        if (r.images?.[0]) {
          setPreviews([{ label: "Sample", image: mediaUrl(r.images[0]) }]);
        }
      })
      .catch(() => {});
    return undefined;
  }, [creating, viewDs]);

  const setAug = (key, val) => setAugSettings((s) => ({
    ...s,
    [key]: { ...(DEFAULT_AUG_SETTINGS[key] || {}), ...(s[key] || {}), ...val },
  }));

  const switchMode = (next) => {
    setDsMode(next);
    if (next === "create") {
      setViewDs(null);
      setPreviews([]);
    }
  };

  const doImport = async () => {
    if (!importPath || !trimmedName) return;
    try {
      const r = await api.post("/datasets/draft/import", { name: trimmedName, path: importPath, mode: "copy" });
      toast(`Imported ${r.imported} files`, "success");
      reload();
    } catch (e) { toast(e.message, "error"); }
  };

  const doUpload = async (e) => {
    const files = e.target.files;
    if (!files?.length || !trimmedName) return;
    const fd = new FormData();
    fd.append("name", trimmedName);
    for (const f of files) fd.append("files", f);
    try {
      const r = await api.upload("/datasets/draft/upload", fd);
      toast(`Uploaded ${r.imported} files`, "success");
      reload();
    } catch (err) { toast(err.message, "error"); }
  };

  const removeDraftFile = async (path) => {
    if (!trimmedName) return;
    try {
      await api.del("/datasets/draft/file", { name: trimmedName, path });
      toast("Removed from dataset", "success");
      reload();
    } catch (e) { toast(e.message, "error"); }
  };

  const doCreate = async () => {
    if (!nameAvailable || localNameTaken) {
      toast(`Dataset name “${trimmedName}” is already taken`, "error");
      return;
    }
    try {
      const { job } = await api.post("/datasets/build", {
        name: trimmedName,
        width: w,
        height: nonSquare ? h : w,
        resize_mode: mode,
        padding_mode: padding,
        augmentations: augs,
        augment_settings: augSettings,
        video_fps: videoFps,
      });
      setBuilding(job);
      const done = await pollJob(job.id, setBuilding);
      if (done.status === "done") {
        toast(`${done.message} — ready to train`, "success");
        reload();
        setDsMode("view");
        setViewDs(trimmedName);
      } else {
        toast(done.message, "error");
      }
      setBuilding(null);
    } catch (e) { toast(e.message, "error"); setBuilding(null); }
  };

  const confirmDelete = async () => {
    const d = pendingDs;
    const wipe = deleteFiles;
    setPendingDs(null);
    if (!d) return;
    try {
      await api.del(`/datasets/${encodeURIComponent(d.name)}`, { delete_files: wipe });
      toast(wipe ? `Deleted “${d.name}”` : `Removed “${d.name}” (archived)`, "success");
      if (viewDs === d.name) setViewDs(null);
      reload();
    } catch (e) { toast(e.message, "error"); }
  };

  const draftListPath = `/datasets/draft/files?name=${encodeURIComponent(trimmedName)}`;
  const gallerySourceKey = creating
    ? `draft:${trimmedName}:${draftCount}:${thumbLevel}:${sort}`
    : `ds:${viewDs}:${thumbLevel}:${sort}`;
  const galleryPath = creating
    ? draftListPath
    : `/datasets/${encodeURIComponent(viewDs)}/files`;
  const galleryCount = creating ? draftCount : (selectedMeta?.count || 0);
  const canCreate = creating && !building && draftCount > 0 && nameAvailable && !localNameTaken && trimmedName;
  const hasDatasets = (info?.datasets?.length || 0) > 0;

  return (
    <div className="col">
      <div className="card prepare-mode-card">
        <div className="row between center wrap gap-2 mb-2">
          <h3 className="mb-0">Dataset</h3>
          <Seg
            ariaLabel="Dataset mode"
            tabs={MODE_TABS.map((t) => (t.id === "view" && !hasDatasets ? { ...t, tip: "Create a dataset first" } : t))}
            value={dsMode}
            onChange={(m) => {
              if (m === "view" && !hasDatasets) {
                toast("Create a dataset first", "error");
                return;
              }
              switchMode(m);
            }}
          />
        </div>
        {creating ? (
          <p className="hint mb-0">Import or upload files into this dataset draft, then create it when ready.</p>
        ) : (
          <>
            <p className="hint mb-2">Viewing a finished dataset.</p>
            {hasDatasets ? (
              <Select
                label="Dataset"
                value={viewDs || ""}
                onChange={setViewDs}
                options={info.datasets.map((d) => ({
                  value: d.name,
                  label: `${d.name} (${d.count} images)`,
                }))}
              />
            ) : (
              <Empty>No datasets yet.</Empty>
            )}
          </>
        )}
      </div>

      <div className="work-split">
        <div className="col">
          {creating && (
            <>
              <div className="card">
                <h3>Import</h3>
                <Text label="Name" value={dsName} onChange={setDsName} tip="Dataset name — used as-is when created." />
                <p className={`hint ${(!nameAvailable || localNameTaken) ? "warn-text" : ""}`}>
                  {trimmedName && (!nameAvailable || localNameTaken) && "Name already taken"}
                  {trimmedName && nameAvailable && !localNameTaken && "Name available"}
                </p>
                <Text label="Folder or file path" value={importPath} onChange={setImportPath} placeholder="C:\path\to\images"
                  tip="Copied into this dataset draft only." />
                <div className="row gap-2">
                  <button type="button" className="btn" onClick={doImport} disabled={!importPath || !trimmedName}>Import</button>
                  <button type="button" className="btn ghost" onClick={() => fileRef.current.click()} disabled={!trimmedName}>Upload…</button>
                  <input ref={fileRef} type="file" multiple hidden onChange={doUpload} accept="image/*,video/*" />
                </div>
              </div>

              <div className="card">
                <h3>Pre-Processing</h3>
                <Select label="Resize" value={mode} onChange={setMode} options={RESIZE_MODES} />
                <div className="row gap-2">
                  <div className="grow"><Num label="Width" value={w} onChange={setW} min={32} max={2048} step={32} /></div>
                  <div className="grow"><Num label="Height" value={nonSquare ? h : w} onChange={setH} min={32} max={2048} step={32} /></div>
                </div>
                <label className="row center gap-2 mb-2">
                  <input type="checkbox" checked={nonSquare} onChange={(e) => setNonSquare(e.target.checked)} />
                  <span className="sub">Non-square</span>
                </label>
                {mode === "pad" && (
                  <Select label="Padding" value={padding} onChange={setPadding} options={["edge", "reflect", "constant"]} />
                )}
                <div className="section-title mt-2">Augmentations</div>
                <div className="row wrap gap-2 mb-2">
                  {AUGS.map((a) => (
                    <button
                      key={a.id}
                      type="button"
                      className={`pill chip ${augs.includes(a.id) ? "on" : ""}`}
                      onClick={() => setAugs((list) => list.includes(a.id) ? list.filter((x) => x !== a.id) : [...list, a.id])}
                    >
                      {a.label}
                    </button>
                  ))}
                </div>
                {augs.includes("rotate") && (
                  <>
                    <Select label="Rotate mode" value={augSettings.rotate.mode} onChange={(v) => setAug("rotate", { mode: v })}
                      options={[{ value: "random", label: "Random angle" }, { value: "fixed", label: "Fixed angle" }]} />
                    {augSettings.rotate.mode === "fixed" ? (
                      <Num label="Angle (°)" value={augSettings.rotate.angle} onChange={(v) => setAug("rotate", { angle: v })} min={0} max={359} step={1} />
                    ) : (
                      <label className="row center gap-2 mb-2">
                        <input type="checkbox" checked={augSettings.rotate.all_angles} onChange={(e) => setAug("rotate", { all_angles: e.target.checked })} />
                        <span className="sub">Generate 90°, 180°, and 270° variants</span>
                      </label>
                    )}
                  </>
                )}
                {augs.includes("brightness") && (
                  <div className="row gap-2">
                    <div className="grow"><Slider label="Brightness min" value={augSettings.brightness.min} min={0.4} max={1.5} step={0.05} onChange={(v) => setAug("brightness", { min: v })} /></div>
                    <div className="grow"><Slider label="Brightness max" value={augSettings.brightness.max} min={0.4} max={1.5} step={0.05} onChange={(v) => setAug("brightness", { max: v })} /></div>
                  </div>
                )}
                {augs.includes("contrast") && (
                  <div className="row gap-2">
                    <div className="grow"><Slider label="Contrast min" value={augSettings.contrast.min} min={0.4} max={1.5} step={0.05} onChange={(v) => setAug("contrast", { min: v })} /></div>
                    <div className="grow"><Slider label="Contrast max" value={augSettings.contrast.max} min={0.4} max={1.5} step={0.05} onChange={(v) => setAug("contrast", { max: v })} /></div>
                  </div>
                )}
                <Slider label="Video fps" value={videoFps} min={0.5} max={12} step={0.5} onChange={setVideoFps}
                  tip="Frames extracted per second when importing video." />
                {draftCount > 0 && (
                  <p className="hint mb-0 mt-2">
                    Creates <b>{totalOutputImages.toLocaleString()}</b> image{totalOutputImages === 1 ? "" : "s"} from{" "}
                    <b>{draftCount}</b> source file{draftCount === 1 ? "" : "s"} ({perSourceImages} per source
                    {augs.length ? ", incl. augmentations" : ""}).
                    {videoFps > 0 && " Videos expand to more frames at create time."}
                  </p>
                )}
              </div>

              {building && (
                <div className="mb-2">
                  <Progress value={building.progress} />
                  <div className="sub mt-1">{building.message}</div>
                </div>
              )}
              <button type="button" className="btn primary w-full" onClick={doCreate} disabled={!canCreate}>
                {building ? "Creating…" : "Create dataset"}
              </button>
            </>
          )}

          {!creating && selectedMeta && (
            <div className="card">
              <div className="row between center mb-2">
                <h3 className="mb-0">Details</h3>
                <DeleteBtn label={`Delete ${selectedMeta.name}`} onClick={() => { setDeleteFiles(false); setPendingDs(selectedMeta); }} />
              </div>
              <div className="kv"><span>Size</span><b>{selectedMeta.width || "?"}×{selectedMeta.height || "?"}</b></div>
              <div className="kv"><span>Resize</span><b>{selectedMeta.resize_mode || "—"}</b></div>
              <div className="kv"><span>Augmentations</span><b>{(selectedMeta.augmentations || []).join(", ") || "none"}</b></div>
              <div className="kv"><span>Images</span><b>{selectedMeta.count}</b></div>
              <button
                type="button"
                className="btn primary w-full mt-2"
                onClick={() => { setTrainDataset(selectedMeta.name); openPrepare({ tab: "train" }); }}
              >
                Train on this dataset
              </button>
            </div>
          )}
        </div>

        <div className="col">
          <div className="card">
            <div className="row between center mb-2">
              <h3 className="mb-0">Preview</h3>
              {previewBusy && <span className="sub">Updating…</span>}
            </div>
            <p className="hint mb-2">
              {creating
                ? "Live pre-processing preview from the first draft file, including augmentation variants."
                : viewDs ? `Sample from “${viewDs}”.` : "Pick a dataset."}
            </p>
            {previews.length > 0 ? (
              <div className="preview-var-grid">
                {previews.map((p) => (
                  <div key={p.label} className="preview-var-cell">
                    <div className="preview-box preview-scroll">
                      <img src={p.image} alt={p.label} />
                    </div>
                    <span className="sub">{p.label}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="preview-box preview-scroll preview-h">
                <span className="sub">{creating ? "Import files to see preview." : "Loading preview…"}</span>
              </div>
            )}
          </div>

          <div className="card">
            {(creating ? trimmedName : viewDs) ? (
              <>
                <ThumbGalleryToolbar
                  title={creating ? "Files" : "Images"}
                  count={galleryCount}
                  sort={sort}
                  onSortChange={setSort}
                  thumbLevel={thumbLevel}
                  onThumbLevelChange={setThumbLevel}
                  sortOptions={GALLERY_SORT_OPTS}
                />
                <LazySourceGallery
                  sourceKey={gallerySourceKey}
                  listPath={galleryPath}
                  totalCount={galleryCount}
                  thumbLevel={thumbLevel}
                  sort={sort}
                  deletable={creating}
                  onRemove={creating ? removeDraftFile : undefined}
                />
              </>
            ) : (
              <>
                <h3 className="mb-0">Files</h3>
                <Empty>Enter a dataset name and import files.</Empty>
              </>
            )}
          </div>
        </div>
      </div>

      {pendingDs && (
        <ConfirmModal
          title="Remove dataset"
          body={`Remove dataset “${pendingDs.name}”?`}
          confirmLabel={deleteFiles ? "Delete files" : "Remove"}
          danger={deleteFiles}
          extra={
            <label className="row center gap-2 mt-2">
              <input type="checkbox" checked={deleteFiles} onChange={(e) => setDeleteFiles(e.target.checked)} />
              <span className="sub">Also permanently delete the files from disk</span>
            </label>
          }
          onCancel={() => setPendingDs(null)}
          onConfirm={confirmDelete}
        />
      )}
    </div>
  );
}
