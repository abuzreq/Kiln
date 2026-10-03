import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../state.jsx";
import { Modal, Text } from "./ui.jsx";

/** The workspace folders, in the order the app actually uses them. */
const FOLDERS = [
  { key: "root", label: "Workspace", hint: "Everything Kiln writes lives here" },
  { key: "source", label: "Source", hint: "Imported media before it becomes a dataset" },
  { key: "datasets", label: "Datasets", hint: "Prepared image folders used for training" },
  { key: "runs", label: "Training runs", hint: "Checkpoints, samples and logs per run" },
  { key: "models", label: "Models", hint: "Named library models (.pt)" },
  { key: "captures", label: "Captures", hint: "Images you saved, with their settings embedded" },
  { key: "sweeps", label: "Sweeps", hint: "Parameter grids and sweep animations" },
  { key: "library", label: "Library", hint: "Saved bends, merge recipes and pins" },
  // The install's old model folders. Not workspace folders, and no longer
  // shipped, but still scanned; the server names them only when they exist.
  { key: "models_pretrained", label: "Install models (old)", legacy: true,
    hint: "Models dropped into the install's models/pretrained. Still scanned; Models is the better home" },
  { key: "models_fine_tuned", label: "Install fine-tunes (old)", legacy: true,
    hint: "Still scanned; Models is the better home" },
];

export default function WorkspaceModal({ onClose }) {
  const { workspace, toast } = useApp();
  const [busy, setBusy] = useState("");
  const [moving, setMoving] = useState(false);
  const [newRoot, setNewRoot] = useState("");

  const open = async (key) => {
    setBusy(key);
    try {
      await api.post("/workspace/open", { key });
    } catch (e) {
      toast(e.message, "error");
    }
    setBusy("");
  };

  const changeRoot = async () => {
    const path = newRoot.trim();
    if (!path) return;
    setMoving(true);
    try {
      await api.post("/workspace", { path });
      toast("Workspace changed — reloading", "success");
      setTimeout(() => window.location.reload(), 600);
    } catch (e) {
      toast(e.message, "error");
      setMoving(false);
    }
  };

  return (
    <Modal wide title="Workspace" onClose={onClose}>
      <p className="hint">
        Kiln keeps your data outside the app folder so it survives updates.
        Open any location below in your file manager.
      </p>

      {!workspace ? (
        <p className="sub">Loading…</p>
      ) : (
        <div className="col gap-2">
          {FOLDERS.filter((f) => !f.legacy || workspace[f.key]).map((f) => (
            <div key={f.key} className="asset-row static ws-row">
              <div className="meta">
                <b>{f.label}</b>
                <span className="sub">{f.hint}</span>
                <code className="ws-path">{workspace[f.key] || "—"}</code>
              </div>
              <button
                type="button"
                className="btn sm"
                onClick={() => open(f.key)}
                disabled={busy === f.key || !workspace[f.key]}
              >
                {busy === f.key ? "Opening…" : "Open"}
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="section-title mt-2">Change workspace</div>
      <p className="hint mb-2">
        Point Kiln at a different folder. Existing data stays where it is — the
        new folder starts empty and Kiln reloads.
      </p>
      <div className="row center gap-2">
        <div className="grow">
          <Text
            label=""
            value={newRoot}
            onChange={setNewRoot}
            placeholder={workspace?.root || "C:\\path\\to\\workspace"}
          />
        </div>
        <button
          type="button"
          className="btn"
          onClick={changeRoot}
          disabled={!newRoot.trim() || moving}
        >
          {moving ? "Switching…" : "Switch"}
        </button>
      </div>
    </Modal>
  );
}
