import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../state.jsx";
import { ConfirmModal, DeleteBtn, Text, Modal } from "./ui.jsx";

export function modelSubtitle(m) {
  if (!m) return "";
  const bits = [];
  if (m.original_name && m.original_name !== m.name) bits.push(`trained as ${m.original_name}`);
  else {
    const t = (m.trained_as || []).find((x) => x && x !== m.name);
    if (t) bits.push(`trained as ${t}`);
  }
  if (m.step != null) bits.push(`step ${m.step}`);
  if (m.mtype) bits.push(m.mtype);
  if (m.attn) bits.push(`attention ${m.attn}`);
  if (m.size_mb != null) bits.push(`${m.size_mb} MB`);
  return bits.join(" · ");
}

/** Delete for models Kiln owns, Hide for models it only found (see library.owned_by_kiln). */
export function RemoveModelBtn({ m, onRemove }) {
  if (m.owned) return <DeleteBtn onClick={() => onRemove?.(m)} label={`Delete ${m.name}`} />;
  return (
    <button
      type="button"
      className="btn ghost sm"
      onClick={() => onRemove?.(m)}
      title="This file is not in Kiln's workspace. Hiding takes it off Kiln's lists and leaves the file alone."
    >
      Hide
    </button>
  );
}

/** Confirms and performs Delete or Hide, whichever applies to the model. */
export function RemoveModelModal({ model, onClose, onDone }) {
  const { toast, modelPath, setModelPath } = useApp();
  const run = async () => {
    onClose();
    try {
      if (model.owned) {
        await api.del("/library/model", { path: model.path });
        toast(`Deleted ${model.name}`, "success");
      } else {
        await api.post("/library/model/hide", { path: model.path, hidden: true });
        toast(`Hid ${model.name}. The file is still on disk.`, "success");
      }
      if (modelPath === model.path) setModelPath("");
      onDone?.();
    } catch (e) { toast(e.message, "error"); }
  };
  return model.owned ? (
    <ConfirmModal
      title="Delete model"
      body={`Delete “${model.name}”? This removes the file from Kiln's workspace.`}
      confirmLabel="Delete"
      danger
      onCancel={onClose}
      onConfirm={run}
    />
  ) : (
    <ConfirmModal
      title="Hide model"
      body={<>
        <p className="hint">
          “{model.name}” lives outside Kiln's workspace, so Kiln won't delete it. Hiding
          takes it off Kiln's lists; the file stays where it is:
        </p>
        <p className="sub mb-0" style={{ wordBreak: "break-all" }}>{model.path}</p>
      </>}
      confirmLabel="Hide from Kiln"
      onCancel={onClose}
      onConfirm={run}
    />
  );
}

export function RenameModal({ model, onClose, onRenamed }) {
  const { toast, modelPath, setModelPath, refreshStars } = useApp();
  const [name, setName] = useState(model?.name || "");
  const [busy, setBusy] = useState(false);
  const trained = model?.original_name && model.original_name !== model.name
    ? model.original_name
    : (model?.trained_as || []).find((x) => x && x !== model?.name);

  const submit = async () => {
    const next = name.trim();
    if (!next) return;
    setBusy(true);
    try {
      const r = await api.post("/library/model/rename", { path: model.path, name: next });
      toast(`Renamed to “${r.name}”`, "success");
      if (modelPath === model.path) setModelPath(r.path);
      await refreshStars();
      onRenamed?.(r);
      onClose();
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      title="Rename model"
      onClose={onClose}
      footer={<>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" onClick={submit} disabled={!name.trim() || busy}>
          {busy ? "Renaming…" : "Rename"}
        </button>
      </>}
    >
      <p className="hint mb-0">
        Changes the file name. Training identity stays on the model so you can still tell where it came from.
      </p>
      {(trained || model?.original_name) && (
        <p className="sub mt-0">
          Trained as <b>{trained || model.original_name}</b>
          {model?.step != null ? ` · step ${model.step}` : ""}
        </p>
      )}
      <Text
        label="New name"
        value={name}
        onChange={setName}
        tip="Letters, numbers, spaces, dashes, dots, and underscores."
      />
    </Modal>
  );
}
