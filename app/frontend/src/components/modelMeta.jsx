import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../state.jsx";
import { Text, Modal } from "./ui.jsx";

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
