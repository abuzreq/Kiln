import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../state.jsx";
import { FolderIcon } from "./icons.jsx";
import { Tooltip } from "./ui.jsx";

/**
 * Opens one workspace folder in the file manager, from the page it belongs to.
 *
 * The header's Workspace button lists every folder, but it is easy to miss,
 * and on the page about datasets the folder you want is the datasets one. The
 * tooltip gives the full path, so it can also be read off without opening.
 * The folder opens on the machine Kiln runs on, as the header's does.
 */
export default function OpenFolder({ folder = "root", label, className = "" }) {
  const { workspace, toast } = useApp();
  const [busy, setBusy] = useState(false);
  const path = workspace?.[folder];

  const open = async () => {
    setBusy(true);
    try {
      await api.post("/workspace/open", { key: folder });
    } catch (e) { toast(e.message, "error"); }
    setBusy(false);
  };

  return (
    <Tooltip text={path ? `Open ${path}` : "Open in your file manager"}>
      <button type="button" className={`btn ghost sm open-folder ${className}`.trim()} onClick={open} disabled={busy}>
        <FolderIcon size={14} />
        <span>{label}</span>
      </button>
    </Tooltip>
  );
}
