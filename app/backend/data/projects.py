"""Workspace store: source images, datasets, and training runs at the workspace root.

Legacy folders under ``<workspace>/projects/<name>/`` are still listed so existing
data remains visible; new imports, datasets, and runs are written at the root.
"""
import json
from pathlib import Path

from app.core.config import workspace
from utils.exceptions import NotFoundError
from utils.process_control import registry

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class Store:
    """Duck-compatible with the old Project object (``.dir`` / listings)."""

    def __init__(self):
        self.name = "workspace"
        self.dir = workspace.root

    def ensure(self):
        for sub in ("source", "datasets", "runs", "models", "captures"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

    def count_source(self) -> int:
        n = 0
        for root in _source_roots():
            n += sum(1 for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        return n

    def list_datasets(self) -> list[dict]:
        out, seen = [], set()
        for d in _dataset_dirs():
            if d.name in seen:
                continue
            seen.add(d.name)
            info = {"name": d.name, "path": str(d), "count": 0}
            meta = d / "dataset.json"
            if meta.exists():
                try:
                    info.update(json.loads(meta.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    pass
            info["name"] = d.name
            info["path"] = str(d)
            info["count"] = sum(1 for p in d.glob("*") if p.suffix.lower() in IMAGE_EXTS)
            out.append(info)
        return out

    def list_runs(self) -> list[dict]:
        out, seen = [], set()
        for r in _run_dirs():
            if r.name in seen:
                continue
            seen.add(r.name)
            ckpts = sorted(r.glob("model-*.pt"))
            samples = sorted(r.glob("sample-*.png"))
            run_meta = {}
            rj = r / "run.json"
            if rj.exists():
                try:
                    run_meta = json.loads(rj.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    run_meta = {}
            out.append({
                "name": r.name,
                "path": str(r),
                "checkpoints": len(ckpts),
                "latest_sample": str(samples[-1]) if samples else None,
                "status": run_meta.get("status"),
                "message": run_meta.get("message"),
                "step": run_meta.get("step"),
            })
        return out

    def to_dict(self, counts: bool = False) -> dict:
        """Workspace summary.

        ``counts`` walks the whole source tree to total the images, which is far
        too expensive to do on a poll — it stays opt-in.
        """
        self.ensure()
        out = {
            "path": str(self.dir),
            "datasets": self.list_datasets(),
            "runs": self.list_runs(),
            "train": train_status_for(self),
        }
        if counts:
            out["source_count"] = self.count_source()
        return out


store = Store()


def _legacy_project_dirs() -> list[Path]:
    root = workspace.projects
    if not root.exists():
        return []
    return [d for d in sorted(root.iterdir()) if d.is_dir()]


def _source_roots() -> list[Path]:
    roots = [workspace.root / "source"]
    for d in _legacy_project_dirs():
        s = d / "source"
        if s.exists():
            roots.append(s)
    return roots


def _dataset_dirs() -> list[Path]:
    out = []
    droot = workspace.root / "datasets"
    if droot.exists():
        out.extend(sorted(p for p in droot.iterdir() if p.is_dir()))
    for d in _legacy_project_dirs():
        nested = d / "datasets"
        if nested.exists():
            out.extend(sorted(p for p in nested.iterdir() if p.is_dir()))
    return out


def _run_dirs() -> list[Path]:
    out = []
    rroot = workspace.root / "runs"
    if rroot.exists():
        out.extend(sorted(p for p in rroot.iterdir() if p.is_dir()))
    for d in _legacy_project_dirs():
        nested = d / "runs"
        if nested.exists():
            out.extend(sorted(p for p in nested.iterdir() if p.is_dir()))
    return out


def find_dataset(name: str) -> Path:
    for d in _dataset_dirs():
        if d.name == name:
            return d
    raise NotFoundError(f"dataset '{name}' not found")


def find_run(name: str) -> Path:
    for d in _run_dirs():
        if d.name == name:
            return d
    raise NotFoundError(f"run '{name}' not found")


def train_status_live() -> dict | None:
    """Running-training status only — O(1), no filesystem walk.

    The topbar badge polls every few seconds; resolving the full workspace for
    that meant thousands of stat calls a minute on a large dataset.
    """
    for j in registry.list("train"):
        if j.get("status") != "running":
            continue
        detail = j.get("detail") or {}
        out = detail.get("out_dir") or ""
        return {
            "status": "training",
            "label": "Training",
            "job_id": j.get("id"),
            "run": Path(out).name if out else None,
            "message": j.get("message") or "",
        }
    return None


def train_status_for(p: Store | None = None) -> dict:
    p = p or store
    for j in registry.list("train"):
        if j.get("status") != "running":
            continue
        detail = j.get("detail") or {}
        out = detail.get("out_dir") or ""
        return {
            "status": "training",
            "label": "Training",
            "job_id": j.get("id"),
            "run": Path(out).name if out else None,
            "message": j.get("message") or "",
        }

    runs = p.list_runs()
    if not runs:
        return {"status": "not_started", "label": "Not started"}

    with_ckpts = [r for r in runs if r.get("checkpoints")]
    latest = with_ckpts[-1] if with_ckpts else runs[-1]
    raw = latest.get("status")
    if raw == "training":
        ended = "interrupted"
    elif raw == "done":
        ended = "completed"
    elif raw == "cancelled":
        ended = "cancelled"
    elif raw == "error":
        ended = "error"
    elif latest.get("checkpoints") or raw:
        ended = "unknown"
    else:
        return {"status": "not_started", "label": "Not started"}

    return {
        "status": "stopped",
        "label": "Stopped",
        "ended": ended,
        "run": latest.get("name"),
        "message": latest.get("message") or "",
        "step": latest.get("step"),
    }


def get_store() -> Store:
    store.ensure()
    return store
