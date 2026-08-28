"""Continue training from a checkpoint or library model as a new project.

Copies the chosen ``.pt`` into the new project's ``models/`` folder and optionally
copies source images + datasets from the origin project so the new run can start
with ``--load``.
"""
import re
import shutil
from pathlib import Path

from app.backend.data import projects
from app.core import library
from app.core.config import workspace
from app.core.engine.trainer import _read_run_meta
from app.core.model_manager import read_meta
from utils.exceptions import NotFoundError, ValidationError
from utils.validators import safe_name

DEFAULT_EXTRA_STEPS = 50000


def _infer_origin(path: Path, card: dict) -> dict:
    out = {
        "project": card.get("origin_project"),
        "run": card.get("origin_run"),
        "kind": "library",
    }
    p = path.resolve()
    try:
        rel = p.relative_to(workspace.root.resolve())
    except (ValueError, OSError):
        return out
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "runs":
        return {"project": None, "run": parts[1], "kind": "checkpoint"}
    if len(parts) >= 4 and parts[0] == "projects" and parts[2] == "runs":
        return {"project": parts[1], "run": parts[3], "kind": "checkpoint"}
    if len(parts) >= 3 and parts[0] == "projects" and parts[2] == "models":
        out["kind"] = "project_model"
    return out


def _run_meta(origin: dict) -> dict:
    run = origin.get("run")
    if not run:
        return {}
    try:
        return _read_run_meta(projects.find_run(run))
    except (NotFoundError, ValidationError, FileNotFoundError):
        return {}


def _origin_data(origin_project: str | None) -> dict:
    p = projects.get_store()
    datasets = p.list_datasets()
    return {
        "has_source": p.count_source() > 0,
        "has_dataset": any(d.get("count") for d in datasets),
        "datasets": [d["name"] for d in datasets],
    }


def _step_from_filename(path: Path, save_every) -> int | None:
    if not path.stem.startswith("model-") or not save_every:
        return None
    try:
        return int(path.stem.split("-", 1)[1]) * int(save_every)
    except (ValueError, IndexError, TypeError):
        return None


def inspect_source(path: str | Path) -> dict:
    """Describe a checkpoint/library model for the continue-training modal."""
    path = Path(path)
    if not path.exists() or path.suffix != ".pt":
        raise NotFoundError("model not found")

    card = library.read_card(path)
    origin = _infer_origin(path, card)
    run_meta = _run_meta(origin)
    data = _origin_data(origin.get("project"))

    step = card.get("step")
    if step is None:
        step = _step_from_filename(path, run_meta.get("save_every"))

    mtype = run_meta.get("mtype") or card.get("mtype")
    mults = run_meta.get("mults") or card.get("mults")
    pred = run_meta.get("pred") or card.get("pred")
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)

    if step is None or not mtype or not mults or not pred:
        meta = read_meta(path)
        step = step if step is not None else meta.step
        mtype = mtype or meta.mtype
        mults = mults or meta.mults
        pred = pred or meta.pred

    name = card.get("name") or path.stem
    original = card.get("original_name") or path.stem
    trained_as = card.get("trained_as") or ([original] if original != name else [path.stem])

    config = {}
    for k in ("image_size", "batch_size", "train_steps", "save_every", "lr", "accum",
              "loss_type", "fit", "mtype", "mults", "pred", "edge_loss"):
        if run_meta.get(k) is not None:
            config[k] = run_meta[k]
        elif card.get(k) is not None:
            config[k] = card[k]
    config.setdefault("mtype", mtype)
    config.setdefault("mults", mults)
    config.setdefault("pred", pred)

    current = int(step or 0)
    origin_target = int(config.get("train_steps") or 0)
    suggested_steps = origin_target if origin_target > current else current + DEFAULT_EXTRA_STEPS

    label = name if name == original else f"{name} (trained as {original})"
    base = origin.get("project") or re.sub(r"-step\d+$", "", str(name or "model"), flags=re.I)
    raw = f"{base}-cont"
    suggested_project = safe_name(raw, "project name") if _looks_safe(raw) else "continued"

    return {
        "path": str(path),
        "name": name,
        "original_name": original,
        "trained_as": trained_as,
        "step": step,
        "size_mb": size_mb,
        "mtype": mtype,
        "mults": mults,
        "pred": pred,
        "label": label,
        "origin_project": origin.get("project"),
        "origin_run": origin.get("run"),
        "kind": origin.get("kind"),
        "has_source": data["has_source"],
        "has_dataset": data["has_dataset"],
        "datasets": data["datasets"],
        "can_copy_data": data["has_source"] or data["has_dataset"],
        "config": config,
        "suggested_steps": suggested_steps,
        "suggested_project": suggested_project,
    }


def _looks_safe(name: str) -> bool:
    try:
        safe_name(name, "project name")
        return True
    except ValidationError:
        return False


def _copy_data(src: projects.Store, dest: projects.Store):
    for sub in ("source", "datasets"):
        s, d = src.dir / sub, dest.dir / sub
        if s.exists():
            shutil.copytree(s, d, dirs_exist_ok=True)


def continue_from_model(
    path: str | Path,
    project_name: str,
    copy_data: bool = True,
    train_steps: int | None = None,
    nostrict: bool = False,
) -> dict:
    """Create a new project seeded with the given checkpoint and return its dict."""
    info = inspect_source(path)
    path = Path(info["path"])
    p = projects.create_project(
        project_name,
        description=f"Continued from {info['label']}"
        + (f" at step {info['step']}" if info.get("step") is not None else ""),
    )

    copied = False
    if copy_data and info.get("origin_project"):
        try:
            src = projects.get_project(info["origin_project"])
            _copy_data(src, p)
            copied = True
        except (NotFoundError, ValidationError):
            copied = False

    try:
        stem = safe_name(info["name"] or info["original_name"] or "seed", "model name")
    except ValidationError:
        stem = "seed"
    dest = p.dir / "models" / f"{stem}.pt"
    if dest.exists():
        dest = p.dir / "models" / "seed.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    thumb = library.sibling_thumb(path)
    if thumb is not None:
        shutil.copy2(thumb, dest.with_suffix(".png"))

    cfg = dict(info.get("config") or {})
    if train_steps:
        cfg["train_steps"] = int(train_steps)

    library.ensure_card(
        dest,
        name=info["name"],
        original_name=info["original_name"],
        trained_as=info.get("trained_as") or [info["original_name"]],
        origin_project=info.get("origin_project"),
        origin_run=info.get("origin_run"),
        step=info.get("step"),
        mtype=info.get("mtype"),
        mults=info.get("mults"),
        pred=info.get("pred"),
        image_size=cfg.get("image_size"),
        kind="continue",
    )

    continued = {
        "source_path": str(path),
        "source_name": info["name"],
        "original_name": info["original_name"],
        "trained_as": info.get("trained_as") or [],
        "step": info.get("step"),
        "resume": str(dest),
        "origin_project": info.get("origin_project"),
        "origin_run": info.get("origin_run"),
        "copied_data": copied,
        "nostrict": bool(nostrict),
        **cfg,
        "mtype": info.get("mtype"),
        "mults": info.get("mults"),
        "pred": info.get("pred"),
    }
    meta = p.load_meta()
    meta["continued_from"] = continued
    p.save_meta(meta)
    return p.to_dict()
