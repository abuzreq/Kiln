"""Training routes: presets, architectures, start/monitor a run, checkpoints."""
import shutil
from pathlib import Path

from flask import Blueprint, request

from app.backend.data import projects
from app.core.config import workspace, get_device_info
from app.core.engine.arch import available_architectures
from app.core.engine.trainer import (
    TrainConfig, start_training, list_checkpoints, load_run_view,
    estimate_peak_mib, recommended_batch, config_from_run,
)
from app.core.model_manager import read_meta
from utils.api_responses import ok, err
from utils.validators import require, as_int, as_float, safe_name

bp = Blueprint("train", __name__, url_prefix="/api")

# Reliable starting points per target resolution (name -> overrides).
# Batch sizes are chosen to stay well under a typical 6-8 GB consumer GPU; going
# larger at 512px pushes a laptop card past its VRAM and into a ~10x slowdown.
CONFIG_PRESETS = {
    "quick-256": {"image_size": 256, "batch_size": 8, "mults": [1, 2, 2, 2], "lr": 4e-4,
                  "train_steps": 120000, "save_every": 100, "label": "Quick 256 (fast iteration)"},
    "standard-512": {"image_size": 512, "batch_size": 4, "mults": [1, 2, 2, 2], "lr": 4e-4,
                     "train_steps": 280000, "save_every": 100, "label": "Standard 512 (recommended)"},
    "detailed-512": {"image_size": 512, "batch_size": 2, "mults": [1, 2, 2, 4], "lr": 3e-4,
                     "train_steps": 400000, "save_every": 100, "label": "Detailed 512 (larger model)"},
}


def _gpu_total_mib() -> int:
    dev = get_device_info()
    gpus = dev.get("gpus") or []
    return gpus[0]["total_mem_mb"] if gpus else 0


@bp.get("/train/estimate")
def estimate():
    """Estimate peak VRAM for a given image size + batch, and flag risky configs."""
    image_size = as_int(request.args.get("image_size", 512), "image_size", 32, 4096)
    batch_size = as_int(request.args.get("batch_size", 8), "batch_size", 1, 256)
    dev = get_device_info()
    gpus = dev.get("gpus") or []
    total = gpus[0]["total_mem_mb"] if gpus else 0
    free = gpus[0].get("free_mem_mb", 0) if gpus else 0
    est = estimate_peak_mib(image_size, batch_size)
    return ok({
        "estimate_mib": est,
        "total_mib": total,
        "free_mib": free,
        "recommended_batch": recommended_batch(image_size, total) if total else None,
        "risky": bool(total and est > 0.85 * total),
    })


@bp.get("/architectures")
def architectures():
    return ok({"architectures": available_architectures()})


@bp.get("/train/presets")
def presets():
    return ok(CONFIG_PRESETS)


@bp.get("/train/continue/info")
def continue_info():
    """Inspect a checkpoint or library model for continue / fine-tune."""
    from app.core.continue_train import inspect_source

    path = request.args.get("path")
    if not path:
        return err("path is required", 400)
    return ok(inspect_source(path))


@bp.get("/runs/available")
def run_available():
    try:
        name = safe_name(request.args.get("name", ""), "run name")
    except Exception as e:
        return err(str(e), 400)
    d = projects.get_store().dir / "runs" / name
    return ok({"name": name, "available": not (d.exists() and any(d.iterdir()))})


@bp.post("/train")
def start():
    p = projects.get_store()
    body = request.get_json(force=True, silent=True) or {}
    (dataset, run_name) = require(body, "dataset", "run_name")

    try:
        ds_dir = projects.find_dataset(dataset)
    except Exception:
        return err(f"dataset '{dataset}' not found", 404)

    from utils.validators import safe_name
    run_name = safe_name(run_name, "run name")
    p.ensure()
    out_dir = p.dir / "runs" / run_name

    resume = body.get("resume") or None
    if resume and not Path(resume).exists():
        return err("starting checkpoint not found", 404)

    from utils.process_control import registry
    if any(j.get("status") == "running" for j in registry.list("train")):
        return err("a training job is already running — stop it before starting another", 409)
    if out_dir.exists() and any(out_dir.iterdir()):
        return err(f"a run named '{run_name}' already exists — pick a new name", 409)

    cfg = TrainConfig(
        dataset=str(ds_dir),
        out_dir=str(out_dir),
        name=body.get("model_name", run_name),
        image_size=as_int(body.get("image_size", 512), "image_size", 32, 4096),
        batch_size=as_int(body.get("batch_size", 8), "batch_size", 1, 64),
        diffusion_steps=as_int(body.get("diffusion_steps", 1000), "diffusion_steps", 10, 4000),
        train_steps=as_int(body.get("train_steps", 280000), "train_steps", 100, 5_000_000),
        accum=as_int(body.get("accum", 10), "accum", 1, 128),
        lr=as_float(body.get("lr", 4e-4), "lr", 1e-6, 1.0),
        loss_type=body.get("loss_type", "l1"),
        l1w=as_float(body.get("l1w", 1.0), "l1w", 0, 100),
        ssimw=as_float(body.get("ssimw", 0.0), "ssimw", 0, 100),
        pred=body.get("pred", "x0"),
        mtype=body.get("mtype", "tinyunet_with_attention3"),
        mults=body.get("mults", [1, 2, 2, 2]),
        fit=body.get("fit", "resize"),
        nsamples=as_int(body.get("nsamples", 1), "nsamples", 1, 16),
        sample_seed=as_int(body.get("sample_seed", 42), "sample_seed", -1, 2**31 - 1),
        save_every=as_int(body.get("save_every", 1000), "save_every", 10, 100000),
        amp=bool(body.get("amp", False)),
        resume=resume,
        nostrict=bool(body.get("nostrict", False)),
    )
    job = start_training(cfg)

    total = _gpu_total_mib()
    est = estimate_peak_mib(cfg.image_size, cfg.batch_size)
    job.detail["vram_estimate_mib"] = est
    warning = None
    if total and est > 0.85 * total:
        rec = recommended_batch(cfg.image_size, total)
        warning = (
            f"Estimated peak VRAM ~{est} MiB may exceed your {total} MiB GPU — training "
            f"can spill to system RAM and run ~10x slower. Try batch size ≤ {rec} at "
            f"{cfg.image_size}px, or a smaller image size."
        )
        job.detail["warning"] = warning
    return ok({"job": job.to_dict(), "warning": warning})


@bp.post("/runs/<run>/continue")
def continue_run(run):
    run = safe_name(run, "run name")
    try:
        run_dir = projects.find_run(run)
    except Exception:
        return err("run not found", 404)

    from utils.process_control import registry
    if any(j.get("status") == "running" for j in registry.list("train")):
        return err("a training job is already running — stop it before continuing", 409)

    body = request.get_json(force=True, silent=True) or {}
    train_steps = as_int(body.get("train_steps"), "train_steps", 100, 5_000_000)
    checkpoint = body.get("filename") or body.get("checkpoint")

    try:
        cfg = config_from_run(run_dir, train_steps, checkpoint)
    except Exception as e:
        return err(str(e), 400)

    job = start_training(cfg)
    total = _gpu_total_mib()
    est = estimate_peak_mib(cfg.image_size, cfg.batch_size)
    job.detail["vram_estimate_mib"] = est
    warning = None
    if total and est > 0.85 * total:
        rec = recommended_batch(cfg.image_size, total)
        warning = (
            f"Estimated peak VRAM ~{est} MiB may exceed your {total} MiB GPU — training "
            f"can spill to system RAM and run ~10x slower. Try batch size ≤ {rec} at "
            f"{cfg.image_size}px, or a smaller image size."
        )
        job.detail["warning"] = warning
    return ok({"job": job.to_dict(), "warning": warning})


@bp.delete("/runs/<run>")
def delete_run(run):
    from utils.process_control import registry
    from app.core import library

    run = safe_name(run, "run name")
    try:
        run_dir = projects.find_run(run).resolve()
    except Exception:
        return err("run not found", 404)
    for j in registry.list("train"):
        if j.get("status") != "running":
            continue
        out = (j.get("detail") or {}).get("out_dir") or ""
        if out and Path(out).resolve() == run_dir:
            return err("stop the running job before deleting this run", 409)
    library.remove_stars_under(run_dir)
    shutil.rmtree(run_dir)
    return ok({"deleted": run})


@bp.get("/runs/<run>")
def get_run(run):
    run = safe_name(run, "run name")
    try:
        run_dir = projects.find_run(run)
    except Exception:
        return err("run not found", 404)
    return ok(load_run_view(run_dir))


@bp.get("/runs/<run>/checkpoints")
def run_checkpoints(run):
    run = safe_name(run, "run name")
    try:
        run_dir = projects.find_run(run)
    except Exception:
        return ok({"checkpoints": [], "run": run})
    return ok({"checkpoints": list_checkpoints(run_dir), "run": run})


@bp.post("/runs/<run>/checkpoints/save")
def save_checkpoint(run):
    run = safe_name(run, "run name")
    body = request.get_json(force=True, silent=True) or {}
    (filename, save_as) = require(body, "filename", "save_as")

    try:
        run_dir = projects.find_run(run).resolve()
    except Exception:
        return err("run not found", 404)
    src = (run_dir / Path(filename).name).resolve()
    if src.parent != run_dir or src.suffix != ".pt" or not src.exists():
        return err("checkpoint not found in run", 404)

    save_as = safe_name(save_as, "model name")
    workspace.models.mkdir(parents=True, exist_ok=True)
    dest = workspace.models / f"{save_as}.pt"
    if dest.exists() and not body.get("overwrite"):
        return err(f"a model named '{save_as}' already exists", 409)

    shutil.copy2(src, dest)
    from app.core import library
    thumb = library.sibling_thumb(src)
    if thumb is not None:
        try:
            shutil.copy2(thumb, dest.with_suffix(".png"))
        except Exception:  # noqa: BLE001
            pass

    step = None
    run_meta = {}
    try:
        from app.core.engine.trainer import _read_run_meta
        run_meta = _read_run_meta(run_dir)
        save_every = run_meta.get("save_every")
        if src.stem.startswith("model-") and save_every:
            step = int(src.stem.split("-", 1)[1]) * int(save_every)
    except Exception:  # noqa: BLE001
        step = None
    trained_as = f"{run}-step{step}" if step is not None else f"{run}-{src.stem}"
    library.ensure_card(
        dest,
        name=save_as,
        original_name=trained_as,
        trained_as=[src.stem, trained_as],
        origin_run=run,
        origin_file=src.name,
        step=step,
        image_size=run_meta.get("image_size"),
        mtype=run_meta.get("mtype"),
        mults=run_meta.get("mults"),
        pred=run_meta.get("pred"),
        train_steps=run_meta.get("train_steps"),
        save_every=run_meta.get("save_every"),
        lr=run_meta.get("lr"),
        batch_size=run_meta.get("batch_size"),
    )

    meta = read_meta(dest)
    meta.source = "workspace"
    meta.thumbnail = str(dest.with_suffix(".png")) if dest.with_suffix(".png").exists() else None
    d = meta.to_dict()
    d["name"] = save_as
    d["original_name"] = trained_as
    return ok({"model": d})
