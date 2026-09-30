"""Training routes: presets, architectures, start/monitor a run, checkpoints."""
import shutil
from pathlib import Path

from flask import Blueprint, request

from app.backend.data import projects
from app.core.config import workspace, get_device_info
from app.core.engine import lr_plan as lrplan
from app.core.engine.arch import available_architectures
from app.core.engine.trainer import (
    TrainConfig, start_training, list_checkpoints, load_run_view,
    estimate_peak_mib, recommended_batch, config_from_run, patch_run_meta, run_meta,
)
from app.core.model_manager import read_meta
from utils.api_responses import ok, err
from utils.validators import require, as_int, as_float, safe_name

bp = Blueprint("train", __name__, url_prefix="/api")

# Reliable starting points per target resolution (name -> overrides).
# Batch sizes are chosen to stay well under a typical 6-8 GB consumer GPU; going
# larger at 512px pushes a laptop card past its VRAM and into a ~10x slowdown.
# ``mults`` are channel multipliers on a base width of 64. Channels need room to
# double more than once for a model to carry much detail: 1,2,2,2 doubles exactly
# once and is a 3.8M-parameter net, which is a probe rather than a result. The
# widths here follow the vendored engine's own README command lines.
CONFIG_PRESETS = {
    "quick-256": {"image_size": 256, "batch_size": 8, "mults": [1, 2, 2, 2], "lr": 4e-4,
                  "train_steps": 120000, "save_every": 1000, "label": "Quick 256",
                  "blurb": "Fastest way to find out whether your dataset works at all."},
    "standard-512": {"image_size": 512, "batch_size": 4, "mults": [1, 2, 2, 4], "lr": 4e-4,
                     "train_steps": 280000, "save_every": 1000, "label": "Standard 512",
                     "blurb": "The default. Good detail without a long wait."},
    "detailed-512": {"image_size": 512, "batch_size": 2, "mults": [1, 2, 2, 4, 4], "lr": 3e-4,
                     "train_steps": 400000, "save_every": 1000, "label": "Detailed 512",
                     "blurb": "A deeper, wider network. Most detail, slowest, wants a bigger "
                              "GPU, and its image sizes go in steps of 32 rather than 16."},
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
    # Width changes what the run costs -- mostly through optimizer state rather
    # than activations -- so the estimate has to see it.
    mults = request.args.get("mults") or None
    mtype = request.args.get("mtype") or None
    attn = request.args.get("attn") or None
    # Mixed precision roughly halves activation memory, so the readout is wrong
    # by ~2x on the whole batch term if it does not know which way the form is set.
    amp = str(request.args.get("amp", "")).lower() in ("1", "true", "yes", "on")
    dev = get_device_info()
    gpus = dev.get("gpus") or []
    total = gpus[0]["total_mem_mb"] if gpus else 0
    free = gpus[0].get("free_mem_mb", 0) if gpus else 0
    try:
        est = estimate_peak_mib(image_size, batch_size, mults, mtype, attn, amp)
        rec = recommended_batch(image_size, total, mults, mtype, attn, amp) if total else None
    except Exception as e:  # noqa: BLE001
        return err(f"could not size that architecture: {e}", 400)
    return ok({
        "estimate_mib": est,
        "total_mib": total,
        "free_mib": free,
        "recommended_batch": rec,
        "risky": bool(total and est > 0.85 * total),
    })


@bp.get("/architectures")
def architectures():
    """The xurdif architectures this install can build, plus the attention
    layouts the configurable one offers by name, so the Train screen does not
    keep its own copy of either list."""
    from app.core.backends.xurdif import attn

    return ok({
        "architectures": available_architectures(),
        "conf_mtype": attn.MTYPE,
        "attn_kinds": list(attn.KINDS),
        "layouts": attn.NAMED_LAYOUTS,
        "default_attn": attn.DEFAULT_SPEC,
        # The trainer builds every xurdif net with dim=64; the Train screen's
        # sketch multiplies it out so channel widths read as real numbers.
        "base_dim": 64,
    })


@bp.get("/train/backends")
def train_backends():
    """What each engine can train, so the UI offers only real options."""
    from app.core import backends

    out = []
    for name in backends.available():
        b = backends.get(name)
        out.append({
            "name": name,
            "modes": list(b.training_modes),
            "capabilities": b.capabilities.to_dict(),
            "presets": b.training_presets(),
        })
    return ok({"backends": out, "default": "xurdif"})


def presets_view() -> dict:
    """The presets, each annotated with what it would cost on this machine.

    Computed here rather than in the browser so the pre-selection and the
    warning ``POST /train`` raises afterwards come from the same numbers -- they
    agree by construction instead of by two copies of the same constants.

    Fit uses 0.82, the budget ``recommended_batch`` works to, not the 0.85 alarm
    line ``_vram_warning`` fires on: pre-select conservatively, warn liberally.
    Anything else would highlight a preset the same server then calls risky.
    """
    total = _gpu_total_mib()
    out = {}
    for rank, (pid, p) in enumerate(CONFIG_PRESETS.items()):
        # No preset sets AMP, so these are costed the way a fresh form will run
        # them -- EMPTY_FORM.amp is true, matching TrainConfig. Costing them the
        # other way would badge a preset as too big when it fits.
        est = estimate_peak_mib(p["image_size"], p["batch_size"], p["mults"], amp=True)
        # ``order`` because Flask sorts JSON keys alphabetically, which would
        # hand the picker "Detailed, Quick, Standard" -- declaration order here
        # runs cheapest to richest and is what the list should show.
        out[pid] = {**p, "estimate_mib": est, "order": rank,
                    "fits": (est <= 0.82 * total) if total else None}
    out[_recommended(out, total)]["recommended"] = True
    return out


# The preset to land on when it fits. Not simply the richest that fits: batch
# size dominates the estimate, so the detailed preset at batch 2 can still land
# below standard at batch 4 and would win a pure "richest that fits" contest on
# the very laptop GPUs its own blurb warns away from. It is also much slower per
# step, which no VRAM number expresses.
PREFERRED_PRESET = "standard-512"


def _recommended(view: dict, total: int) -> str:
    """Which preset to pre-select for this machine."""
    cheapest = min(view, key=lambda k: view[k]["estimate_mib"])
    if not total:
        return cheapest          # no GPU to measure against; start small
    if view.get(PREFERRED_PRESET, {}).get("fits"):
        return PREFERRED_PRESET
    fitting = [k for k, v in view.items() if v["fits"]]
    return max(fitting, key=lambda k: view[k]["estimate_mib"]) if fitting else cheapest


@bp.get("/train/presets")
def presets():
    return ok(presets_view())


@bp.get("/train/lr_presets")
def lr_presets():
    """The learning-rate schedules, each compiled against this run's own numbers.

    Compiled here rather than in the browser for the same reason as the config
    presets: the step counts the UI shows and the ones the trainer will use are
    the same numbers by construction, not two copies of the arithmetic.
    """
    lr = as_float(request.args.get("lr", 4e-4), "lr", 1e-6, 1.0)
    train_steps = as_int(request.args.get("train_steps", 280000), "train_steps", 100, 5_000_000)
    save_every = as_int(request.args.get("save_every", 1000), "save_every", 10, 1_000_000)
    out = {}
    for rank, (pid, meta) in enumerate(lrplan.PRESETS.items()):
        plan = lrplan.compile_plan({"preset": pid}, lr=lr, train_steps=train_steps,
                                   save_every=save_every)
        # ``order`` for the same reason as the config presets: Flask sorts JSON
        # keys, and declaration order here runs from no decay to most.
        out[pid] = {"label": meta["label"], "blurb": meta["blurb"], "order": rank,
                    "plan": plan, "summary": lrplan.summarize(plan)}
    return ok(out)


def _running_job_for(run_dir):
    """The training job writing into this run folder, if one is."""
    from utils.process_control import registry

    for j in registry.list("train"):
        if j.get("status") != "running":
            continue
        out = (j.get("detail") or {}).get("out_dir") or ""
        try:
            if out and Path(out).resolve() == Path(run_dir).resolve():
                return j
        except OSError:
            continue
    return None


@bp.put("/runs/<run>/lr_plan")
def set_lr_plan(run):
    """Set or change a run's learning-rate schedule.

    One endpoint for a running run, a stopped one, and one whose Kiln has been
    restarted, because it writes a file rather than poking a job. No pause is
    needed: Adam reads param_groups[*]["lr"] fresh every step, so unlike the
    sampler's multistep solvers there is no state a mid-run change can corrupt.
    """
    run = safe_name(run, "run name")
    try:
        run_dir = projects.find_run(run)
    except Exception:
        return err("run not found", 404)

    body = request.get_json(force=True, silent=True) or {}
    meta = run_meta(run_dir)
    job = _running_job_for(run_dir)
    step = (job.get("detail") or {}).get("step") if job else None

    try:
        if body.get("from_now"):
            # The one-click drop: hold a new rate from wherever the run is now.
            if step is None:
                return err("that run is not training, so there is no step to change from", 409)
            rate = as_float(body.get("lr"), "lr", 1e-6, 1.0)
            plan = lrplan.override_from(lrplan.read(run_dir) or {}, int(step), rate)
        else:
            base = as_float(body.get("lr", meta.get("lr", 4e-4)), "lr", 1e-6, 1.0)
            plan = lrplan.compile_plan(
                body.get("lr_plan") or body,
                lr=base,
                train_steps=int(meta.get("train_steps") or 280000),
                save_every=int(meta.get("save_every") or 1000))
    except Exception as e:
        return err(str(e), 400)

    lrplan.write(run_dir, plan)
    summary = lrplan.summarize(plan)
    if step is not None:
        lrplan.append_event(run_dir, int(step), lrplan.lr_at(plan, int(step)), "live")
    patch_run_meta(run_dir, lr_schedule=plan.get("preset") or "custom",
                   lr_schedule_summary=summary)
    return ok({
        "plan": plan,
        "summary": summary,
        # Never a silent "I set it and nothing happened".
        "applies": "next step" if step is not None else "next continue",
        "at_step": step,
        "lr_now": lrplan.lr_at(plan, int(step)) if step is not None else None,
    })


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

    from app.core import backends

    backend_name = body.get("backend") or "xurdif"
    mode = (body.get("mode") or ("continue" if body.get("resume") else "scratch")).lower()
    try:
        backend = backends.get(backend_name)
    except Exception:
        return err(f"unknown backend: {backend_name}", 400)
    if not backend.supports_training_mode(mode):
        return err(
            f"the {backend.name} backend cannot do '{mode}' training "
            f"(it supports: {', '.join(backend.training_modes) or 'nothing'})", 400)
    needed = {"scratch": backend.capabilities.train_from_scratch,
              "continue": backend.capabilities.finetune,
              "finetune": backend.capabilities.finetune,
              "lora": backend.capabilities.lora}.get(mode, False)
    if not needed:
        return err(f"the {backend.name} backend does not support '{mode}'", 400)

    from utils.process_control import registry
    if any(j.get("status") == "running" for j in registry.list("train")):
        return err("a training job is already running — stop it before starting another", 409)
    if out_dir.exists() and any(out_dir.iterdir()):
        return err(f"a run named '{run_name}' already exists — pick a new name", 409)

    try:
        cfg = backend.training_config(body, ds_dir, out_dir)
        # A record dataset trains from a frozen copy of its file list and recipe,
        # written into the run so resuming later sees the same images.
        from app.backend.data import manifest

        snap = manifest.snapshot(ds_dir, out_dir)
        if snap is not None:
            cfg.manifest = str(snap)
    except Exception as e:
        return err(str(e), 400)

    job = backend.start_training(cfg)
    job.detail["backend"] = backend.name

    warning = _vram_warning(job, cfg, backend.name)
    return ok({"job": job.to_dict(), "warning": warning})


def _vram_warning(job, cfg, backend_name: str) -> str | None:
    """Flag a config likely to spill into system RAM.

    The estimate is fitted to the xurdif tinyunet, so it is only quoted for
    that backend -- a made-up number for a different architecture is worse than
    no number.
    """
    if backend_name != "xurdif":
        return None
    total = _gpu_total_mib()
    est = estimate_peak_mib(cfg.image_size, cfg.batch_size, cfg.mults, cfg.mtype, cfg.attn,
                            cfg.amp)
    job.detail["vram_estimate_mib"] = est
    if not (total and est > 0.85 * total):
        return None
    rec = recommended_batch(cfg.image_size, total, cfg.mults, cfg.mtype, cfg.attn, cfg.amp)
    warning = (
        f"Estimated peak VRAM ~{est} MiB may exceed your {total} MiB GPU — training "
        f"can spill to system RAM and run ~10x slower. Try batch size ≤ {rec} at "
        f"{cfg.image_size}px, or a smaller image size."
    )
    job.detail["warning"] = warning
    return warning


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
        # Continuing at a lower rate is the habit this whole feature grew out of,
        # so it is reachable here too and not only as a live edit. `lr` reseats the
        # plan's base; `lr_plan` or `preset` replaces the schedule outright.
        if body.get("lr") is not None:
            cfg.lr = as_float(body["lr"], "lr", 1e-6, 1.0)
        if body.get("lr_plan") or body.get("preset") or body.get("lr") is not None:
            cfg.lr_plan = lrplan.compile_plan(
                body.get("lr_plan") or {"preset": body.get("preset") or "constant"},
                lr=cfg.lr, train_steps=cfg.train_steps, save_every=cfg.save_every)
    except Exception as e:
        return err(str(e), 400)

    job = start_training(cfg)
    warning = _vram_warning(job, cfg, "xurdif")
    return ok({"job": job.to_dict(), "warning": warning})


@bp.delete("/runs/<run>")
def delete_run(run):
    from utils.process_control import registry
    from app.core import library

    run = safe_name(run, "run name")
    try:
        run_dir = projects.find_run(run)   # not resolved: a linked run loses only the link
    except Exception:
        return err("run not found", 404)
    for j in registry.list("train"):
        if j.get("status") != "running":
            continue
        out = (j.get("detail") or {}).get("out_dir") or ""
        if out and Path(out).resolve() == run_dir.resolve():
            return err("stop the running job before deleting this run", 409)
    library.remove_stars_under(run_dir)
    library.remove_hidden_under(run_dir)
    from utils.fs import safe_rmtree

    safe_rmtree(run_dir)
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
        attn=run_meta.get("attn"),
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
