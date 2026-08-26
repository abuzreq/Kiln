"""Perform routes: model listing, guided sampling (as a streamed job), postproc."""
import io
import json
import threading

from flask import Blueprint, request, send_file

from app.core.engine.inpaint import fill_size
from app.core.engine.sampler import (
    DEFAULT_SAMPLER, RECOMMENDED_SAMPLER, SampleParams, postprocess_only, resolve_seed,
    sampler, sampler_catalog,
)
from app.core.model_manager import manager
from utils.api_responses import ok, err
from utils.imaging import (
    build_card, bytes_with_params, data_url, from_data_url, from_data_url_mask,
    preview_url, read_params, save_with_params,
)
from utils.process_control import registry
from utils.validators import require

bp = Blueprint("perform", __name__, url_prefix="/api")


@bp.get("/models")
def list_models():
    return ok(manager.scan_public())


@bp.get("/samplers")
def list_samplers():
    """Solvers the UI can offer, with their capabilities."""
    return ok({
        "samplers": sampler_catalog(),
        "default": DEFAULT_SAMPLER,
        "recommended": RECOMMENDED_SAMPLER,
    })


def _params_from_body(body: dict) -> SampleParams:
    pp = body.get("postproc") or {}
    batch_size = int(body.get("batch_size") or body.get("variations") or 1)
    batch_size = max(1, min(4, batch_size))
    return SampleParams(
        model_path=body["model_path"],
        ema=bool(body.get("ema", True)),
        image_size=int(body.get("image_size", 512)),
        steps=int(body.get("steps", 50)),
        train_steps=int(body.get("train_steps", 1000)),
        eta=float(body.get("eta", 0.5)),
        skip=int(body.get("skip", 0)),
        seed=resolve_seed(body.get("seed")),
        batch_size=batch_size,
        text=body.get("text", ""),
        text_weight=float(body.get("text_weight", 0.0)),
        guidance_step=float(body.get("guidance_step", 0.02)),
        guidance_power=float(body.get("guidance_power", 1.0)),
        spherical=bool(body.get("spherical", False)),
        image_prompt_weight=float(body.get("image_prompt_weight", 0.0)),
        cuts=float(body.get("cuts", 0.5)),
        noise_level=float(body.get("noise_level", 1.0)),
        simplify=float(body.get("simplify", 0.0)),
        postproc=pp,
        device=body.get("device", "auto"),
        sampler=body.get("sampler") or DEFAULT_SAMPLER,
        # Region fill only; 1 is the single-pass behaviour every existing
        # recipe was made with.
        resample=max(1, min(20, int(body.get("resample") or 1))),
        jump_length=max(0, min(100, int(body.get("jump_length") or 0))),
    )


def _build_bend_runtime(bends, meta, backend=None):
    """Optional: construct a bending runtime if a bend stack is supplied."""
    if not bends:
        return None
    if backend is not None and not backend.capabilities.bend:
        from utils.logger import get_logger

        get_logger("perform").info("bends ignored: %s cannot bend", backend.name)
        return None
    try:
        from app.core.craft.bending import build_runtime

        return build_runtime(bends, meta, backend=backend)
    except Exception as e:  # noqa: BLE001
        # bending is optional; never block sampling because of it
        from utils.logger import get_logger

        get_logger("perform").warning("bends ignored: %s", e)
        return None


def _publish_card(job, params, *, bends=None, bend_preset=None,
                  init_image=False, mask=False, kind="sample", extra=None):
    """Attach the resolved recipe to the job so the client can carry it with the
    image and hand it back on capture / export."""
    from app.core import library

    name = None
    try:
        name = (library.read_card(params.model_path) or {}).get("name")
    except Exception:  # noqa: BLE001
        pass
    card = build_card(
        params, model_name=name, bends=bends, bend_preset=bend_preset,
        postproc=params.postproc, init_image=init_image, mask=mask, kind=kind,
        extra=extra,
    )
    job.detail["seed"] = params.seed
    job.detail["card"] = card
    if params.batch_size > 1:
        seeds = [params.seed + i for i in range(params.batch_size)]
        job.detail["seeds"] = seeds
        job.detail["cards"] = [
            {**card, "params": {**card["params"], "seed": s, "batch_size": 1}}
            for s in seeds
        ]
    return card


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


# Previews are for watching progress, so there is no point rendering more of
# them than a person can see. Cap the rate rather than the step count so the
# cost stays flat however many steps the sampler takes.
PREVIEW_INTERVAL = 0.1  # seconds


def _throttled_preview(job, frame):
    """Rate-limited JPEG preview, shared by the batch and sequential paths."""
    import time

    now = time.time()
    if now - job.detail.get("_preview_at", 0.0) < PREVIEW_INTERVAL:
        return
    job.detail["_preview_at"] = now
    job.detail["frame"] = preview_url(frame["image_pp"])


def _apply_frame_to_job(job, frame, *, final=False):
    import time

    job.progress = frame["step"] / max(frame["total"], 1)
    job.detail["step"] = frame["step"]
    job.detail["total"] = frame["total"]

    last = job.detail.get("_preview_at", 0.0)
    now = time.time()
    is_last = frame["step"] >= frame["total"]
    if final or is_last or (now - last) >= PREVIEW_INTERVAL:
        job.detail["_preview_at"] = now
        if final:
            # the run is over — hand back the real thing
            job.detail["frame"] = data_url(frame["image_pp"])
            job.detail["frame_raw"] = data_url(frame["image"])
            if frame.get("images_pp"):
                job.detail["frames"] = [data_url(im) for im in frame["images_pp"]]
                job.detail["frames_raw"] = [data_url(im) for im in frame["images"]]
        else:
            # cheap JPEG thumbnail; frame_raw is only needed once, at the end
            job.detail["frame"] = preview_url(frame["image_pp"])

    if not job.paused():
        n = len(frame.get("images_pp") or [])
        suffix = f" · {n} vars" if n > 1 else ""
        job.message = f"step {frame['step']}/{frame['total']}{suffix}"


def _sample_worker(job, params, init_image, image_prompt, bend_runtime, mask=None,
                   feather=8.0, mults=None):
    from dataclasses import replace

    from app.core.engine.inpaint import run_inpaint

    def _frames(p):
        if mask is not None:
            return run_inpaint(
                p, init_image, mask,
                feather=feather,
                image_prompt=image_prompt,
                bend_runtime=bend_runtime,
                cancel=job.cancelled,
                control=job,
                mults=mults,
            )
        return sampler.run(
            p, init_image, image_prompt, bend_runtime,
            cancel=job.cancelled, control=job,
        )

    try:
        last_frame = None
        try:
            for frame in _frames(params):
                last_frame = frame
                _apply_frame_to_job(job, frame)
                if job.cancelled():
                    break
            if last_frame is not None:
                _apply_frame_to_job(job, last_frame, final=True)
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e) or params.batch_size <= 1:
                raise
            # VRAM too tight for a true batch — fall back to sequential seeds.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            n = params.batch_size
            base_seed = params.seed
            job.message = f"batch OOM — running {n} sequentially"
            job.detail["batch_fallback"] = "sequential"
            collected_pp = []
            collected_raw = []
            for i in range(n):
                if job.cancelled():
                    break
                seed_i = None if base_seed is None else int(base_seed) + i
                p = replace(params, batch_size=1, seed=seed_i)
                last = None
                for frame in _frames(p):
                    last = frame
                    job.progress = (i + frame["step"] / max(frame["total"], 1)) / n
                    job.detail["step"] = frame["step"]
                    job.detail["total"] = frame["total"]
                    _throttled_preview(job, frame)
                    if not job.paused():
                        job.message = f"var {i + 1}/{n} · step {frame['step']}/{frame['total']}"
                    if job.cancelled():
                        break
                if last is not None:
                    collected_pp.append(data_url(last["image_pp"]))
                    collected_raw.append(data_url(last["image"]))
                    job.detail["frames"] = list(collected_pp)
                    job.detail["frames_raw"] = list(collected_raw)

        if job.status == "running":
            job.status = "cancelled" if job.cancelled() else "done"
            job.progress = 1.0 if job.status == "done" else job.progress
            job.detail["paused"] = False
            job.message = "done" if job.status == "done" else "stopped"
    except Exception as e:  # noqa: BLE001
        job.status = "error"
        job.detail["paused"] = False
        job.message = str(e)


@bp.post("/perform/sample")
def sample():
    body = request.get_json(force=True, silent=True) or {}
    require(body, "model_path")
    params = _params_from_body(body)

    init_image = from_data_url(body["init_image"]) if body.get("init_image") else None
    image_prompt = from_data_url(body["image_prompt"]) if body.get("image_prompt") else None

    bundle = manager.load(params.model_path, ema=params.ema)
    meta = bundle["meta"]
    bend_runtime = _build_bend_runtime(body.get("bends"), meta, bundle["backend"])

    job = registry.create("sample")
    job.message = "sampling..."
    job.detail["total"] = 0
    _publish_card(
        job, params,
        bends=body.get("bends"), bend_preset=body.get("bend_preset"),
        init_image=init_image is not None, kind="sample",
    )

    t = threading.Thread(
        target=_sample_worker,
        args=(job, params, init_image, image_prompt, bend_runtime, None),
        daemon=True,
    )
    job.thread = t
    t.start()
    return ok({"job": job.to_dict()})


@bp.post("/perform/inpaint")
def inpaint():
    """Masked img2img: restyle the painted region of the canvas with a chosen model."""
    body = request.get_json(force=True, silent=True) or {}
    require(body, "model_path", "init_image", "mask")
    params = _params_from_body(body)
    init_image = from_data_url(body["init_image"])
    mask = from_data_url_mask(body["mask"])

    bundle = manager.load(params.model_path, ema=params.ema)
    meta = bundle["meta"]
    bend_runtime = _build_bend_runtime(body.get("bends"), meta, bundle["backend"])

    job = registry.create("inpaint")
    job.message = "filling..."
    job.detail["total"] = 0
    feather = float(body.get("feather", 8) or 0)
    # Region fill works at the canvas's own size, not params.image_size — surface
    # it so the UI can stop implying that Sample settings governs fills.
    fill_w, fill_h = fill_size(init_image, meta.mults)
    job.detail["fill_size"] = [fill_w, fill_h]
    job.detail["canvas_size"] = list(init_image.size)
    _publish_card(
        job, params,
        bends=body.get("bends"), bend_preset=body.get("bend_preset"),
        init_image=True, mask=True, kind="inpaint",
        extra={"fill_size": [fill_w, fill_h], "canvas_size": list(init_image.size)},
    )

    t = threading.Thread(
        target=_sample_worker,
        args=(job, params, init_image, None, bend_runtime, mask, feather, meta.mults),
        daemon=True,
    )
    job.thread = t
    t.start()
    return ok({"job": job.to_dict()})


@bp.post("/perform/postproc")
def postproc():
    body = request.get_json(force=True, silent=True) or {}
    (image,) = require(body, "image")
    img = from_data_url(image)
    out = postprocess_only(img, body.get("postproc") or {})
    return ok({"image": data_url(out)})


@bp.post("/perform/capture")
def capture():
    """Save an output image into the workspace captures folder.

    The generation recipe travels with the client (``job.detail.card``) and is
    written into the PNG itself, so the file stays reproducible on its own.
    """
    import time
    from app.core.config import workspace
    from utils.validators import safe_name

    body = request.get_json(force=True, silent=True) or {}
    (image,) = require(body, "image")
    img = from_data_url(image)
    name = safe_name(body.get("name") or f"capture_{int(time.time())}", "capture name")
    card = body.get("card") or read_params(img)
    out = workspace.captures / f"{name}.png"
    save_with_params(img, out, card)
    return ok({"path": str(out), "card": card})


@bp.post("/perform/export")
def export_image():
    """Stream an image back as a PNG download with its recipe embedded.

    The canvas lives in the browser as a data URL, which carries no metadata —
    so a plain <a download> would hand the user a file stripped of the very
    thing that makes it reproducible. Downloads go through here instead.
    """
    from utils.validators import safe_name

    body = request.get_json(force=True, silent=True) or {}
    (image,) = require(body, "image")
    img = from_data_url(image)
    card = body.get("card") or read_params(img)
    name = safe_name(body.get("filename") or "kiln", "filename")
    return send_file(
        io.BytesIO(bytes_with_params(img, card)),
        mimetype="image/png",
        as_attachment=True,
        download_name=f"{name}.png",
    )


@bp.post("/perform/read-params")
def read_image_params():
    """Read an embedded Kiln recipe out of an uploaded / dropped image."""
    body = request.get_json(force=True, silent=True) or {}
    (image,) = require(body, "image")
    return ok({"card": read_params(from_data_url(image))})


@bp.get("/captures")
def captures():
    from app.core.config import workspace

    out = []
    paths = sorted(workspace.captures.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths[:120]:
        entry = {"path": str(p), "name": p.stem, "mtime": p.stat().st_mtime}
        try:
            entry["card"] = read_params(p)
        except Exception:  # noqa: BLE001
            entry["card"] = None
        out.append(entry)
    # ``files`` kept for older clients that only wanted the paths
    return ok({"captures": out, "files": [e["path"] for e in out]})


@bp.get("/sweeps")
def sweeps():
    """Sweep grids and animations.

    Kept apart from captures: a capture is an image you chose to keep, a sweep is
    a study of how one parameter behaves. Mixing them buried the captures.
    """
    from app.core.config import workspace

    out = []
    files = [p for ext in ("*.png", "*.gif") for p in workspace.sweeps.glob(ext)]
    for p in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:120]:
        entry = {"path": str(p), "name": p.stem, "mtime": p.stat().st_mtime,
                 "animated": p.suffix.lower() == ".gif"}
        try:
            # GIFs carry no metadata chunk, so their recipe sits in a sibling .json
            if entry["animated"]:
                sidecar = p.with_suffix(".json")
                entry["card"] = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None
            else:
                entry["card"] = read_params(p)
        except Exception:  # noqa: BLE001
            entry["card"] = None
        out.append(entry)
    return ok({"sweeps": out})


@bp.delete("/sweeps")
def delete_sweep():
    from pathlib import Path

    from app.core.config import workspace

    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    p = Path(path)
    try:
        p.resolve().relative_to(workspace.sweeps.resolve())
    except (ValueError, OSError):
        return err("refusing to delete outside the sweeps folder", 400)
    if not p.exists():
        return err("not found", 404)
    p.unlink()
    p.with_suffix(".json").unlink(missing_ok=True)   # the GIF's recipe sidecar
    return ok({"deleted": str(p)})


@bp.delete("/captures")
def delete_capture():
    from pathlib import Path

    from app.core.config import workspace

    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    p = Path(path)
    try:
        p.resolve().relative_to(workspace.captures.resolve())
    except (ValueError, OSError):
        return err("refusing to delete outside the captures folder", 400)
    if not p.exists():
        return err("not found", 404)
    p.unlink()
    return ok({"deleted": str(p)})
