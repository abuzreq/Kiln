"""Perform routes: model listing, guided sampling (as a streamed job), postproc."""
import io
import json
from dataclasses import replace
from functools import partial

from flask import Blueprint, Response, request, send_file, stream_with_context

from app.core.engine.inpaint import fill_size
from app.core.engine.lanes import enqueue, is_oom
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
from utils.validators import require, safe_name

bp = Blueprint("perform", __name__, url_prefix="/api")


@bp.get("/models")
def list_models():
    include_hidden = request.args.get("hidden") == "1"
    if request.args.get("stream") != "1":
        return ok(manager.scan_public(include_hidden=include_hidden))

    # One JSON model per line, written as each checkpoint is read, so the Start
    # hub can show the first cards while the rest are still being opened. The
    # last line is {"done": true}: a stream cut short without it is a failure,
    # not a short list.
    def lines():
        try:
            for d in manager.iter_public(include_hidden=include_hidden):
                yield json.dumps({"model": d}, default=str) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps({"error": str(e)}) + "\n"

    return Response(stream_with_context(lines()), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


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
        text_weight=float(body.get("text_weight", 1.0)),
        guidance_step=float(body.get("guidance_step", 0.02)),
        guidance_power=float(body.get("guidance_power", 1.0)),
        spherical=bool(body.get("spherical", False)),
        image_prompt_weight=float(body.get("image_prompt_weight", 0.0)),
        cuts=float(body.get("cuts", 0.5)),
        noise_level=float(body.get("noise_level", 1.0)),
        simplify=float(body.get("simplify", 0.0)),
        attenuation=float(body.get("attenuation", 1.0)),
        postproc=pp,
        device=body.get("device", "auto"),
        sampler=body.get("sampler") or DEFAULT_SAMPLER,
        # Region fill only; 1 is the single-pass behaviour every existing
        # recipe was made with.
        resample=max(1, min(20, int(body.get("resample") or 1))),
        jump_length=max(0, min(100, int(body.get("jump_length") or 0))),
    )


def _wants_live(body: dict) -> bool:
    """Whether the client wants in-progress previews (Create's Live preview switch).

    A viewing preference, not part of the recipe: it never reaches SampleParams
    or the card, so it cannot change what an image is or how it replays.
    """
    return body.get("live_preview", True) is not False


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


# Previews are for watching progress, so there is no point rendering more of
# them than anyone fetches. The rate is capped rather than the step count, so
# the cost stays flat however fast the steps are, and the cap is the browser's
# poll interval (pollJob, every 300 ms from Create): a preview published in
# between was overwritten before it was ever seen.
PREVIEW_INTERVAL = 0.3  # seconds


def _live_preview(job, frame):
    """The in-progress picture: raw, as a cheap JPEG.

    Raw on purpose. Finish (post-processing) is applied to the result only:
    running it on every preview was most of what a preview cost, 11-22% of a
    run with Finish on, for a picture that is replaced a moment later.
    """
    job.detail["frame"] = preview_url(frame["image"])


def _throttled_preview(job, frame, live=True):
    """Rate-limited preview, shared by the batch and sequential paths."""
    import time

    if not live:
        return
    now = time.time()
    if now - job.detail.get("_preview_at", 0.0) < PREVIEW_INTERVAL:
        return
    job.detail["_preview_at"] = now
    _live_preview(job, frame)


def _apply_frame_to_job(job, frame, *, final=False, live=True):
    import time

    job.progress = frame["step"] / max(frame["total"], 1)
    job.detail["step"] = frame["step"]
    job.detail["total"] = frame["total"]

    if final:
        # the run is over — hand back the real thing
        job.detail["frame"] = data_url(frame["image_pp"])
        job.detail["frame_raw"] = data_url(frame["image"])
        if frame.get("images_pp"):
            job.detail["frames"] = [data_url(im) for im in frame["images_pp"]]
            job.detail["frames_raw"] = [data_url(im) for im in frame["images"]]
    elif live and (time.time() - job.detail.get("_preview_at", 0.0)) >= PREVIEW_INTERVAL:
        job.detail["_preview_at"] = time.time()
        _live_preview(job, frame)

    if not job.paused():
        n = frame.get("batch", 1)
        suffix = f" · {n} vars" if n > 1 else ""
        job.message = f"step {frame['step']}/{frame['total']}{suffix}"


def _sample_worker(job, params, init_image, image_prompt, bends=None, mask=None,
                   feather=8.0, live=True):
    """Run one sample or fill in its lane.

    The model is described here, not in the route: on a cache miss that reads
    the whole checkpoint, which made the POST slow and turned a bad model into a
    failed request rather than a failed job.
    """
    from app.core.engine.inpaint import run_inpaint

    meta = bend_runtime = None

    def _frames(p):
        if mask is not None:
            return run_inpaint(
                p, init_image, mask,
                feather=feather,
                image_prompt=image_prompt,
                bend_runtime=bend_runtime,
                cancel=job.cancelled,
                control=job,
                mults=meta.mults,
            )
        return sampler.run(
            p, init_image, image_prompt, bend_runtime,
            cancel=job.cancelled, control=job,
        )

    try:
        meta, backend = manager.describe(params.model_path)
        bend_runtime = _build_bend_runtime(bends, meta, backend)
        if mask is not None:
            # Region fill works at the canvas's own size, not params.image_size.
            fill_w, fill_h = fill_size(init_image, meta.mults)
            job.detail["fill_size"] = [fill_w, fill_h]
            job.detail["card"] = {**job.detail["card"], "fill_size": [fill_w, fill_h]}
            if job.detail.get("cards"):
                job.detail["cards"] = [{**cd, "fill_size": [fill_w, fill_h]}
                                       for cd in job.detail["cards"]]

        last_frame = None
        try:
            for frame in _frames(params):
                last_frame = frame
                _apply_frame_to_job(job, frame, live=live)
                if job.cancelled():
                    break
            if last_frame is not None:
                _apply_frame_to_job(job, last_frame, final=True)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or params.batch_size <= 1:
                raise
            # VRAM too tight for a true batch — fall back to sequential seeds.
            from app.core import devices
            devices.empty_cache()
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
                    _throttled_preview(job, frame, live)
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
        if is_oom(e):
            raise           # the lane decides: retry alone, or fail
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

    job = registry.create("sample", status="queued")
    job.message = "queued"
    job.detail["total"] = 0
    _publish_card(
        job, params,
        bends=body.get("bends"), bend_preset=body.get("bend_preset"),
        init_image=init_image is not None, kind="sample",
    )
    enqueue(job, params.model_path, partial(
        _sample_worker, params=params, init_image=init_image, image_prompt=image_prompt,
        bends=body.get("bends"), live=_wants_live(body)))
    return ok({"job": job.to_dict()})


@bp.post("/perform/inpaint")
def inpaint():
    """Masked img2img: restyle the painted region of the canvas with a chosen model."""
    body = request.get_json(force=True, silent=True) or {}
    require(body, "model_path", "init_image", "mask")
    params = _params_from_body(body)
    init_image = from_data_url(body["init_image"])
    mask = from_data_url_mask(body["mask"])

    job = registry.create("inpaint", status="queued")
    job.message = "queued"
    job.detail["total"] = 0
    feather = float(body.get("feather", 8) or 0)
    # Region fill works at the canvas's own size, not params.image_size; the
    # worker adds detail.fill_size (and puts it on the card) once it knows the
    # model's downsampling. Surfaced so the UI can stop implying that Sample
    # settings govern fills.
    job.detail["canvas_size"] = list(init_image.size)
    _publish_card(
        job, params,
        bends=body.get("bends"), bend_preset=body.get("bend_preset"),
        init_image=True, mask=True, kind="inpaint",
        extra={"canvas_size": list(init_image.size)},
    )
    enqueue(job, params.model_path, partial(
        _sample_worker, params=params, init_image=init_image, image_prompt=None,
        bends=body.get("bends"), mask=mask, feather=feather, live=_wants_live(body)))
    return ok({"job": job.to_dict()})


# --- Randomize ---------------------------------------------------------
# One button that composes something over the canvas out of what the user
# already has: procedural shapes, the models in the library, the bend presets.
# How it decides is hidden; everything it decided is in the card, and one seed
# drives every choice, so a roll regenerates from its PNG. See
# docs/exploration-design.md, "Randomize".

def _change_params(change: float, steps: int) -> dict:
    """The Create panel's Change slider, mapped as changeToParams does in JS."""
    c = min(1.0, max(0.0, float(change)))
    return {"skip": int(round((1 - c) * max(steps - 4, 0) * 0.9)),
            "noise_level": 0.25 + c * 0.75}


def _randomize_models(rng, current: str) -> list[str]:
    """Up to two other models a roll may reach for, besides the current one."""
    try:
        others = [m["path"] for m in manager.scan_public()
                  if m.get("role") != "checkpoint" and m.get("path") != current]
    except Exception:  # noqa: BLE001
        others = []
    rng.shuffle(others)
    return others[:2]


def _randomize_presets(rng, model_path: str) -> list[dict]:
    """Bend stacks a roll may apply: starters, saved presets, and discoveries."""
    from app.core import library
    from app.core.craft import starters

    out = [{"name": e["name"], "bends": e["bends"]}
           for e in starters.entries() + library.list_entries("bends") if e.get("bends")]
    try:
        from app.core.craft import explore

        found = explore.list_discoveries(model_path=model_path, limit=100)["entries"]
        if found and rng.random() < 0.3:
            out = [{"name": f"discovery {e['id']}", "bends": e["bends"]} for e in found]
    except Exception:  # noqa: BLE001
        pass
    return out


def _plan_roll(body: dict, seed: int) -> dict:
    """Every choice a roll makes, from one seed."""
    import random

    from app.core.tools import masks

    rng = random.Random(seed)
    current = body["model_path"]
    steps = int(body.get("steps", 50))
    others = [] if rng.random() < 0.5 else _randomize_models(rng, current)
    n = rng.choice([2, 2, 3, 3, 4])
    regions = []
    for i in range(n):
        model = current if i == 0 or not others else rng.choice([current] + others)
        change = round(rng.uniform(0.45, 0.9), 2)
        preset = None
        if rng.random() < 0.4:
            options = _randomize_presets(rng, model)
            if options:
                preset = rng.choice(options)
        regions.append({
            "name": f"Area {i + 1}",
            "mask": masks.random_record(rng),
            "overlap": rng.random() < 0.3,
            "model_path": model,
            "bends": preset["bends"] if preset else None,
            "bend_preset": preset["name"] if preset else "",
            "change": change,
            **_change_params(change, steps),
            "feather": rng.randint(4, 16),
            "resample": rng.randint(1, 3),
            "seed": seed + i + 1,
        })
    return {"seed": seed, "regions": regions}


def _roll_summary(plan: dict, ground: str) -> str:
    from pathlib import Path

    parts = []
    for r in plan["regions"]:
        bits = [r["mask"]["kind"], Path(r["model_path"]).stem, f"{int(r['change'] * 100)} %"]
        if r["bend_preset"]:
            bits.append(r["bend_preset"])
        parts.append(" · ".join(bits))
    head = "fresh ground, then " if ground == "sampled" else ""
    return head + " → ".join(parts)


def _randomize_worker(job, body, plan, init_image):
    from pathlib import Path

    from PIL import Image, ImageChops

    from app.core import library
    from app.core.engine.inpaint import run_inpaint
    from app.core.tools import masks

    w, h = int(body["width"]), int(body["height"])
    shared = {k: body.get(k) for k in ("steps", "eta", "ema", "sampler", "train_steps", "device")
              if body.get(k) is not None}
    regions = plan["regions"]
    ground = "canvas" if init_image is not None else "sampled"
    n = len(regions) + (1 if ground == "sampled" else 0)
    state = {"done": 0}

    def _run_region(params, current, mask, feather, bends):
        meta, backend = manager.describe(params.model_path)
        runtime = _build_bend_runtime(bends, meta, backend)
        last = None
        for frame in run_inpaint(params, current, mask, feather=feather, bend_runtime=runtime,
                                 cancel=job.cancelled, control=job, mults=meta.mults):
            last = frame
            job.progress = (state["done"] + frame["step"] / max(frame["total"], 1)) / n
            job.detail["step"] = frame["step"]
            job.detail["total"] = frame["total"]
            _throttled_preview(job, frame, _wants_live(body))
            if job.cancelled():
                break
        return last

    try:
        current = init_image
        if current is None:
            # A blank canvas gets a ground first: a plain sample with the
            # current model, through the same path as a full-change fill.
            job.message = "ground"
            job.detail["region_name"] = "ground"
            params = _params_from_body({**shared, "model_path": body["model_path"],
                                        "seed": plan["seed"], "skip": 0, "noise_level": 1.0,
                                        "batch_size": 1, "resample": 1})
            blank = Image.new("RGB", (w, h), (128, 128, 128))
            last = _run_region(params, blank, Image.new("L", (w, h), 255), 0.0, None)
            if last is None or job.cancelled():
                raise RuntimeError("stopped")
            current = last["image_pp"]
            state["done"] += 1

        taken = Image.new("L", (w, h), 0)
        union = Image.new("L", (w, h), 0)
        for i, r in enumerate(regions):
            if job.cancelled():
                break
            job.detail["region"] = i
            job.detail["region_name"] = r["name"]
            job.message = f"area {i + 1} of {len(regions)}"
            mask = masks.from_record(r["mask"], w, h)
            if not r["overlap"] and i > 0:
                mask = masks.carve(mask, taken)
            if masks.coverage_of(mask) < 0.01:
                # carved away to nothing: let it overlap after all
                mask = masks.from_record(r["mask"], w, h)
            params = _params_from_body({**shared, "model_path": r["model_path"], "seed": r["seed"],
                                        "skip": r["skip"], "noise_level": r["noise_level"],
                                        "resample": r["resample"], "batch_size": 1})
            last = _run_region(params, current, mask, float(r["feather"]), r["bends"])
            if last is None:
                break
            current = last["image_pp"]
            taken = ImageChops.lighter(taken, mask)
            union = ImageChops.lighter(union, mask)
            r["model"] = ((library.read_card(r["model_path"]) or {}).get("name")
                          or Path(r["model_path"]).stem)
            state["done"] += 1

        job.detail["frame"] = data_url(current)
        job.detail["frame_raw"] = data_url(current)
        job.detail["mask_union"] = data_url(union)
        job.detail["ground"] = ground
        job.detail["summary"] = _roll_summary(plan, ground)
        job.detail["card"] = {**job.detail["card"], "regions": plan["regions"],
                              "ground": ground, "summary": job.detail["summary"]}
        if job.status == "running":
            job.status = "cancelled" if job.cancelled() else "done"
            job.progress = 1.0 if job.status == "done" else job.progress
            job.message = "done" if job.status == "done" else "stopped"
    except Exception as e:  # noqa: BLE001
        if is_oom(e):
            raise           # the lane decides: retry alone, or fail
        if str(e) == "stopped":
            job.status = "cancelled"
            job.message = "stopped"
        else:
            job.status = "error"
            job.message = str(e)


@bp.post("/perform/randomize")
def randomize():
    """Compose something over the canvas from shapes, models and presets."""
    body = request.get_json(force=True, silent=True) or {}
    require(body, "model_path", "width", "height")
    seed = resolve_seed(body.get("seed"))
    size = (int(body["width"]), int(body["height"]))
    init_image = from_data_url(body["init_image"]).convert("RGB") if body.get("init_image") else None
    if init_image is not None and init_image.size != size:
        init_image = init_image.resize(size)
    plan = _plan_roll(body, seed)

    job = registry.create("randomize", status="queued")
    job.message = "queued"
    job.detail["total"] = 0
    job.detail["regions"] = len(plan["regions"])
    params = _params_from_body({**body, "seed": seed, "batch_size": 1})
    _publish_card(job, params, init_image=init_image is not None, mask=True, kind="randomize",
                  extra={"roll_seed": seed, "canvas_size": list(size)})

    # A roll reaches for up to two other models; it holds every one it uses.
    models = {body["model_path"], *(r["model_path"] for r in plan["regions"])}
    enqueue(job, models, partial(_randomize_worker, body=body, plan=plan, init_image=init_image))
    return ok({"job": job.to_dict()})


@bp.post("/perform/postproc")
def postproc():
    body = request.get_json(force=True, silent=True) or {}
    (image,) = require(body, "image")
    img = from_data_url(image)
    seed = body.get("seed")
    out = postprocess_only(img, body.get("postproc") or {},
                           seed=None if seed in (None, "") else int(seed))
    return ok({"image": data_url(out)})


@bp.post("/perform/capture")
def capture():
    """Save an output image into the workspace captures folder.

    The generation recipe travels with the client (``job.detail.card``) and is
    written into the PNG itself, so the file stays reproducible on its own.
    ``folder`` puts it one level down: Create saves a queue's results there.
    """
    import time
    from app.core.config import workspace

    body = request.get_json(force=True, silent=True) or {}
    (image,) = require(body, "image")
    img = from_data_url(image)
    name = safe_name(body.get("name") or f"capture_{int(time.time())}", "capture name")
    card = body.get("card") or read_params(img)
    folder = workspace.captures
    if body.get("folder"):
        folder = folder / safe_name(body["folder"], "folder")
        folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.png"
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
    # One level of folders: each queue's results are saved into their own.
    root = workspace.captures
    files = [*root.glob("*.png"), *root.glob("*/*.png")]
    paths = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths[:120]:
        entry = {"path": str(p), "name": p.stem, "mtime": p.stat().st_mtime,
                 "folder": "" if p.parent == root else p.parent.name}
        try:
            entry["card"] = read_params(p)
        except Exception:  # noqa: BLE001
            entry["card"] = None
        out.append(entry)
    # ``files`` kept for older clients that only wanted the paths
    return ok({"captures": out, "files": [e["path"] for e in out]})


ASSET_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _asset_entry(p):
    from PIL import Image

    entry = {"path": str(p), "name": p.stem, "mtime": p.stat().st_mtime}
    try:
        with Image.open(p) as im:
            entry["size"] = list(im.size)
    except Exception:  # noqa: BLE001
        entry["size"] = None
    return entry


@bp.get("/assets")
def assets():
    """Images the user has brought in, newest first. See Workspace.assets."""
    from app.core.config import workspace

    paths = [p for p in workspace.assets.iterdir() if p.suffix.lower() in ASSET_EXTS]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ok({"assets": [_asset_entry(p) for p in paths[:200]]})


@bp.post("/assets")
def add_asset():
    """Keep an image as an asset. Saved as PNG so any embedded recipe survives."""
    import re
    import time
    from app.core.config import workspace

    body = request.get_json(force=True, silent=True) or {}
    (image,) = require(body, "image")
    img = from_data_url(image)
    # The name comes from a filename the user did not choose for us, so it is
    # cleaned rather than refused: "Screenshot (3).png" is a perfectly good
    # asset, and rejecting it made the panel look like it had ignored the drop.
    stem = re.sub(r"[^\w \-.]+", "-", str(body.get("name") or "")).strip(" -.")[:96]
    stem = stem or f"asset_{int(time.time())}"
    out = workspace.assets / f"{stem}.png"
    # Never overwrite: two drops of files with the same name are two assets.
    n = 2
    while out.exists():
        out = workspace.assets / f"{stem} {n}.png"
        n += 1
    save_with_params(img, out, body.get("card") or read_params(img))
    return ok({"asset": _asset_entry(out)})


@bp.delete("/assets")
def delete_asset():
    from pathlib import Path

    from app.core.config import workspace

    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    p = Path(path)
    try:
        p.resolve().relative_to(workspace.assets.resolve())
    except (ValueError, OSError):
        return err("refusing to delete outside the assets folder", 400)
    if not p.exists():
        return err("not found", 404)
    p.unlink()
    return ok({"deleted": str(p)})


@bp.get("/sweeps")
def sweeps():
    """Sweep grids and animations.

    Kept apart from captures: a capture is an image you chose to keep, a sweep is
    a study of how one parameter behaves. Mixing them buried the captures.
    """
    from app.core.config import workspace

    out = []
    files = [p for ext in ("*.png", "*.gif", "*.mp4", "*.webm") for p in workspace.sweeps.glob(ext)]
    for p in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:120]:
        kind = p.suffix.lower()
        entry = {"path": str(p), "name": p.stem, "mtime": p.stat().st_mtime,
                 "animated": kind == ".gif", "video": kind in (".mp4", ".webm"),
                 "format": kind.lstrip(".")}
        try:
            # GIFs and videos carry no Kiln card, so their recipe sits in a sibling .json
            if entry["animated"] or entry["video"]:
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
    if p.suffix.lower() != ".png":
        p.with_suffix(".json").unlink(missing_ok=True)   # a GIF's or video's recipe sidecar
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
