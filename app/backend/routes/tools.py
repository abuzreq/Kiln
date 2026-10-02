"""Tools routes: super-resolution, sweep runner / contact sheets, GIF and video export."""
import base64
import io
import time

from flask import Blueprint, request, send_file

from app.core.config import workspace
from app.core.engine.lanes import enqueue, is_oom
from app.core.engine.sampler import resolve_seed
from app.core.tools import sweep as sweep_mod
from app.core.tools.superres import upscale
from utils.api_responses import ok, err
from utils.imaging import build_card, bytes_with_params, data_url, from_data_url, save_with_params
from utils.process_control import registry
from utils.exceptions import ValidationError
from utils.validators import require, as_int, safe_name

bp = Blueprint("tools", __name__, url_prefix="/api/tools")


@bp.post("/superres")
def superres():
    body = request.get_json(force=True, silent=True) or {}
    (image,) = require(body, "image")
    img = from_data_url(image)
    out = upscale(img, factor=as_int(body.get("factor", 2), "factor", 2, 4), sharpen=float(body.get("sharpen", 0.0)))
    return ok({"image": data_url(out), "size": out.size})


@bp.post("/gif")
def make_gif():
    """Assemble frames into an animated GIF.

    GIF has no metadata chunk to carry a Kiln card, so the recipe is written to
    a sibling .json instead — the animation stays traceable to its run.
    """
    import json
    import time

    from PIL import Image

    body = request.get_json(force=True, silent=True) or {}
    (images,) = require(body, "images")
    if len(images) < 2:
        return err("a GIF needs at least two frames", 400)
    fps = max(1.0, min(30.0, float(body.get("fps", 8))))
    frames = [from_data_url(im) for im in images]
    if body.get("pingpong"):
        # bounce back so a parameter sweep loops smoothly instead of snapping
        frames = frames + frames[-2:0:-1]

    size = frames[0].size
    frames = [f if f.size == size else f.resize(size, Image.LANCZOS) for f in frames]
    # a shared adaptive palette keeps colours stable across frames
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]

    name = safe_name(body.get("name") or f"kiln_{int(time.time())}", "gif name")
    # A sweep animation is a study, not a kept image: it files with the sweeps.
    dest = workspace.sweeps if body.get("kind") == "sweep" else workspace.captures
    out = dest / f"{name}.gif"
    pal[0].save(
        out, save_all=True, append_images=pal[1:],
        duration=int(1000 / fps), loop=0, optimize=True, disposal=2,
    )
    if body.get("card"):
        out.with_suffix(".json").write_text(
            json.dumps(body["card"], indent=2), encoding="utf-8"
        )
    buf = io.BytesIO(out.read_bytes())
    return ok({
        "path": str(out),
        "frames": len(pal),
        "gif": "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
    })


@bp.post("/video")
def make_video():
    """Assemble frames into a video: H.264 MP4, or WebM where H.264 is missing.

    Same inputs as /gif. A video holds full colour where a GIF has 256, and is
    what most places a sweep gets posted to want. Like the GIF, its recipe goes
    in a sibling .json. The file is served by path rather than inlined: a few
    seconds of 512px video is no size for a data URL.
    """
    import json

    from app.core.tools.video import write_video

    body = request.get_json(force=True, silent=True) or {}
    (images,) = require(body, "images")
    if len(images) < 2:
        return err("a video needs at least two frames", 400)
    fps = max(1.0, min(30.0, float(body.get("fps", 8))))
    frames = [from_data_url(im) for im in images]
    if body.get("pingpong"):
        frames = frames + frames[-2:0:-1]
    # A sweep of six frames at 8 fps is under a second; a player that does not
    # loop would show it once and stop, so the whole run can be repeated.
    frames = frames * as_int(body.get("repeat", 1), "repeat", 1, 20)

    name = safe_name(body.get("name") or f"kiln_{int(time.time())}", "video name")
    dest = workspace.sweeps if body.get("kind") == "sweep" else workspace.captures
    try:
        out, codec = write_video(frames, dest / name, fps)
    except RuntimeError as e:
        return err(str(e), 500)
    if body.get("card"):
        out.with_suffix(".json").write_text(json.dumps(body["card"], indent=2), encoding="utf-8")
    return ok({
        "path": str(out),
        "frames": len(frames),
        "codec": codec,
        "format": out.suffix.lstrip("."),
        "seconds": round(len(frames) / fps, 2),
    })


@bp.post("/zip")
def make_zip():
    """Bundle a set of frames into one .zip download.

    Every frame goes in as a PNG with its recipe embedded, not as a bare image
    beside a manifest -- a sidecar stops travelling with the picture the moment
    someone moves or re-shares it.
    """
    import re
    import zipfile

    body = request.get_json(force=True, silent=True) or {}
    (images,) = require(body, "images")
    if not images:
        return err("nothing to download", 400)
    name = safe_name(body.get("name") or f"kiln_{int(time.time())}", "zip name")
    # Entry names are built from a parameter name and a float, so coerce rather
    # than validate: a stray character should not fail the whole download.
    slug = lambda v: re.sub(r"[^A-Za-z0-9._-]+", "-", str(v)).strip("-.") or "frame"
    label = slug(body.get("label") or "frame")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, item in enumerate(images):
            entry = item if isinstance(item, dict) else {"image": item}
            src = entry.get("image")
            if not src:
                continue
            value = entry.get("value")
            stem = f"{i + 1:02d}_{label}" if value is None else f"{i + 1:02d}_{label}_{value}"
            z.writestr(
                f"{slug(stem)}.png",
                bytes_with_params(from_data_url(src), entry.get("card") or body.get("card")),
            )
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name=f"{name}.zip")


def _axis_from_body(body, prefix=""):
    param = body.get(f"{prefix}param") or body.get("param")
    if not param:
        return None
    count_key = f"{prefix}count" if prefix else "count"
    max_n = 8 if prefix else 12
    count = as_int(body.get(count_key, body.get("count", 4)), count_key or "count", 2, max_n)
    values = body.get(f"{prefix}values")
    frm = float(body.get(f"{prefix}from", body.get("from", 0)))
    to = float(body.get(f"{prefix}to", body.get("to", 1)))
    out = sweep_mod.axis_values(param, frm, to, count, values)

    # The client computes values for the non-linear axes, so check them here
    # rather than trusting the payload. An unknown sampler name would otherwise
    # fall back to DDIM silently and produce a grid of identical cells.
    if param == "sampler":
        from app.core.engine.sampler import SAMPLERS

        bad = [v for v in out if v not in SAMPLERS]
        if bad:
            raise ValidationError(f"unknown sampler(s): {', '.join(map(str, bad))}")
    elif param == "image_size":
        bad = [v for v in out if not (32 <= int(v) <= 2048)]
        if bad:
            raise ValidationError(
                f"image size must be between 32 and 2048: {', '.join(map(str, bad))}")
    if len(out) > max_n:
        raise ValidationError(f"at most {max_n} values on this axis")
    return {"param": param, "values": out}


@bp.post("/sweep")
def sweep():
    body = request.get_json(force=True, silent=True) or {}
    (model_path,) = require(body, "model_path")
    ax0 = _axis_from_body(body, "")
    if not ax0:
        return err("param is required", 400)
    axes = [ax0]
    param2 = body.get("param2")
    if param2 and param2 != ax0["param"]:
        axes.append(_axis_from_body({
            "param": param2,
            "from": body.get("from2", 0),
            "to": body.get("to2", 1),
            "count": body.get("count2", 4),
            "values": body.get("values2"),
        }, ""))

    base = {
        "image_size": int(body.get("image_size", 512)),
        "steps": int(body.get("steps", 50)),
        "eta": float(body.get("eta", 0.5)),
        "skip": int(body.get("skip", 0)),
        "seed": resolve_seed(body.get("seed")),
        "text": body.get("text", ""),
        "text_weight": float(body.get("text_weight", 0.0)),
        "guidance_step": float(body.get("guidance_step", 0.02)),
        "guidance_power": float(body.get("guidance_power", 1.0)),
        "image_prompt_weight": float(body.get("image_prompt_weight", 0.0)),
        "cuts": float(body.get("cuts", 0.5)),
        "noise_level": float(body.get("noise_level", 1.0)),
        "ema": bool(body.get("ema", True)),
        "postproc": body.get("postproc") or {},
        "device": body.get("device", "auto"),
        "sampler": body.get("sampler") or "unipc",
    }

    job = registry.create("sweep", status="queued")
    job.message = "queued"
    job.detail["cols"] = len(ax0["values"])
    job.detail["rows"] = len(axes[1]["values"]) if len(axes) > 1 else 1
    job.detail["axis_x"] = ax0
    job.detail["axis_y"] = axes[1] if len(axes) > 1 else None
    job.detail["planned"] = [
        f"{ax0['param']}={sweep_mod._fmt(ax0['param'], x)}"
        + (f" · {axes[1]['param']}={sweep_mod._fmt(axes[1]['param'], y)}" if len(axes) > 1 else "")
        for y in (axes[1]["values"] if len(axes) > 1 else [None])
        for x in ax0["values"]
    ]

    def worker(job):
        try:
            sheet = sweep_mod.run_sweep(job, model_path, axes, base)
            if sheet is None:
                return
            names = "_".join(a["param"] for a in axes)
            out = workspace.sweeps / f"sweep_{names}_{int(time.time())}.png"
            card = build_card(
                base, model_path=model_path, postproc=base.get("postproc"), kind="sweep",
                extra={"axes": [{"param": a["param"], "values": a["values"]} for a in axes]},
            )
            save_with_params(sheet, out, card)
            job.detail["card"] = card
            job.detail["sheet"] = data_url(sheet)
            job.detail["path"] = str(out)
            job.status = "done"
            job.progress = 1.0
            n = len(job.detail.get("planned") or [])
            job.message = f"sweep ready ({n} samples)"
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                raise       # the lane decides: retry alone, or fail
            job.status = "error"
            job.message = str(e)

    enqueue(job, model_path, worker)
    return ok({"job": job.to_dict()})


@bp.post("/mask")
def generate_mask():
    """A procedural mask (white = selected) for the Shape tool's Generate arm.

    Fast enough (tens of milliseconds) to answer inline, so no job. The seed
    comes back so the panel can show what it used when it was left blank.
    """
    from app.core.tools import masks

    body = request.get_json(force=True, silent=True) or {}
    kind = body.get("kind") or "blobs"
    if kind not in masks.KINDS:
        return err(f"unknown shape kind: {kind}", 400)
    w = as_int(body.get("width", 512), "width", 8, 4096)
    h = as_int(body.get("height", 512), "height", 8, 4096)
    seed = resolve_seed(body.get("seed"))
    img = masks.generate(
        kind, w, h, seed=seed,
        coverage=float(body.get("coverage", 0.3)),
        soften=float(body.get("soften", 0.0)),
        invert=bool(body.get("invert", False)),
        params=body.get("params") or {},
    )
    return ok({"mask": data_url(img), "seed": seed, "coverage": round(masks.coverage_of(img), 4)})
