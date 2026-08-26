"""Tools routes: super-resolution, sweep runner / contact sheets, GIF export."""
import base64
import io
import threading
import time

from flask import Blueprint, request, send_file

from app.core.config import workspace
from app.core.engine.sampler import resolve_seed
from app.core.tools import sweep as sweep_mod
from app.core.tools.superres import upscale
from utils.api_responses import ok, err
from utils.imaging import build_card, bytes_with_params, data_url, from_data_url, save_with_params
from utils.process_control import registry
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
    return {"param": param, "values": sweep_mod.axis_values(param, frm, to, count, values)}


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

    job = registry.create("sweep")
    job.message = "starting sweep..."
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

    def worker():
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
            job.status = "error"
            job.message = str(e)

    t = threading.Thread(target=worker, daemon=True)
    job.thread = t
    t.start()
    return ok({"job": job.to_dict()})
