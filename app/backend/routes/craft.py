"""Craft routes: introspection, op catalog, bend preview, bend presets."""
from flask import Blueprint, request

from app.core import library
from app.core.craft import bending, ops, starters
from app.core.model_manager import manager
from utils.api_responses import ok, err
from utils.exceptions import NotFoundError
from utils.logger import get_logger
from utils.imaging import data_url, from_data_url, save_with_params
from utils.validators import require

bp = Blueprint("craft", __name__, url_prefix="/api/craft")
log = get_logger("craft")


@bp.get("/ops")
def op_catalog():
    return ok({"ops": ops.catalog(), "groups": list(bending.GROUPS)})


@bp.post("/introspect")
def introspect():
    body = request.get_json(force=True, silent=True) or {}
    (model_path,) = require(body, "model_path")
    bundle = manager.load(model_path, ema=body.get("ema", True))
    graph = bundle["backend"].layer_graph(
        bundle["model"], image_size=int(body.get("ref_size", 64)))
    graph["model"] = bundle["meta"].to_dict()
    graph["capabilities"] = bundle["backend"].capabilities.to_dict()
    return ok(graph)


# --- bend presets (Library-backed) -----------------------------------
@bp.get("/bends")
def list_bends():
    """Saved bend stacks, with Kiln's own starters folded in.

    Starters come first: an empty Library is the common case, and they are there
    to be tried before anything is saved. A user entry of the same name wins --
    saving over a starter is how you adapt one.
    """
    saved = library.list_entries("bends")
    taken = {e.get("name") for e in saved}
    builtin = [e for e in starters.entries() if e["name"] not in taken]
    return ok(builtin + saved)


@bp.post("/bends")
def save_bend():
    body = request.get_json(force=True, silent=True) or {}
    (name, bends) = require(body, "name", "bends")
    entry = library.save_entry("bends", name, {
        "bends": bends,
        "notes": body.get("notes", ""),
        "model_hint": body.get("model_hint", ""),
    })
    return ok(entry)


@bp.delete("/bends/<name>")
def delete_bend(name):
    try:
        library.delete_entry("bends", name)
    except NotFoundError:
        # A starter has no file to delete. Say so rather than reporting a missing
        # entry for something the user can plainly see in the list.
        if starters.is_starter(name):
            return err(f"'{name}' ships with Kiln and cannot be deleted. Save a stack "
                       f"under the same name to replace it with your own.", 400)
        raise
    return ok({"deleted": name})


@bp.post("/bends/export")
def bends_export():
    """Download a bend stack in the shared network-bending JSON format.

    Targets should already be resolved to concrete layer names by the caller --
    the UI knows the model's layer list, and no other tool understands Kiln's
    group names like "encoder".
    """
    import io
    import json as _json

    from flask import send_file

    from app.core.craft import interchange
    from utils.validators import safe_name

    body = request.get_json(force=True, silent=True) or {}
    (bends,) = require(body, "bends")
    if not isinstance(bends, list):
        return err("bends must be a list", 400)
    doc, report = interchange.to_external(
        bends, max_denoising_steps=body.get("max_denoising_steps") or 200)
    name = safe_name(body.get("name") or "kiln-bends", "file name")
    buf = io.BytesIO(_json.dumps(doc, indent=2).encode("utf-8"))
    resp = send_file(buf, mimetype="application/json",
                     as_attachment=True, download_name=f"{name}.json")
    # the caller cannot read a streamed body, so surface the lossiness in a header
    resp.headers["X-Kiln-Export"] = _json.dumps(report)
    return resp


@bp.post("/bends/import")
def bends_import():
    """Parse a shared-format bend file into a Kiln stack."""
    from app.core.craft import interchange

    body = request.get_json(force=True, silent=True) or {}
    (doc,) = require(body, "doc")
    try:
        bends, report = interchange.from_external(doc, known_layers=body.get("layers"))
    except ValueError as e:
        return err(str(e), 400)
    return ok({"bends": bends, "report": report})


@bp.post("/bend/sweep")
def bend_sweep():
    """Full samples across one bend parameter, everything else held fixed.

    Same seed and settings for every frame, so the only thing moving is the bend
    — which is what makes the strip (and the GIF) legible.
    """
    import threading

    from app.core.engine.sampler import SampleParams, resolve_seed, sampler
    from utils.imaging import build_card, data_url, preview_url
    from utils.process_control import registry

    body = request.get_json(force=True, silent=True) or {}
    (model_path, bends, param) = require(body, "model_path", "bends", "param")
    index = int(body.get("bend_index", 0))
    count = max(2, min(24, int(body.get("count", 5))))
    lo = float(body.get("from", 0.0))
    hi = float(body.get("to", 1.0))
    stack = list(bends or [])
    if not (0 <= index < len(stack)):
        return err("bend_index is outside the stack", 400)
    if param not in (stack[index].get("params") or {}):
        return err(f"'{param}' is not a parameter of that bend", 400)

    values = [lo + (hi - lo) * (i / max(count - 1, 1)) for i in range(count)]
    seed = resolve_seed(body.get("seed"))

    def make_params():
        return SampleParams(
            model_path=model_path,
            image_size=int(body.get("image_size", 256)),
            steps=int(body.get("steps", 25)),
            eta=float(body.get("eta", 0.5)),
            seed=seed,
            ema=bool(body.get("ema", True)),
            sampler=body.get("sampler") or "ddim",
        )

    job = registry.create("bend_sweep")
    job.message = "sweeping bend..."
    job.detail.update({
        "param": param, "bend_index": index, "seed": seed,
        "values": [round(v, 4) for v in values], "frames": [], "total": len(values),
    })

    def worker():
        from app.core.craft.bending import build_runtime

        try:
            bundle = manager.load(model_path, ema=bool(body.get("ema", True)))
            meta = bundle["meta"]
            frames = []
            for i, v in enumerate(values):
                if job.cancelled():
                    job.finish("cancelled")
                    return
                mod = [dict(b) for b in stack]
                mod[index] = {**mod[index],
                              "params": {**(mod[index].get("params") or {}), param: v}}
                runtime = None
                try:
                    runtime = build_runtime([b for b in mod if b.get("active", True)], meta,
                                            backend=bundle["backend"])
                except Exception as e:  # noqa: BLE001
                    log.warning("bend sweep: bends ignored (%s)", e)
                last = None
                for frame in sampler.run(make_params(), bend_runtime=runtime,
                                         cancel=job.cancelled):
                    last = frame
                if last is None:
                    continue
                # Each frame carries its own card: a frame can be opened in Create
                # or exported on its own, and the recipe has to travel with it.
                frames.append({
                    "value": round(v, 4),
                    "image": data_url(last["image_pp"]),
                    "card": build_card(
                        make_params(), model_path=model_path, kind="bend-sweep",
                        bends=[b for b in mod if b.get("active", True)],
                        extra={"bend_param": param, "bend_value": round(v, 4)},
                    ),
                })
                job.detail["frames"] = frames
                job.detail["frame"] = preview_url(last["image_pp"])
                job.progress = (i + 1) / len(values)
                job.message = f"{param} = {round(v, 4)}  ({i + 1}/{len(values)})"
            job.detail["card"] = build_card(
                make_params(), model_path=model_path, kind="bend-sweep",
                extra={"bend_param": param, "bend_values": [round(v, 4) for v in values]},
            )
            job.finish("done")
            job.progress = 1.0
            job.message = f"{len(frames)} frames"
        except Exception as e:  # noqa: BLE001
            job.finish("error")
            job.message = str(e)

    t = threading.Thread(target=worker, daemon=True)
    job.thread = t
    t.start()
    return ok({"job": job.to_dict()})


# --- 2-way merge ------------------------------------------------------
@bp.post("/merge/check")
def merge_check():
    from app.core.craft.merging import check_compat

    body = request.get_json(force=True, silent=True) or {}
    (a, b) = require(body, "model_a", "model_b")
    return ok(check_compat(a, b))


@bp.post("/merge/preview")
def merge_preview():
    """Write a temporary merge into the cache for same-seed A/B/merged compare."""
    from app.core.config import workspace
    from app.core.craft.merging import merge as do_merge

    body = request.get_json(force=True, silent=True) or {}
    (a, b) = require(body, "model_a", "model_b")
    res = do_merge(
        a, b, "_merge_preview",
        method=body.get("method", "linear"),
        alpha=float(body.get("alpha", 0.5)),
        block_weights=body.get("block_weights") or {},
        which=body.get("which", "both"),
        out_dir=str(workspace.cache),
    )
    # only the preview file changed — keep other models warm on the GPU
    manager.evict(res["path"])
    return ok(res)


@bp.post("/merge")
def merge():
    from app.core.craft.merging import merge as do_merge

    body = request.get_json(force=True, silent=True) or {}
    (a, b, out_name) = require(body, "model_a", "model_b", "out_name")
    res = do_merge(
        a, b, out_name,
        method=body.get("method", "linear"),
        alpha=float(body.get("alpha", 0.5)),
        block_weights=body.get("block_weights") or {},
        which=body.get("which", "both"),
    )
    manager.evict(res["path"])  # in case this name overwrote a cached model
    if res.get("path"):
        library.ensure_card(
            res["path"],
            name=out_name,
            original_name=out_name,
            trained_as=res.get("merged_from") or [out_name],
            kind="merge",
        )
        # The compare step already rendered this exact recipe; saving that image
        # beside the .pt gives the model a real thumbnail for free.
        if body.get("thumbnail"):
            try:
                from pathlib import Path

                png = Path(res["path"]).with_suffix(".png")
                save_with_params(from_data_url(body["thumbnail"]), png, body.get("card"))
                res["thumbnail"] = str(png)
            except Exception as e:  # noqa: BLE001
                from utils.logger import get_logger

                get_logger("craft").warning("merge thumbnail not saved: %s", e)
    # optionally record a reusable recipe
    if body.get("save_recipe"):
        library.save_entry("recipes", out_name, {
            "method": res["method"], "alpha": res["alpha"],
            "block_weights": res.get("block_weights", {}), "which": body.get("which", "both"),
        })
    return ok(res)


# --- novelty explorer -------------------------------------------------
# A background worker over random bend stacks; see app/core/craft/explore.py
# and docs/exploration-design.md. Run state is in memory, the archive on disk.
@bp.post("/explore")
def explore_toggle():
    from app.core.craft import explore

    body = request.get_json(force=True, silent=True) or {}
    if body.get("run", True):
        (model_path,) = require(body, "model_path")
        return ok(explore.explorer.start(model_path))
    return ok(explore.explorer.stop())


@bp.get("/explore/status")
def explore_status():
    from app.core.craft import explore

    return ok(explore.explorer.status())


@bp.get("/discoveries")
def list_discoveries():
    from app.core.craft import explore

    args = request.args
    try:
        since = float(args.get("since") or 0)
    except ValueError:
        since = 0.0
    try:
        limit = max(1, min(500, int(args.get("limit") or 200)))
    except ValueError:
        limit = 200
    return ok(explore.list_discoveries(
        model_path=args.get("model_path") or None, since=since,
        sort=args.get("sort") or "newest", limit=limit,
    ))


@bp.delete("/discoveries/<did>")
def delete_discovery(did):
    from app.core.craft import explore

    if not explore.delete_discovery(did, request.args.get("model_path") or None):
        raise NotFoundError(f"discovery {did} not found")
    return ok({"deleted": did})


@bp.post("/discoveries/<did>/star")
def star_discovery(did):
    from app.core.craft import explore

    body = request.get_json(force=True, silent=True) or {}
    entry = explore.star_discovery(did, body.get("model_path") or None,
                                   starred=bool(body.get("starred", True)))
    if entry is None:
        raise NotFoundError(f"discovery {did} not found")
    return ok(entry)
