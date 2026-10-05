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
    payload = {
        "bends": bends,
        "notes": body.get("notes", ""),
        "model_hint": body.get("model_hint", ""),
    }
    thumb = _small_thumb(body.get("thumbnail"))
    if thumb:
        payload["thumbnail"] = thumb
    entry = library.save_entry("bends", name, payload)
    return ok(entry)


def _small_thumb(thumb):
    """A recipe's picture, or None.

    Made by the page from the render the recipe was saved from, and kept in the
    entry itself; anything that is not a modest image data URL is left out
    rather than stored.
    """
    if isinstance(thumb, str) and thumb.startswith("data:image/") and len(thumb) <= 100_000:
        return thumb
    return None


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
    from app.core.engine.lanes import enqueue, is_oom
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

    job = registry.create("bend_sweep", status="queued")
    job.message = "queued"
    job.detail.update({
        "param": param, "bend_index": index, "seed": seed,
        "values": [round(v, 4) for v in values], "frames": [], "total": len(values),
    })

    def worker(job):
        from app.core.craft.bending import build_runtime

        try:
            meta, backend = manager.describe(model_path)
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
                                            backend=backend)
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
            if is_oom(e):
                raise       # the lane decides: retry alone, or fail
            job.finish("error")
            job.message = str(e)

    enqueue(job, model_path, worker)
    return ok({"job": job.to_dict()})


# --- 2-way merge ------------------------------------------------------
@bp.post("/merge/check")
def merge_check():
    from app.core.craft.merging import check_compat

    body = request.get_json(force=True, silent=True) or {}
    (a, b) = require(body, "model_a", "model_b")
    return ok(check_compat(a, b))


def _recipe_entry(body: dict, model_a: str, model_b: str, thumbnail=None) -> dict:
    """A merge recipe as the Library keeps it: the mix, and the two models.

    Like a bend preset's ``model_hint``, ``model_a`` says what it was made on;
    used from Create, the recipe blends into whatever model is selected.
    ``model_b`` is the partner Create offers by default.
    """
    from app.core.craft import ladder

    entry = {
        "model_a": model_a,
        "model_b": model_b,
        **ladder.recipe(body.get("method") or "linear", body.get("alpha", 0.5),
                        body.get("block_weights")),
        "which": body.get("which", "both"),
        "notes": body.get("notes", ""),
    }
    thumb = _small_thumb(thumbnail)
    if thumb:
        entry["thumbnail"] = thumb
    return entry


@bp.post("/merge/recipes")
def save_merge_recipe():
    """Keep a blend of two models as a recipe, without writing a model file.

    Body: ``name``, ``model_a``, ``model_b``, ``method``, ``alpha`` or
    ``block_weights``, optional ``notes`` and ``thumbnail`` (a small data URL
    of the picked sample).
    """
    body = request.get_json(force=True, silent=True) or {}
    (name, a, b) = require(body, "name", "model_a", "model_b")
    entry = library.save_entry("recipes", name,
                               _recipe_entry(body, a, b, body.get("thumbnail")))
    return ok(entry)


@bp.post("/merge/ladder")
def merge_ladder():
    """Sample A, B and a ladder of blends between them, all on one seed.

    Body: ``model_a``, ``model_b``, ``method``, ``fixed`` (``alpha``,
    ``block_weights``), up to two ``axes`` of ``{param, values}``, ``sample``
    (Create's sampling fields), ``refs`` (default true: also sample A and B,
    which a zoom that already has them skips) and ``known`` (recipes the client
    already holds samples of, which are not sampled again). A blank seed is resolved here,
    once, so every cell and both references share it; ``detail.seed`` says
    which. Cells are published as they finish -- see ``ladder.plan`` for their
    shape -- and blends are never written to disk. Each finished cell and
    reference is stamped with ``rev``, and ``detail.rev`` is the latest, so a
    poll can ask for only what is new (``GET /api/jobs/<id>?since=``).
    """
    from dataclasses import replace

    from app.backend.routes.perform import _params_from_body, _throttled_preview
    from app.core.craft import ladder
    from app.core.craft.merging import check_compat
    from app.core.engine.lanes import enqueue, is_oom
    from app.core.engine.sampler import pick_device, sampler
    from utils.exceptions import IncompatibleModelError
    from utils.imaging import build_card
    from utils.process_control import registry

    body = request.get_json(force=True, silent=True) or {}
    (a, b) = require(body, "model_a", "model_b")
    cells = ladder.plan(body.get("method", "linear"), body.get("fixed"), body.get("axes"),
                        body.get("known"))
    compat = check_compat(a, b)
    if not compat["compatible"]:
        raise IncompatibleModelError("; ".join(compat["reasons"]))
    # One image per cell: variations would multiply the cost for pictures the
    # ladder has nowhere to show.
    base = _params_from_body({**(body.get("sample") or {}), "model_path": a, "batch_size": 1})
    want_refs = body.get("refs", True) is not False
    live = (body.get("sample") or {}).get("live_preview", True) is not False
    todo = sorted((i for i, c in enumerate(cells) if c["order"] is not None),
                  key=lambda i: cells[i]["order"])
    merge_of = {"model_a": a, "model_b": b}

    job = registry.create("merge_ladder", status="queued")
    job.message = "queued"
    axes = body.get("axes") or []
    job.detail.update({
        "seed": base.seed,
        "axes": axes,
        "rows": len(axes[0]["values"]) if len(axes) == 2 else 1,
        "cols": len(axes[-1]["values"]) if axes else 1,
        "cells": [{**c, "image": None, "card": None, "rev": None} for c in cells],
        "refs": {"a": None, "b": None} if want_refs else None,
        "planned": len(todo) + (2 if want_refs else 0),
        "current": None,
        "rev": 0,
    })

    def publish(slots, key, entry):
        """Put a finished picture in place, then move ``detail.rev`` past it.

        In that order: the jobs route reads ``rev`` before it copies the
        entries, so a picture is always in some poll before ``since`` can skip it.
        """
        rev = job.detail["rev"] + 1
        slots[key] = {**entry, "rev": rev}
        job.detail["rev"] = rev

    def sample(path, bundle=None):
        """The final frame of one run, or None if the job was cancelled."""
        last = None
        for last in sampler.run(replace(base, model_path=path), bundle=bundle,
                                cancel=job.cancelled):
            _throttled_preview(job, last, live)
        return None if job.cancelled() or last is None else last

    def worker(job):
        detail = job.detail
        total, done = detail["planned"], 0
        try:
            steps = [("a", a, "model A"), ("b", b, "model B")] if want_refs else []
            for key, path, label in steps:
                detail["current"] = key
                job.message = f"sampling {label}"
                last = sample(path)
                if last is None:
                    job.finish("cancelled")
                    return
                publish(detail["refs"], key, {"image": data_url(last["image_pp"]),
                                              "card": build_card(base, model_path=path)})
                done += 1
                job.progress = done / total
            rig = None
            for n, i in enumerate(todo, 1):
                c = cells[i]
                detail["current"] = i
                job.message = f"merge {n} of {len(todo)}"
                if rig is None:
                    rig = ladder.LadderRig(a, b, pick_device(base.device), ema=base.ema)
                r = c["recipe"]
                last = sample(a, rig.apply(r["method"], r.get("alpha", 0.5),
                                           r.get("block_weights")))
                if last is None:
                    job.finish("cancelled")
                    return
                # A rung is not a model yet, so its card names no model_path;
                # the recipe travels in "merge" until the blend is saved.
                publish(detail["cells"], i, {
                    **detail["cells"][i],
                    "image": data_url(last["image_pp"]),
                    "card": build_card(base, model_path="", kind="merge-ladder",
                                       extra={"merge": {**merge_of, **r}}),
                })
                done += 1
                job.progress = done / total
            detail["current"] = None
            detail.pop("frame", None)
            job.message = f"{len(todo)} merges"
            job.progress = 1.0
            job.finish("done")
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                raise       # the lane decides: retry alone, or fail
            log.exception("merge ladder failed")
            job.finish("error")
            job.message = str(e)

    enqueue(job, [a, b], worker)
    return ok({"job": job.to_dict()})


def _thumbnail_mismatch(res: dict, card: dict) -> str | None:
    """Why a sample can't stand for the saved merge, or None if it can.

    The sample shows a blend of the slot the sampler reads. Saving with
    "Weights to merge" set to the other slot leaves that one as model A's, so
    the file would sample as plain A, not as the picture.
    """
    from app.core import backends

    backend, _ = backends.resolve(res["path"])
    ema = (card.get("params") or {}).get("ema", True) is not False
    slot = backend.sampled_slot(dict.fromkeys(res.get("slots") or ()), ema)
    if slot in (res.get("blended") or ()):
        return None
    return f"the sample shows blended {slot} weights, which this merge keeps from model A"


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
            lineage=library.lineage("merge", [library.parent_ref(a, "a"),
                                              library.parent_ref(b, "b")]),
        )
        # The ladder already rendered this exact recipe; saving that image as
        # the model's thumbnail gives it a real one for free.
        if body.get("thumbnail"):
            card = dict(body.get("card") or {})
            why_not = _thumbnail_mismatch(res, card)
            if why_not:
                res["thumbnail_skipped"] = why_not
            else:
                # The sample was of a blend in memory; from now on it is this
                # file, so the recipe must replay from it.
                card.update(model_path=res["path"], model=out_name)
                try:
                    png = library.thumb_path(res["path"], ensure_dir=True)
                    save_with_params(from_data_url(body["thumbnail"]), png, card)
                    res["thumbnail"] = str(png)
                except Exception as e:  # noqa: BLE001
                    log.warning("merge thumbnail not saved: %s", e)
    # optionally record a reusable recipe, the same entry "Save as recipe" makes
    if body.get("save_recipe"):
        library.save_entry("recipes", out_name, _recipe_entry(
            {**body, "method": res["method"], "alpha": res["alpha"],
             "block_weights": res.get("block_weights")},
            a, b, body.get("recipe_thumbnail")))
    return ok(res)


# --- novelty explorer -------------------------------------------------
# A background worker over random bend stacks; see app/core/craft/explore.py.
# Run state is in memory, the archive on disk.
@bp.post("/explore")
def explore_toggle():
    from app.core.craft import explore

    body = request.get_json(force=True, silent=True) or {}
    if body.get("run", True):
        (model_path,) = require(body, "model_path")
        metric = body.get("metric") or explore.DEFAULT_METRIC
        if metric not in explore.METRICS:
            return err(f"unknown novelty metric: {metric}", 400)
        return ok(explore.explorer.start(model_path, metric))
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


@bp.delete("/discoveries")
def clear_discoveries():
    """Everything in Discoveries, or one model's, starred entries included."""
    from app.core.craft import explore

    removed = explore.clear_discoveries(request.args.get("model_path") or None)
    return ok({"removed": removed})


@bp.get("/discoveries/map")
def discovery_map():
    from app.core.craft import explore

    metric = request.args.get("metric") or explore.DEFAULT_METRIC
    if metric not in explore.METRICS:
        return err(f"unknown novelty metric: {metric}", 400)
    return ok(explore.discovery_map(metric, request.args.get("model_path") or None))


@bp.get("/discoveries/<did>/similar")
def similar_discoveries(did):
    from app.core.craft import explore

    try:
        limit = max(1, min(100, int(request.args.get("limit") or 24)))
    except ValueError:
        limit = 24
    out = explore.similar_discoveries(did, request.args.get("model_path") or None, limit)
    if out is None:
        raise NotFoundError(f"discovery {did} not found")
    return ok(out)
