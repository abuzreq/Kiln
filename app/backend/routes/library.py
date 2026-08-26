"""Library routes: reusable bends, merge recipes, sampling presets, model cards."""
import threading

from flask import Blueprint, request

from app.core import library
from app.core.config import workspace
from app.core.model_manager import manager, read_meta
from utils.api_responses import ok, err
from utils.process_control import registry
from utils.validators import require, safe_name

bp = Blueprint("library", __name__, url_prefix="/api/library")

# Curated starting points. xurdif ships no official model zoo, so this is a small
# seed list plus support for any direct .pt URL; entries download into the workspace.
DOWNLOAD_CATALOG = [
    {
        "name": "xurdif-source",
        "label": "xurdif (engine source & docs)",
        "kind": "info",
        "url": "https://github.com/htoyryla/xurdif",
        "description": "The training/sampling engine Kiln is built on. Train your own compact models on the Train screen.",
    },
]


@bp.get("/bends")
def bends():
    return ok(library.list_entries("bends"))


@bp.get("/recipes")
def recipes():
    return ok(library.list_entries("recipes"))


@bp.get("/presets")
def presets():
    return ok(library.list_entries("presets"))


@bp.post("/presets")
def save_preset():
    body = request.get_json(force=True, silent=True) or {}
    (name, params) = require(body, "name", "params")
    return ok(library.save_entry("presets", name, {"params": params, "notes": body.get("notes", "")}))


@bp.delete("/<kind>/<name>")
def delete(kind, name):
    if kind not in ("bends", "recipes", "presets"):
        from utils.api_responses import err
        return err("unknown library kind", 400)
    library.delete_entry(kind, name)
    return ok({"deleted": name})


@bp.get("/stars")
def stars():
    return ok({"paths": library.list_stars()})


@bp.post("/stars")
def toggle_star():
    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    return ok({"paths": library.toggle_star(path)})


@bp.get("/models")
def models():
    return ok(manager.scan_public())


@bp.post("/previews")
def previews():
    """Cached multi-seed previews for the Start hub, keyed by model path.

    Takes the paths rather than rescanning: ``scan_public`` opens every
    checkpoint to read its metadata, which is far too heavy for something the
    hub polls. This is a directory listing and nothing more.

    With ``generate``, missing models are also queued for background filling, so
    the hub's poll both reports progress and keeps the queue topped up.
    """
    from app.core import previews as previews_mod

    body = request.get_json(force=True, silent=True) or {}
    paths = body.get("paths") or []
    if not isinstance(paths, list):
        return err("paths must be a list", 400)
    paths = [p for p in paths[:200] if isinstance(p, str)]

    out = {p: previews_mod.list_previews(p) for p in paths}
    queued = 0
    if body.get("generate"):
        queued = sum(1 for p in paths if previews_mod.enqueue(p))
    return ok({
        "previews": out,
        "count": previews_mod.PREVIEW_COUNT,
        "queued": queued,
        "pending": previews_mod.pending_count(),
    })


@bp.post("/model/rename")
def rename_model():
    body = request.get_json(force=True, silent=True) or {}
    (path, name) = require(body, "path", "name")
    result = library.rename_model(path, name)
    manager.clear_cache()
    return ok(result)


@bp.delete("/model")
def delete_model():
    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    library.delete_model_files(path)
    manager.clear_cache()
    return ok({"deleted": path})


@bp.get("/catalog")
def catalog():
    return ok(DOWNLOAD_CATALOG)


@bp.post("/download")
def download():
    """Download a .pt model from a direct URL into the workspace models folder."""
    body = request.get_json(force=True, silent=True) or {}
    (url, name) = require(body, "url", "name")
    name = safe_name(name, "model name")
    dest = workspace.models / f"{name}.pt"

    job = registry.create("download")
    job.message = f"downloading {name}..."

    def worker():
        import requests

        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                got = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if job.cancelled():
                            job.status = "cancelled"
                            return
                        f.write(chunk)
                        got += len(chunk)
                        if total:
                            job.progress = got / total
                            job.message = f"{got // (1<<20)} / {total // (1<<20)} MB"
            job.status = "done"
            job.progress = 1.0
            job.message = f"saved {name}.pt"
            job.detail["path"] = str(dest)
            try:
                read_meta(dest)
            except Exception as e:
                dest.unlink(missing_ok=True)
                dest.with_suffix(".png").unlink(missing_ok=True)
                card_file = library.card_path(dest)
                if card_file.exists():
                    card_file.unlink()
                raise ValueError(f"downloaded file is not a valid model checkpoint: {e}") from e
            library.ensure_card(dest, name=name, original_name=name, trained_as=[name])
            manager.clear_cache()
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.message = str(e)
            dest.unlink(missing_ok=True)

    t = threading.Thread(target=worker, daemon=True)
    job.thread = t
    t.start()
    return ok({"job": job.to_dict()})
