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
        "name": "xurdif-sample-models",
        "label": "Sample models from the xurdif author",
        "kind": "info",
        "url": (
            "https://www.dropbox.com/scl/fo/flh4pczukrrlb3ar1rfuc/"
            "AAT22M2b21Tf1yKe3Ji0HS0?rlkey=f1zdhexy36p3hffcun686m77c&dl=0"
        ),
        "description": (
            "Hannu Toyryla, who wrote the xurdif engine, publishes a folder of "
            "trained models. Open it, copy the download link for any .pt file, "
            "then paste that link into the box below. Kiln links to the folder "
            "and downloads nothing on its own."
        ),
    },
    {
        "name": "xurdif-source",
        "label": "xurdif (engine source & docs)",
        "kind": "info",
        "url": "https://github.com/htoyryla/xurdif",
        "description": "The training/sampling engine Kiln is built on. Train your own compact models on the Train screen.",
    },
]


def _direct_download_url(url: str) -> str:
    """Turn a share link into one that returns the file itself.

    A Dropbox share link ends in `dl=0`, which serves an HTML preview page.
    Downloading that yields a .pt full of markup, and the only thing the user
    sees is "downloaded file is not a valid model checkpoint" for a link that
    looked perfectly correct.
    """
    if "dropbox.com" not in url:
        return url
    if "dl=1" in url or "raw=1" in url:
        return url
    if "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    return url + ("&" if "?" in url else "?") + "dl=1"


def _diffusers():
    """The Diffusers discovery module, or None if the backend is unavailable."""
    try:
        from app.core.backends.hfdiffusers import discovery

        return discovery
    except Exception as e:  # noqa: BLE001
        from utils.logger import get_logger

        get_logger("library").info("diffusers discovery unavailable: %s", e)
        return None


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
    return ok(manager.scan_public(include_hidden=request.args.get("hidden") == "1"))


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
    failed = {p: why for p in paths if (why := previews_mod.failure(p))}
    return ok({
        "previews": out,
        "failed": failed,
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


@bp.post("/model/hide")
def hide_model():
    """Hide (or unhide) a model from Kiln's lists. The file is never touched."""
    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    paths = library.set_hidden(path, bool(body.get("hidden", True)))
    return ok({"hidden": paths})


@bp.get("/catalog")
def catalog():
    return ok(DOWNLOAD_CATALOG)


@bp.post("/download")
def download():
    """Download a .pt model from a direct URL into the workspace models folder."""
    body = request.get_json(force=True, silent=True) or {}
    (url, name) = require(body, "url", "name")
    url = _direct_download_url(url)
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


# --- Hugging Face models ---------------------------------------------
@bp.get("/hf/suggested")
def hf_suggested():
    """A short, verified starting list -- not a whitelist.

    Any repo the user pastes still goes through /hf/validate; this exists so the
    browser is not an empty text box.
    """
    d = _diffusers()
    if d is None:
        return ok({"models": [], "available": False})
    return ok({"models": d.suggested(), "available": True})


@bp.post("/hf/validate")
def hf_validate():
    """Can Kiln use this model? Reads metadata only -- never downloads weights.

    Always 200: an unsupported model is a normal answer with a reason, not a
    request error, and the browser renders the reason next to the input.
    """
    d = _diffusers()
    if d is None:
        return err("the Diffusers backend is not available in this install", 501)
    body = request.get_json(force=True, silent=True) or {}
    (ref,) = require(body, "ref")
    return ok(d.validate(ref))


@bp.post("/hf/import")
def hf_import():
    """Download a validated model into the workspace as a job."""
    d = _diffusers()
    if d is None:
        return err("the Diffusers backend is not available in this install", 501)
    body = request.get_json(force=True, silent=True) or {}
    (ref,) = require(body, "ref")

    verdict = d.validate(ref)
    if not verdict["ok"]:
        return err(verdict.get("reason") or "unsupported model", 400)
    name = safe_name(body.get("name") or verdict["model"]["name"], "model name")

    job = registry.create("hf_import")
    job.message = f"importing {name}..."
    job.detail["model"] = verdict["model"]

    def worker():
        try:
            res = d.import_to_workspace(
                ref, name, workspace.models,
                progress=lambda m: setattr(job, "message", m),
            )
            job.detail.update(res)
            library.ensure_card(res["path"], name=name, original_name=name,
                                trained_as=[name], kind="import")
            manager.clear_cache()
            job.status = "done"
            job.progress = 1.0
            job.message = f"imported {name}"
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.message = str(e)

    t = threading.Thread(target=worker, daemon=True)
    job.thread = t
    t.start()
    return ok({"job": job.to_dict()})


@bp.post("/model/convert")
def convert_model():
    """Re-home a xurdif .pt into a Diffusers model directory.

    Lossless, and verified so: the network is the same network, so the weights
    transfer verbatim and the converted model samples bit-identically to the
    original (scripts/smoke_tinyunet_parity.py). The original .pt is left in
    place -- this adds a model, it does not replace one.
    """
    from pathlib import Path

    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    src = Path(path)
    if not src.exists():
        return err("model not found", 404)

    try:
        from app.core.backends.hfdiffusers.convert import convert_checkpoint
    except Exception as e:  # noqa: BLE001
        return err(f"the Diffusers backend is not available: {e}", 501)

    name = safe_name(body.get("name") or f"{src.stem}-diffusers", "model name")
    dest = workspace.models / name
    try:
        res = convert_checkpoint(
            src, dest,
            ema=bool(body.get("ema", True)),
            num_train_timesteps=body.get("num_train_timesteps"),
            overwrite=bool(body.get("overwrite", False)),
        )
    except Exception as e:  # noqa: BLE001
        return err(str(e), 400)

    library.ensure_card(res["path"], name=name, original_name=name,
                        trained_as=[res["source_name"]], kind="convert")
    manager.clear_cache()
    return ok(res)


@bp.get("/model/convertible")
def convertible_models():
    """xurdif models that can be re-homed, and whether they already have been."""
    try:
        from app.core.backends.hfdiffusers.convert import (
            DEFAULT_TRAIN_TIMESTEPS, resolve_train_timesteps,
        )
        from app.core.backends.hfdiffusers.tinyunet import SOURCE_MTYPE
    except Exception:  # noqa: BLE001
        return ok({"models": [], "available": False})

    from pathlib import Path

    out = []
    for m in manager.scan():
        if m.backend != "xurdif" or m.mtype != SOURCE_MTYPE:
            continue
        steps, source = resolve_train_timesteps(Path(m.path))
        out.append({
            "path": m.path, "name": m.name, "mults": m.mults, "step": m.step,
            "size_mb": round(m.size_mb, 2), "source": m.source,
            "num_train_timesteps": steps,
            # Worth surfacing: xurdif does not record the schedule length in the
            # checkpoint, and guessing wrong rescales the whole noise schedule.
            "timesteps_known": source != "default",
            "default_timesteps": DEFAULT_TRAIN_TIMESTEPS,
        })
    return ok({"models": out, "available": True})
