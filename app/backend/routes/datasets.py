"""Dataset routes: dataset records, their files, previews and deletion.

A dataset is Kiln's record of where its images are (see
``app.backend.data.manifest``); nothing here copies or rewrites the user's
images. Augmentations are a recipe applied at training time.
"""
import os
import threading
from pathlib import Path

from flask import Blueprint, request

from app.backend.data import dataset_builder, manifest, projects
from utils.api_responses import ok, err
from utils.fs import is_link, safe_move, safe_rmtree, unlink_link
from utils.process_control import registry
from utils.validators import require, safe_name

bp = Blueprint("datasets", __name__, url_prefix="/api/datasets")


def _dataset_taken(name: str) -> bool:
    try:
        projects.find_dataset(name)
        return True
    except Exception:
        return False


def _find(ds: str) -> Path:
    return projects.find_dataset(safe_name(ds, "dataset name"))


def _page(paths: list[str], offset, limit, sort: str) -> tuple[list[str], int]:
    if sort == "date":
        def mtime(p):
            try:
                return Path(p).stat().st_mtime
            except OSError:
                return 0
        paths = sorted(paths, key=mtime, reverse=True)
    else:
        paths = sorted(paths, key=lambda p: Path(p).name.lower())
    off = max(int(offset), 0)
    lim = max(min(int(limit), 200), 1)
    return paths[off:off + lim], len(paths)


def _summary(ds_dir: Path) -> dict:
    """Record plus what it currently resolves to, for the Data screen."""
    record = manifest.load(ds_dir)
    res = manifest.resolve(ds_dir, record)
    out = {
        "name": ds_dir.name,
        "path": str(ds_dir),
        "kind": "record" if record else "folder",
        "linked": is_link(ds_dir),
        "count": len(res["files"]),
        "videos_pending": len(res["videos_pending"]),
        "missing": res["missing"],
        "excluded_count": res["excluded"],
    }
    if record:
        uploads = ds_dir / manifest.UPLOADS
        out.update(
            recipe=record["recipe"],
            # What a run will see: every image in its variations.
            variants=record["recipe"]["augment_variants"],
            total=len(res["files"]) * record["recipe"]["augment_variants"],
            sources=[s["path"] for s in record["sources"]],
            added=record["added"],
            uploads=sum(1 for f in res["files"]
                        if manifest.norm(f).startswith(manifest.norm(uploads) + os.sep)),
        )
    return out


def _run_job(kind: str, message: str, fn):
    job = registry.create(kind)
    job.message = message

    def worker():
        try:
            fn(job)
            if job.status == "running":
                job.status = "done"
                job.progress = 1.0
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.message = str(e)

    t = threading.Thread(target=worker, daemon=True)
    job.thread = t
    t.start()
    return job


@bp.get("")
def list_datasets():
    manifest.adopt_drafts()
    return ok(projects.get_store().list_datasets())


@bp.get("/available")
def available():
    try:
        name = safe_name(request.args.get("name", ""), "dataset name")
    except Exception as e:
        return err(str(e), 400)
    return ok({"name": name, "available": not _dataset_taken(name)})


@bp.post("")
def create():
    """Create an empty dataset record. Images are linked or uploaded afterwards."""
    body = request.get_json(force=True, silent=True) or {}
    (name,) = require(body, "name")
    try:
        name = safe_name(name, "dataset name")
    except Exception as e:
        return err(str(e), 400)
    if _dataset_taken(name):
        return err(f"dataset '{name}' already exists; choose another name", 409)
    ds_dir = projects.get_store().dir / "datasets" / name
    manifest.new(ds_dir, body.get("recipe"))
    return ok(_summary(ds_dir))


@bp.get("/<ds>")
def get_dataset(ds):
    try:
        return ok(_summary(_find(ds)))
    except Exception:
        return err("dataset not found", 404)


@bp.post("/<ds>/link")
def link(ds):
    """Add a folder or file to the dataset where it is. Nothing is copied."""
    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    try:
        ds_dir = _find(ds)
    except Exception:
        return err("dataset not found", 404)
    try:
        manifest.add_path(ds_dir, path)
    except Exception as e:
        return err(str(e), 400)
    return ok(_summary(ds_dir))


@bp.delete("/<ds>/source")
def unlink_source(ds):
    """Stop using a linked folder or file. The folder itself is untouched."""
    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    try:
        ds_dir = _find(ds)
        manifest.remove_source(ds_dir, path)
    except Exception as e:
        return err(str(e), 400)
    return ok(_summary(ds_dir))


@bp.post("/<ds>/upload")
def upload(ds):
    """Browser uploads have no origin path, so these are copied into the dataset."""
    try:
        ds_dir = _find(ds)
        manifest.save_uploads(ds_dir, request.files.getlist("files"))
    except Exception as e:
        return err(str(e), 400)
    return ok(_summary(ds_dir))


@bp.post("/<ds>/exclude")
def exclude(ds):
    body = request.get_json(force=True, silent=True) or {}
    try:
        ds_dir = _find(ds)
        if body.get("all") and body.get("restore"):
            manifest.restore_all(ds_dir)
        else:
            (path,) = require(body, "path")
            manifest.exclude(ds_dir, path, restore=bool(body.get("restore")))
    except Exception as e:
        return err(str(e), 400)
    return ok(_summary(ds_dir))


@bp.post("/<ds>/recipe")
def recipe(ds):
    body = request.get_json(force=True, silent=True) or {}
    try:
        ds_dir = _find(ds)
        manifest.set_recipe(ds_dir, body.get("recipe") or {})
    except Exception as e:
        return err(str(e), 400)
    return ok(_summary(ds_dir))


@bp.post("/<ds>/frames")
def frames(ds):
    """Extract frames for videos that do not have them at the recipe's fps yet."""
    try:
        ds_dir = _find(ds)
    except Exception:
        return err("dataset not found", 404)

    def work(job):
        n = manifest.extract_pending(ds_dir, job=job)
        job.message = f"extracted {n} frames"

    job = _run_job("dataset", "extracting video frames...", work)
    return ok({"job": job.to_dict()})


@bp.post("/preview")
def preview():
    """Examples of one image framed and augmented by a recipe (not yet saved)."""
    from utils.imaging import data_url

    body = request.get_json(force=True, silent=True) or {}
    (ds,) = require(body, "dataset")
    try:
        ds_dir = _find(ds)
        record = manifest.load(ds_dir)
        if record is None:
            return err("only editable datasets have a recipe to preview", 400)
        recipe_ = manifest.clean_recipe({**record["recipe"], **(body.get("recipe") or {})})
        path = body.get("source_path")
        files = manifest.files(ds_dir)
        index = {manifest.norm(f): i for i, f in enumerate(files)}.get(manifest.norm(path or ""))
        if index is None:
            if not files:
                return ok({"previews": []})
            path, index = files[0], 0
        # The variations depend on the image's place in the dataset, so the
        # preview shows what this image will really be trained on.
        items = dataset_builder.preview_variants(path, recipe_, index)
    except Exception as e:
        return err(str(e), 400)
    return ok({
        "source": path,
        "previews": [{"label": x["label"], "image": data_url(x["image"])} for x in items],
    })


@bp.delete("/<ds>")
def delete_dataset(ds):
    """Remove a dataset.

    For a record-based dataset only Kiln's own folder goes: the record, browser
    uploads and extracted frames. Linked folders and files are never touched.
    A dataset folder that is itself a link loses the link, never its target.
    """
    from app.core.config import workspace

    ds = safe_name(ds, "dataset name")
    body = request.get_json(force=True, silent=True) or {}
    try:
        ds_dir = projects.find_dataset(ds)   # deliberately not resolved
    except Exception:
        return err("dataset not found", 404)

    if is_link(ds_dir):
        unlink_link(ds_dir)
        return ok({"deleted": ds, "files": False, "unlinked": True})
    if manifest.load(ds_dir) is not None:
        safe_rmtree(ds_dir)
        return ok({"deleted": ds, "files": False})

    # Legacy folder of built images: delete, or archive (the default in the UI).
    if bool(body.get("delete_files", True)):
        safe_rmtree(ds_dir)
        return ok({"deleted": ds, "files": True})
    archive = workspace.root / "archive" / "datasets" / ds
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() or is_link(archive):
        safe_rmtree(archive)
    safe_move(ds_dir, archive)
    return ok({"deleted": ds, "files": False, "archived": str(archive)})


@bp.get("/<ds>/files")
def dataset_files(ds):
    try:
        ds_dir = _find(ds)
    except Exception:
        return err("dataset not found", 404)
    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 80)), 200)
    sort = request.args.get("sort", "name")
    files, total = _page(manifest.files(ds_dir), offset, limit, sort)
    return ok({"files": files, "count": total, "offset": offset, "limit": limit})


@bp.get("/<ds>/excluded")
def excluded_files(ds):
    try:
        ds_dir = _find(ds)
    except Exception:
        return err("dataset not found", 404)
    record = manifest.load(ds_dir)
    paths = [p for p in (record or {}).get("excluded", []) if Path(p).exists()]
    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 80)), 200)
    files, total = _page(paths, offset, limit, request.args.get("sort", "name"))
    return ok({"files": files, "count": total, "offset": offset, "limit": limit})


@bp.get("/<ds>/health")
def health(ds):
    try:
        ds_dir = _find(ds)
    except Exception:
        return err("dataset not found", 404)
    return ok(dataset_builder.health(ds_dir))


@bp.get("/<ds>/samples")
def samples(ds):
    try:
        ds_dir = _find(ds)
    except Exception:
        return err("dataset not found", 404)
    imgs = manifest.files(ds_dir)
    return ok({"images": imgs[:60], "count": len(imgs)})
