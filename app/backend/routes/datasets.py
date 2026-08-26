"""Dataset routes: preview processing and build training-ready datasets."""
import threading

from flask import Blueprint, request, send_file
import io

from app.backend.data import dataset_builder, drafts, projects
from utils.api_responses import ok, err
from utils.process_control import registry
from utils.validators import require, as_int, safe_name

bp = Blueprint("datasets", __name__, url_prefix="/api/datasets")


def _dataset_taken(name: str) -> bool:
    try:
        projects.find_dataset(name)
        return True
    except Exception:
        return False


def _list_dataset_files(ds_dir, offset: int = 0, limit: int = 80, sort: str = "name") -> tuple[list[str], int]:
    from app.backend.data.projects import IMAGE_EXTS
    imgs = [x for x in ds_dir.glob("*") if x.suffix.lower() in IMAGE_EXTS]
    if sort == "date":
        imgs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        imgs.sort(key=lambda p: p.name.lower())
    paths = [str(x) for x in imgs]
    off = max(int(offset), 0)
    lim = max(min(int(limit), 200), 1)
    return paths[off:off + lim], len(paths)


@bp.get("")
def list_datasets():
    return ok(projects.get_store().list_datasets())


@bp.get("/available")
def available():
    try:
        name = safe_name(request.args.get("name", ""), "dataset name")
    except Exception as e:
        return err(str(e), 400)
    return ok({"name": name, "available": not _dataset_taken(name)})


@bp.get("/draft/files")
def draft_files():
    try:
        name = safe_name(request.args.get("name", ""), "dataset name")
    except Exception as e:
        return err(str(e), 400)
    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 80)), 200)
    sort = request.args.get("sort", "name")
    files, total = drafts.list_files(name, offset, limit, sort)
    return ok({"files": files, "count": total, "offset": offset, "limit": limit})


@bp.post("/draft/import")
def draft_import():
    body = request.get_json(force=True, silent=True) or {}
    (name, path) = require(body, "name", "path")
    try:
        return ok(drafts.import_path(name, path, body.get("mode", "copy")))
    except Exception as e:
        return err(str(e), 400)


@bp.post("/draft/upload")
def draft_upload():
    name = request.form.get("name", "")
    if not name:
        return err("name required", 400)
    files = request.files.getlist("files")
    try:
        return ok(drafts.save_uploads(name, files))
    except Exception as e:
        return err(str(e), 400)


@bp.delete("/draft/file")
def draft_remove_file():
    body = request.get_json(force=True, silent=True) or {}
    (name, path) = require(body, "name", "path")
    try:
        return ok(drafts.remove_file(name, path))
    except Exception as e:
        return err(str(e), 404)


@bp.post("/preview")
def preview():
    from utils.imaging import data_url

    body = request.get_json(force=True, silent=True) or {}
    w = as_int(body.get("width", 512), "width", 32, 4096)
    h = as_int(body.get("height", 512), "height", 32, 4096)
    try:
        name = safe_name(body.get("name", ""), "dataset name")
    except Exception as e:
        return err(str(e), 400)
    try:
        items = dataset_builder.preview_draft_variants(
            name,
            w,
            h,
            body.get("resize_mode", "center_crop"),
            body.get("padding_mode", "edge"),
            body.get("augmentations", []),
            body.get("augment_settings", {}),
            source_path=body.get("source_path"),
        )
    except Exception as e:
        return err(str(e), 400)
    return ok({
        "previews": [{"label": x["label"], "image": data_url(x["image"])} for x in items],
    })


@bp.post("/build")
def build():
    p = projects.get_store()
    body = request.get_json(force=True, silent=True) or {}
    (ds_name,) = require(body, "name")
    try:
        safe_name(ds_name, "dataset name")
    except Exception as e:
        return err(str(e), 400)
    if _dataset_taken(ds_name):
        return err(f"dataset '{ds_name}' already exists; choose another name", 409)
    w = as_int(body.get("width", 512), "width", 32, 4096)
    h = as_int(body.get("height", 512), "height", 32, 4096)
    resize_mode = body.get("resize_mode", "center_crop")
    padding_mode = body.get("padding_mode", "edge")
    aug = body.get("augmentations", [])
    aug_settings = body.get("augment_settings", {})
    fps = float(body.get("video_fps", 2.0))

    job = registry.create("dataset")
    job.message = "creating dataset..."

    def worker():
        try:
            dataset_builder.build(
                p, ds_name, w, h, resize_mode, padding_mode, aug, aug_settings, fps, job=job, from_draft=True,
            )
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.message = str(e)

    t = threading.Thread(target=worker, daemon=True)
    job.thread = t
    t.start()
    return ok({"job": job.to_dict()})


@bp.delete("/<ds>")
def delete_dataset(ds):
    import shutil
    from app.core.config import workspace

    ds = safe_name(ds, "dataset name")
    body = request.get_json(force=True, silent=True) or {}
    delete_files = bool(body.get("delete_files", True))
    try:
        ds_dir = projects.find_dataset(ds).resolve()
    except Exception:
        return err("dataset not found", 404)
    if delete_files:
        shutil.rmtree(ds_dir)
        return ok({"deleted": ds, "files": True})
    archive = workspace.root / "archive" / "datasets" / ds
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        shutil.rmtree(archive)
    shutil.move(str(ds_dir), str(archive))
    return ok({"deleted": ds, "files": False, "archived": str(archive)})


@bp.get("/<ds>/files")
def dataset_files(ds):
    try:
        ds_dir = projects.find_dataset(ds)
    except Exception:
        return err("dataset not found", 404)
    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 80)), 200)
    sort = request.args.get("sort", "name")
    files, total = _list_dataset_files(ds_dir, offset, limit, sort)
    return ok({"files": files, "count": total, "offset": offset, "limit": limit})


@bp.get("/<ds>/health")
def health(ds):
    try:
        ds_dir = projects.find_dataset(ds)
    except Exception:
        return err("dataset not found", 404)
    return ok(dataset_builder.health(ds_dir))


@bp.get("/<ds>/samples")
def samples(ds):
    try:
        ds_dir = projects.find_dataset(ds)
    except Exception:
        return err("dataset not found", 404)
    imgs = sorted(str(x) for x in ds_dir.glob("*.png"))
    return ok({"images": imgs[:60], "count": len(imgs)})
