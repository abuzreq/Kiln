"""System routes: health, device/GPU info, workspace management, media serving."""
import io
import os
from pathlib import Path

from flask import Blueprint, request, send_file

from app import __version__
from app.core.config import get_device_info, workspace
from utils.api_responses import ok, err
from utils.process_control import registry
from utils.validators import require

bp = Blueprint("system", __name__, url_prefix="/api")


def _within_allowed(path: Path) -> bool:
    """Only serve files Kiln has a reason to show.

    That is the workspace, the install's bundled models and vendor folders, and
    the folders and files a dataset record links to -- datasets read the user's
    images in place. The path is checked both as given (so a folder linked into
    the workspace is served under the workspace path it appears at) and resolved
    (so ``..`` or a link cannot walk out of those roots).
    """
    from app.backend.data import manifest

    proj_root = Path(__file__).resolve().parents[3]
    roots = [workspace.root, proj_root / "models", proj_root / "vendor"]

    def under(p: str, root: str) -> bool:
        return p == root or p.startswith(root.rstrip(os.sep) + os.sep)

    try:
        given = manifest.norm(path)
        real = os.path.normcase(str(path.resolve()))
    except Exception:  # noqa: BLE001
        return False
    if ".." in Path(str(path)).parts:
        return False
    for r in roots:
        if under(given, manifest.norm(r)) or under(real, os.path.normcase(str(r.resolve()))):
            return True
    return manifest.is_member_path(given)


@bp.get("/media")
def media():
    """Serve an image file by absolute path (restricted to the workspace)."""
    raw = request.args.get("path", "")
    if not raw:
        return err("path required", 400)
    p = Path(raw)
    if not p.exists() or not p.is_file() or not _within_allowed(p):
        return err("not found", 404)
    return send_file(str(p))


@bp.get("/thumb")
def thumb():
    """Serve a downscaled thumbnail of an image (max side ~320px)."""
    from PIL import Image

    raw = request.args.get("path", "")
    p = Path(raw)
    if not raw or not p.exists() or not p.is_file() or not _within_allowed(p):
        return err("not found", 404)
    try:
        st = p.stat()
        im = Image.open(p).convert("RGB")
        im.thumbnail((320, 320))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
        buf.seek(0)
        resp = send_file(buf, mimetype="image/jpeg")
        # Unlike /media, this streams a BytesIO, so Flask attaches no validators
        # and every repeat view re-decodes the source and re-encodes a JPEG.
        # Galleries can hold a lot of these, so give them something to revalidate
        # against; the source path plus mtime and size identifies the output.
        resp.last_modified = st.st_mtime
        resp.set_etag(f"{int(st.st_mtime)}-{st.st_size}-320")
        resp.cache_control.max_age = 3600
        resp.cache_control.public = True
        return resp.make_conditional(request)
    except Exception as e:  # noqa: BLE001
        return err(str(e), 500)


@bp.get("/health")
def health():
    return ok({"app": "kiln", "version": __version__, "status": "ok"})


@bp.get("/device")
def device():
    return ok(get_device_info())


@bp.get("/workspace")
def get_workspace():
    return ok(workspace.to_dict())


@bp.post("/workspace")
def set_workspace():
    (path,) = require(request.get_json(force=True, silent=True) or {}, "path")
    workspace.set_root(path)
    return ok(workspace.to_dict())


@bp.post("/workspace/open")
def open_workspace_folder():
    """Reveal one of the workspace folders in the OS file manager.

    Kiln writes everything outside the project directory, so without this the
    only way to find your datasets, runs or captures is to already know where
    the workspace is.
    """
    import subprocess
    import sys

    body = request.get_json(force=True, silent=True) or {}
    key = body.get("key", "root")
    # to_dict also names the install's own model folders: discovery scans them, so
    # a tester who guesses and drops a .pt into models/pretrained is right, and
    # then cannot find it again from inside Kiln. They are openable for that reason.
    paths = workspace.to_dict()
    if key not in paths:
        return err(f"unknown workspace folder: {key}", 400)

    target = Path(paths[key])
    # Still never anything outside the places Kiln itself owns.
    proj_root = Path(__file__).resolve().parents[3]
    allowed = [workspace.root, proj_root / "models", proj_root / "vendor"]
    try:
        resolved = target.resolve()
        if not any(_within(resolved, root) for root in allowed):
            raise ValueError(target)
    except (ValueError, OSError):
        return err("refusing to open a path outside Kiln's own folders", 400)
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)

    try:
        if sys.platform == "win32":
            os.startfile(str(target))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as e:  # noqa: BLE001
        return err(f"could not open the folder: {e}", 500)
    return ok({"opened": str(target)})


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


# Detail keys that hold images. A queue strip polling every live job wants
# progress and position, not megabytes of data URLs it will not draw.
_HEAVY_DETAIL = ("frame", "frames", "frame_raw", "frames_raw", "sheet", "cells",
                 "mask_union")


@bp.get("/jobs")
def list_jobs():
    """Jobs, newest first.

    ``kind`` or ``kinds=sample,inpaint`` filter by kind, ``live=1`` keeps only
    queued and running jobs, and ``light=1`` leaves the images out of detail.
    """
    from utils.process_control import LIVE

    kinds = {k for k in (request.args.get("kinds") or "").split(",") if k}
    if request.args.get("kind"):
        kinds.add(request.args["kind"])
    jobs = registry.list()
    if kinds:
        jobs = [j for j in jobs if j["kind"] in kinds]
    if request.args.get("live") == "1":
        jobs = [j for j in jobs if j["status"] in LIVE]
    if request.args.get("light") == "1":
        jobs = [{**j, "detail": {k: v for k, v in j["detail"].items() if k not in _HEAVY_DETAIL}}
                for j in jobs]
    return ok(jobs)


@bp.get("/jobs/<job_id>")
def get_job(job_id):
    registry.prune()
    job = registry.get(job_id)
    if not job:
        return ok(None)
    return ok(job.to_dict())


@bp.post("/jobs/<job_id>/cancel")
def cancel_job(job_id):
    return ok({"cancelled": registry.cancel(job_id)})


@bp.post("/jobs/group/<group_id>/cancel")
def cancel_job_group(group_id):
    """Stop a repeat run: every one of its jobs that has not finished."""
    return ok({"cancelled": registry.cancel_group(group_id)})


@bp.post("/jobs/<job_id>/pause")
def pause_job(job_id):
    return ok({"paused": registry.pause(job_id)})


@bp.post("/jobs/<job_id>/resume")
def resume_job(job_id):
    body = request.get_json(force=True, silent=True) or {}
    return ok({"resumed": registry.resume(job_id, body)})


@bp.post("/gpu/free")
def free_gpu():
    """Release cached CUDA memory (best effort)."""
    freed = False
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            freed = True
    except Exception:  # noqa: BLE001
        pass
    return ok({"freed": freed})
