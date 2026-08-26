"""Source ingest: import and upload into the workspace source folder."""
from flask import Blueprint, request

from app.backend.data import ingest, projects
from utils.api_responses import ok
from utils.validators import require

bp = Blueprint("source", __name__, url_prefix="/api")


@bp.get("/studio")
def studio():
    counts = request.args.get("counts") in ("1", "true", "yes")
    return ok(projects.get_store().to_dict(counts=counts))


@bp.get("/studio/train")
def studio_train():
    """Just the live training status, for the topbar badge's frequent poll."""
    return ok({"train": projects.train_status_live()})


@bp.get("/source")
def source():
    p = projects.get_store()
    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 80)), 200)
    files, total = ingest.list_source(p, offset, limit)
    return ok({"files": files, "count": total, "offset": offset, "limit": limit})


@bp.post("/source/import")
def import_media():
    p = projects.get_store()
    body = request.get_json(force=True, silent=True) or {}
    (path,) = require(body, "path")
    return ok(ingest.import_path(p, path, body.get("mode", "copy")))


@bp.post("/source/upload")
def upload():
    p = projects.get_store()
    files = request.files.getlist("files")
    return ok(ingest.save_uploads(p, files))
