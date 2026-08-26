"""Import raw media into a project's source folder."""
import os
import shutil
from pathlib import Path

from app.backend.data.projects import IMAGE_EXTS, Store, store as workspace_store
from utils.exceptions import ValidationError
from utils.logger import get_logger

log = get_logger("ingest")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}


def _iter_media(path: Path):
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS or path.suffix.lower() in VIDEO_EXTS:
            yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in VIDEO_EXTS:
                yield p


def import_path(project: Store | None, source: str, mode: str = "copy") -> dict:
    """Import a file or folder into workspace source/. mode = copy | symlink."""
    project = project or workspace_store
    src = Path(source).expanduser()
    if not src.exists():
        raise ValidationError(f"source path does not exist: {source}")
    dest_root = project.dir / "source"
    dest_root.mkdir(parents=True, exist_ok=True)

    imported, skipped = 0, 0
    for media in _iter_media(src):
        dest = dest_root / media.name
        i = 1
        while dest.exists():
            dest = dest_root / f"{media.stem}_{i}{media.suffix}"
            i += 1
        try:
            if mode == "symlink":
                os.symlink(media, dest)
            else:
                shutil.copy2(media, dest)
            imported += 1
        except Exception as e:  # noqa: BLE001
            log.warning("could not import %s: %s", media, e)
            skipped += 1
    return {"imported": imported, "skipped": skipped, "source_count": project.count_source()}


def save_uploads(project: Store | None, files) -> dict:
    """Persist uploaded file objects (werkzeug FileStorage) into source/."""
    project = project or workspace_store
    dest_root = project.dir / "source"
    dest_root.mkdir(parents=True, exist_ok=True)
    imported = 0
    for f in files:
        name = Path(f.filename).name
        if not name:
            continue
        ext = Path(name).suffix.lower()
        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
            continue
        dest = dest_root / name
        i = 1
        while dest.exists():
            dest = dest_root / f"{Path(name).stem}_{i}{ext}"
            i += 1
        f.save(str(dest))
        imported += 1
    return {"imported": imported, "source_count": project.count_source()}


def collect_source_files(project: Store | None = None) -> list[str]:
    from app.backend.data.projects import _source_roots
    files = []
    for src in _source_roots():
        if not src.exists():
            continue
        files.extend(
            str(p) for p in sorted(src.rglob("*"))
            if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in VIDEO_EXTS
        )
    return files


def list_source(project: Store | None = None, offset: int = 0, limit: int = 80) -> tuple[list[str], int]:
    """Return a page of source media paths and the total count."""
    files = collect_source_files(project)
    off = max(int(offset), 0)
    lim = max(min(int(limit), 200), 1)
    return files[off:off + lim], len(files)


def list_source_legacy(project: Store | None = None, limit: int = 500) -> list[str]:
    files, _ = list_source(project, 0, limit)
    return files
