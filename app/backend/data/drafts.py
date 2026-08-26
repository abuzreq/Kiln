"""Per-dataset draft staging: source files for a dataset being created."""
import os
import shutil
from pathlib import Path

from app.backend.data.projects import IMAGE_EXTS, store as workspace_store
from app.backend.data import video_frames
from app.core.config import workspace
from utils.exceptions import ValidationError, NotFoundError
from utils.validators import safe_name

VIDEO_EXTS = video_frames.VIDEO_EXTS


def draft_dir(name: str) -> Path:
    name = safe_name(name, "dataset name")
    d = workspace.root / "drafts" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _iter_media(path: Path):
    if path.is_file():
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
            yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                yield p


def import_path(name: str, source: str, mode: str = "copy") -> dict:
    dest_root = draft_dir(name)
    src = Path(source).expanduser()
    if not src.exists():
        raise ValidationError(f"source path does not exist: {source}")
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
        except Exception:
            skipped += 1
    return {"imported": imported, "skipped": skipped, "count": count_files(name)}


def save_uploads(name: str, files) -> dict:
    dest_root = draft_dir(name)
    imported = 0
    for f in files:
        fname = Path(f.filename).name
        if not fname:
            continue
        ext = Path(fname).suffix.lower()
        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
            continue
        dest = dest_root / fname
        i = 1
        while dest.exists():
            dest = dest_root / f"{Path(fname).stem}_{i}{ext}"
            i += 1
        f.save(str(dest))
        imported += 1
    return {"imported": imported, "count": count_files(name)}


def count_files(name: str) -> int:
    d = draft_dir(name)
    n = 0
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
            n += 1
    return n


def list_files(name: str, offset: int = 0, limit: int = 80, sort: str = "name") -> tuple[list[str], int]:
    d = draft_dir(name)
    files = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]
    if sort == "date":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        files.sort(key=lambda p: p.name.lower())
    paths = [str(p) for p in files]
    off = max(int(offset), 0)
    lim = max(min(int(limit), 200), 1)
    return paths[off:off + lim], len(paths)


def remove_file(name: str, path: str) -> dict:
    d = draft_dir(name).resolve()
    p = Path(path).resolve()
    if not str(p).startswith(str(d)) or not p.is_file():
        raise NotFoundError("file not in this draft")
    p.unlink(missing_ok=True)
    return {"removed": str(p), "count": count_files(name)}


def gather_images(name: str, video_fps: float = 2.0) -> list[Path]:
    d = draft_dir(name)
    images, videos = [], []
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in VIDEO_EXTS:
            videos.append(p)
    if videos:
        frames_dir = d / "_frames"
        frames_dir.mkdir(exist_ok=True)
        for v in videos:
            video_frames.extract_frames(str(v), str(frames_dir), target_fps=video_fps)
        images += sorted(frames_dir.glob("*.png"))
    return images
