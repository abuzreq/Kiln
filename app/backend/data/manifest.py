"""Dataset records: Kiln's own account of a set of training images.

A dataset used to be a folder of framed, augmented PNG copies. A record-based
("v2") dataset is instead a small ``dataset.json`` naming where the images are:

- ``sources``  -- folders anywhere on disk, read in place and never modified
- ``added``    -- single files added from elsewhere, also read in place
- ``excluded`` -- files the user removed from the dataset (the files stay put)
- ``recipe``   -- framing and augmentation, applied at training time

Two folders inside the dataset belong to Kiln and may be deleted with it:
``files/`` holds browser uploads (which have no origin path to point at) and is
always part of the dataset; ``_frames/`` caches frames extracted from videos.

A dataset folder without a v2 record is a legacy dataset: a folder of finished
images. It resolves to the images in its own folder, has no recipe, and trains
exactly as it always did.
"""
import hashlib
import json
import os
import time
from pathlib import Path

from app.backend.data.projects import IMAGE_EXTS
from app.backend.data.video_frames import VIDEO_EXTS
from utils.exceptions import ValidationError

VERSION = 2
RECORD = "dataset.json"
UPLOADS = "files"
FRAMES = "_frames"

DEFAULT_RECIPE = {
    "width": 512,
    "height": 512,
    "resize_mode": "center_crop",
    "padding_mode": "edge",
    # H flip on by default: both trainers always flipped at random before
    # recipes existed, so a new dataset trains the way an old one did.
    # Each augmentation's options are combined, so H flip alone makes two
    # versions of every image -- what both trainers did before recipes existed.
    "augmentations": ["hflip"],
    "augment_settings": {},
    "video_fps": 2.0,
}


def norm(path: str | Path) -> str:
    """Absolute, normalised, *unresolved* path: links stay the path the user gave."""
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def load(ds_dir: str | Path) -> dict | None:
    """The v2 record for a dataset folder, or None for a legacy dataset."""
    f = Path(ds_dir) / RECORD
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return None
    data.setdefault("sources", [])
    data.setdefault("added", [])
    data.setdefault("excluded", [])
    data["recipe"] = clean_recipe(data.get("recipe"))
    return data


def save(ds_dir: str | Path, record: dict) -> dict:
    ds_dir = Path(ds_dir)
    ds_dir.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["version"] = VERSION
    record["name"] = ds_dir.name
    record.setdefault("created_at", time.time())
    record["updated_at"] = time.time()
    record["recipe"] = clean_recipe(record.get("recipe"))
    tmp = ds_dir / (RECORD + ".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(tmp, ds_dir / RECORD)
    return record


def new(ds_dir: str | Path, recipe: dict | None = None) -> dict:
    return save(ds_dir, {"sources": [], "added": [], "excluded": [], "recipe": recipe or {}})


def clean_recipe(recipe: dict | None) -> dict:
    from app.backend.data import augment, framing

    r = {**DEFAULT_RECIPE, **(recipe or {})}
    try:
        r["width"] = max(32, min(4096, int(r["width"])))
        r["height"] = max(32, min(4096, int(r["height"])))
        r["video_fps"] = max(0.1, min(60.0, float(r["video_fps"])))
    except (TypeError, ValueError) as e:
        raise ValidationError(f"invalid dataset size or fps: {e}") from e
    if r["resize_mode"] not in framing.RESIZE_MODES:
        r["resize_mode"] = DEFAULT_RECIPE["resize_mode"]
    if r["padding_mode"] not in framing.PADDING_MODES:
        r["padding_mode"] = DEFAULT_RECIPE["padding_mode"]
    r["augmentations"] = [a for a in (r.get("augmentations") or []) if a in augment.AUGMENTATIONS]
    if not isinstance(r.get("augment_settings"), dict):
        r["augment_settings"] = {}
    r["augment_settings"] = augment.clean_settings(r["augment_settings"])
    return {k: r[k] for k in DEFAULT_RECIPE}


def versions(recipe: dict) -> int:
    """How many versions of each image this recipe trains on."""
    from app.backend.data import augment

    return augment.count(recipe["augmentations"], recipe["augment_settings"])


# --- resolving ------------------------------------------------------------

def _walk(root: Path, recursive: bool = True):
    """Media files under a folder, following links (the user's links are the point)."""
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    if not recursive:
        for p in sorted(root.iterdir()):
            if p.is_file():
                yield p
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        # Kiln's own cache never counts as source material.
        if Path(dirpath) == root and FRAMES in dirnames:
            dirnames.remove(FRAMES)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def frames_dir(ds_dir: Path, video: Path, fps: float) -> Path:
    tag = hashlib.sha1(norm(video).encode("utf-8")).hexdigest()[:10]
    return ds_dir / FRAMES / f"{video.stem}-{tag}-{fps:g}fps"


def _frames_ready(d: Path, video: Path) -> bool:
    done = d / ".done"
    if not done.exists():
        return False
    try:
        return done.read_text(encoding="utf-8").strip() == str(int(video.stat().st_mtime))
    except OSError:
        return False


def resolve(ds_dir: str | Path, record: dict | None = None) -> dict:
    """Every image in a dataset, plus what is missing or still needs work.

    Returns ``{"files": [...], "videos_pending": [...], "missing": [...],
    "excluded": n}``. Legacy datasets resolve to the images in their folder.
    """
    ds_dir = Path(ds_dir)
    record = record if record is not None else load(ds_dir)
    if record is None:
        files = sorted(str(p) for p in _walk(ds_dir) if p.suffix.lower() in IMAGE_EXTS)
        return {"files": files, "videos_pending": [], "missing": [], "excluded": 0}

    fps = float(record["recipe"]["video_fps"])
    excluded = {norm(p) for p in record.get("excluded", [])}
    seen: set[str] = set()
    files: list[str] = []
    pending: list[str] = []
    missing: list[str] = []
    dropped = 0

    def take(p: Path):
        nonlocal dropped
        key = norm(p)
        if key in seen:
            return
        seen.add(key)
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            if key in excluded:
                dropped += 1
            else:
                files.append(str(p))
        elif ext in VIDEO_EXTS:
            if key in excluded:
                dropped += 1
                return
            d = frames_dir(ds_dir, p, fps)
            if _frames_ready(d, p):
                for f in sorted(d.glob("*.png")):
                    fk = norm(f)
                    if fk in excluded:
                        dropped += 1
                    elif fk not in seen:
                        seen.add(fk)
                        files.append(str(f))
            else:
                pending.append(str(p))

    roots = [(ds_dir / UPLOADS, True)] + [
        (Path(s["path"]), bool(s.get("recursive", True))) for s in record.get("sources", [])
    ]
    for root, recursive in roots:
        if not root.exists():
            if root != ds_dir / UPLOADS:
                missing.append(str(root))
            continue
        for p in _walk(root, recursive):
            take(p)
    for a in record.get("added", []):
        p = Path(a)
        if p.exists():
            take(p)
        else:
            missing.append(str(p))
    return {"files": files, "videos_pending": pending, "missing": missing, "excluded": dropped}


def files(ds_dir: str | Path) -> list[str]:
    return resolve(ds_dir)["files"]


def extract_pending(ds_dir: str | Path, job=None) -> int:
    """Extract frames for every video that has none at the recipe's fps."""
    from app.backend.data import video_frames

    ds_dir = Path(ds_dir)
    record = load(ds_dir)
    if record is None:
        return 0
    fps = float(record["recipe"]["video_fps"])
    pending = resolve(ds_dir, record)["videos_pending"]
    total = 0
    for i, v in enumerate(pending):
        if job is not None and job.cancelled():
            break
        video = Path(v)
        d = frames_dir(ds_dir, video, fps)
        if d.exists():
            from utils.fs import safe_rmtree

            safe_rmtree(d)
        total += video_frames.extract_frames(str(video), str(d), target_fps=fps)
        (d / ".done").write_text(str(int(video.stat().st_mtime)), encoding="utf-8")
        if job is not None:
            job.progress = (i + 1) / max(len(pending), 1)
            job.message = f"extracted frames from {i + 1}/{len(pending)} videos"
    return total


# --- editing --------------------------------------------------------------

def add_path(ds_dir: str | Path, path: str) -> dict:
    """Link a folder (as a source) or a file (as an added file). Nothing is copied."""
    ds_dir = Path(ds_dir)
    record = load(ds_dir)
    if record is None:
        raise ValidationError("this dataset was built as a folder of images and cannot be edited")
    p = Path(os.path.expanduser(path.strip().strip('"')))
    if not p.exists():
        raise ValidationError(f"path does not exist: {path}")
    p = Path(os.path.abspath(p))
    key = norm(p)
    if p.is_dir():
        if key in {norm(s["path"]) for s in record["sources"]}:
            raise ValidationError("that folder is already part of this dataset")
        if norm(ds_dir) == key or key.startswith(norm(ds_dir) + os.sep):
            raise ValidationError("a dataset cannot include its own folder")
        record["sources"].append({"path": str(p), "recursive": True})
    else:
        if p.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
            raise ValidationError(f"not an image or video file: {p.name}")
        if key not in {norm(a) for a in record["added"]}:
            record["added"].append(str(p))
    # Re-adding something the user removed earlier brings it back.
    record["excluded"] = [e for e in record["excluded"]
                          if not (norm(e) == key or norm(e).startswith(key + os.sep))]
    return save(ds_dir, record)


def remove_source(ds_dir: str | Path, path: str) -> dict:
    ds_dir = Path(ds_dir)
    record = _editable(ds_dir)
    key = norm(path)
    record["sources"] = [s for s in record["sources"] if norm(s["path"]) != key]
    record["added"] = [a for a in record["added"] if norm(a) != key]
    record["excluded"] = [e for e in record["excluded"] if not norm(e).startswith(key + os.sep)]
    return save(ds_dir, record)


def owned(ds_dir: str | Path, path: str | Path) -> bool:
    """Is this file one Kiln wrote into the dataset (an upload or an extracted frame)?"""
    from app.core.library import owned_by_kiln

    key = norm(path)
    base = norm(ds_dir)
    inside = any(key.startswith(norm(Path(base) / sub) + os.sep) for sub in (UPLOADS, FRAMES))
    return inside and owned_by_kiln(path)


def exclude(ds_dir: str | Path, path: str, restore: bool = False) -> dict:
    """Take a file out of the dataset (or put it back).

    Uploads are Kiln's own copies and are deleted; anything else is only marked
    excluded and stays exactly where it is on disk.
    """
    ds_dir = Path(ds_dir)
    record = _editable(ds_dir)
    key = norm(path)
    if restore:
        record["excluded"] = [e for e in record["excluded"] if norm(e) != key]
        return save(ds_dir, record)
    if key.startswith(norm(ds_dir / UPLOADS) + os.sep) and owned(ds_dir, path):
        Path(path).unlink(missing_ok=True)
        return save(ds_dir, record)
    if key not in {norm(e) for e in record["excluded"]}:
        record["excluded"].append(os.path.abspath(path))
    return save(ds_dir, record)


def restore_all(ds_dir: str | Path) -> dict:
    ds_dir = Path(ds_dir)
    record = _editable(ds_dir)
    record["excluded"] = []
    return save(ds_dir, record)


def set_recipe(ds_dir: str | Path, recipe: dict) -> dict:
    ds_dir = Path(ds_dir)
    record = _editable(ds_dir)
    record["recipe"] = clean_recipe({**record["recipe"], **(recipe or {})})
    return save(ds_dir, record)


def save_uploads(ds_dir: str | Path, uploads) -> int:
    ds_dir = Path(ds_dir)
    _editable(ds_dir)
    dest_root = ds_dir / UPLOADS
    dest_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in uploads:
        name = Path(f.filename or "").name
        ext = Path(name).suffix.lower()
        if not name or ext not in IMAGE_EXTS | VIDEO_EXTS:
            continue
        dest = dest_root / name
        i = 1
        while dest.exists():
            dest = dest_root / f"{Path(name).stem}_{i}{ext}"
            i += 1
        f.save(str(dest))
        n += 1
    return n


def _editable(ds_dir: Path) -> dict:
    record = load(ds_dir)
    if record is None:
        raise ValidationError("this dataset was built as a folder of images and cannot be edited")
    return record


def adopt_drafts() -> int:
    """Turn drafts left by the old import-then-build flow into datasets.

    Before records, media was copied into ``drafts/<name>/`` and only became a
    dataset when built. Those copies are Kiln's, so each non-empty draft moves
    into a new dataset's ``files/`` folder under the same name. Drafts whose
    name is already taken are left where they are.
    """
    from app.backend.data.projects import _dataset_dirs
    from app.core.config import workspace
    from utils.fs import safe_move, safe_rmtree

    root = workspace.root / "drafts"
    if not root.is_dir():
        return 0
    taken = {d.name for d in _dataset_dirs()}
    adopted = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in taken:
            continue
        media = [p for p in _walk(d) if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
                 and FRAMES not in p.parts]
        if not media:
            safe_rmtree(d)
            continue
        ds_dir = workspace.root / "datasets" / d.name
        stale = d / FRAMES
        if stale.exists():
            safe_rmtree(stale)
        safe_move(d, ds_dir / UPLOADS)
        new(ds_dir)
        adopted += 1
    return adopted


# --- media access ---------------------------------------------------------

_roots_cache: dict = {"key": None, "roots": []}


def allowed_roots() -> list[str]:
    """Every source folder and added file named by any dataset record.

    ``/api/media`` serves files under these. Cached against the records' mtimes,
    because the thumbnail route asks for every image in a gallery.
    """
    from app.backend.data.projects import _dataset_dirs

    records = [d / RECORD for d in _dataset_dirs()]
    key = tuple((str(r), r.stat().st_mtime) for r in records if r.exists())
    if key == _roots_cache["key"]:
        return _roots_cache["roots"]
    roots: list[str] = []
    for r in records:
        rec = load(r.parent) if r.exists() else None
        if rec is None:
            continue
        roots += [norm(s["path"]) for s in rec["sources"]]
        roots += [norm(a) for a in rec["added"]]
    _roots_cache.update(key=key, roots=roots)
    return roots


def is_member_path(path: str | Path) -> bool:
    key = norm(path)
    return any(key == r or key.startswith(r + os.sep) for r in allowed_roots())


# --- training snapshot ----------------------------------------------------

def snapshot(ds_dir: str | Path, out_dir: str | Path) -> Path | None:
    """Freeze a v2 dataset's file list and recipe into a run folder.

    Returns the snapshot path, or None for a legacy dataset (which trains from
    its folder as it always has).
    """
    ds_dir = Path(ds_dir)
    record = load(ds_dir)
    if record is None:
        return None
    res = resolve(ds_dir, record)
    if res["videos_pending"]:
        raise ValidationError(
            f"{len(res['videos_pending'])} video(s) in this dataset still need their frames "
            "extracted; open the dataset in Data and extract them first")
    if not res["files"]:
        raise ValidationError(f"dataset '{ds_dir.name}' has no images")
    out = Path(out_dir) / RECORD
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": VERSION,
        "dataset": ds_dir.name,
        "dataset_dir": str(ds_dir),
        "recipe": record["recipe"],
        "files": res["files"],
        # What the run will actually see: every image in every version.
        "variants": versions(record["recipe"]),
        "total": len(res["files"]) * versions(record["recipe"]),
        "created_at": time.time(),
    }, indent=2), encoding="utf-8")
    return out
