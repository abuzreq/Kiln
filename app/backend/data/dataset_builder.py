"""Compile draft or source media into a training-ready dataset."""
import io
import json
import shutil
import time
from pathlib import Path

from PIL import Image

from app.backend.data import augment, drafts, framing
from app.backend.data.projects import IMAGE_EXTS, Store, store as workspace_store
from utils.exceptions import ValidationError
from utils.logger import get_logger
from utils.validators import safe_name

log = get_logger("dataset_builder")


def preview_draft_variants(
    draft_name: str,
    width: int,
    height: int,
    resize_mode: str,
    padding_mode: str,
    augment_ops: list[str] | None = None,
    augment_settings: dict | None = None,
    source_path: str | None = None,
) -> list[dict]:
    """Return labeled preview images for the first draft source file."""
    images = drafts.gather_images(draft_name, video_fps=1)
    if source_path:
        p = Path(source_path)
        if p.exists() and p.is_file():
            images = [p] + [x for x in images if str(x) != str(p)]
    if not images:
        raise ValidationError("no media in this draft; import or upload files first")
    sample = Image.open(images[0])
    base = framing.process_image(sample, width, height, resize_mode, padding_mode)
    out = [{"label": "Base", "image": base}]
    augment_ops = augment_ops or []
    for label, var in augment.variants_labeled(base, augment_ops, augment_settings, seed=0):
        out.append({"label": label, "image": framing.process_image(var, width, height, resize_mode, padding_mode)})
    return out


def preview_draft(
    draft_name: str,
    width: int,
    height: int,
    mode: str,
    padding_mode: str,
    source_path: str | None = None,
) -> bytes:
    variants = preview_draft_variants(draft_name, width, height, mode, padding_mode, [], {}, source_path)
    buf = io.BytesIO()
    variants[0]["image"].save(buf, format="PNG")
    return buf.getvalue()


def build(
    project: Store | None,
    name: str,
    width: int,
    height: int,
    resize_mode: str = "center_crop",
    padding_mode: str = "edge",
    augment_ops: list[str] | None = None,
    augment_settings: dict | None = None,
    video_fps: float = 2.0,
    job=None,
    from_draft: bool = True,
) -> dict:
    project = project or workspace_store
    ds_name = safe_name(name, "dataset name")
    ds_dir = project.dir / "datasets" / ds_name
    if ds_dir.exists() and any(ds_dir.glob("*.png")):
        raise ValidationError(f"dataset '{ds_name}' already exists")
    ds_dir.mkdir(parents=True, exist_ok=True)

    if from_draft:
        images = drafts.gather_images(ds_name, video_fps=video_fps)
    else:
        from app.backend.data.ingest import collect_source_files
        images = [Path(p) for p in collect_source_files(project)]

    if not images:
        raise ValidationError("no source images found; add media to this dataset first")

    augment_ops = augment_ops or []
    total = len(images)
    written = 0
    for idx, img_path in enumerate(images):
        if job is not None and job.cancelled():
            break
        try:
            img = Image.open(img_path)
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", img_path, e)
            continue
        base = framing.process_image(img, width, height, resize_mode, padding_mode)
        base.save(ds_dir / f"{written:06d}.png")
        written += 1
        for var in augment.variants(base, augment_ops, augment_settings, seed=idx):
            var = framing.process_image(var, width, height, resize_mode, padding_mode)
            var.save(ds_dir / f"{written:06d}.png")
            written += 1
        if job is not None:
            job.progress = (idx + 1) / total
            job.message = f"processed {idx + 1}/{total} sources -> {written} images"

    meta = {
        "name": ds_name,
        "width": width,
        "height": height,
        "resize_mode": resize_mode,
        "padding_mode": padding_mode,
        "augmentations": augment_ops,
        "augment_settings": augment_settings or {},
        "video_fps": video_fps,
        "count": written,
        "created_at": time.time(),
    }
    (ds_dir / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if from_draft:
        draft_path = drafts.draft_dir(ds_name)
        if draft_path.exists():
            shutil.rmtree(draft_path, ignore_errors=True)

    if job is not None:
        job.status = "done"
        job.progress = 1.0
        job.message = f"dataset '{ds_name}' ready: {written} images"
        job.detail["dataset"] = str(ds_dir)
    return meta


def health(ds_dir: Path) -> dict:
    imgs = [p for p in ds_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS]
    sizes = set()
    for p in imgs[:200]:
        try:
            with Image.open(p) as im:
                sizes.add(im.size)
        except Exception:  # noqa: BLE001
            pass
    warnings = []
    if len(imgs) < 50:
        warnings.append(f"only {len(imgs)} images; 50-200+ recommended for stable training")
    if len(sizes) > 1:
        warnings.append("images are not all the same size")
    return {"count": len(imgs), "distinct_sizes": len(sizes), "warnings": warnings}
