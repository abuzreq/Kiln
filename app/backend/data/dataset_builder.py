"""Previews and health checks for datasets."""
from pathlib import Path

from PIL import Image

from app.backend.data import augment, framing
from utils.logger import get_logger

log = get_logger("dataset_builder")


def preview_variants(image_path: str | Path, recipe: dict) -> list[dict]:
    """Labeled examples of what training will see for one image of a v2 dataset.

    The base is the image framed to the recipe; each augmentation then shows its
    possible outcomes. At training time one random combination is drawn per step
    (``augment.random_augment``), so these are examples, not files.
    """
    with Image.open(image_path) as im:
        base = framing.process_image(
            im, recipe["width"], recipe["height"], recipe["resize_mode"], recipe["padding_mode"])
    out = [{"label": "Framed", "image": base}]
    for label, var in augment.variants_labeled(
            base, recipe["augmentations"], recipe["augment_settings"], seed=0):
        out.append({"label": label, "image": var})
    return out


def health(ds_dir: Path) -> dict:
    from app.backend.data import manifest

    imgs = [Path(f) for f in manifest.files(ds_dir)]
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
