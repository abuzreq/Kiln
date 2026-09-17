"""Previews and health checks for datasets."""
from pathlib import Path

from PIL import Image

from app.backend.data import augment, framing
from utils.logger import get_logger

log = get_logger("dataset_builder")


def preview_variants(image_path: str | Path, recipe: dict) -> list[dict]:
    """What training will build for one image of a v2 dataset -- the real set.

    The base is the image framed to the recipe, then every combination of its
    augmentations, exactly as ``ManifestDataset`` builds them, so a run with
    this recipe trains on the images shown here.
    """
    with Image.open(image_path) as im:
        base = framing.process_image(
            im, recipe["width"], recipe["height"], recipe["resize_mode"], recipe["padding_mode"])
    combos = augment.combinations(recipe["augmentations"], recipe["augment_settings"])
    return [{"label": augment.label(c) if i else "Framed", "image": augment.apply(base, c)}
            for i, c in enumerate(combos)]


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
