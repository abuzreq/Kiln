"""Dataset-expansion augmentations, applied when building a dataset."""
import random

from PIL import Image, ImageEnhance

AUGMENTATIONS = ("hflip", "vflip", "rotate", "brightness", "contrast")

DEFAULT_SETTINGS = {
    "rotate": {"mode": "random", "angles": [90, 180, 270], "angle": 90},
    "brightness": {"min": 0.8, "max": 1.2},
    "contrast": {"min": 0.8, "max": 1.2},
}


def _merge_settings(settings: dict | None) -> dict:
    out = {k: dict(v) for k, v in DEFAULT_SETTINGS.items()}
    if settings:
        for k, v in settings.items():
            if isinstance(v, dict) and k in out:
                out[k].update(v)
            else:
                out[k] = v
    return out


def variants_labeled(
    img: Image.Image, ops: list[str], settings: dict | None = None, seed: int = 0,
) -> list[tuple[str, Image.Image]]:
    """Return labeled extra variants of ``img`` for the selected augmentation ops."""
    rnd = random.Random(seed)
    cfg = _merge_settings(settings)
    out: list[tuple[str, Image.Image]] = []
    if "hflip" in ops:
        out.append(("H flip", img.transpose(Image.FLIP_LEFT_RIGHT)))
    if "vflip" in ops:
        out.append(("V flip", img.transpose(Image.FLIP_TOP_BOTTOM)))
    if "rotate" in ops:
        rot = cfg.get("rotate", DEFAULT_SETTINGS["rotate"])
        if rot.get("mode") == "fixed":
            angle = int(rot.get("angle", 90))
            out.append((f"Rotate {angle}°", img.rotate(angle, expand=False)))
        else:
            angles = rot.get("angles") or [90, 180, 270]
            if rot.get("all_angles"):
                for angle in angles:
                    out.append((f"Rotate {int(angle)}°", img.rotate(int(angle), expand=False)))
            else:
                angle = int(rnd.choice(angles))
                out.append((f"Rotate {angle}°", img.rotate(angle, expand=False)))
    if "brightness" in ops:
        b = cfg.get("brightness", DEFAULT_SETTINGS["brightness"])
        lo, hi = float(b.get("min", 0.8)), float(b.get("max", 1.2))
        out.append((f"Brightness {lo:.2f}", ImageEnhance.Brightness(img).enhance(lo)))
        if hi != lo:
            out.append((f"Brightness {hi:.2f}", ImageEnhance.Brightness(img).enhance(hi)))
    if "contrast" in ops:
        c = cfg.get("contrast", DEFAULT_SETTINGS["contrast"])
        lo, hi = float(c.get("min", 0.8)), float(c.get("max", 1.2))
        out.append((f"Contrast {lo:.2f}", ImageEnhance.Contrast(img).enhance(lo)))
        if hi != lo:
            out.append((f"Contrast {hi:.2f}", ImageEnhance.Contrast(img).enhance(hi)))
    return out


def variants(img: Image.Image, ops: list[str], settings: dict | None = None, seed: int = 0) -> list[Image.Image]:
    """Return extra variants of ``img`` for the selected augmentation ops."""
    return [im for _, im in variants_labeled(img, ops, settings, seed)]


def random_augment(img: Image.Image, ops: list[str], settings: dict | None = None,
                   rng: random.Random | None = None) -> Image.Image:
    """One random variation of ``img``, drawn fresh each time. Used at training time.

    Where ``variants_labeled`` writes a fixed set of copies, this picks one
    outcome per call, so every training step sees a different mix:

    - hflip / vflip: flipped with probability 1/2
    - rotate: random mode picks one of 0° and the chosen angles; fixed mode
      rotates by the angle with probability 1/2
    - brightness / contrast: a factor drawn uniformly between min and max
    """
    rng = rng or random
    cfg = _merge_settings(settings)
    out = img
    if "hflip" in ops and rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if "vflip" in ops and rng.random() < 0.5:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    if "rotate" in ops:
        rot = cfg.get("rotate", DEFAULT_SETTINGS["rotate"])
        if rot.get("mode") == "fixed":
            angle = int(rot.get("angle", 90)) if rng.random() < 0.5 else 0
        else:
            angle = int(rng.choice([0] + [int(a) for a in (rot.get("angles") or [90, 180, 270])]))
        if angle % 360:
            out = out.rotate(angle, expand=False)
    for op, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast)):
        if op in ops:
            c = cfg.get(op, DEFAULT_SETTINGS[op])
            lo, hi = sorted((float(c.get("min", 0.8)), float(c.get("max", 1.2))))
            out = enhancer(out).enhance(rng.uniform(lo, hi))
    return out
