"""Augmentations: a fixed, seeded set of variations per image.

A recipe names the augmentations to use, how many variations to make of each
image (``augment_variants``) and a seed (``augment_seed``). ``plan`` turns that
into a list of small dicts -- what to do to one image -- drawn once, when
training starts:

    plan(["hflip", "rotate"], settings, 4, seed=0, index=12)
    -> [{}, {"hflip": True}, {"rotate": 90}, {"hflip": True, "rotate": 270}]

The first variation is always the framed image untouched, and no two are the
same. Because the draw is seeded with the image's index, the same recipe and
seed give the same dataset in both engines and on any machine. ``apply`` turns
one of these dicts into an image, which costs about a millisecond, so nothing is
written to disk and originals are never touched.
"""
import random

from PIL import Image, ImageEnhance

AUGMENTATIONS = ("hflip", "vflip", "rotate", "brightness", "contrast")

DEFAULT_SETTINGS = {
    "rotate": {"mode": "random", "angles": [90, 180, 270], "angle": 90},
    "brightness": {"min": 0.8, "max": 1.2},
    "contrast": {"min": 0.8, "max": 1.2},
}

# Brightness and contrast are continuous, so the number of distinct variations
# is only bounded by taste. More than this per image is a very long pass.
MAX_VARIANTS = 64


def _merge_settings(settings: dict | None) -> dict:
    out = {k: dict(v) for k, v in DEFAULT_SETTINGS.items()}
    if settings:
        for k, v in settings.items():
            if isinstance(v, dict) and k in out:
                out[k].update(v)
            else:
                out[k] = v
    return out


def _clean_ops(ops) -> list[str]:
    return [o for o in AUGMENTATIONS if o in (ops or [])]


def _angles(rot: dict) -> list[int]:
    """The non-zero angles a random-mode rotation can pick, deduped in order."""
    seen, out = set(), []
    for a in rot.get("angles") or DEFAULT_SETTINGS["rotate"]["angles"]:
        try:
            a = int(a) % 360
        except (TypeError, ValueError):
            continue
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _draw(ops: list[str], cfg: dict, rng: random.Random) -> dict:
    """One variation. Settings left at their no-op value are dropped, so the
    untouched image is always the empty dict and duplicates compare equal."""
    p = {}
    if "hflip" in ops and rng.random() < 0.5:
        p["hflip"] = True
    if "vflip" in ops and rng.random() < 0.5:
        p["vflip"] = True
    if "rotate" in ops:
        rot = cfg.get("rotate", DEFAULT_SETTINGS["rotate"])
        if rot.get("mode") == "fixed":
            try:
                angle = int(rot.get("angle", 90)) % 360
            except (TypeError, ValueError):
                angle = 90
            angle = angle if rng.random() < 0.5 else 0
        else:
            angle = rng.choice([0] + _angles(rot))
        if angle:
            p["rotate"] = angle
    for op in ("brightness", "contrast"):
        if op in ops:
            c = cfg.get(op, DEFAULT_SETTINGS[op])
            lo, hi = sorted((float(c.get("min", 0.8)), float(c.get("max", 1.2))))
            f = round(rng.uniform(lo, hi), 3)
            if f != 1.0:
                p[op] = f
    return p


def _key(params: dict):
    return tuple(sorted(params.items()))


def max_variants(ops, settings: dict | None = None) -> int:
    """How many distinct variations of one image these augmentations can make."""
    ops = _clean_ops(ops)
    if "brightness" in ops or "contrast" in ops:
        return MAX_VARIANTS
    cfg = _merge_settings(settings)
    n = 1
    if "hflip" in ops:
        n *= 2
    if "vflip" in ops:
        n *= 2
    if "rotate" in ops:
        rot = cfg.get("rotate", DEFAULT_SETTINGS["rotate"])
        n *= 2 if rot.get("mode") == "fixed" else 1 + len(_angles(rot))
    return max(1, min(n, MAX_VARIANTS))


def plan(ops, settings: dict | None, count: int, seed: int = 0, index: int = 0) -> list[dict]:
    """``count`` distinct variations for the image at ``index``.

    The first is the image itself. If the augmentations cannot make that many
    distinct variations the list is shorter; ``max_variants`` says in advance
    how many there are, and recipes are clamped to it.
    """
    ops = _clean_ops(ops)
    count = max(1, int(count))
    out: list[dict] = [{}]
    if not ops or count == 1:
        return out
    cfg = _merge_settings(settings)
    rng = random.Random(f"{seed}:{index}")
    seen = {()}
    tries = 50 * count + 100
    while len(out) < count and tries > 0:
        tries -= 1
        p = _draw(ops, cfg, rng)
        k = _key(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def apply(img: Image.Image, params: dict | None) -> Image.Image:
    """The image with one variation applied. An empty variation returns it as is."""
    out = img
    if not params:
        return out
    if params.get("hflip"):
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if params.get("vflip"):
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    angle = int(params.get("rotate") or 0) % 360
    if angle:
        out = out.rotate(angle, expand=False)
    for op, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast)):
        f = float(params.get(op) or 1.0)
        if f != 1.0:
            out = enhancer(out).enhance(f)
    return out


def label(params: dict | None) -> str:
    """A short name for one variation, for the Data screen's preview."""
    if not params:
        return "Original"
    parts = []
    if params.get("hflip"):
        parts.append("H flip")
    if params.get("vflip"):
        parts.append("V flip")
    if params.get("rotate"):
        parts.append(f"{int(params['rotate'])}°")
    if params.get("brightness"):
        parts.append(f"bright {float(params['brightness']):.2f}")
    if params.get("contrast"):
        parts.append(f"contrast {float(params['contrast']):.2f}")
    return " + ".join(parts) or "Original"
