"""Augmentations: every combination of the options chosen, per image.

Each augmentation offers a small, known set of options -- H flip is on or off,
quarter turns are 0/90/180/270, brightness is a few levels between min and max.
A recipe's augmentations are combined, so one image becomes the product of them:

    combinations(["hflip", "rotate"], settings)
    -> [{}, {"rotate": 90}, {"rotate": 180}, {"rotate": 270},
        {"hflip": True}, {"hflip": True, "rotate": 90}, ...]      # 2 x 4 = 8

The first combination is always the framed image untouched. Nothing is random
and nothing is written to disk: a combination is a few numbers, applied to the
framed image as it loads (``apply``), which costs about a millisecond. The same
recipe therefore gives the same training set every time, and ``count`` says how
big it is before a run starts.
"""
from itertools import product

from PIL import Image, ImageEnhance

AUGMENTATIONS = ("hflip", "vflip", "rotate", "brightness", "contrast")

# How many factors brightness or contrast may take between its min and max.
MAX_LEVELS = 6

DEFAULT_SETTINGS = {
    # "angles": the image itself plus each angle; "fixed": the image and one angle.
    "rotate": {"mode": "angles", "angles": [90, 180, 270], "angle": 90},
    "brightness": {"min": 0.8, "max": 1.2, "levels": 2},
    "contrast": {"min": 0.8, "max": 1.2, "levels": 2},
}


def _merge_settings(settings: dict | None) -> dict:
    out = {k: dict(v) for k, v in DEFAULT_SETTINGS.items()}
    if settings:
        for k, v in settings.items():
            if isinstance(v, dict) and k in out:
                out[k].update(v)
            else:
                out[k] = v
    # "random" was this mode's name while augmentations were drawn per step.
    if out["rotate"].get("mode") not in ("angles", "fixed"):
        out["rotate"]["mode"] = "angles"
    return out


def _clean_ops(ops) -> list[str]:
    return [o for o in AUGMENTATIONS if o in (ops or [])]


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _angles(rot: dict) -> list[int]:
    """The non-zero angles a rotation offers, deduped and in order."""
    seen, out = set(), []
    for a in rot.get("angles") or DEFAULT_SETTINGS["rotate"]["angles"]:
        a = _int(a, 0) % 360
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _factors(cfg: dict) -> list[float]:
    """The factors one enhancement offers: ``levels`` steps from min to max."""
    try:
        lo, hi = sorted((float(cfg.get("min", 0.8)), float(cfg.get("max", 1.2))))
    except (TypeError, ValueError):
        lo, hi = 0.8, 1.2
    n = max(1, min(MAX_LEVELS, _int(cfg.get("levels", 2), 2)))
    if n == 1 or hi == lo:
        vals = [round((lo + hi) / 2, 3)]
    else:
        vals = [round(lo + (hi - lo) * i / (n - 1), 3) for i in range(n)]
    # 1.0 changes nothing, and the untouched image is already in the set.
    return [v for v in dict.fromkeys(vals) if v != 1.0]


def clean_settings(settings: dict | None) -> dict:
    """Augmentation settings with every value in range, ready to store."""
    cfg = _merge_settings(settings)
    rot = cfg["rotate"]
    rot["angles"] = _angles(rot) or list(DEFAULT_SETTINGS["rotate"]["angles"])
    rot["angle"] = _int(rot.get("angle", 90), 90) % 360
    if rot["mode"] == "fixed" and not rot["angle"]:
        rot["angle"] = 90
    for op in ("brightness", "contrast"):
        c = cfg[op]
        try:
            lo, hi = sorted((float(c.get("min", 0.8)), float(c.get("max", 1.2))))
        except (TypeError, ValueError):
            lo, hi = 0.8, 1.2
        c["min"], c["max"] = round(max(0.1, lo), 3), round(min(3.0, hi), 3)
        c["levels"] = max(1, min(MAX_LEVELS, _int(c.get("levels", 2), 2)))
    return cfg


def options(ops, settings: dict | None = None) -> dict[str, list[dict]]:
    """What each chosen augmentation offers, including leaving the image alone."""
    cfg = _merge_settings(settings)
    out: dict[str, list[dict]] = {}
    for op in _clean_ops(ops):
        if op in ("hflip", "vflip"):
            out[op] = [{}, {op: True}]
        elif op == "rotate":
            rot = cfg["rotate"]
            angles = ([_int(rot.get("angle", 90), 90) % 360] if rot["mode"] == "fixed"
                      else _angles(rot))
            out[op] = [{}] + [{"rotate": a} for a in angles if a]
        else:
            out[op] = [{}] + [{op: f} for f in _factors(cfg[op])]
    return {op: opts for op, opts in out.items() if len(opts) > 1}


def counts(ops, settings: dict | None = None) -> dict[str, int]:
    """How many versions each augmentation offers, for the Data screen."""
    return {op: len(opts) for op, opts in options(ops, settings).items()}


def count(ops, settings: dict | None = None) -> int:
    """How many versions of one image this recipe makes, the original included."""
    n = 1
    for k in counts(ops, settings).values():
        n *= k
    return n


def combinations(ops, settings: dict | None = None) -> list[dict]:
    """Every combination of the chosen augmentations; the first changes nothing."""
    picks = list(options(ops, settings).values())
    if not picks:
        return [{}]
    out = []
    for combo in product(*picks):
        merged = {}
        for part in combo:
            merged.update(part)
        out.append(merged)
    return out


def apply(img: Image.Image, params: dict | None) -> Image.Image:
    """The image with one combination applied. An empty one returns it as is."""
    out = img
    if not params:
        return out
    if params.get("hflip"):
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if params.get("vflip"):
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    angle = _int(params.get("rotate"), 0) % 360
    if angle:
        out = out.rotate(angle, expand=False)
    for op, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast)):
        f = float(params.get(op) or 1.0)
        if f != 1.0:
            out = enhancer(out).enhance(f)
    return out


def label(params: dict | None) -> str:
    """A short name for one combination, for the Data screen's preview."""
    if not params:
        return "Original"
    parts = []
    if params.get("hflip"):
        parts.append("H flip")
    if params.get("vflip"):
        parts.append("V flip")
    if params.get("rotate"):
        parts.append(f"{_int(params['rotate'], 0)}°")
    if params.get("brightness"):
        parts.append(f"bright {float(params['brightness']):.2f}")
    if params.get("contrast"):
        parts.append(f"contrast {float(params['contrast']):.2f}")
    return " + ".join(parts) or "Original"
