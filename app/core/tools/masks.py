"""Procedural black-and-white masks: shapes to compose over.

Randomize uses these to pick the areas it fills. Every mask is described by
a small *record* (kind, seed, coverage, soften, invert, params) and can be
regenerated from it at any size, which is what lets a roll's card stay small
enough to live inside the PNG: the card carries records, not rasters.

Kinds: ``blobs`` (thresholded fractal value noise), ``cells`` (jittered
voronoi, cells added until the coverage is reached), ``stripes`` (bands at an
angle with wobbly edges), ``shapes`` (scattered circles and star polygons,
optionally with holes) and ``split`` (two or three pieces along a wavy line).
White is where a fill happens; grey is partial strength, as a soft brush is.
"""
import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

KINDS = ("blobs", "cells", "stripes", "shapes", "split")
# How often Randomize reaches for each kind. Organic first; glyph-like last.
KIND_WEIGHTS = {"blobs": 0.32, "cells": 0.28, "split": 0.16, "stripes": 0.14, "shapes": 0.10}
MAX_SIDE = 1024


def _value_noise(rng: np.random.Generator, w: int, h: int, cells: int,
                 octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
    """Fractal value noise in [0, 1]: random grids, bicubic-upsampled and summed."""
    total = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        c = max(2, int(cells * (2 ** o)))
        grid = (rng.random((c + 1, c + 1)) * 255).astype(np.uint8)
        layer = Image.fromarray(grid).resize((w, h), Image.BICUBIC)
        total += amp * (np.asarray(layer, np.float32) / 255.0)
        norm += amp
        amp *= persistence
    return total / norm


def _noise_1d(rng: np.random.Generator, n: int, cells: int = 6, octaves: int = 3) -> np.ndarray:
    """A wobbly line in [-0.5, 0.5], for edges that are not straight."""
    total = np.zeros(n, np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        c = max(2, cells * (2 ** o))
        pts = rng.random(c + 1).astype(np.float32)
        x = np.linspace(0, c, n)
        total += amp * np.interp(x, np.arange(c + 1), pts)
        norm += amp
        amp *= 0.5
    return total / norm - 0.5


def _blobs(rng, w, h, coverage, params):
    n = _value_noise(rng, w, h, int(params.get("scale", 4)))
    thr = np.quantile(n, 1.0 - coverage)
    return (n >= thr).astype(np.uint8) * 255


def _cells(rng, w, h, coverage, params):
    count = int(params.get("cells", 14))
    jitter = float(params.get("jitter", 0.35))
    sites = rng.random((count, 2)).astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = xs / w
    ys = ys / h
    if jitter > 0:
        xs = xs + (_value_noise(rng, w, h, 4, octaves=2) - 0.5) * 0.12 * jitter
        ys = ys + (_value_noise(rng, w, h, 4, octaves=2) - 0.5) * 0.12 * jitter
    d = (xs[..., None] - sites[:, 0]) ** 2 + (ys[..., None] - sites[:, 1]) ** 2
    label = d.argmin(axis=-1)
    order = rng.permutation(count)
    out = np.zeros((h, w), bool)
    target = coverage * w * h
    for i in order:
        out |= label == i
        if out.sum() >= target:
            break
    return out.astype(np.uint8) * 255


def _stripes(rng, w, h, coverage, params):
    angle = math.radians(float(params.get("angle", rng.uniform(0, 180))))
    count = max(1, int(params.get("count", 5)))
    wobble = float(params.get("wobble", 0.5))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = xs / w - 0.5
    ys = ys / h - 0.5
    t = xs * math.cos(angle) + ys * math.sin(angle) + 0.5      # along the stripes
    u = -xs * math.sin(angle) + ys * math.cos(angle) + 0.5     # across them
    if wobble > 0:
        wave = _noise_1d(rng, 512, 5)
        t = t + np.interp(np.clip(u, 0, 1), np.linspace(0, 1, 512), wave) * 0.25 * wobble
    frac = (t * count) % 1.0
    return (frac < coverage).astype(np.uint8) * 255


def _star(cx, cy, r, points, inner, rot):
    pts = []
    for i in range(points * 2):
        a = rot + i * math.pi / points
        rr = r if i % 2 == 0 else r * inner
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _shapes(rng, w, h, coverage, params):
    count = int(params.get("count", 6))
    holes = bool(params.get("holes", True))
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    target = coverage * w * h
    base = min(w, h)
    placed = 0
    while placed < count * 3:
        placed += 1
        r = base * rng.uniform(0.08, 0.22)
        cx, cy = rng.uniform(r, w - r), rng.uniform(r, h - r)
        if rng.random() < 0.5:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        else:
            draw.polygon(_star(cx, cy, r, int(rng.integers(5, 10)), rng.uniform(0.45, 0.8),
                               rng.uniform(0, math.pi)), fill=255)
        if holes and rng.random() < 0.5:
            hr = r * rng.uniform(0.2, 0.45)
            hx, hy = cx + rng.uniform(-r * 0.3, r * 0.3), cy + rng.uniform(-r * 0.3, r * 0.3)
            draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=0)
        if np.asarray(img).astype(bool).sum() >= target:
            break
    return np.asarray(img, np.uint8)


def _split(rng, w, h, coverage, params):
    vertical = params.get("orientation", "v" if rng.random() < 0.5 else "h") == "v"
    pieces = max(2, min(3, int(params.get("pieces", 2))))
    wave = float(params.get("wave", 0.6))
    n = w if vertical else h
    along = h if vertical else w
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pos = (xs / w) if vertical else (ys / h)            # 0..1 across the cut
    coord = (ys / h) if vertical else (xs / w)          # 0..1 along the cut
    label = np.zeros((h, w), np.int32)
    for k in range(1, pieces):
        edge = k / pieces + np.interp(coord, np.linspace(0, 1, 256),
                                      _noise_1d(rng, 256, 3)) * 0.35 * wave / pieces
        label += (pos > edge).astype(np.int32)
    white = max(1, min(pieces - 1, int(round(coverage * pieces))))
    chosen = rng.choice(pieces, size=white, replace=False)
    return np.isin(label, chosen).astype(np.uint8) * 255


_GEN = {"blobs": _blobs, "cells": _cells, "stripes": _stripes, "shapes": _shapes, "split": _split}


def generate(kind: str, width: int, height: int, seed: int = 0, coverage: float = 0.3,
             soften: float = 0.0, invert: bool = False, params: dict | None = None) -> Image.Image:
    """One mask as a PIL ``L`` image of the requested size."""
    if kind not in _GEN:
        raise ValueError(f"unknown mask kind: {kind}")
    coverage = float(min(0.95, max(0.02, coverage)))
    scale = min(1.0, MAX_SIDE / max(width, height))
    w, h = max(8, int(width * scale)), max(8, int(height * scale))
    rng = np.random.default_rng(int(seed))
    arr = _GEN[kind](rng, w, h, coverage, params or {})
    img = Image.fromarray(arr, "L")
    if invert:
        img = ImageOps.invert(img)
    if soften > 0:
        img = img.filter(ImageFilter.GaussianBlur(float(soften) * scale))
    if (w, h) != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    return img


def from_record(record: dict, width: int, height: int) -> Image.Image:
    return generate(record["kind"], width, height, seed=record.get("seed", 0),
                    coverage=record.get("coverage", 0.3), soften=record.get("soften", 0.0),
                    invert=record.get("invert", False), params=record.get("params") or {})


def random_record(rng, coverage=(0.15, 0.45), soften=(0.0, 12.0)) -> dict:
    """A record Randomize would roll; ``rng`` is a ``random.Random``."""
    kind = rng.choices(list(KIND_WEIGHTS), list(KIND_WEIGHTS.values()))[0]
    params = {}
    if kind == "blobs":
        params = {"scale": rng.randint(3, 6)}
    elif kind == "cells":
        params = {"cells": rng.randint(8, 24), "jitter": round(rng.uniform(0.2, 0.7), 2)}
    elif kind == "stripes":
        params = {"angle": rng.randint(0, 179), "count": rng.randint(3, 9),
                  "wobble": round(rng.uniform(0.2, 0.9), 2)}
    elif kind == "shapes":
        params = {"count": rng.randint(3, 9), "holes": rng.random() < 0.6}
    elif kind == "split":
        params = {"orientation": rng.choice(["h", "v"]), "pieces": rng.choice([2, 2, 3]),
                  "wave": round(rng.uniform(0.3, 0.9), 2)}
    return {
        "kind": kind, "seed": rng.randint(0, 2 ** 31 - 1),
        "coverage": round(rng.uniform(*coverage), 3),
        "soften": round(rng.uniform(*soften), 1),
        "invert": False, "params": params,
    }


def carve(mask: Image.Image, taken: Image.Image) -> Image.Image:
    """``mask`` minus the area ``taken`` already covers."""
    return ImageChops.multiply(mask, ImageOps.invert(taken.convert("L")))


def coverage_of(mask: Image.Image) -> float:
    a = np.asarray(mask.convert("L"), np.float32) / 255.0
    return float(a.mean())
