"""Draw Kiln's mark as the app icon: a multi-size .ico for Windows and a PNG.

    python scripts/make_icon.py

The geometry is the mark in app/frontend/src/components/KilnMark.jsx on the
favicon's dark tile (app/frontend/public/favicon.svg): the chunky variant with
one skip per level for the small sizes, the full one with two from 64 px up. Writes app/assets/kiln.ico (16-256 px, used by the window and
the shortcuts) and app/assets/kiln.png (256 px, for Linux). Rerun it after
changing the mark.
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "assets"

TILE = "#15171e"
BLOCK = "#f1ebe4"

# [x, y, w, h] on the 32-unit grid -- the same numbers as KilnMark.jsx.
BLOCKS = [
    (4, 22, 5, 7), (5.5, 15.6, 5, 5.4), (8, 10.4, 4.6, 4.2), (12, 6.4, 8, 3.2),
    (19.4, 10.4, 4.6, 4.2), (21.5, 15.6, 5, 5.4), (23, 22, 5, 7),
]
# (x0, x1, y, colour) for each horizontal skip.
SKIPS = [
    (12.6, 19.4, 12.5, "#ffe0a8"),
    (10.5, 21.5, 17.4, "#ffc06a"),
    (10.5, 21.5, 19.4, "#ffb347"),
    (9, 23, 24, "#ff8a52"),
    (9, 23, 26.6, "#ff6a2b"),
]
BLOCKS_SMALL = [
    (3.5, 21.5, 6, 7.5), (5.5, 15, 5.5, 5.6), (8, 9.6, 5, 4.6), (12, 5.6, 8, 3.6),
    (19, 9.6, 5, 4.6), (21, 15, 5.5, 5.6), (22.5, 21.5, 6, 7.5),
]
SKIPS_SMALL = [
    (13, 19, 12, "#ffd08a"),
    (11, 21, 17.8, "#ffb347"),
    (9.5, 22.5, 25.2, "#ff6a2b"),
]

SIZES = (16, 24, 32, 48, 64, 128, 256)
SUPERSAMPLE = 8


def _capsule(draw, x0, x1, y, width, color, k):
    """A horizontal stroke with round caps, as SVG's stroke-linecap=round."""
    r = width / 2
    draw.rounded_rectangle(
        [(x0 - r) * k, (y - r) * k, (x1 + r) * k, (y + r) * k], radius=r * k, fill=color)


def draw_mark(size: int) -> Image.Image:
    small = size < 64
    k = size * SUPERSAMPLE / 32
    img = Image.new("RGBA", (round(32 * k), round(32 * k)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, 32 * k - 1, 32 * k - 1], radius=7 * k, fill=TILE)
    for x, y, w, h in BLOCKS_SMALL if small else BLOCKS:
        d.rounded_rectangle([x * k, y * k, (x + w) * k, (y + h) * k], radius=0.8 * k, fill=BLOCK)
    # Solid skips, as the favicon: the app's dashes move, and standing still
    # their round caps run together into beads.
    for x0, x1, y, color in SKIPS_SMALL if small else SKIPS:
        _capsule(d, x0, x1, y, 1.7 if small else 1.1, color, k)
    return img.resize((size, size), Image.LANCZOS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = [draw_mark(s) for s in SIZES]
    largest = images[-1]
    # Pillow writes one frame per entry in sizes, taking each from the matching
    # image in append_images when there is one -- so every size keeps its own
    # drawing instead of being a resample of the 256 px one.
    largest.save(OUT / "kiln.ico", format="ICO", sizes=[(s, s) for s in SIZES],
                 append_images=images[:-1])
    largest.save(OUT / "kiln.png")
    print(f"Wrote {OUT / 'kiln.ico'} and {OUT / 'kiln.png'}")


if __name__ == "__main__":
    main()
