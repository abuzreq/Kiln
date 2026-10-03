"""Lightweight super-resolution / upscaling, and downscaling.

Kiln targets small, private datasets and ships no heavy SR network. This provides
a dependency-free high-quality Lanczos upscale with optional unsharp masking,
which is a reliable finishing step for compact-model outputs. A factor below 1
shrinks instead: Lanczos widens its kernel when reducing, so fine detail is
averaged away rather than aliased into moire.
"""
from PIL import Image, ImageFilter


def upscale(img: Image.Image, factor: float = 2, sharpen: float = 0.0) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    out = img.resize(size, Image.LANCZOS)
    if sharpen and sharpen > 0:
        percent = int(max(0.0, min(sharpen, 3.0)) * 90)
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=2))
    return out
