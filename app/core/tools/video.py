"""Frames to a video file, through the OpenCV Kiln already depends on.

H.264 in an MP4 first: it plays everywhere a video is likely to go next (a
browser, a phone, a social upload). OpenCV's wheels only carry an H.264 encoder
on some platforms, so a VP9 WebM -- which every browser also plays -- is the
fallback, and VP8 after that. Each attempt is read back before it counts:
OpenCV will happily open a writer that then writes nothing, or writes
something other than what was asked for.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from utils.logger import get_logger

log = get_logger("video")

# (container suffix, fourcc asked for, fourccs that count as success, label),
# in order of preference. What is read back is checked, not just that something
# was written: asked for a codec it lacks, OpenCV can fall back to the
# container's default -- MPEG-4 Part 2 in an MP4 -- which browsers do not play.
CODECS = (
    (".mp4", "avc1", {"h264", "avc1"}, "H.264"),
    (".webm", "VP90", {"vp90", "vp09"}, "VP9"),
    (".webm", "VP80", {"vp80", "vp08"}, "VP8"),
)
# Small samples are scaled up by a whole number with no smoothing before
# encoding: the codecs subsample colour and smear detail at 64 or 128 px,
# and a crisp pixel grid is what a low-res model actually made.
MIN_SIDE = 512


def prepare(frames: list[Image.Image]) -> list[Image.Image]:
    """Same size, RGB, scaled up to at least MIN_SIDE, even sides (yuv420)."""
    size = frames[0].size
    frames = [f.convert("RGB") if f.size == size else f.convert("RGB").resize(size, Image.LANCZOS)
              for f in frames]
    k = max(1, math.ceil(MIN_SIDE / max(size)))
    w, h = size[0] * k, size[1] * k
    even = (w - w % 2, h - h % 2)
    out = []
    for f in frames:
        if k > 1:
            f = f.resize((w, h), Image.NEAREST)
        if (w, h) != even:
            f = f.crop((0, 0, *even))
        out.append(f)
    return out


def probe(path: Path) -> tuple[str, int]:
    """(fourcc, lower-cased; frames that decode) of a written file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = code.to_bytes(4, "little").decode("latin-1").lower()
        n = 0
        while cap.isOpened():
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        return fourcc, n
    finally:
        cap.release()


def write_video(frames: list[Image.Image], stem: Path, fps: float) -> tuple[Path, str]:
    """Encode ``frames`` next to ``stem`` (no suffix); returns (path, codec label).

    Raises RuntimeError when no codec on this machine could write the file.
    """
    import cv2
    import numpy as np

    frames = prepare(frames)
    w, h = frames[0].size
    bgr = [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR) for f in frames]
    tried = []
    for suffix, fourcc, accept, label in CODECS:
        out = stem.parent / f"{stem.name}{suffix}"
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
        try:
            if not writer.isOpened():
                tried.append(label)
                continue
            for frame in bgr:
                writer.write(frame)
        finally:
            writer.release()
        got, n = probe(out) if out.exists() else ("", 0)
        if got in accept and n > 0:
            return out, label
        tried.append(label)
        out.unlink(missing_ok=True)
        log.info("asked for %s, got %r with %d frames; trying the next codec", label, got, n)
    raise RuntimeError(f"no video encoder available here (tried {', '.join(tried)})")
