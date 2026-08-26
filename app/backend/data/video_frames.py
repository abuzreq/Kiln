"""Extract still frames from video files for dataset building."""
from pathlib import Path

from PIL import Image

from utils.logger import get_logger

log = get_logger("video_frames")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}


def extract_frames(video_path: str, out_dir: str, target_fps: float = 2.0, max_frames: int = 2000):
    """Sample frames from a video at ``target_fps`` and write them as PNGs."""
    import cv2

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / max(target_fps, 0.01))), 1)

    written, idx = 0, 0
    stem = video_path.stem
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(out_dir / f"{stem}_f{written:05d}.png")
            written += 1
        idx += 1
    cap.release()
    log.info("extracted %d frames from %s", written, video_path.name)
    return written
