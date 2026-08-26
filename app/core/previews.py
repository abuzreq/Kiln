"""Cached multi-seed preview images for the Start hub.

Every model in Kiln has at most one thumbnail. The hub wants several per model so
a card can cycle through them, so this module keeps a small cache of sampled
seeds and fills it in the background.

Two design notes worth knowing before changing anything here:

- Previews live under ``workspace.cache/previews/<key>``, not beside the ``.pt``.
  ``library.delete_model_files`` and ``library.rename_model`` only know about the
  single sibling thumbnail and the card, so extra files next to a checkpoint
  would leak on delete and orphan on rename. The key includes the file's mtime
  and size, so a retrained or overwritten model simply misses the cache instead
  of showing stale pictures.
- Generation deliberately does not check whether anything else is using the GPU.
  Sampling while a training run is active is a supported part of the workflow;
  the CUDA-OOM fallback in the perform routes is the intended pressure valve.
"""
import hashlib
import queue
import threading
from pathlib import Path

from app.core.config import workspace
from app.core.engine.sampler import SampleParams, sampler
from utils.imaging import build_card, save_with_params
from utils.logger import get_logger

log = get_logger("previews")

# A fixed ladder rather than a per-model one: every model is sampled from the
# same noise, so the cards are genuinely comparable side by side. 42 matches the
# seed the trainer already uses for snapshot thumbnails.
FIRST_SEED = 42
PREVIEW_COUNT = 6
# Small and quick — these are ~200px cards on a hub page, not final renders.
PREVIEW_SIZE = 256
PREVIEW_STEPS = 16
PREVIEW_SAMPLER = "unipc"
# One denoise loop yields this many distinct seeds; the sampler caps it at 4.
MAX_BATCH = 4


def seeds(count: int = PREVIEW_COUNT) -> list[int]:
    return [FIRST_SEED + i for i in range(count)]


def cache_key(model_path: str | Path) -> str | None:
    """Stable id for one checkpoint's previews, or None if it is gone.

    Includes mtime and size so the cache self-invalidates when a model is
    retrained or overwritten in place.
    """
    p = Path(model_path)
    try:
        st = p.stat()
    except OSError:
        return None
    raw = f"{p.resolve()}|{int(st.st_mtime)}|{st.st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def preview_dir(model_path: str | Path) -> Path | None:
    key = cache_key(model_path)
    return (workspace.cache / "previews" / key) if key else None


def list_previews(model_path: str | Path) -> list[str]:
    """Cached preview PNGs for one model, in seed order. Directory listing only."""
    d = preview_dir(model_path)
    if not d or not d.is_dir():
        return []
    found = []
    for seed in seeds():
        f = d / f"seed-{seed}.png"
        if f.exists():
            found.append(str(f))
    return found


def _batches(count: int) -> list[tuple[int, int]]:
    """Split a seed count into (start_seed, batch_size) runs of at most MAX_BATCH."""
    out = []
    remaining = count
    seed = FIRST_SEED
    while remaining > 0:
        n = min(MAX_BATCH, remaining)
        out.append((seed, n))
        seed += n
        remaining -= n
    return out


def generate(model_path: str, count: int = PREVIEW_COUNT) -> list[str]:
    """Render any missing previews for one model. Returns every cached path.

    Already-complete models return immediately without touching the GPU, so this
    is safe to call speculatively.
    """
    d = preview_dir(model_path)
    if d is None:
        return []
    existing = list_previews(model_path)
    if len(existing) >= count:
        return existing

    d.mkdir(parents=True, exist_ok=True)
    for start, bs in _batches(count):
        wanted = [start + i for i in range(bs)]
        if all((d / f"seed-{s}.png").exists() for s in wanted):
            continue
        params = SampleParams(
            model_path=model_path,
            ema=True,
            image_size=PREVIEW_SIZE,
            steps=PREVIEW_STEPS,
            sampler=PREVIEW_SAMPLER,
            seed=start,
            batch_size=bs,
            postproc={},
        )
        last = None
        for frame in sampler.run(params):
            last = frame
        if last is None:
            continue
        # batch_size > 1 puts every variation in images_pp; a batch of 1 has none.
        images = last.get("images_pp") or [last["image_pp"]]
        for i, img in enumerate(images):
            seed = start + i
            card = build_card(
                params, model_path=model_path, kind="model-preview",
                extra={"seed_index": seed - FIRST_SEED},
            )
            card["params"] = {**card.get("params", {}), "seed": seed, "batch_size": 1}
            save_with_params(img, d / f"seed-{seed}.png", card)
    out = list_previews(model_path)
    log.info("previews for %s: %d/%d", Path(model_path).name, len(out), count)
    return out


# --- background filling -----------------------------------------------
# One worker, not a pool: the model cache only keeps two UNets resident, so
# generating for several models at once would just thrash VRAM.
_q: "queue.Queue[str]" = queue.Queue()
_pending: set[str] = set()
_lock = threading.Lock()
_worker: threading.Thread | None = None


def _drain():
    while True:
        path = _q.get()
        try:
            generate(path)
        except Exception as e:  # noqa: BLE001
            log.warning("preview generation failed for %s: %s", Path(path).name, e)
        finally:
            with _lock:
                _pending.discard(path)
            _q.task_done()


def _ensure_worker():
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_drain, name="previews", daemon=True)
            _worker.start()


def enqueue(model_path: str) -> bool:
    """Queue one model for background generation. False if it is done or queued."""
    if not model_path or not Path(model_path).exists():
        return False
    if len(list_previews(model_path)) >= PREVIEW_COUNT:
        return False
    with _lock:
        if model_path in _pending:
            return False
        _pending.add(model_path)
    _ensure_worker()
    _q.put(model_path)
    return True


def pending_count() -> int:
    with _lock:
        return len(_pending)
