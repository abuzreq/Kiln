"""Reading and loading xurdif checkpoints.

A checkpoint is a ``.pt`` dict bundling ``{step, model, ema, scaler, mults,
mtype, pred}``. The state dicts are those of a ``GaussianDiffusion`` module, so
the UNet weights are stored under a ``denoise_fn.`` prefix (alongside scheduler
buffers we ignore).

Moved from ``app/core/model_manager.py``; the caching/eviction layer stayed
there because it is engine-independent.
"""
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from app.core.backends.base import ModelDescriptor
from app.core.engine.arch import DEFAULT_MTYPE, build_unet
from utils.exceptions import NotFoundError
from utils.logger import get_logger

log = get_logger("xurdif.loader")

DENOISE_PREFIX = "denoise_fn."
DEFAULT_MULTS = [1, 2, 2, 2]


def _torch():
    import torch  # deferred so the app boots without a torch install

    return torch


# Reading a checkpoint's four metadata fields costs a full torch.load of the
# file, and /models re-reads every .pt on every request. Cache by (mtime, size)
# so a workspace with hundreds of training snapshots stays responsive; the
# signature invalidates itself whenever a file is rewritten.
_META_CACHE: "OrderedDict[str, tuple[tuple[float, int], ModelDescriptor]]" = OrderedDict()
_META_CACHE_MAX = 4096
_meta_lock = threading.Lock()


def size_multiple(mults) -> int:
    """The multiple every image dimension must divide by.

    Each level of ``mults`` halves the feature map and the skip connections
    concatenate, so a dimension that is not a multiple of ``2 ** len(mults)``
    fails on the concat.
    """
    return 2 ** len(list(mults or DEFAULT_MULTS))


def sidecar_thumbnail(path: Path) -> str | None:
    """A training run writes ``sample-N.png`` next to ``model-N.pt``; use it as a thumb."""
    stem = path.stem
    if stem.startswith("model-"):
        cand = path.with_name("sample-" + stem.split("-", 1)[1] + ".png")
        if cand.exists():
            return str(cand)
    for ext in (".png", ".jpg"):
        cand = path.with_suffix(ext)
        if cand.exists():
            return str(cand)
    return None


def ema_status(data: dict) -> str:
    """Does asking for EMA weights change anything for this checkpoint?

    xurdif always writes an ``ema`` slot, but its Trainer only starts averaging
    at ``step_start_ema = 2000`` -- before that it just copies the live weights.
    So an early snapshot has an EMA slot that is byte-identical to the model,
    and the toggle does nothing. Measured across a real library: every
    checkpoint had the slot, but a quarter of them were identical copies.
    """
    torch = _torch()
    has_model, has_ema = "model" in data, "ema" in data
    if not has_ema and not has_model:
        return "none"
    if not (has_model and has_ema):
        return "same"          # only one set of weights to hand back
    m, e = data["model"], data["ema"]
    shared = [k for k in m if k in e]
    if not shared:
        return "none"
    for k in shared:
        a, b = m[k], e[k]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        if a.shape != b.shape or not torch.equal(a, b):
            return "distinct"
    return "same"


def describe(path: str | Path) -> ModelDescriptor:
    torch = _torch()
    path = Path(path)
    if not path.exists():
        raise NotFoundError(f"model not found: {path}")

    st = path.stat()
    key, sig = str(path), (st.st_mtime, st.st_size)
    with _meta_lock:
        hit = _META_CACHE.get(key)
        if hit is not None and hit[0] == sig:
            _META_CACHE.move_to_end(key)
            # hand back a copy -- callers mutate .source / .thumbnail
            cached = replace(hit[1])
            cached.thumbnail = sidecar_thumbnail(path)
            return cached

    data = torch.load(str(path), map_location="cpu", weights_only=False)
    mults = data.get("mults")
    if mults is None and isinstance(data.get("opt"), object):
        mults = getattr(data.get("opt"), "mults", None)
    mtype = data.get("mtype")
    if mtype is None and data.get("opt") is not None:
        mtype = getattr(data["opt"], "model", None)
    pred = data.get("pred")
    if pred is None and data.get("opt") is not None:
        pred = getattr(data["opt"], "pred", None)
    resolved_mults = list(mults) if mults is not None else list(DEFAULT_MULTS)
    meta = ModelDescriptor(
        path=str(path),
        name=path.stem,
        mtype=mtype or DEFAULT_MTYPE,
        mults=resolved_mults,
        pred=pred or "eps",
        step=data.get("step"),
        size_mb=path.stat().st_size / (1024 * 1024),
        backend="xurdif",
        size_multiple=size_multiple(resolved_mults),
        ema=ema_status(data),
    )
    meta.thumbnail = sidecar_thumbnail(path)
    with _meta_lock:
        _META_CACHE[key] = (sig, replace(meta))
        _META_CACHE.move_to_end(key)
        while len(_META_CACHE) > _META_CACHE_MAX:
            _META_CACHE.popitem(last=False)
    return meta


def load_net(path: str, device: str = "cpu", ema: bool = True):
    """Build the UNet for one checkpoint and load its weights. Returns (net, descriptor)."""
    torch = _torch()
    meta = describe(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    which = "ema" if (ema and "ema" in data) else "model"
    if which not in data:
        # A checkpoint with only one of the two slots: use whichever is there
        # rather than raising over a preference we cannot honour.
        which = "ema" if "ema" in data else "model"
    raw = data[which]

    # strip the GaussianDiffusion 'denoise_fn.' prefix; drop scheduler buffers
    unet_state = {}
    for k, v in raw.items():
        if k.startswith(DENOISE_PREFIX):
            unet_state[k[len(DENOISE_PREFIX):]] = v

    model = build_unet(meta.mtype, meta.mults)
    missing, unexpected = model.load_state_dict(unet_state, strict=False)
    if missing:
        log.info("load %s: %d missing keys (ok if buffers)", Path(path).name, len(missing))
    model.eval().to(device)
    return model, meta


def scan(sources: "list[tuple[Path, str]]") -> list[ModelDescriptor]:
    """Every readable checkpoint under the given (directory, source-label) pairs."""
    seen: set[str] = set()
    out: list[ModelDescriptor] = []
    for d, label in sources:
        if not d.exists():
            continue
        for pt in sorted(d.glob("*.pt")):
            if str(pt) in seen:
                continue
            seen.add(str(pt))
            try:
                meta = describe(pt)
                meta.source = label
                thumb = sidecar_thumbnail(pt)
                if thumb:
                    meta.thumbnail = thumb
                out.append(meta)
            except Exception as e:  # noqa: BLE001
                log.warning("skip unreadable checkpoint %s: %s", pt, e)
    return out
