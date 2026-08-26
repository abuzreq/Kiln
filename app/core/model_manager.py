"""Load, inspect, cache and list xurdif model checkpoints.

A checkpoint is a ``.pt`` dict bundling ``{step, model, ema, mults, mtype, pred}``.
The state dicts are those of a ``GaussianDiffusion`` module, so the UNet weights
are stored under a ``denoise_fn.`` prefix (alongside scheduler buffers we ignore).
"""
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.core.config import workspace
from app.core.engine.arch import DEFAULT_MTYPE, build_unet
from utils.exceptions import NotFoundError
from utils.logger import get_logger

log = get_logger("model_manager")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _torch():
    import torch  # deferred so the app boots without a torch install

    return torch


@dataclass
class ModelMeta:
    path: str
    name: str
    mtype: str
    mults: list
    pred: str
    step: int | None = None
    size_mb: float = 0.0
    source: str = "workspace"
    thumbnail: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "path": self.path,
            "name": self.name,
            "mtype": self.mtype,
            "mults": self.mults,
            "pred": self.pred,
            "step": self.step,
            "size_mb": round(self.size_mb, 2),
            "source": self.source,
            "thumbnail": self.thumbnail,
        }


# Reading a checkpoint's four metadata fields costs a full torch.load of the
# file, and /models re-reads every .pt on every request. Cache by (mtime, size)
# so a workspace with hundreds of training snapshots stays responsive; the
# signature invalidates itself whenever a file is rewritten.
_META_CACHE: "OrderedDict[str, tuple[tuple[float, int], ModelMeta]]" = OrderedDict()
_META_CACHE_MAX = 4096
_meta_lock = threading.Lock()


def read_meta(path: str | Path) -> ModelMeta:
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
            # hand back a copy — callers mutate .source / .thumbnail
            cached = replace(hit[1])
            cached.thumbnail = _sidecar_thumbnail(path)
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
    meta = ModelMeta(
        path=str(path),
        name=path.stem,
        mtype=mtype or DEFAULT_MTYPE,
        mults=list(mults) if mults is not None else [1, 2, 2, 2],
        pred=pred or "eps",
        step=data.get("step"),
        size_mb=path.stat().st_size / (1024 * 1024),
    )
    meta.thumbnail = _sidecar_thumbnail(path)
    with _meta_lock:
        _META_CACHE[key] = (sig, replace(meta))
        _META_CACHE.move_to_end(key)
        while len(_META_CACHE) > _META_CACHE_MAX:
            _META_CACHE.popitem(last=False)
    return meta


def _sidecar_thumbnail(path: Path) -> str | None:
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


class ModelManager:
    """Caches loaded UNets keyed by (path, device, ema), most-recent first.

    Bounded on purpose: every cached bundle pins a UNet on the GPU, so an
    unbounded cache meant VRAM only ever grew as you tried models out.
    """

    MAX_LOADED = 2

    def __init__(self):
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._lock = threading.Lock()

    # --- discovery ----------------------------------------------------
    def scan(self, extra_dirs: list[Path] | None = None) -> list[ModelMeta]:
        # (directory, source-label) pairs, scanned in priority order.
        repo_root = Path(__file__).resolve().parents[2]
        sources: list[tuple[Path, str]] = [
            (workspace.models, "workspace"),
            (repo_root / "models" / "pretrained", "pretrained"),
            (repo_root / "models" / "fine_tuned", "fine-tuned"),
            (repo_root / "vendor" / "xurdif" / "models", "vendored"),
        ]
        # workspace runs + any leftover per-project run folders
        runs = workspace.runs
        if runs.exists():
            for r in runs.iterdir():
                if r.is_dir():
                    sources.append((r, f"run:{r.name}"))
        if workspace.projects.exists():
            for proj in workspace.projects.iterdir():
                if proj.is_dir():
                    mdir = proj / "models"
                    if mdir.exists():
                        sources.append((mdir, "workspace"))
                    pruns = proj / "runs"
                    if pruns.exists():
                        for r in pruns.iterdir():
                            if r.is_dir():
                                sources.append((r, f"run:{r.name}"))
        if extra_dirs:
            sources.extend((d, "custom") for d in extra_dirs)

        seen: set[str] = set()
        out: list[ModelMeta] = []
        for d, label in sources:
            if not d.exists():
                continue
            for pt in sorted(d.glob("*.pt")):
                if str(pt) in seen:
                    continue
                seen.add(str(pt))
                try:
                    meta = read_meta(pt)
                    meta.source = label
                    thumb = _sidecar_thumbnail(pt)
                    if thumb:
                        meta.thumbnail = thumb
                    out.append(meta)
                except Exception as e:  # noqa: BLE001
                    log.warning("skip unreadable checkpoint %s: %s", pt, e)
        return out

    def scan_public(self) -> list[dict]:
        """Scan plus role (main vs training checkpoint) and starred flags."""
        from app.core import library

        stars = {library._norm_star_path(p) for p in library.list_stars()}
        out = []
        for m in self.scan():
            d = m.to_dict()
            is_ckpt = (m.source or "").startswith("run:")
            d["role"] = "checkpoint" if is_ckpt else "main"
            d["group"] = (m.source or "")[4:] if is_ckpt else None
            d["starred"] = library._norm_star_path(m.path) in stars
            card = library.read_card(m.path)
            if card:
                d["name"] = card.get("name") or d["name"]
                d["original_name"] = card.get("original_name") or d["name"]
                d["trained_as"] = card.get("trained_as") or []
                if d.get("step") is None and card.get("step") is not None:
                    d["step"] = card["step"]
            else:
                d["original_name"] = d["name"]
                d["trained_as"] = []
            d["renamable"] = library.can_rename(m.path)
            mtime = Path(m.path).stat().st_mtime
            if card and card.get("created_at"):
                d["created_at"] = card["created_at"]
            d["mtime"] = card.get("created_at") if card and card.get("created_at") else mtime
            out.append(d)
        return out

    # --- loading ------------------------------------------------------
    def load(self, path: str, device: str = "cpu", ema: bool = True):
        torch = _torch()
        key = f"{path}|{device}|{'ema' if ema else 'model'}"
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        meta = read_meta(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        which = "ema" if (ema and "ema" in data) else "model"
        raw = data[which]

        # strip the GaussianDiffusion 'denoise_fn.' prefix; drop scheduler buffers
        unet_state = {}
        for k, v in raw.items():
            if k.startswith("denoise_fn."):
                unet_state[k[len("denoise_fn."):]] = v

        model = build_unet(meta.mtype, meta.mults)
        missing, unexpected = model.load_state_dict(unet_state, strict=False)
        if missing:
            log.info("load %s: %d missing keys (ok if buffers)", Path(path).name, len(missing))
        model.eval().to(device)

        bundle = {"model": model, "meta": meta}
        with self._lock:
            self._cache[key] = bundle
            self._cache.move_to_end(key)
            evicted = False
            while len(self._cache) > self.MAX_LOADED:
                old_key, _ = self._cache.popitem(last=False)
                log.info("evicting cached model %s", old_key)
                evicted = True
        if evicted:
            _free_cuda()
        return bundle

    def evict(self, path: str):
        """Drop every cached bundle for one checkpoint (e.g. a stale merge preview)."""
        with self._lock:
            keys = [k for k in self._cache if k.split("|", 1)[0] == str(path)]
            for k in keys:
                del self._cache[k]
        if keys:
            _free_cuda()
        return len(keys)

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
        _free_cuda()


def _free_cuda():
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


manager = ModelManager()
